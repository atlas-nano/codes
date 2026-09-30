// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation. Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software. This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
-------------------------------------------------------------------------*/

/* ----------------------------------------------------------------------
   Contributing author: Stan Moore (Sandia)
-------------------------------------------------------------------------*/

#include "fix_acks2_sam.h"

#include "atom.h"
#include "citeme.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "pair.h"
#include "text_file_reader.h"
#include "update.h"

#include "math_special.h"   // core: square() (replaces reaxff_defs.h SQR)

// SQR formerly came from reaxff_defs.h (via reaxff_api.h); map it to the core
// MathSpecial helper. DANGER_ZONE is an inlined default (drops the REAXFF dep).
#define SQR(x) MathSpecial::square(x)
static constexpr double DANGER_ZONE = 0.90;

#include <cmath>
#include <cstring>
#include <exception>
#include <algorithm>
#include <vector>

using namespace LAMMPS_NS;
using namespace FixConst;

static const char cite_fix_acks2_reax[] =
  "fix acks2/reaxff command: https://doi.org/10.1137/18M1224684\n\n"
  "@Article{O'Hearn2020,\n"
  " author = {K. A. {O'Hearn} and A. Alperen and H. M. Aktulga},\n"
  " title = {Fast Solvers for Charge Distribution Models on Shared Memory Platforms},\n"
  " journal = {SIAM J.\\ Sci.\\ Comput.},\n"
  " year = 2020,\n"
  " volume = 42,\n"
  " number = 1,\n"
  " pages = {1--22}\n"
  "}\n\n";

/* ----------------------------------------------------------------------*/

FixACKS2Sam::FixACKS2Sam(LAMMPS *lmp, int narg, char **arg) :
  FixQEqBaseSam(lmp, narg, arg)
{
  bcut = nullptr;

  X_diag = nullptr;
  Xdia_inv = nullptr;

  // BiCGStab
  g = nullptr;
  q_hat = nullptr;
  r_hat = nullptr;
  y = nullptr;
  z = nullptr;

  // X matrix
  X.firstnbr = nullptr;
  X.numnbrs = nullptr;
  X.jlist = nullptr;
  X.val = nullptr;

  // Update comm sizes for this fix
  comm_forward = comm_reverse = 2;
  // : exchange payload = s_hist + s_hist_X (nprev each) + q_hist (6, ASPC) — see pack_exchange.
  maxexchange = 2*nprev + 6;

  s_hist_X = s_hist_last = nullptr;

  last_rows_rank = 0;
  last_rows_flag = (comm->me == last_rows_rank);

  if (lmp->citeme) lmp->citeme->add(cite_fix_acks2_reax);
}

/* ----------------------------------------------------------------------*/

FixACKS2Sam::~FixACKS2Sam()
{
  if (copymode) return;

  memory->destroy(bcut);

  if (!reaxflag)
    memory->destroy(bcut_acks2);

  memory->destroy(s_hist_X);
  memory->destroy(s_hist_last);

  FixACKS2Sam::deallocate_storage();
  FixACKS2Sam::deallocate_matrix();
}

/* ----------------------------------------------------------------------*/

void FixACKS2Sam::post_constructor()
{
  memory->create(s_hist_last,2,nprev,"acks2/reax:s_hist_last");
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < nprev; ++j)
      s_hist_last[i][j] = 0.0;

  grow_arrays(atom->nmax);
  for (int i = 0; i < atom->nmax; i++)
    for (int j = 0; j < nprev; ++j)
      s_hist[i][j] = s_hist_X[i][j] = 0.0;

  pertype_parameters(pertype_option);
}

/* ----------------------------------------------------------------------*/


/* ----------------------------------------------------------------------*/

void FixACKS2Sam::allocate_storage()
{
  nmax = atom->nmax;
  NN = atom->nlocal + atom->nghost;
  const int size = nmax*2 + 2;

  // 0 to nn-1: owned atoms related to H matrix
  // nn to NN-1: ghost atoms related to H matrix
  // NN to NN+nn-1: owned atoms related to X matrix
  // NN+nn to 2*NN-1: ghost atoms related X matrix
  // 2*NN to 2*NN+1: last two rows, owned by proc 0

  memory->create(s,size,"acks2:s");
  memory->create(b_s,size,"acks2:b_s");

  memory->create(Hdia_inv,nmax,"acks2:Hdia_inv");
  memory->create(chi_field,nmax,"acks2:chi_field");

  memory->create(X_diag,nmax,"acks2:X_diag");
  memory->create(Xdia_inv,nmax,"acks2:Xdia_inv");

  memory->create(p,size,"acks2:p");
  memory->create(q,size,"acks2:q");
  memory->create(r,size,"acks2:r");
  memory->create(d,size,"acks2:d");

  memory->create(g,size,"acks2:g");
  memory->create(q_hat,size,"acks2:q_hat");
  memory->create(r_hat,size,"acks2:r_hat");
  memory->create(y,size,"acks2:y");
  memory->create(z,size,"acks2:z");

  // (uninit_reads_s86): start every augmented vector at zero. Each is created fresh here (on every run setup and
  // nmax growth) and glibc hands back recycled chunks, so a slot read before it is written -- as init_matvec's
  // constraint tail was on ranks != last_rows_rank (D2) -- saw whatever the previous holder left. D2 is fixed at its
  // site; this makes any future read of a never-written slot a deterministic 0. Nothing is kept across a reallocation.
  for (double *v : {s, b_s, p, q, r, d, g, q_hat, r_hat, y, z})
    for (int k = 0; k < size; k++) v[k] = 0.0;
}

/* ----------------------------------------------------------------------*/

void FixACKS2Sam::deallocate_storage()
{
  FixQEqBaseSam::deallocate_storage();

  memory->destroy(X_diag);
  memory->destroy(Xdia_inv);

  memory->destroy(g);
  memory->destroy(q_hat);
  memory->destroy(r_hat);
  memory->destroy(y);
  memory->destroy(z);
}

/* ----------------------------------------------------------------------*/

void FixACKS2Sam::allocate_matrix()
{
  FixQEqBaseSam::allocate_matrix();

  X.n = n_cap;
  X.m = m_cap;
  memory->create(X.firstnbr,n_cap,"acks2:X.firstnbr");
  memory->create(X.numnbrs,n_cap,"acks2:X.numnbrs");
  memory->create(X.jlist,m_cap,"acks2:X.jlist");
  memory->create(X.val,m_cap,"acks2:X.val");
}

/* ----------------------------------------------------------------------*/

void FixACKS2Sam::deallocate_matrix()
{
  FixQEqBaseSam::deallocate_matrix();

  memory->destroy(X.firstnbr);
  memory->destroy(X.numnbrs);
  memory->destroy(X.jlist);
  memory->destroy(X.val);
}

/* ----------------------------------------------------------------------*/

void FixACKS2Sam::init()
{
  FixQEqBaseSam::init();

  init_bondcut();
}

/* ----------------------------------------------------------------------*/

void FixACKS2Sam::init_bondcut()
{
  int i,j;
  int ntypes;

  ntypes = atom->ntypes;
  if (bcut == nullptr)
    memory->create(bcut,ntypes+1,ntypes+1,"acks2:bondcut");

  for (i = 1; i <= ntypes; ++i)
    for (j = 1; j <= ntypes; ++j) {
      bcut[i][j] = 0.5*(bcut_acks2[i] + bcut_acks2[j]);
    }
}

/* ----------------------------------------------------------------------*/

void FixACKS2Sam::init_storage()
{
  if (efield) get_chi_field();

  for (int ii = 0; ii < NN; ii++) {
    int i = ilist[ii];
    if (atom->mask[i] & groupbit) {
      b_s[i] = -chi[atom->type[i]];
      if (efield) b_s[i] -= chi_field[i];
      b_s[NN + i] = 0.0;
      s[i] = 0.0;
      s[NN + i] = 0.0;
    }
  }

  for (int i = 0; i < 2; i++) {
    b_s[2*NN + i] = 0.0;
    s[2*NN + i] = 0.0;
  }
}

/* ----------------------------------------------------------------------*/

void FixACKS2Sam::pre_force(int /*vflag*/)
{
  if (update->ntimestep % nevery) return;

  NN = atom->nlocal + atom->nghost;

  {
    nn = list->inum;
    ilist = list->ilist;
    numneigh = list->numneigh;
    firstneigh = list->firstneigh;
  }

  // grow arrays if necessary
  // need to be atom->nmax in length

  if (atom->nmax > nmax) reallocate_storage();
  if (atom->nlocal > n_cap*DANGER_ZONE || m_fill > m_cap*DANGER_ZONE)
    reallocate_matrix();

  if (efield) get_chi_field();

  init_matvec();   // sets s = cubic predictor of the augmented vector (charges s[i] DIRECTLY -> q-direct)

  // : ENFORCE the exclusion that aspc_setup() has always documented -- "NOT engaged for the ACKS2
  // saddle (it keeps aspc_on=0 -> BO)". It was a comment, never code:
  // `fix_modify <id> aspc N` sets aspc_on=1 regardless of which solve path will run, and reaching HERE
  // means the ACKS2 saddle. Measured on an ACKS2 saddle deck, ASPC buys 9.6% wall-clock for a 581x loss
  // of charge conservation (qtot RMS 1.4e-2 vs 2.5e-5) and ~30% corruption of the CT observable -- and
  // aspc_rtol has no usable window: the committed error is LINEAR in it, and tightening it toward the
  // deck's own solve tolerance drives the accept rate to zero. Refuse by default, loudly, once.
  if (aspc_on && !aspc_saddle_allow && !acks2_saddle_refused) {
    acks2_saddle_refused = 1;
    if (comm->me == 0)
      error->warning(FLERR, "samqeq: ASPC is not supported on the ACKS2 saddle ("
                            "bond-softness or the gas ACKS2 path) -- IGNORING it and running the exact"
                            "Born-Oppenheimer solve. The predictor does not satisfy the saddle's constraint"
                            "rows, so an accepted corrector commits a charge error linear in aspc_rtol."
                            "Drop `fix_modify {} aspc` from this deck; `fix_modify {} aspc saddle allow`"
                            "re-enables it behind the accept/reject gate if you are experimenting.", id, id);
  }
  if (aspc_on && !aspc_saddle_allow) {
    // exact BO, exactly as if aspc had never been requested
    matvecs = acks2_use_minres ? acks2_minres(b_s, s) : BiCGStab(b_s, s);
  } else if (aspc_on) {
    // ★ ASPC (#16) — OPERATIVE hook for fix qeq/sam. The ACKS2 augmented solve is ALREADY q-direct
    // (atom->q[i] = s[i]; init_matvec already predicts s), so the prototype's s/t-cancellation problem does
    // NOT apply. ASPC = cap BiCGStab at n_corr iterations + omega-damp toward the predictor, store corrected
    // s (calculate_Q backs it up). UNTESTED draft (session 21) — verify on a water-NVE energy-drift test.
    // NOTE: predictor currently = the existing cubic (4,-6,4,-1) coeffs (polynomial-exact); the ASPC-optimal
    // time-reversible coeffs (aspc_B) are an optional refinement if the cubic+omega drifts.
    const int sz = 2*NN + 2;
    double *spred = new double[sz];
    for (int k = 0; k < sz; ++k) spred[k] = s[k];     // predictor (pre-solve)
    int imax_save = imax; imax = aspc_ncorr;          // corrector: n_corr solver iterations
    // FIX (a): honour the selected saddle solver (MINRES under bondsoft) on the ASPC branch too.
    acks2_capped = 1;
    matvecs = acks2_use_minres ? acks2_minres(b_s, s) : BiCGStab(b_s, s);
    acks2_capped = 0;
    imax = imax_save;
    const double rresid = acks2_relresid;             // capped-corrector solve quality (globally reduced)
    const double w = aspc_omega;
    for (int ii = 0; ii < nn; ++ii) {                 // omega-mix the OWNED entries calculate_Q uses
      int i = ilist[ii];
      if (atom->mask[i] & groupbit) {
        s[i]      = w*s[i]      + (1.0 - w)*spred[i];        // charge block
        s[NN + i] = w*s[NN + i] + (1.0 - w)*spred[NN + i];   // X-aux block
      }
    }
    for (int i = 0; i < 2; ++i) s[2*NN + i] = w*s[2*NN + i] + (1.0 - w)*spred[2*NN + i];  // 2 constraint rows

    // ---- FIX (b): ACCEPT/REJECT GATE -- the contract the CG path has always had ------------------
    // Previously this branch COMMITTED the omega-mixed result of an n_corr-iteration solve
    // UNCONDITIONALLY. On the saddle, n_corr=2 iterations of an indefinite KKT
    // system reduces the residual hardly at all, so an unconverged correction was committed every step,
    // compounded, and the solve ran away: measured q_total = -190.9 e on a 1727-atom box whose true
    // total is -1 e, diverging by step 5-6 with per-atom charges still looking innocuous.
    // Mirrors FixQEqSam's projected-CG corrector: accept only if the correction is BOTH non-runaway AND
    // actually converged; otherwise fall through to the exact (uncapped) Born-Oppenheimer solve. The 4.0
    // e threshold is the CG path's own default (lr_qfreeze lives in the derived class, out of scope here).
    double mq = 0.0;
    for (int ii = 0; ii < nn; ++ii) {
      int i = ilist[ii];
      if (!(atom->mask[i] & groupbit)) continue;
      if (!std::isfinite(s[i])) { mq = 1.0e30; break; }
      if (fabs(s[i]) > mq) mq = fabs(s[i]);
    }
    MPI_Allreduce(MPI_IN_PLACE, &mq, 1, MPI_DOUBLE, MPI_MAX, world);

    // (adjudication D2): the gate must bound what is actually COMMITTED. Testing `rresid` alone was
    // wrong: residual(w*s + (1-w)*s_pred) = w*r_s + (1-w)*r_pred, and r_pred was never bounded. That hole
    // is what produced the "second defect" -- MINRES converged (r_s ~ 0), the gate passed, and the mix was
    // garbage. Re-capping MINRES only restored an accidental correlation between r_s and r_pred; this
    // closes it. Both residuals are already computed by both solvers, so this costs ZERO extra matvecs.
    const double rcommit = w*rresid + (1.0 - w)*acks2_relresid0;   // triangle-inequality bound on the mix
    if (mq < 4.0 && rcommit < aspc_rtol) {
      aspc_naccept++;                                 // SOLVER-DIAG acc/rej -- now live on the saddle too
    } else {
      aspc_nreject++;
      for (int k = 0; k < sz; ++k) s[k] = spred[k];   // discard the bad correction, restart from the predictor
      matvecs += acks2_use_minres ? acks2_minres(b_s, s)   // ...and solve it exactly (uncapped)
                                  : BiCGStab(b_s, s);
    }
    delete[] spred;
  } else {
    matvecs = acks2_use_minres ? acks2_minres(b_s, s)   // robust indefinite-saddle solver (bond-softness)
                               : BiCGStab(b_s, s);       // default BiCGStab (byte-identical); Born-Oppenheimer
  }

  // ---- : NEVER COMMIT A GROSSLY UNCONVERGED UNCAPPED SOLVE ------------------------------------------
  // BiCGStab's exit on exhausting imax was a WARNING followed by calculate_Q() on whatever iterate it held;
  // only the |q-q0|>50 e tripwire stood between that and the trajectory. Measured on a Li box
  // (mode2_libox, ion-state Li row): diag-BiCGStab at 1000 matvecs,
  // resid/b = 3.1, committed q_total +0.0796 e and PE +9956 eV (true: -194) as a "converged" step -- garbage
  // that passed the tripwire. The same operator converges in 14 matvecs under the ILU preconditioner (a
  // preconditioner cannot change the answer), so: escalate once via saddle_fallback(), then refuse. The 10x
  // band tolerates BiCGStab's early-exit quirk (resid 1.3e-5 at tol 1e-5 is a normal converged exit; see the
  // A7 note in BiCGStab) and catches every gross failure in the measured set (0.3, 3.1, 4e3).
  // The gate keys on acks2_exhausted (budget exhausted or breakdown) AND the residual: BiCGStab's early exit can
  // report 1.8e-4 at tol 1e-5 on a converged solve (17/34 goldens did, under ILU), which is not a failure.
  {
    const double gross = 10.0 * tolerance;
    if (acks2_exhausted && acks2_relresid > gross) {     // exhausted/breakdown AND grossly off (see acks2_exhausted)
      const double r_first = acks2_relresid;
      const int extra = saddle_fallback(b_s, s);
      if (extra > 0) matvecs += extra;
      if (extra < 0 || (acks2_exhausted && acks2_relresid > gross))
        error->all(FLERR, "samqeq: ACKS2 saddle solve did not converge at step {} (rel-residual {:.2e} after"
                          "{} matvecs, tolerance {:.1e}; {}) -- refusing to commit an unconverged charge set."
                          "Remedies: `fix_modify {} precond ilu`, a larger"
                          "`maxiter`. A monatomic-ion fragment has an almost empty X row, which makes this saddle"
                          "ill-conditioned under the diagonal preconditioner",
                   update->ntimestep, r_first, matvecs, tolerance,
                   extra < 0 ? "no further fallback available" : "the ILU retry did not converge either", id, id);
    }
  }

  // SOLVER-DIAG on the saddle path. The CG path has printed this since #28, but the ACKS2 saddle
  // printed nothing, so the campaign-wide "watch the ASPC acc/rej
  // counter" rule was unenforceable exactly where it was needed: the counters read 0/0, indistinguishable
  // from ASPC being off. A permanently-rejecting corrector (every step falling through to the full solve)
  // is now visible as acc/rej = 0/N.
  if (comm->me == 0 && (update->ntimestep == 0 || update->ntimestep % 200 == 0)) {
    if (aspc_on && aspc_saddle_allow)   // : a REFUSED aspc would print a permanent 0/0 and read as "dead ASPC"

      utils::logmesg(lmp, "samqeq SOLVER-DIAG step {}: {} {} matvecs, resid/b={:.2e}, ASPC acc/rej={}/{}\n",
                     update->ntimestep, acks2_use_minres ? "saddle-MINRES" : "saddle-BiCGStab",
                     matvecs, acks2_relresid, aspc_naccept, aspc_nreject);
    else
      utils::logmesg(lmp, "samqeq SOLVER-DIAG step {}: {} {} matvecs, resid/b={:.2e}\n",
                     update->ntimestep, acks2_use_minres ? "saddle-MINRES" : "saddle-BiCGStab",
                     matvecs, acks2_relresid);
  }

  calculate_Q();
}

/* ----------------------------------------------------------------------*/


/* ----------------------------------------------------------------------*/

void FixACKS2Sam::compute_X() // NOLINT
{
  int jnum;
  int i, j, ii, jj, flag;
  double dx, dy, dz, r_sqr;
  constexpr double SMALL = 0.0001;

  int *type = atom->type;
  tagint *tag = atom->tag;
  double **x = atom->x;
  int *mask = atom->mask;

  memset(X_diag,0,atom->nmax*sizeof(double));

  // fill in the X matrix
  m_fill = 0;
  r_sqr = 0;
  for (ii = 0; ii < nn; ii++) {
    i = ilist[ii];
    if (mask[i] & groupbit) {
      jlist = firstneigh[i];
      jnum = numneigh[i];
      X.firstnbr[i] = m_fill;

      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        j &= NEIGHMASK;

        dx = x[j][0] - x[i][0];
        dy = x[j][1] - x[i][1];
        dz = x[j][2] - x[i][2];
        r_sqr = SQR(dx) + SQR(dy) + SQR(dz);

        flag = 0;
        if (r_sqr <= SQR(swb)) {
          if (j < atom->nlocal) flag = 1;
          else if (tag[i] < tag[j]) flag = 1;
          else if (tag[i] == tag[j]) {
            if (dz > SMALL) flag = 1;
            else if (fabs(dz) < SMALL) {
              if (dy > SMALL) flag = 1;
              else if (fabs(dy) < SMALL && dx > SMALL)
                flag = 1;
            }
          }
        }

        if (flag) {
          double bcutoff = bcut[type[i]][type[j]];
          double bcutoff2 = bcutoff*bcutoff;
          if (r_sqr <= bcutoff2) {
            X.jlist[m_fill] = j;
            double X_val = calculate_X(sqrt(r_sqr), bcutoff);
            X.val[m_fill] = X_val;
            X_diag[i] -= X_val;
            X_diag[j] -= X_val;
            m_fill++;
          }
        }
      }

      X.numnbrs[i] = m_fill - X.firstnbr[i];
    }
  }

  if (m_fill >= X.m)
    error->all(FLERR,"Fix acks2/reaxff has insufficient ACKS2 X matrix size: m_fill={} X.m={}\n",m_fill,X.m);
}

/* ----------------------------------------------------------------------*/

double FixACKS2Sam::calculate_X(double r, double bcut)
{
  double d = r/bcut;
  double d3 = d*d*d;
  double omd = 1.0 - d;
  double omd2 = omd*omd;
  double omd6 = omd2*omd2*omd2;

  return bond_softness*d3*omd6;
}

/* ----------------------------------------------------------------------*/

/* ----------------------------------------------------------------------
   preconditioner apply: out = M^{-1} in. Default = diagonal Jacobi on the s
   (Hdia_inv) and u (Xdia_inv) blocks + identity on the 2 constraint rows.
   FixQEqSam overrides this to optionally use an ILU saddle preconditioner.
-------------------------------------------------------------------------*/
void FixACKS2Sam::precond_apply(double *in, double *out)
{
  for (int jj = 0; jj < nn; ++jj) {
    int j = ilist[jj];
    if (atom->mask[j] & groupbit) {
      out[j]      = in[j]      * Hdia_inv[j];
      out[NN + j] = in[NN + j] * Xdia_inv[j];
    }
  }
  if (last_rows_flag) {
    out[2*NN]     = in[2*NN];
    out[2*NN + 1] = in[2*NN + 1];
  }
}

/* ----------------------------------------------------------------------*/

int FixACKS2Sam::BiCGStab(double *b, double *x)
{
  int  i;
  double tmp, alpha, beta, omega, sigma, rho, rho_old, rnorm, bnorm;

  sparse_matvec_acks2(&H, &X, x, d);
  pack_flag = 1;
  comm->reverse_comm(this); //Coll_Vector(d);
  more_reverse_comm(d);

  vector_sum(r , 1.,  b, -1., d, nn);
  bnorm = parallel_norm(b, nn);
  rnorm = parallel_norm(r, nn);

  if (bnorm == 0.0) bnorm = 1.0;
  acks2_relresid0 = rnorm / bnorm;   // : residual of the ENTRY guess x0 (= the ASPC predictor)
  vector_copy(r_hat, r, nn);
  omega = 1.0;
  rho = 1.0;

  // #12: BiCGStab breakdown RESTART. The ACKS2 saddle is indefinite + moderately ill-conditioned;
  // the parallel-reduction order (np-dependent) can drive the shadow residual r_hat ⊥ r -> rho/omega = 0
  // -> the classic BiCGStab breakdown (seen at np=2 on benzene/Au; np=1/4/6 converge on the SAME operator).
  // The cure is to RE-ANCHOR r_hat = r and continue (not give up with a wrong x). Only triggers on exact
  // breakdown, so converging solves (all bit-tests, np=1) are byte-identical.
  // A7/R3 dimension note: rnorm/bnorm is a ratio of norms of the SAME augmented [q;u;rows] vector
  // space (residual vs RHS). The dominant chi-block scales together in numerator and denominator
  // (ratio unit-invariant), so unlike the base-class CG (e/sqrt(E)) no ev_scale factor is derivable
  // here; the sub-dominant charge-block admixture is a pre-existing norm inhomogeneity (present in
  // metal too). Left UNSCALED by design -- see "Applied ".
  int nrestart = 0; const int maxrestart = 8; bool fresh = true;
  for (i = 1; i < imax && rnorm / bnorm > tolerance; ++i) {
    rho = parallel_dot(r_hat, r, nn);
    if (rho == 0.0) {
      if (nrestart++ < maxrestart) { vector_copy(r_hat, r, nn); omega = 1.0; fresh = true; continue; }
      break;
    }

    if (!fresh) {
      beta = (rho / rho_old) * (alpha / omega);
      vector_sum(q , 1., p, -omega, z, nn);
      vector_sum(p , 1., r, beta, q, nn);
    } else {
      vector_copy(p, r, nn);
      fresh = false;
    }

    // pre-conditioning (diagonal by default; FixQEqSam may override with ILU)
    precond_apply(p, d);

    pack_flag = 1;
    comm->forward_comm(this); //Dist_vector(d);
    more_forward_comm(d);
    sparse_matvec_acks2(&H, &X, d, z);
    pack_flag = 2;
    comm->reverse_comm(this); //Coll_vector(z);
    more_reverse_comm(z);

    tmp = parallel_dot(r_hat, z, nn);
    alpha = rho / tmp;

    vector_sum(q , 1., r, -alpha, z, nn);

    tmp = parallel_dot(q, q, nn);

    // early convergence check
    // A7 note: tmp = |q|^2 (SQUARED, unnormalized) vs the linear-scale tolerance -- a pre-existing
    // dimensional oddity inherited from fix acks2/reaxff (already inconsistent in metal units).
    // Left verbatim: mechanical unit-scaling cannot fix it and any change breaks metal byte-id.
    if (tmp < tolerance) {
      vector_add(x, alpha, d, nn);
      break;
    }

    // pre-conditioning (diagonal by default; FixQEqSam may override with ILU)
    precond_apply(q, q_hat);

    pack_flag = 3;
    comm->forward_comm(this); //Dist_vector(q_hat);
    more_forward_comm(q_hat);
    sparse_matvec_acks2(&H, &X, q_hat, y);
    pack_flag = 3;
    comm->reverse_comm(this); //Dist_vector(y);
    more_reverse_comm(y);

    sigma = parallel_dot(y, q, nn);
    tmp = parallel_dot(y, y, nn);
    omega = sigma / tmp;

    vector_sum(g , alpha, d, omega, q_hat, nn);
    vector_add(x, 1., g, nn);
    vector_sum(r , 1., q, -omega, y, nn);

    rnorm = parallel_norm(r, nn);
    if (omega == 0) {
      if (nrestart++ < maxrestart) { vector_copy(r_hat, r, nn); rho = 1.0; fresh = true; continue; }
      break;
    }
    rho_old = rho;
  }

  acks2_relresid = rnorm / bnorm;   // : expose the solve quality so the ASPC corrector can be gated on it
  acks2_exhausted = (omega == 0 || rho == 0 || i >= imax) ? 1 : 0;   // : failure signature for the no-commit gate

  if (comm->me == 0) {
    if (omega == 0 || rho == 0) {
      error->warning(FLERR,"Fix acks2/reaxff BiCGStab numerical breakdown, omega = {:.8}, rho = {:.8}",
                      omega,rho);
    } else if (i >= imax && !acks2_capped) {
      // : inside the ASPC corrector the cap (imax = aspc_ncorr) is BY DESIGN, so "failed after n
      // iterations" every step is noise, not news -- the accept/reject gate is the real signal there.
      // Gated on acks2_capped, NOT on aspc_on, so a genuine non-convergence of the uncapped
      // fall-through solve still warns even with ASPC enabled.
      error->warning(FLERR,"Fix acks2/reaxff BiCGStab convergence failed after {} iterations"
                           "at step {}", i, update->ntimestep);
    }
  }

  return i;
}

/* ----------------------------------------------------------------------
   MINRES (Paige–Saunders) on the symmetric-INDEFINITE ACKS2 KKT saddle (augmented size 2NN+2). The saddle is
   structurally indefinite (the constraint rows + the negative-Laplacian X-block), so plain BiCGStab breaks down
   (rho=0) — exactly the failure seen for the bond-softness X-block. MINRES is
   designed for symmetric-indefinite systems and is the robust cure. Unpreconditioned (the diagonal Jacobi precon
   is itself indefinite via Xdia_inv<0). Adapted from FixQEqSam::qeq_minres, generalized to the augmented vector.
   Matvec is the VIRTUAL sparse_matvec_acks2 (FixQEqSam adds the reciprocal): input in `d` (forward pack_flag=1),
   output in `z` (reverse pack_flag=2). Work arrays (all 2NN+2): r1=r, r2=q, v=d, Av=z, w=p, w2=q_hat.
-------------------------------------------------------------------------*/
int FixACKS2Sam::acks2_minres(double *b, double *x)
{
  int *mask = atom->mask;
  // r0 = b - A x0 (x0 = warm-start guess). matvec input must be in d.
  for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit){ d[i]=x[i]; d[NN+i]=x[NN+i]; } }
  if (last_rows_flag){ d[2*NN]=x[2*NN]; d[2*NN+1]=x[2*NN+1]; }
  pack_flag=1; comm->forward_comm(this); more_forward_comm(d);
  sparse_matvec_acks2(&H, &X, d, z);
  pack_flag=2; comm->reverse_comm(this); more_reverse_comm(z);
  for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit){
    r[i]=b[i]-z[i];   q[i]=r[i];   p[i]=0.0;   q_hat[i]=0.0;
    r[NN+i]=b[NN+i]-z[NN+i]; q[NN+i]=r[NN+i]; p[NN+i]=0.0; q_hat[NN+i]=0.0; } }
  if (last_rows_flag){ for(int k=2*NN;k<2*NN+2;k++){ r[k]=b[k]-z[k]; q[k]=r[k]; p[k]=0.0; q_hat[k]=0.0; } }
  double beta1 = sqrt(parallel_dot(q, q, nn));
  // (adjudication D4): report/gate on ||r||/||b||, the SAME normalisation BiCGStab uses. phibar/beta1 is
  // ||r_k||/||r_0||, which for a warm start is a DIFFERENT quantity -- comparing both against one aspc_rtol
  // was an apples-to-oranges test, and the SOLVER-DIAG label "resid/b" was simply wrong for MINRES.
  // The loop's CONVERGENCE test below is deliberately left as phibar/beta1 so this stays byte-identical.
  double bnorm_m = parallel_norm(b, nn); if (bnorm_m == 0.0) bnorm_m = 1.0;
  if (beta1 == 0.0) { acks2_relresid = 0.0; acks2_relresid0 = 0.0; return 0; }   // exact warm start
  acks2_relresid0 = beta1 / bnorm_m;   // residual of the ENTRY guess x0 (= the ASPC predictor)
  double beta=beta1, oldb=0.0, dbar=0.0, epsln=0.0, phibar=beta1, cs=-1.0, sn=0.0;
  int it = 1;
  // : the 4000 FLOOR exists because the indefinite saddle needs more iters than the QEq H-block -- but it
  // must not override the ASPC corrector's DELIBERATE cap (imax = aspc_ncorr). It did, and the consequence was
  // subtle: MINRES ran to full convergence, its tiny residual passed the corrector's accept gate, and what was
  // then COMMITTED was the omega-MIXED vector w*solution + (1-w)*predictor -- far from the solution whenever the
  // predictor is cold. Measured: PE -250.7 at step 0 then +6049 by step 20, accepted every step, no guard tripped.
  // BiCGStab never showed this because it honours the cap, fails the gate, and falls through to the exact solve.
  const int cap = acks2_capped ? imax : ((imax < 4000) ? 4000 : imax);
  for (; it <= cap; it++) {
    double sca = 1.0/beta;                                            // v = r2/beta -> d (matvec input)
    for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit){ d[i]=sca*q[i]; d[NN+i]=sca*q[NN+i]; } }
    if (last_rows_flag){ d[2*NN]=sca*q[2*NN]; d[2*NN+1]=sca*q[2*NN+1]; }
    pack_flag=1; comm->forward_comm(this); more_forward_comm(d);      // y = A v -> z
    sparse_matvec_acks2(&H, &X, d, z);
    pack_flag=2; comm->reverse_comm(this); more_reverse_comm(z);
    if (it >= 2) { double f=beta/oldb;
      for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit){ z[i]-=f*r[i]; z[NN+i]-=f*r[NN+i]; } }
      if (last_rows_flag){ z[2*NN]-=f*r[2*NN]; z[2*NN+1]-=f*r[2*NN+1]; } }
    double alfa = parallel_dot(d, z, nn);
    double g = alfa/beta;
    for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit){ z[i]-=g*q[i]; z[NN+i]-=g*q[NN+i]; } }
    if (last_rows_flag){ z[2*NN]-=g*q[2*NN]; z[2*NN+1]-=g*q[2*NN+1]; }
    for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit){ r[i]=q[i]; q[i]=z[i]; r[NN+i]=q[NN+i]; q[NN+i]=z[NN+i]; } }
    if (last_rows_flag){ for(int k=2*NN;k<2*NN+2;k++){ r[k]=q[k]; q[k]=z[k]; } }
    oldb = beta;
    beta = sqrt(parallel_dot(q, q, nn));
    double oldeps=epsln;
    double delta=cs*dbar+sn*alfa;
    double gbar=sn*dbar-cs*alfa;
    epsln=sn*beta; dbar=-cs*beta;
    double gamma=sqrt(gbar*gbar+beta*beta); if(gamma<1.0e-300) gamma=1.0e-300;
    cs=gbar/gamma; sn=beta/gamma;
    double phi=cs*phibar; phibar=sn*phibar;
    double denom=1.0/gamma;                                           // x += phi*w ; w-recurrence (v still in d)
    for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit){
      double wk =(d[i]   -oldeps*q_hat[i]   -delta*p[i]   )*denom; q_hat[i]=p[i];     p[i]=wk;     x[i]+=phi*wk;
      double w2 =(d[NN+i]-oldeps*q_hat[NN+i]-delta*p[NN+i])*denom; q_hat[NN+i]=p[NN+i]; p[NN+i]=w2; x[NN+i]+=phi*w2; } }
    if (last_rows_flag){ for(int k=2*NN;k<2*NN+2;k++){
      double wk=(d[k]-oldeps*q_hat[k]-delta*p[k])*denom; q_hat[k]=p[k]; p[k]=wk; x[k]+=phi*wk; } }
    // A7/R3 dimension note: phibar/beta1 = ||resid||/||b|| over the SAME augmented [q;u;rows] vector space
    // (unpreconditioned) -- same "ratio unit-invariant, no ev_scale factor derivable" analysis as the
    // BiCGStab rnorm/bnorm criterion above. Left UNSCALED by design. 1e-300 = IEEE underflow guard.
    if (phibar/beta1 <= tolerance || beta <= 1.0e-300) break;
  }
  acks2_relresid = phibar / bnorm_m;  // : ||r||/||b||, same normalisation as BiCGStab (D4)
  acks2_exhausted = (it > cap) ? 1 : 0;   // : failure signature for the no-commit gate
  // (adjudication D1): gate on acks2_capped. Making MINRES honour the ASPC cap made `it > cap` true on
  // EVERY capped corrector call, so this warned ~97 times in 200 steps and tripped LAMMPS's 100-warning
  // budget -- silencing every genuine warning for the rest of the run. A by-design cap is not news.
  if (it > cap && !acks2_capped && comm->me == 0)
    error->warning(FLERR, "samqeq ACKS2 MINRES did not converge ({} iters, resid/b={:.2e}) at step {}",
                   it, phibar/beta1, update->ntimestep);
  return it;
}

/* ----------------------------------------------------------------------*/

void FixACKS2Sam::sparse_matvec_acks2(sparse_matrix *H, sparse_matrix *X, double *x, double *b)
{
  int i, j, itr_j;
  int ii;

  for (ii = 0; ii < nn; ++ii) {
    i = ilist[ii];
    if (atom->mask[i] & groupbit) {
      b[i] = (eta[atom->type[i]] + h_ridge + (onsite_extra ? onsite_extra[i] : 0.0)) * x[i];   // +h_ridge: SAMQEQ metal-limit Tikhonov [+ the quartic secant via onsite_extra]
      b[NN + i] = X_diag[i] * x[NN + i];
    }
  }

  for (i = atom->nlocal; i < NN; ++i) {
    if (atom->mask[i] & groupbit) {
      b[i] = 0;
      b[NN + i] = 0;
    }
  }
  // last two rows
  b[2*NN] = 0;
  b[2*NN + 1] = 0;

  for (ii = 0; ii < nn; ++ii) {
    i = ilist[ii];
    if (atom->mask[i] & groupbit) {
      // H Matrix
      for (itr_j=H->firstnbr[i]; itr_j<H->firstnbr[i]+H->numnbrs[i]; itr_j++) {
        j = H->jlist[itr_j];
        b[i] += H->val[itr_j] * x[j];
        b[j] += H->val[itr_j] * x[i];
      }

      // X Matrix
      for (itr_j=X->firstnbr[i]; itr_j<X->firstnbr[i]+X->numnbrs[i]; itr_j++) {
        j = X->jlist[itr_j];
        b[NN + i] += X->val[itr_j] * x[NN + j];
        b[NN + j] += X->val[itr_j] * x[NN + i];
      }

      // Identity Matrix
      b[NN + i] += x[i];
      b[i] += x[NN + i];

      // Second-to-last row/column
      b[2*NN] += x[NN + i];
      b[NN + i] += x[2*NN];

      // Last row/column
      b[2*NN + 1] += x[i];
      b[i] += x[2*NN + 1];
    }
  }

}

/* ----------------------------------------------------------------------*/


/* ----------------------------------------------------------------------*/

int FixACKS2Sam::pack_forward_comm(int n, int *list, double *buf,
                                  int /*pbc_flag*/, int * /*pbc*/)
{
  int m = 0;

  if (pack_flag == 1) {
    for(int i = 0; i < n; i++) {
      int j = list[i];
      buf[m++] = d[j];
      buf[m++] = d[NN+j];
    }
  } else if (pack_flag == 2) {
    for(int i = 0; i < n; i++) {
      int j = list[i];
      buf[m++] = s[j];
      buf[m++] = s[NN+j];
    }
  } else if (pack_flag == 3) {
    for(int i = 0; i < n; i++) {
      int j = list[i];
      buf[m++] = q_hat[j];
      buf[m++] = q_hat[NN+j];
    }
  }
  return m;
}

/* ----------------------------------------------------------------------*/

void FixACKS2Sam::unpack_forward_comm(int n, int first, double *buf)
{
  int i, m;

  int last = first + n;
  m = 0;

  if (pack_flag == 1) {
    for(i = first; i < last; i++) {
      d[i] = buf[m++];
      d[NN+i] = buf[m++];
    }
  } else if (pack_flag == 2) {
    for(i = first; i < last; i++) {
      s[i] = buf[m++];
      s[NN+i] = buf[m++];
    }
  } else if (pack_flag == 3) {
    for(i = first; i < last; i++) {
      q_hat[i] = buf[m++];
      q_hat[NN+i] = buf[m++];
    }
  }
}

/* ----------------------------------------------------------------------*/

int FixACKS2Sam::pack_reverse_comm(int n, int first, double *buf)
{
  int i, m;
  m = 0;
  int last = first + n;

  if (pack_flag == 1) {
    for(i = first; i < last; i++) {
      buf[m++] = d[i];
      buf[m++] = d[NN+i];
    }
  } else if (pack_flag == 2) {
    for(i = first; i < last; i++) {
      buf[m++] = z[i];
      buf[m++] = z[NN+i];
    }
  } else if (pack_flag == 3) {
    for(i = first; i < last; i++) {
      buf[m++] = y[i];
      buf[m++] = y[NN+i];
    }
  } else if (pack_flag == 4) {
    for(i = first; i < last; i++)
      buf[m++] = X_diag[i];
  }

  return m;
}

/* ----------------------------------------------------------------------*/

void FixACKS2Sam::unpack_reverse_comm(int n, int *list, double *buf)
{
  int j;
  int m = 0;
  if (pack_flag == 1) {
    for(int i = 0; i < n; i++) {
      j = list[i];
      d[j] += buf[m++];
      d[NN+j] += buf[m++];
    }
  } else if (pack_flag == 2) {
    for(int i = 0; i < n; i++) {
      j = list[i];
      z[j] += buf[m++];
      z[NN+j] += buf[m++];
    }
  } else if (pack_flag == 3) {
    for(int i = 0; i < n; i++) {
      j = list[i];
      y[j] += buf[m++];
      y[NN+j] += buf[m++];
    }
  } else if (pack_flag == 4) {
    for(int i = 0; i < n; i++) {
      j = list[i];
      X_diag[j] += buf[m++];
    }
  }
}

/* ----------------------------------------------------------------------
   one proc broadcasts last two rows of vector to everyone else
-------------------------------------------------------------------------*/

void FixACKS2Sam::more_forward_comm(double *vec)
{
  MPI_Bcast(&vec[2*NN],2,MPI_DOUBLE,last_rows_rank,world);
}

/* ----------------------------------------------------------------------
   reduce last two rows of vector and give to one proc
-------------------------------------------------------------------------*/

void FixACKS2Sam::more_reverse_comm(double *vec)
{
  if (last_rows_flag)
    MPI_Reduce(MPI_IN_PLACE,&vec[2*NN],2,MPI_DOUBLE,MPI_SUM,last_rows_rank,world);
  else
    MPI_Reduce(&vec[2*NN],nullptr,2,MPI_DOUBLE,MPI_SUM,last_rows_rank,world);
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
-------------------------------------------------------------------------*/

double FixACKS2Sam::memory_usage()
{
  double bytes;
  const double size = 2.0*nmax + 2.0;

  bytes = size*nprev * sizeof(double); // s_hist
  bytes += nmax*4.0 * sizeof(double); // storage
  bytes += size*11.0 * sizeof(double); // storage
  bytes += n_cap*4.0 * sizeof(int); // matrix...
  bytes += m_cap*2.0 * sizeof(int);
  bytes += m_cap*2.0 * sizeof(double);

  return bytes;
}

/* ----------------------------------------------------------------------
   allocate solution history array
-------------------------------------------------------------------------*/

void FixACKS2Sam::grow_arrays(int nmax)
{
  memory->grow(s_hist,nmax,nprev,"acks2:s_hist");
  memory->grow(s_hist_X,nmax,nprev,"acks2:s_hist_X");
  memory->grow(q_hist,nmax,6,"acks2:q_hist");   // ASPC (#16): q-history (this is the grow_arrays actually called
                                                // for FixQEqSam; the base FixQEqBaseSam::grow_arrays is bypassed)
}

/* ----------------------------------------------------------------------
   copy values within solution history array
-------------------------------------------------------------------------*/

void FixACKS2Sam::copy_arrays(int i, int j, int /*delflag*/)
{
  for (int m = 0; m < nprev; m++) {
    s_hist[j][m] = s_hist[i][m];
    s_hist_X[j][m] = s_hist_X[i][m];
  }
  for (int m = 0; m < 6; m++) q_hist[j][m] = q_hist[i][m];   // ASPC (#16): migrate q-history (atom sort/exchange)
}

/* ----------------------------------------------------------------------
   pack values in local atom-based array for exchange with another proc
-------------------------------------------------------------------------*/

int FixACKS2Sam::pack_exchange(int i, double *buf)
{
  for (int m = 0; m < nprev; m++) buf[m] = s_hist[i][m];
  for (int m = 0; m < nprev; m++) buf[nprev+m] = s_hist_X[i][m];
  for (int m = 0; m < 6; m++) buf[2*nprev+m] = q_hist[i][m];   // ASPC (#16): migrate q-history
  return nprev*2 + 6;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based array from exchange with another proc
-------------------------------------------------------------------------*/

int FixACKS2Sam::unpack_exchange(int nlocal, double *buf)
{
  for (int m = 0; m < nprev; m++) s_hist[nlocal][m] = buf[m];
  for (int m = 0; m < nprev; m++) s_hist_X[nlocal][m] = buf[nprev+m];
  for (int m = 0; m < 6; m++) q_hist[nlocal][m] = buf[2*nprev+m];   // ASPC (#16): migrate q-history
  return nprev*2 + 6;
}

/* ----------------------------------------------------------------------*/

double FixACKS2Sam::parallel_norm(double *v, int n)
{
  int  i;
  double my_sum, norm_sqr;

  int ii;

  my_sum = 0.0;
  norm_sqr = 0.0;
  for (ii = 0; ii < n; ++ii) {
    i = ilist[ii];
    if (atom->mask[i] & groupbit) {
      my_sum += SQR(v[i]);
      my_sum += SQR(v[NN+i]);
    }
  }

  // last two rows
  if (last_rows_flag) {
    my_sum += SQR(v[2*NN]);
    my_sum += SQR(v[2*NN + 1]);
  }

  MPI_Allreduce(&my_sum, &norm_sqr, 1, MPI_DOUBLE, MPI_SUM, world);

  return sqrt(norm_sqr);
}

/* ----------------------------------------------------------------------*/

double FixACKS2Sam::parallel_dot(double *v1, double *v2, int n)
{
  int  i;
  double my_dot, res;

  int ii;

  my_dot = 0.0;
  res = 0.0;
  for (ii = 0; ii < n; ++ii) {
    i = ilist[ii];
    if (atom->mask[i] & groupbit) {
      my_dot += v1[i] * v2[i];
      my_dot += v1[NN+i] * v2[NN+i];
    }
  }

  // last two rows
  if (last_rows_flag) {
    my_dot += v1[2*NN] * v2[2*NN];
    my_dot += v1[2*NN + 1] * v2[2*NN + 1];
  }

  MPI_Allreduce(&my_dot, &res, 1, MPI_DOUBLE, MPI_SUM, world);

  return res;
}

/* ----------------------------------------------------------------------*/

double FixACKS2Sam::parallel_vector_acc(double *v, int n)
{
  int  i;
  double my_acc, res;

  int ii;

  my_acc = 0.0;
  res = 0.0;
  for (ii = 0; ii < n; ++ii) {
    i = ilist[ii];
    if (atom->mask[i] & groupbit) {
      my_acc += v[i];
      my_acc += v[NN+i];
    }
  }

  // last two rows
  if (last_rows_flag) {
    my_acc += v[2*NN];
    my_acc += v[2*NN + 1];
  }

  MPI_Allreduce(&my_acc, &res, 1, MPI_DOUBLE, MPI_SUM, world);

  return res;
}

/* ----------------------------------------------------------------------*/

void FixACKS2Sam::vector_sum(double* dest, double c, double* v,
                                double d, double* y, int k)
{
  int kk;

  for (--k; k>=0; --k) {
    kk = ilist[k];
    if (atom->mask[kk] & groupbit) {
      dest[kk] = c * v[kk] + d * y[kk];
      dest[NN + kk] = c * v[NN + kk] + d * y[NN + kk];
    }
  }

  // last two rows
  if (last_rows_flag) {
    dest[2*NN] = c * v[2*NN] + d * y[2*NN];
    dest[2*NN + 1] = c * v[2*NN + 1] + d * y[2*NN + 1];
  }
}

/* ----------------------------------------------------------------------*/

void FixACKS2Sam::vector_add(double* dest, double c, double* v, int k)
{
  int kk;

  for (--k; k>=0; --k) {
    kk = ilist[k];
    if (atom->mask[kk] & groupbit) {
      dest[kk] += c * v[kk];
      dest[NN + kk] += c * v[NN + kk];
    }
  }

  // last two rows
  if (last_rows_flag) {
    dest[2*NN] += c * v[2*NN];
    dest[2*NN + 1] += c * v[2*NN + 1];
  }
}


/* ----------------------------------------------------------------------*/

void FixACKS2Sam::vector_copy(double* dest, double* v, int k)
{
  int kk;

  for (--k; k>=0; --k) {
    kk = ilist[k];
    if (atom->mask[kk] & groupbit) {
      dest[kk] = v[kk];
      dest[NN + kk] = v[NN + kk];
    }
  }

  // last two rows
  if (last_rows_flag) {
    dest[2*NN] = v[2*NN];
    dest[2*NN + 1] = v[2*NN + 1];
  }
}

