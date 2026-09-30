// clang-format off
/* ----------------------------------------------------------------------
   samQEq (fix qeq/sam): long-range Ewald (reciprocal-in-matvec) projected-CG plain-QEq path.
   Split out of fix_qeq_sam.cpp; all routines are members of class FixQEqSam
   (declared in fix_qeq_sam.h) -- a separate translation unit, not a new class.
-------------------------------------------------------------------------*/

#include "fix_qeq_sam.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "platform.h"   // (#30) host phase timers
#include "force.h"
#include "group.h"
#include "neighbor.h"   // ASPC (#16): neighbor->ncalls (rebuild detector for q_hist reset)
#include "memory.h"
#include "neigh_list.h"
#include "update.h"
#include "kspace.h"
#include "pppm_samqeq.h"
#include "math_const.h"
#include "modify.h"

#include <cmath>
#include <cstring>
#if defined(_OPENMP)
#include <omp.h>
#include <vector>
#include <limits>
#include <climits>
#endif

using namespace LAMMPS_NS;
using namespace FixConst;

/* ----------------------------------------------------------------------
   solve_diag_of: the lr solve-diagonal base at site i: eta[type] (gself-folded), plus the quartic secant under
   lr_quartic. apply_quartic_eta seeds eta_diag from the CHEMICAL eta0, so eta[t] + (eta_diag - eta0) carries the
   gself self-energy once and adds the secant.
-------------------------------------------------------------------------*/
double FixQEqSam::solve_diag_of(int i)
{
  const int t = atom->type[i];
  if (lr_quartic) return eta[t] + (eta_diag[i] - (eta0 ? eta0[t] : eta[t]));
  return eta[t];
}

/* ----------------------------------------------------------------------
   eself_diag_of: the HARMONIC on-site curvature the solve places on row i -- the part of solve_diag_of
   that multiplies s = q - q0 in the energy 1/2 D s^2 -- EXCLUDING the ridge (a regulariser,
   never energy) and the anharmonic secant (anh_energy_add reports that energy exactly). This is the only
   curvature compute_scalar may report: the gself-folded eta[type].
-------------------------------------------------------------------------*/
double FixQEqSam::eself_diag_of(int i, bool /*in_group*/)
{
  return eta[atom->type[i]];
}
using MathConst::MY_PIS;

#if defined(_OPENMP)
namespace {
/* ----------------------------------------------------------------------
   OpenMP threading -- P1 deterministic thread-ordered reduction helper.
   Partitions ii in [0,n) across `nth` OpenMP threads with schedule(static) -- a FIXED, contiguous
   chunking that depends only on (n,nth), never on runtime scheduling/timing -- each thread folds its
   own chunk into a private accumulator via `combine(acc, body(ii))`, then the nth per-thread partials
   are folded together in FIXED thread-index order 0..nth-1 (a plain serial loop, NOT
   `#pragma omp reduction`, whose pairwise/tree combine order is unspecified by the OpenMP standard and
   can vary between runs/compilers/thread counts). Net effect: bit-identical results run-to-run at a
   FIXED nth (the SPEC's determinism requirement) -- though NOT bit-identical to the nth==1 serial
   accumulation (FP reorder; same accepted standing policy as `-sf opt`, never for bit-tests). Only
   ever called from an `if (omp_go(nth, n))` branch (nth>1 AND n>=OMP_GRAIN, fix_qeq_sam.h) -- the
   gate-fails branch always takes the ORIGINAL untouched serial loop instead (see call sites below), so
   this template is never exercised by the 24-case suite (no `package omp`) NOR on small per-rank loops
   (< OMP_GRAIN iterations), where the fork/join overhead exceeds the work (measured : 639-atom cell,
   OMP4 36% slower without the grain gate).
   `combine` must be associative+commutative (sum or max); non-finite guards fold naturally into a max
   (returning a saturating sentinel, e.g. 1e30, from `body` for a non-finite element).
-------------------------------------------------------------------------*/
template <typename Body, typename Combine>
inline double omp_reduce(int n, int nth, double init, Combine combine, Body body)
{
  std::vector<double> parts(nth, init);
#pragma omp parallel num_threads(nth)
  {
    int tid = omp_get_thread_num();
    double local = init;
#pragma omp for schedule(static)
    for (int ii = 0; ii < n; ii++) local = combine(local, body(ii));
    parts[tid] = local;
  }
  double acc = init;
  for (int t = 0; t < nth; t++) acc = combine(acc, parts[t]);   // deterministic thread-ORDERED combine
  return acc;
}
static inline double omp_max(double a, double b) { return a > b ? a : b; }
static inline double omp_sum(double a, double b) { return a + b; }
}
#endif

/* ----------------------------------------------------------------------
   Route B: PLAIN-QEq long-range solve (not the ACKS2 saddle). compute_H supplies the erfc(g_ewald r)/r
   real-space split; qeq_solve does two SPD CG solves of H = η + full 1/r Coulomb (reciprocal + Ewald self
   added in qeq_matvec via the reliable pppm/samqeq compute_vector) and enforces per-molecule neutrality.
   The SPD operator is well-conditioned (diagonally dominant, η≈16) ⇒ CG converges fast where the ACKS2
   saddle stagnates. lr_ewald=0 -> delegate to the base (legacy/DSF) path unchanged.
-------------------------------------------------------------------------*/
void FixQEqSam::pre_force(int vflag)
{
  if (lr_xl) return;                                   // XL mode: charges propagated in initial/final_integrate
  if (!lr_ewald) {
    // BONDSOFT ON THE GAS PATH (2026-08-04). compute_X dispatches to
    // FixACKS2Sam::compute_X when lr_bondsoft is set, and that reads bcut[i][j].
    // The lazy allocation below (line ~163) sits BELOW this early return, so a
    // gas-path deck (lr_ewald=0) with `fix_modify <id> bondsoft <kappa> <bcut>`
    // reached compute_X with bcut == nullptr and segfaulted before printing
    // anything. Allocate here too. Guarded by lr_bondsoft, which is off by
    // default, so every other deck is byte-identical.
    if (lr_bondsoft && bcut == nullptr) setup_bondsoft_bcut();
    FixACKS2Sam::pre_force(vflag);
    return;
  }

  if (update->ntimestep % nevery) return;

  // #9: SKIP the re-solve at a run-BOUNDARY setup (ntimestep>0: 2nd+ run, restart, NVT->NVE, heating). The original
  // reason -- verlet.cpp runs fix setup_pre_force BEFORE kspace->setup(), so PPPM's grid was not finalized ("Out of
  // range atoms") -- is stale since : pppm/samqeq (and /kk) re-arm compute_step at init and call setup() from
  // start_compute, which is exactly the path the `reset_timestep 0` idiom has always taken. What remains is that a
  // cold re-solve can over-polarize a state the previous run held (#9). The previous run's CONVERGED charges in
  // atom->q are kept (init_storage preserves them): refresh ghost q for the setup force eval; the first pre_force
  // in the run loop re-solves. A fresh first setup has ntimestep==0 -> setup_resolve=0 -> normal solve.
  // `reset_timestep 0` turns the skip into a solve (cold CG from the stored q).
  // The skip is silent, and a `run 0` after read_restart (or after a
  // model change between runs) then reports forces on the STORED charges with no solve at all -- two campaign
  // single points looked like results with max|q_solve - q_raw| = 0. Say so: always for `run 0` (nothing will ever
  // re-solve), once per fix otherwise (the first in-loop pre_force re-solves; loop decks must not flood the log).
  if (setup_resolve && comm->me == 0 && (update->nsteps == 0 || !setup_skip_logged)) {
    utils::logmesg(lmp, "samqeq: setup solve SKIPPED at run-boundary step {} (2nd+ run / restart): the setup forces"
                        "use the STORED atom->q{} -- use `reset_timestep 0` before the run to force a solve\n",
                   update->ntimestep,
                   update->nsteps == 0 ? "; this `run 0` performs NO charge solve, so model changes since the"
                                         "charges were stored (gself, params, ...) are NOT reflected"
                                       : " (the first in-loop step re-solves)");
    if (update->nsteps != 0) setup_skip_logged = 1;
  }
  if (setup_resolve) { comm_v = atom->q; pack_flag = 6; comm->forward_comm(this);
                                         return; }

  lr_alpha = force->kspace->g_ewald;   // authoritative (g_ewald is set in kspace->setup, after fix init)
  NN = atom->nlocal + atom->nghost;
  nn = list->inum; ilist = list->ilist; numneigh = list->numneigh; firstneigh = list->firstneigh;
  if (atom->nmax > nmax) reallocate_storage();
  if (atom->nlocal > n_cap*0.90 || m_fill > m_cap*0.90) reallocate_matrix();   // 0.90 = reaxff DANGER_ZONE
  if (efield) get_chi_field();

  if (imax < 500) imax = 500;     // headroom for the long-range projected CG

  // A pinned recip_self is a number for one g_ewald. On a converged mesh the grid self-term tends to
  // 2 g_ewald/sqrt(pi) (within 0.05% on the shipped decks), so a pinned value far from it belongs to another
  // split: changing the cutoff or the kspace accuracy retunes g_ewald and leaves the pin behind (a pin of 0.37933
  // carried from 10 to 12 A cutoffs is 19% off and drives the liquid near-critical). Checked once per g_ewald.
  if (lr_self_pinned && lr_pppm && lr_alpha > 0.0 && lr_alpha != lr_pin_gchk) {
    lr_pin_gchk = lr_alpha;
    const double lim = 2.0*lr_alpha/MathConst::MY_PIS, rel = (lr_self_meas - lim)/lim;
    if (fabs(rel) > 0.02 && comm->me == 0)
      error->warning(FLERR, "samqeq: pinned recip_self {:.5f} differs by {:+.1f}% from 2 g_ewald/sqrt(pi) = {:.5f}"
                            "at the current g_ewald {:.6f}; a pin is valid only for the split it was measured at."
                            "Drop `fix_modify {} recip_self` to calibrate, or re-measure the pin",
                     lr_self_meas, 100.0*rel, lim, lr_alpha, id);
  }
  if (lr_self_peratom) { if (!lr_nrecip) compute_self_peratom_now(); }                       // R4: exact per-atom self, every solve
  else if (!lr_nrecip && !lr_self_pinned && (!lr_calibrated || grid_changed())) calibrate_recip_self();   // (re)measure recip_self iff the PPPM grid changed (skip in taper-cutoff mode: no reciprocal)
  if (lr_bondsoft) {              // STANDARD bond-softness ACKS2: run the saddle with the BASE bcut X-block
    if (bcut == nullptr) setup_bondsoft_bcut();   // (compute_X dispatches to FixACKS2Sam::compute_X)
    FixACKS2Sam::pre_force(vflag);// init_matvec -> Gaussian cutoff H (lr_nrecip) + bcut X-block -> gaps the soft

    return;                       // mode so the η≈J_shield(0) operator stays solvable at LOW η (target polarization)
  }
  if (host_H_needed())            // (#30): skipped when the solve will run entirely on device
    compute_H();                  // SPD H off-diagonals = erfc(g_ewald r)/r (Ewald real-space split)
  if (ionfield_flag) add_fixed_charge_field();                 // ionfield: fixed non-group ION charges -> field
                                                                             //   (no find_drude(): it hard-errors without `fix drude`)
  if (lr_quartic) compute_quartic_gate();   // (B') field gate for the quartic (geometry-only; no-op gate=1 when fld0<=0)
  // A4 (audit #5, RESOLVED BY MEASUREMENT): gate(x) is position-dependent but carries NO dgate/dx
  // force/virial. Measured cost (216-water Zn2+ cell, 4 ps NVE, fld0=8 vs 0): NVE drift +4.0 -> +6.9
  // kcal/mol/ps system-wide (~+70% over the endemic BO-FQ baseline; <~1 K across a 4 ps xPT window;
  // fully absorbed by any thermostat). DECISION: documented approximation — the analytic force needs
  // per-pair dJ/dr + a reciprocal gradient, unjustified at this cost. If fld0>0 NVE runs ever need
  // >few-ps conservation, revisit (decks: examples/drude_water/hbfit/in.gate_drift_*).
  if (lr_quartic) {
    // ASPC-quartic: once history is built, run the LIGHT path (eta_eff at the predictor + the regular capped-CG
    // ASPC corrector, inside qeq_solve) — the inner "regular CG" is ASPC-accelerated, ~2 matvecs/step vs the
    // ~128 of the full Picard SCF. Warmup and aspc-off use the full Picard SCF (quartic_scf); a corrector
    // reject does NOT re-run quartic_scf -- it falls through, INSIDE qeq_solve, to a single fixed-eta_diag
    // adaptive-ridge CG solve at the rejected predictor's eta_eff (see qeq_solve's ASPC branch below).
    if (aspc_on && aspc_have >= aspc_nhist) qeq_solve();
    else quartic_scf();
  }
  else {
  { if (th_on < 0) { const char *e = getenv("SAMQEQ_HOST_TIME"); th_on = (e && atoi(e)) ? 1 : 0; }
    const double t0 = (th_on == 1) ? platform::walltime() : 0.0;
    qeq_solve();                  // projected CG (η+full Coulomb, per-molecule-neutral subspace) -> atom->q
    if (th_on == 1) { th_solve += platform::walltime() - t0; th_steps++; } }
  }
}

/* ----------------------------------------------------------------------
   (B) QUARTIC near-crit cure: damped-Picard SCF. The on-site quartic makes the solve diagonal eta_eff(q)
   charge-dependent, so iterate { fill eta_diag(q) -> qeq_solve -> damp-mix -> converge }. ASPC is disabled
   inside (it predicts ACROSS MD steps; this SCF is WITHIN one step). qeq_solve warm-starts from atom->q, so
   the mixed charge seeds the next iteration (Picard). Converged when the in-group max |q_new - q_old| < tol.
   qsave is free scratch here (only used by calibrate_recip_self, which does not run during a solve).
-------------------------------------------------------------------------*/
void FixQEqSam::quartic_scf()
{
  int *type = atom->type, *mask = atom->mask; double *qa = atom->q;
  const double scf_tol = 1.0e-5;                 // charge-change convergence (Picard; CG tol is much tighter)
  // FULL cold-Picard SCF — the warmup (build q_hist) and ASPC-OFF path. Once aspc_have>=nhist the ASPC-quartic
  // LIGHT path (eta_eff at the predictor + the regular capped-CG corrector) runs inside qeq_solve instead
  // (pre_force dispatches there); see qeq_solve's ASPC branch, which ALSO owns the corrector-reject fallback
  // itself (a single fixed-eta_diag CG solve, NOT a re-entry into this Picard loop). So this exact SCF only
  // runs while history fills or when aspc is off. aspc_on=0 ⇒ byte-identical original.
  if (aspc_on && (aspc_nhist != aspc_korder + 2 || ngroup_fq <= 0)) aspc_setup();   // ensure coeffs/count for q_hist
  int as = aspc_on; aspc_on = 0;                 // no cross-step predictor inside the within-step SCF
  int it = 0, tot_mv = 0;
  double dq = 0.0;
  for (; it < quartic_niter; it++) {
    apply_quartic_eta();                          // eta_diag <- eta_eff(LAGGED q)
    for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit) qsave[i]=qa[i]; }   // q_old
  { if (th_on < 0) { const char *e = getenv("SAMQEQ_HOST_TIME"); th_on = (e && atoi(e)) ? 1 : 0; }
    const double t0 = (th_on == 1) ? platform::walltime() : 0.0;
    qeq_solve();                                  // linear solve at fixed eta_diag -> new qa (warm-started)
    if (th_on == 1) { th_solve += platform::walltime() - t0; th_steps++; } }
    tot_mv += matvecs;
    dq = 0.0;
    for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(!(mask[i]&groupbit)) continue;
      double d = fabs(qa[i]-qsave[i]); if (d>dq) dq=d;
      qa[i] = quartic_mix*qa[i] + (1.0-quartic_mix)*qsave[i]; }   // damped mix
    MPI_Allreduce(MPI_IN_PLACE,&dq,1,MPI_DOUBLE,MPI_MAX,world);
    comm_v=qa; pack_flag=6; comm->forward_comm(this);             // sync mixed q to ghosts for the next iter
    if (dq < scf_tol) break;
  }
  if (it >= quartic_niter && dq >= scf_tol && comm->me == 0 && warn_budget(nwarn_picard))   // : budgeted
    error->warning(FLERR, "samqeq: quartic Picard SCF did not converge in {} iterations (dq={:.3e}) at step {}"
                          "[{} so far]", quartic_niter, dq, update->ntimestep, nwarn_picard);
  aspc_on = as;

  // store the CONVERGED quartic charge in q_hist so the ASPC-quartic predictor can extrapolate it next step
  // (neutral perturbation qs = qa - q0). Done on EVERY quartic_scf call (warmup + aspc-off); the
  // corrector-reject fallback runs inside qeq_solve instead and deliberately does NOT push q_hist there
  // (aspc_reject) -- it is an SCF-inconsistent single solve, not a converged Picard sample.
  if (aspc_on) {
    for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(!(mask[i]&groupbit)) continue;
      double qsi = qa[i] - q0[type[i]];
      for (int j=aspc_nhist-1;j>0;j--) q_hist[i][j]=q_hist[i][j-1];
      q_hist[i][0]=qsi; }
    if (aspc_have < aspc_nhist) aspc_have++;
  }

  matvecs = tot_mv;
  if (comm->me == 0 && (update->ntimestep == 0 || update->ntimestep % 200 == 0))
    utils::logmesg(lmp, "samqeq QUARTIC-SCF step {}: {} Picard iters, {} matvecs (c4={:.4g} c={:.4g}) [full]\n",
                   update->ntimestep, it, tot_mv, quartic_c4, quartic_c);
}

/* ----------------------------------------------------------------------
   one-time calibration of the PPPM grid reciprocal self-term recip_self (= prec's i=i coefficient).
   Measure with a NEUTRAL +1/−1 pair (A,B) so there is NO tinfoil background contamination; the A–B cross
   term is the analytic reciprocal complement erf(αR)/R, so recip_self = prec[A] + erf(α R_AB)/R_AB.
   Subtracting recip_self cancels prec's i=i grid self -> QEq diagonal = eta (PD, force-consistent i≠j).
   Requires nn/ilist set and force->kspace->g_ewald valid (call after kspace setup).
   APPROXIMATION: the grid self is measured at ONE position (atom A) and applied to ALL atoms. PPPM's i=i
   coefficient has a small sub-grid-spacing position dependence (the grid breaks exact translational
   invariance below one spacing); at standard density/order (accuracy 1e-4, order 5) this spread is ≪ the
   self itself, so a single homogeneous value keeps the diagonal ≈ eta. A per-atom recip_self would be more
   rigorous (TODO; would change q* slightly ⇒ not byte-identical, needs golden regen). ---------------------*/
/* ----------------------------------------------------------------------
   : shared commit path for BOTH calibration routes -- guardrails, then store.
   Keeping this in one place means a future third route cannot skip the checks.
-------------------------------------------------------------------------*/
void FixQEqSam::finish_recip_self(double rs, int nprobe)
{
  // ---- GUARDRAILS on the calibration itself -----------------------------------------------
  // G1 CONTINUITY. A re-calibration at unchanged grid COUNTS must not move the self-term much: the
  // only legitimate driver there is the analytic dalpha response, measured at ~0.34% for a 0.34%
  // g_ewald move. A larger jump means a different probe pair was drawn, which silently re-tunes the
  // model (measured: 0.23 eV/e -> the liquid dipole falls 2.4655 -> 2.1513 D and stays there). This
  // is the check that would have caught the field failure at the boundary instead of letting it
  // produce 100 ps of depolarized liquid with a converged residual at every step.
  // (row 183): two holes closed. (i) The comparison is now alpha-SCALED: rs follows g_ewald (-> 2 g_ewald/sqrt(pi)
  // on a converged mesh), so the reference is lr_self_meas * g_now/g_prev -- the legitimate dalpha response no
  // longer eats the budget (recalib_on_grid_change WARNED "moved 0.73 % at an unchanged grid" on the host for a pure
  // alpha retune). (ii) A re-calibration across a mesh-COUNT change used to be exempt, and the R1 origin also
  // resets there, so nothing looked: the kk second-run defect (device weight table zeroed, recip_self 17 % low on
  // gself_cluster_energy, whose second init() re-derived the mesh 36^3 -> 40^3 from the solved charges' q2) was
  // ACCEPTED silently, while the same defect at unchanged counts (recip_self 3.95 % low) was refused. Across a count
  // change the jump now WARNS above 2 % -- not an abort, because a legitimate count change on a COARSE mesh can move
  // the grid self by more than a fine mesh's 0.003 % (0.33830 -> 0.33829 here) and an abort would be a new hard stop
  // for NPT/auto-mesh production decks; the empty-operator tripwire in measure_recip_probe() is what refuses the
  // defect itself. lr_gewald is still the PREVIOUS calibration's alpha here (updated below).
  if (lr_calibrated && lr_pppm && lr_self_meas != 0.0) {
    const double gnow = force->kspace ? force->kspace->g_ewald : lr_alpha;
    const double ascale = (lr_gewald > 0.0 && gnow > 0.0) ? gnow/lr_gewald : 1.0;
    const double ref = ascale*lr_self_meas;
    const double rel = fabs(rs - ref)/fabs(ref);
    const bool samecounts = (lr_nx == lr_pppm->nx_pppm && lr_ny == lr_pppm->ny_pppm && lr_nz == lr_pppm->nz_pppm);
    if (samecounts && rel > 0.02)
      error->all(FLERR, "samqeq: grid reciprocal self-term jumped {:.2f}% on re-calibration at an"
                        "UNCHANGED grid ({:.5f} -> {:.5f} raw; g_ewald x{:.5f} divided out). This silently"
                        "re-tunes the model: the self-term is subtracted from the solve diagonal, so a low value"
                        "over-stiffens it and de-polarizes the charges. Pin the calibration with"
                        "`fix_modify {} recip_self {:.6f}` to reproduce the previous operator, or investigate the"
                        "probe-pair selection / the kspace backend state.",
                 100.0*rel, lr_self_meas, rs, ascale, id, lr_self_meas);
    else if (samecounts && rel > 0.005 && comm->me == 0)
      error->warning(FLERR, "samqeq: grid reciprocal self-term moved {:.2f}% on re-calibration at an"
                            "unchanged grid ({:.5f} -> {:.5f} raw; g_ewald x{:.5f} divided out); treat results"
                            "either side as different models until checked", 100.0*rel, lr_self_meas, rs, ascale);
    else if (!samecounts && rel > 0.02 && comm->me == 0)
      error->warning(FLERR, "samqeq: grid reciprocal self-term moved {:.2f}% on re-calibration across a mesh"
                            "change {}x{}x{} -> {}x{}x{} ({:.5f} -> {:.5f} raw; g_ewald x{:.5f} divided out). A"
                            "mesh-count change alone moves it far less on a converged mesh -- check the kspace"
                            "state; treat results either side as different models until checked",
                     100.0*rel, lr_nx, lr_ny, lr_nz, lr_pppm->nx_pppm, lr_pppm->ny_pppm, lr_pppm->nz_pppm,
                     lr_self_meas, rs, ascale);
  }
  // G4 CROSS-RANK CONSENSUS. rs is Bcast above, so every rank must hold bit-identical values. This is
  // a permanent tripwire against reintroducing rank-local logic into the calibration path -- the exact
  // defect class this guard exists for (measured: recip_self 0.35516/0.35516/0.35311/0.33655 at
  // np 1/2/4/8 on ONE configuration, moving the liquid dipole 2.4377 -> 2.1055 D).
  {
    double rmin = rs, rmax = rs;
    MPI_Allreduce(MPI_IN_PLACE, &rmin, 1, MPI_DOUBLE, MPI_MIN, world);
    MPI_Allreduce(MPI_IN_PLACE, &rmax, 1, MPI_DOUBLE, MPI_MAX, world);
    if (rmin != rmax)
      error->all(FLERR, "samqeq: recip_self disagrees across ranks ({:.10g} vs {:.10g}) -- the"
                        "calibration must be a collective, decomposition-independent measurement", rmin, rmax);
  }

  lr_self_meas = rs; lr_calibrated = 1;
  if (lr_pppm) { lr_nx = lr_pppm->nx_pppm; lr_ny = lr_pppm->ny_pppm; lr_nz = lr_pppm->nz_pppm; }
  // ---- R1: DRIFT against the FIRST calibration at this mesh (REQUEST_recip_self_wander_guard) ---------
  // G1 above compares against the PREVIOUS calibration only, so a random walk of <2% steps passes it:
  // measured: 2 of 3 surviving replicas walked to -2.2% alive and one carried -2.17%
  // through a 2M-step NVT production (<q_min> 0.0036 e less negative than its clean twin). Under NPT the box
  // test re-arms this calibration on every >0.1% box-length change (300-3700 per segment); each LEGACY
  // single-pair re-draw lands anywhere in the per-atom spread (in production runs a sixth of the directories wandered
  // >2%, down to -7.6% in production). The legitimate spacing response at fixed counts is <~0.03% (519
  // re-calibrations over a P0 NPT segment), so the same 2% abort / 0.5% warn thresholds apply against the
  // ORIGIN. The origin resets only when the grid COUNTS change (a genuinely different mesh).
  {
    const int nx = lr_pppm ? lr_pppm->nx_pppm : -1, ny = lr_pppm ? lr_pppm->ny_pppm : -1,
              nz = lr_pppm ? lr_pppm->nz_pppm : -1;
    if (lr_self_first == 0.0 || lr_first_nx != nx || lr_first_ny != ny || lr_first_nz != nz) {
      lr_self_first = rs; lr_first_nx = nx; lr_first_ny = ny; lr_first_nz = nz;
      lr_first_step = update->ntimestep; lr_ncalib = 1; lr_drift_band = 0;
      lr_first_vol = domain->xprd*domain->yprd*domain->zprd;
    } else {
      lr_ncalib++;
      const double drift = fabs(rs - lr_self_first)/fabs(lr_self_first);
      // The origin is keyed on mesh COUNTS only, so a box that
      // genuinely changed size under a pinned `kspace_modify mesh` (an NPT arm that evaporated, V/V0 = 16-18)
      // trips this guard too. Stopping is still right -- the operator changed -- but the diagnosis must say so.
      const double vr = lr_first_vol > 0.0 ? domain->xprd*domain->yprd*domain->zprd/lr_first_vol : 1.0;
      const bool boxmoved = fabs(vr - 1.0) > 0.05;
      if (drift > 0.02) {
        if (boxmoved)
          error->all(FLERR, "samqeq: grid reciprocal self-term DRIFTED {:.2f}% from the FIRST calibration at this"
                            "{}x{}x{} mesh ({:.5f} raw at step {} -> {:.5f} now, calibration #{}; box volume x{:.3f}"
                            "since then) -- the BOX changed at fixed mesh counts, so the grid spacing and the operator"
                            "changed with it (a phase change, or a pinned `kspace_modify mesh` under NPT); this is not"
                            "a calibration random walk. If the volume change is intended, let the mesh follow the box"
                            "(drop the pinned mesh) or re-origin with `fix_modify {} recip_self <value>`.",
                     100.0*drift, nx, ny, nz, lr_self_first, lr_first_step, rs, lr_ncalib, vr, id);
        error->all(FLERR, "samqeq: grid reciprocal self-term DRIFTED {:.2f}% from the FIRST calibration at this"
                          "{}x{}x{} mesh ({:.5f} raw at step {} -> {:.5f} now, calibration #{}; box volume x{:.3f}"
                          "since then) -- a random walk of"
                          "sub-2% steps that the previous-value check cannot see; the operator is no longer the"
                          "one this run started with. Use `fix_modify {} recip_probes 16` (deterministic"
                          "multi-probe) or pin the origin with `fix_modify {} recip_self {:.6f}`.",
                   100.0*drift, nx, ny, nz, lr_self_first, lr_first_step, rs, lr_ncalib, vr, id, id, lr_self_first);
      }
      const int band = (int)(drift/0.005);
      if (band > lr_drift_band) {              // warn once per 0.5% band, not at every re-calibration
        lr_drift_band = band;
        if (comm->me == 0)
          error->warning(FLERR, "samqeq: grid reciprocal self-term has drifted {:.2f}% from the FIRST calibration"
                                "at this mesh ({:.5f} raw at step {} -> {:.5f}, calibration #{}; box volume x{:.3f}"
                                "since then{}); treat results either side as different models until checked",
                         100.0*drift, lr_self_first, lr_first_step, rs, lr_ncalib, vr,
                         boxmoved ? " -- the box changed at fixed mesh counts" : "");
      }
    }
    lr_self_step = update->ntimestep;
  }
  lr_gewald = force->kspace ? force->kspace->g_ewald : lr_alpha;
  lr_prd[0] = domain->xprd; lr_prd[1] = domain->yprd; lr_prd[2] = domain->zprd;
  if (comm->me==0) {
    // G3: disclose the spread. A wide spread means the homogeneous-scalar model is strained
    // on this grid and the single subtracted value is a poorer approximation for every atom.
    if (nprobe > 1)
      utils::logmesg(lmp, "samqeq: calibrated grid reciprocal self-term recip_self={:.5f} +/- {:.5f} (raw,"
                          "K={} deterministic probes) = {:.4f} eV/e (subtracted so the QEq diagonal stays = eta)\n",
                     lr_self_meas, lr_self_sigma, nprobe, lr_self_meas*force->qqrd2e/ev_scale);
    else
      utils::logmesg(lmp, "samqeq: calibrated grid reciprocal self-term recip_self={:.5f} (raw)"
                          "= {:.4f} eV/e (subtracted so the QEq diagonal stays = eta)\n",
                     lr_self_meas, lr_self_meas*force->qqrd2e/ev_scale);
    if (nprobe > 1 && lr_self_sigma*force->qqrd2e/ev_scale > 0.35)
      error->warning(FLERR, "samqeq: recip_self spread over probes is {:.3f} eV/e -- the homogeneous"
                            "self-term is a poor approximation on this grid; consider a finer mesh",
                     lr_self_sigma*force->qqrd2e/ev_scale);
  }
}

/* ----------------------------------------------------------------------
   : DETERMINISTIC multi-probe calibration of the grid reciprocal self-term.

   WHY. The legacy path below picks its probe pair as "the first two group atoms in ilist order,
   on the lowest rank owning two". ilist order depends on the domain decomposition and on the atom
   ordering of the data file, and the measured self-term carries a per-atom spread this file already
   documents at ~0.3-0.5 eV/e. So the SAME system runs a different Hamiltonian at different -np.
   Measured on a 256-molecule TIP4P-FQ liquid (one configuration, `run 0`, only -np varied):

       np 1 2 4 8
       rs 0.35516 0.35516 0.35311 0.33655 -> liquid dipole 2.4377/2.4377/2.3960/2.1055 D

   and permuting the atom order of the data file alone, at fixed np=8, moves rs 0.33655 -> 0.34339
   (dipole 2.1055 -> 2.2166 D). Sensitivity is ~1.24 D per eV/e of diagonal error.

   HOW. Probe k is the k-th LOWEST GLOBAL TAG in the fix group -- a property of the system, not of
   the decomposition. Its partner is the globally nearest owned group atom by minimum image, tie-broken
   by lower tag. A and B need NOT be co-resident: each owner sets its own charge, PPPM sums over all
   ranks' owned atoms, and only prec[A] is read on A's owner. That removes the co-residency
   requirement and with it the migration fallback that re-opened this bug at run boundaries.
   Averaging over K probes also centres the estimate in the per-atom spread instead of sampling one
   arbitrary point of it; the spread itself is reported so a strained homogeneous model is visible.

   STATUS: : the DEFAULT route (K=16, clamped to the group size). Validated -(NPT
   drift 0.004% vs legacy 0.6-7.6%). `fix_modify <id> recip_probes 0` restores the legacy single pair (the earlier
   goldens were regenerated at the flip). (row 182): floor 4 -> 2 atoms; a G2-window miss under the default K
   still falls back to the legacy pair, which is now tag-keyed and decomposition-independent (measure_recip_probe).
-------------------------------------------------------------------------*/
/* ----------------------------------------------------------------------
   (row 182): ONE co-residency-free pair measurement, shared by both routes.
   A = the group atom with global tag tA. B = tB when tB > 0 on entry (the STICKY pair on a re-calibration),
   else the nearest owned group atom to A by minimum image, tie-broken by LOWER TAG. hmax > 0 restricts the
   partner to the G2 window [0.1, hmax] (probe route); hmax <= 0 = no window (legacy route: the historical
   pair never had one). Each owner sets its own charge, PPPM sums over all ranks, only prec[A] is read on
   A's owner and B's owner publishes the raw dz_AB (EW3DC), so A and B may live on different ranks.
   Returns false when A has no admissible partner (charges are the caller's to restore); on success tB
   carries the partner and sk = prec[A] + erf(aR)/R - phi_slab(A). The body is the /probe loop,
   moved here unchanged, so the probe route is byte-identical.
   WHY (measured 2026-09-26, spcfq_gas_ewald with the orientation pinned by `rotate`): the legacy route
   picked "first two group atoms in ilist order on the lowest rank owning two" -- ilist order is the
   setup-sort BIN order and the owner is the decomposition -- so ONE configuration measured rs 0.32565
   (np 1, 2: an O-H pair) and 0.32548 (np 4, 1x2x2 grid: the H-H pair), q_O moving 7.0e-4 e, and
   REFUSED ("no rank owning two fix-group atoms") once the three atoms landed on three ranks. With
   recip_self pinned, np 1/2/4 were bit-identical: the self-term was the whole np-dependence. On the
   375-atom shield_special00 liquid (a G2 miss, hence legacy) `atom_modify sort 0` alone moved q by 0.056 e., ledger L52.
-------------------------------------------------------------------------*/
bool FixQEqSam::measure_recip_probe(tagint tA, tagint &tB, double hmax, double &sk)
{
  int *mask = atom->mask; double *qa = atom->q; double **x = atom->x;
  tagint *tag = atom->tag;
  const int nall = atom->nlocal + atom->nghost;
  for (int i=0;i<nall;i++) qa[i]=0.0;

  // locate A and publish its position
  int A=-1; for (int ii=0;ii<nn;ii++){ int i=ilist[ii]; if((mask[i]&groupbit)&&tag[i]==tA){A=i;break;} }
  int ownA_in = (A>=0 && A<atom->nlocal) ? comm->me : comm->nprocs, ownA;
  MPI_Allreduce(&ownA_in,&ownA,1,MPI_INT,MPI_MIN,world);
  if (ownA == comm->nprocs) error->all(FLERR, "samqeq: recip_self probe tag {} is owned by no rank", tA);
  double xA[3] = {0,0,0};
  if (comm->me==ownA) { xA[0]=x[A][0]; xA[1]=x[A][1]; xA[2]=x[A][2]; }
  MPI_Bcast(xA,3,MPI_DOUBLE,ownA,world);

  // partner: the pinned tag, or the nearest owned group atom to A, min-image, tie-broken by LOWER TAG
  struct { double d; int t; } cand, best;
  cand.d = 1.0e300; cand.t = INT_MAX;
  for (int i=0;i<atom->nlocal;i++) {
    if (!(mask[i]&groupbit) || tag[i]==tA || (tB > 0 && tag[i]!=tB)) continue;
    double dx=x[i][0]-xA[0], dy=x[i][1]-xA[1], dz=x[i][2]-xA[2];
    domain->minimum_image(FLERR,dx,dy,dz);
    double d = sqrt(dx*dx+dy*dy+dz*dz);
    if (hmax > 0.0 && (d < 0.1 || d > hmax)) continue;         // G2: probe-validity window (probe route only)
    if (d < cand.d || (d == cand.d && (int)tag[i] < cand.t)) { cand.d=d; cand.t=(int)tag[i]; }
  }
  MPI_Allreduce(&cand,&best,1,MPI_DOUBLE_INT,MPI_MINLOC,world);
  if (best.d > 1.0e299) return false;
  tB = (tagint)best.t;

  // both owners set their own charge; PPPM sums over all ranks' owned atoms, so A and B may live
  // on different ranks. This is what removes the legacy co-residency requirement.
  if (comm->me==ownA) qa[A] = 1.0;
  // : the partner's owner also publishes the pair's z-separation (raw, no z-wrap: slab PPPM requires a
  // non-periodic z) for the EXACT Yeh-Berkowitz removal below. Previously this route used the full distance
  // d^2 ("conservative: exact only for a z-aligned pair"), which over-subtracts 2pi/V (d^2 - dz^2) for every
  // in-plane pair -- a per-pair bias that the legacy dz^2 path never had (exposed by the default flip on
  // the slab goldens).
  double dzAB_loc = 0.0, dzAB = 0.0;
  for (int i=0;i<atom->nlocal;i++)
    if ((mask[i]&groupbit) && (int)tag[i]==best.t) { qa[i] = -1.0; dzAB_loc = x[i][2] - xA[2]; }
  MPI_Allreduce(&dzAB_loc,&dzAB,1,MPI_DOUBLE,MPI_SUM,world);   // exactly one rank owns B

  for (int i=0;i<atom->nlocal;i++) prec[i]=0.0;                 // E2: compute_vector writes/reads locals only
  eksp->compute_vector(prec, groupbit, groupbit, false);

  // (row 183) G5 EMPTY-OPERATOR TRIPWIRE. A +1 charge at A cannot produce an exactly zero reciprocal potential
  // at A: prec[A] == 0.0 means the kspace backend deposited nothing (measured: the device rho1d weight table zeroed by
  // PPPMKokkos::allocate() at a run boundary -> recip_self = erf(aR)/R, 17-66 % low, ACCEPTED because the mesh counts
  // had changed). The cheapest point to catch the whole "fresh zero View" class (u_brick #29, rho1d), once
  // per probe on both routes. Collective: reduced flag.
  {
    int empty_loc = (comm->me == ownA && prec[A] == 0.0) ? 1 : 0, empty = 0;
    MPI_Allreduce(&empty_loc, &empty, 1, MPI_INT, MPI_MAX, world);
    if (empty)
      error->all(FLERR, "samqeq: recip_self probe tag {}: the reciprocal potential at the probe atom is IDENTICALLY"
                        "ZERO for a +1/-1 pair -- the kspace backend returned an empty operator (zeroed or stale"
                        "device grid/weight state at a run boundary?); refusing to calibrate on it", tA);
  }

  double s = 0.0;
  if (comm->me==ownA) {
    double slabAB = 0.0;
    if (force->kspace && force->kspace->slabflag == 1) {
      // analytic Yeh-Berkowitz removal: phi_slab(A) = (2pi/V_slab) dz_AB^2 (a pair-GEOMETRY term, not grid self)
      double vslab = domain->xprd*domain->yprd*domain->zprd*force->kspace->slab_volfactor;
      slabAB = MathConst::MY_2PI/vslab * dzAB*dzAB;
    }
    s = prec[A] + erf(lr_alpha*best.d)/best.d - slabAB;
  }
  MPI_Bcast(&s,1,MPI_DOUBLE,ownA,world);
  sk = s;
  return true;
}

bool FixQEqSam::calibrate_recip_self_probes()
{
  int *mask = atom->mask; double *qa = atom->q;
  tagint *tag = atom->tag;
  const int nall = atom->nlocal + atom->nghost;
  const int K = lr_recip_probes;

  for (int i=0;i<nall;i++) qsave[i]=qa[i];

  // ---- the K lowest global tags in the group. K Allreduces, trivially order-independent. --------
  std::vector<tagint> ptag; ptag.reserve(K);
  tagint prev = 0;
  for (int k=0;k<K;k++) {
    tagint loc = std::numeric_limits<tagint>::max(), got;
    for (int ii=0;ii<nn;ii++){ int i=ilist[ii];
      if ((mask[i]&groupbit) && tag[i]>prev && tag[i]<loc) loc=tag[i]; }
    MPI_Allreduce(&loc,&got,1,MPI_LMP_TAGINT,MPI_MIN,world);
    if (got == std::numeric_limits<tagint>::max()) break;   // group exhausted
    ptag.push_back(got); prev = got;
  }
  // (row 182): floor 4 -> 2. The 4 was parser policy (an EXPLICIT K of 1..3 is refused), not a limit of the
  // construction: a probe needs one partner inside the G2 window (spcfq_gas_ewald: hmax 3.5 A vs r_OH 1.0), and a
  // 2- or 3-atom group averaged over ALL its sites is the per-atom mean -- the best homogeneous value there is.
  // Below 4 the default used to fall to the legacy pair, i.e. to the decomposition-dependent route.
  if ((int)ptag.size() < 2) {
    if (lr_recip_probes_user)
      error->all(FLERR, "samqeq: recip_self needs >=2 group atoms for the multi-probe calibration (found {})",
                 (int)ptag.size());
    for (int i=0;i<nall;i++) qa[i]=qsave[i];
    return false;                                        // the legacy route reports the empty/1-atom group
  }

  const double hmax = 2.5 * (domain->xprd / (lr_pppm ? std::max(1,lr_pppm->nx_pppm) : 1));
  std::vector<double> S; S.reserve(ptag.size());

  for (size_t k=0;k<ptag.size();k++) {
    tagint tB = 0; double sk = 0.0;
    if (!measure_recip_probe(ptag[k], tB, hmax, sk)) {
      if (lr_recip_probes_user)
        error->all(FLERR, "samqeq: recip_self probe tag {} found no partner within [0.1, {:.3f}] Ang --"
                          "the erf(aR)/R complement is only valid for a close pair", (tagint)ptag[k], hmax);
      // : under the DEFAULT K=16 a lattice whose nearest neighbour exceeds 2.5 grid spacings (Au slab:
      // nn 2.88 A vs hmax 1.84 A on the metal_slab_gself golden) keeps its earlier operator: legacy pair.
      if (comm->me == 0)
        utils::logmesg(lmp, "samqeq: recip_self: probe tag {} has no partner within the G2 window [0.1, {:.3f}] A"
                            "-- the default K=16 probe calibration falls back to the legacy single pair for this"
                            "group (set `fix_modify {} recip_probes 16` explicitly to make this an error)\n",
                       (tagint)ptag[k], hmax, id);
      for (int i=0;i<nall;i++) qa[i]=qsave[i];
      return false;
    }
    S.push_back(sk);
  }

  double mean = 0.0; for (double v : S) mean += v; mean /= (double)S.size();
  double var = 0.0; for (double v : S) var += (v-mean)*(v-mean);
  lr_self_sigma = (S.size()>1) ? sqrt(var/(double)(S.size()-1)) : 0.0;

  for (int i=0;i<nall;i++) qa[i]=qsave[i];
  finish_recip_self(mean, (int)S.size());
  return true;
}

void FixQEqSam::calibrate_recip_self()
{
  if (lr_recip_probes > 0 && calibrate_recip_self_probes()) return;   // /: false => G2 miss or <2 atoms -> legacy below
  int *mask = atom->mask; double *qa = atom->q; tagint *tag = atom->tag;
  const int nall = atom->nlocal + atom->nghost;
  for (int i=0;i<nall;i++) qsave[i]=qa[i];
  // probe pair, STICKY : the first calibration picks the pair and CACHES its tags (lr_tagA/lr_tagB,
  // restart-carried); every RE-calibration re-measures the SAME tagged pair by minimum image, so run-boundary
  // re-draws (PBC-straddling pairs ~40 eV/e low, a different point of the ~0.3-0.5 eV/e per-atom spread) stay gone.
  // (row 182): the pair is now a property of the SYSTEM, not of the decomposition. First calibration: A = the
  // LOWEST GLOBAL TAG in the group, B = its nearest group atom (min-image, tie -> lower tag; no distance window,
  // the historical pair had none), measured by measure_recip_probe() wherever the two atoms live. Replaces "first
  // two group atoms in ilist order on the lowest rank owning two" (= the setup-sort bin order on ONE rank), the
  // "pair migrated to different ranks -> fresh pick" re-draw (a decomposition-dependent walk under NPT at np>1)
  // and the co-residency refusal. Pre-draws are NOT reproduced (the bin order is an artefact of Atom::sort);
  // the goldens on this route were re-baselined at the flip (ledger L52).
  tagint tA = lr_tagA, tB = lr_tagB;
  const bool fresh = (tA <= 0);
  if (fresh) {
    tagint loc = std::numeric_limits<tagint>::max();
    for (int i=0;i<atom->nlocal;i++) if ((mask[i]&groupbit) && tag[i]<loc) loc=tag[i];
    MPI_Allreduce(&loc,&tA,1,MPI_LMP_TAGINT,MPI_MIN,world);
    if (tA == std::numeric_limits<tagint>::max())
      error->all(FLERR, "samqeq: recip_self calibration: the fix group is empty");
    tB = 0;
  }
  double rs = 0.0;
  if (!measure_recip_probe(tA, tB, 0.0, rs))
    error->all(FLERR, "samqeq: recip_self calibration needs >=2 fix-group atoms (probe tag {} has no group"
                      "partner{})", tA, fresh ? "" : " -- its cached partner left the group");
  lr_tagA = tA; lr_tagB = tB;
  if (fresh && comm->me == 0)
    utils::logmesg(lmp, "samqeq: recip_self legacy pair = tags {} (lowest in group) / {} (nearest, min-image)\n",
                   tA, tB);
  // grid-signature caching, the eV/e log line and the guardrails all live in finish_recip_self()
  // so that neither route can skip them.
  for (int i=0;i<nall;i++) qa[i]=qsave[i];
  finish_recip_self(rs, 1);
}

/* ----------------------------------------------------------------------
   Has the PPPM grid signature ({nx,ny,nz}_pppm + g_ewald + box lengths) changed since the last
   calibrate_recip_self()? recip_self depends only on this signature, so we recalibrate iff it changes.
   NB (correction of an older claim here): the grid dims/g_ewald are fixed by KSpace::init(), which runs
   BEFORE modify->init()/setup_pre_force (lammps.cpp: force->init() precedes modify->init()) — so pre_force
   always sees the CURRENT run's finalized signature; PPPM::setup() only re-derives volume factors from it.
   Returns false if not in lr_ewald/PPPM mode.
-------------------------------------------------------------------------*/
bool FixQEqSam::grid_changed()
{
  if (!lr_pppm) return false;
  double gw = force->kspace ? force->kspace->g_ewald : lr_gewald;
  if ((lr_pppm->nx_pppm != lr_nx) || (lr_pppm->ny_pppm != lr_ny) ||
      (lr_pppm->nz_pppm != lr_nz)) return true;
  // ROOT CAUSE of the spurious every-run-boundary recalibration (: 5 calibrations for 5 runs
  // on a FIXED box): PPPM::init() re-derives g_ewald from qsum_qsq() at EVERY run start, and q2 = Σq²
  // tracks the live FQ charges — the auto-tuned g_ewald jitters at each run boundary purely from
  // charge drift, so an exact != compare trips every time. The jitter is far below the calibration's
  // own ~0.3 eV/e context spread, and the pinning rationale WANTS the original calibration kept
  // (params were tuned against it). Gate g_ewald on the same 0.1% relative test as the box lengths:
  // real accuracy/grid changes move it far more; charge jitter never does. (Decks that pin
  // `kspace_modify gewald` were never exposed.) Sentinel (-1, pre-first-calibration) stays exact.
  // MEASURED CORRECTION to the "charge jitter never does" clause above -- it does. On a fixed-box
  // 256-molecule TIP4P-FQ liquid the retune between the data-file charges and the solved charges moves
  // g_ewald 0.31499941 -> 0.3160651, i.e. +0.34%, so this gate trips at EVERY run boundary. But the
  // 0.1% THRESHOLD is right and must stay: recip_self is a function of g_ewald, so freezing it while
  // g_ewald moves leaves the diagonal inconsistent with the operator actually being applied.
  // Demonstrated by tests/cases/recalib_on_grid_change: widening this to 1% makes the second run
  // reuse a recip_self measured for the first run's g_ewald, and the charges MOVE across the boundary
  // (0.353392 -> 0.359845 on atom 81) -- the opposite of that case's stated invariant. Reverted.
  // What was wrong at the boundary was never that it recalibrated; it was that the legacy single-pair
  // measurement is not deterministic, so one run in twelve re-picked and drew 0.33940 instead of
  // 0.3564x, over-stiffening the diagonal by 0.23 eV/e and depolarizing the liquid for 100 ps with no
  // error reported. That is cured by `fix_modify <id> recip_probes <K>` and caught by the G1 guard,
  // not by suppressing a correct re-measurement.
  if (lr_gewald > 0.0) { if (fabs(gw - lr_gewald) > 1.0e-3*lr_gewald) return true; }
  else if (gw != lr_gewald) return true;
  // recip_self also scales with the grid SPACING: under NPT the box changes at FIXED counts (PPPM setup()
  // rescales, counts stay) so the {nx,ny,nz,g_ewald} signature alone goes silently stale. Re-arm on a
  // >0.1% box-length change (below that the drift is well under the measurement's own ~0.3 eV/e spread).
  const double prd[3] = {domain->xprd, domain->yprd, domain->zprd};
  for (int k=0;k<3;k++)
    if (lr_prd[k] > 0.0 && fabs(prd[k]-lr_prd[k]) > 1.0e-3*lr_prd[k]) return true;
  return false;
}

/* ----------------------------------------------------------------------
   add the reciprocal-space Coulomb to the H-block of the matvec result b = M·x.
   The trial H-block x[0..nlocal) IS a set of (perturbation) charges; feed it to the ELECTRODE per-atom
   kspace potential (PPPMSamqeq::compute_vector — reliable, dedicated, no reentrant stock-PPPM), then:
       b[i] += qqrd2e * (φ_recip,i(x) − recip_self·x[i])
   φ_recip,i is the reciprocal potential at i from all trial charges (incl. i's own recip self, grid-global
   so no comm needed). We subtract the MEASURED grid self recip_self (= lr_self_meas, calibrated in
   calibrate_recip_self via a neutral +1/−1 pair), NOT the analytic point self 2·g_ewald/√π — the analytic
   value is wrong on the finite PPPM grid, and what we need cancelled is precisely PPPM's i=i grid coefficient,
   so the QEq diagonal stays = η + ξ_image (PD; ξ = ψ_{k≠0}(0) − 2α/√π → −2.837/L, the charge's interaction with its
   own periodic images, which the Ewald energy the forces integrate also keeps -- R4 measured this: a neutral probe
   pair cancels ξ and so measures 2α/√π − smearing, NOT the lone-charge grid self). Together with the erfc real-space already in H, this is the full
   shielded 1/r Coulomb operator (i≠j) applied to x. compute_vector reads atom->q of the source group, so we
   temporarily stash the trial charges there and restore.
   NB NET CHARGE (adjudication): the grid has G(k=0)=0 and nothing is added here, so the operator OMITS the
   Ewald background potential -pi·Q/(alpha²V) that PPPM's ENERGY carries (-(pi/2)qsum²/(alpha²V)). Zero for a
   neutral cell, which every per-molecule-pinned solve is.
   NB SELF-ENERGY/ENERGY CONSISTENCY (Ewald path): the off-diagonal here matches the physical i≠j Coulomb in pe
   (coul/long's −g_ewald/√π·Σq² self-correction is internal to the Ewald sum, so the REPORTED elong+ecoul is the
   i≠j sum — no extra reported self-energy), and the on-site χq+½ηq² is reported separately by compute_scalar,
   so q* is variational w.r.t. pe + E_self. (The DSF path lr_ewald=0 differs: coul/dsf reports a per-atom
   e_self ∝ q² in pe that the solve's bare-η diagonal omits — a FORCELESS offset, see compute_H setup.)*/
void FixQEqSam::add_reciprocal(double *x, double *b)
{
  double *q = atom->q; int *mask = atom->mask;
  for (int ii = 0; ii < nn; ii++) { int i=ilist[ii]; if(!(mask[i]&groupbit)) continue;
    qsave[i] = q[i]; q[i] = x[i]; }                       // stash physical q; feed trial charges
  for (int i = 0; i < atom->nlocal; i++) prec[i] = 0.0;   // compute_vector ACCUMULATES -> must pre-zero.
                                                          // E2: it only ever writes LOCAL sensor atoms
                                                          // (project_psi + slab term: i < nlocal) and prec is
                                                          // only read back for owned in-group atoms, so
                                                          // zeroing nlocal (not nmax) suffices — bit-identical.
  eksp->compute_vector(prec, groupbit, groupbit, false);  // raw reciprocal potential (no qqrd2e)
  // subtract the MEASURED grid recip_self so prec's i=i grid self cancels -> H_ii = eta (PD, force-consistent
  // i≠j Coulomb). (The analytic 2α/√π is wrong on the finite PPPM grid; lr_self_meas is calibrated in pre_force.)
  double pref = force->qqrd2e, selfc = lr_self_meas;
  // NaN TRIPWIRE (2026-08-22, gamma_align K5 / metal-slab Ewald). A pure Au(111) slab with a vacuum
  // gap returned resid/b=nan on the FIRST CG iteration under pppm/samqeq (bulk Au and the water slab
  // golden are fine). Localise it: report the first non-finite reciprocal potential with the atom,
  // its trial charge and the inputs, then error out -- a silent all-zero charge set is worse than a stop.
  for (int ii = 0; ii < nn; ii++) { int i=ilist[ii]; if(!(mask[i]&groupbit)) continue;
    if (!std::isfinite(prec[i]) || !std::isfinite(x[i]))
      error->one(FLERR, "samqeq add_reciprocal: non-finite value at local atom {} (tag {}): prec={} x={}"
                        "recip_self={} alpha={} -- pppm/samqeq compute_vector returned NaN/Inf",
                 i, atom->tag[i], prec[i], x[i], selfc, lr_alpha); }
  if (!lr_self_peratom && !std::isfinite(selfc))
    error->all(FLERR, "samqeq add_reciprocal: calibrated recip_self is non-finite ({})", selfc);
  if (lr_self_peratom) {                                  // R4: exact per-atom grid self removed
    for (int ii = 0; ii < nn; ii++) { int i=ilist[ii]; if(!(mask[i]&groupbit)) continue;
      b[i] += pref*(prec[i] - lr_self_atom[i]*x[i]);
      q[i] = qsave[i]; }
    return;
  }
  for (int ii = 0; ii < nn; ii++) { int i=ilist[ii]; if(!(mask[i]&groupbit)) continue;
    b[i] += pref*(prec[i] - selfc*x[i]);                  // reciprocal i≠j part (grid self removed)
    q[i] = qsave[i]; }                                    // restore physical q
}

/* OMP : lazily (re)allocate the per-thread project_neutral molsum scratch (nth*nmol_ doubles). nmol_
   is rebuilt every solve (build_molinv), so this regrows often but is cheap (nmol_ = fragment COUNT, not
   raw max(mol)+1, post-B9). Grown on either an nmol_ increase or an nthreads change; never shrunk (avoids
   realloc churn when nmol_ oscillates near a boundary).*/
void FixQEqSam::ensure_ps_molsum(int nth)
{
  if (ps_molsum_nth == nth && ps_molsum_cap >= nmol_) return;
  int cap = nmol_ > ps_molsum_cap ? nmol_ : ps_molsum_cap;
  memory->destroy(ps_molsum_t);
  memory->create(ps_molsum_t, nth*cap, "samqeq:ps_molsum_t");
  ps_molsum_nth = nth; ps_molsum_cap = cap;
}

/* P: remove each molecule's mean so Σ_{i∈mol} v_i = 0 (project onto the per-molecule-neutral subspace).
   molinv[m] = 1/(in-group atom count of molecule m), precomputed in qeq_solve.
   A5: mol == nullptr (atom_style charge) ⇒ every atom is fragment 0 ⇒ GLOBAL neutrality.
   B9 (audit): molsum/molinv are COMPACT-indexed (length nmol_ = nactive, not raw max(mol)+1) — index by the
   per-atom compact slot cmol[i] (built in build_molinv), never by the raw atom->molecule[i]. Elementwise
   MPI_SUM is per-slot independent of how many OTHER slots exist or which slot a molecule occupies, so this
   is BYTE-IDENTICAL to the old raw-indexed sums (same atoms, same accumulation order, just a smaller/denser
   array to reduce).
   S2 (scaling audit, fixes #1-minimal + #3): the small reductions that used to be SEPARATE Allreduces now
   ride THIS reduce as tail slots of the molsum message (molsum is over-allocated by PN_TAIL_MAX):
     [nmol_] optional fused dot partial <dot_a,dot_b> (caller-requested via dot_out; qeq_dot_local => local
               accumulation byte-identical to a standalone qeq_dot). LEGALITY: only requested by callers whose
               dot operands are NOT touched by this projection (see the header comment) -- the per-CG-iteration
               dots read the PROJECTED vector and stay separate.
   At np=1 MPI_Allreduce is elementwise-trivial and every local partial is unchanged => byte-identical; at
   np>1 the reduced values are unchanged in exact arithmetic (elementwise MPI_SUM of the same partials).
   DEADLOCK SAFETY: the reduce length nred = nmol_ + tail depends only on (dot_out!=nullptr) [same literal at
   every collective call site] -- identical on all ranks. Saves one Allreduce per solve for the fused ||b||.*/
void FixQEqSam::project_neutral(double *v, double *dot_a, double *dot_b, double *dot_out)
{
  const double th_t0 = (th_on == 1) ? platform::walltime() : 0.0;   // (#30) host phase timing
  int *mask = atom->mask;
  int nth = comm->nthreads;
  for (int m = 0; m < nmol_; m++) molsum[m] = 0.0;
#if defined(_OPENMP)
  if (omp_go(nth, nn)) {
    // P1 : molsum[cmol[i]] += v[i] is a SCATTER-reduction -- two atoms of the same molecule can land in
    // DIFFERENT threads' static ii-chunks and race on the same molsum[m] slot. Each thread accumulates into
    // its OWN private nmol_-length slice (ps_molsum_t), then a deterministic thread-ORDERED sweep (t=0..nth-1,
    // fixed order) folds the slices into molsum before the (unchanged) MPI_Allreduce below.
    ensure_ps_molsum(nth);
    int cap = ps_molsum_cap;
#pragma omp parallel num_threads(nth)
    {
      int tid = omp_get_thread_num();
      double *ms = ps_molsum_t + (size_t)tid*cap;
      for (int m = 0; m < nmol_; m++) ms[m] = 0.0;
#pragma omp for schedule(static)
      for (int ii = 0; ii < nn; ii++){ int i=ilist[ii]; if (mask[i]&groupbit) ms[cmol[i]] += v[i]; }
    }
    for (int t = 0; t < nth; t++) { double *ms = ps_molsum_t + (size_t)t*cap;
      for (int m = 0; m < nmol_; m++) molsum[m] += ms[m]; }
  } else
#endif
  { for (int ii = 0; ii < nn; ii++){ int i=ilist[ii]; if (mask[i]&groupbit) molsum[cmol[i]] += v[i]; } }
  // S2 fused-collective tail (see the function banner): fill the requested tail slots from PRE-reduce local
  // state, then ONE Allreduce carries molsum + tail. Slot indices are pure functions of rank-uniform state.
  int nred = nmol_;
  int dslot = -1;
  if (dot_out) { dslot = nred++; molsum[dslot] = qeq_dot_local(dot_a, dot_b); }
  MPI_Allreduce(MPI_IN_PLACE, molsum, nred, MPI_DOUBLE, MPI_SUM, world);
  if (dot_out) *dot_out = molsum[dslot];
#if defined(_OPENMP)
  if (omp_go(nth, nn)) {
    // P1: the subtract is purely elementwise (each ii only touches its own v[i]) -- embarrassingly parallel,
    // no reduction, no race; molsum/molinv/cmol are read-only here (already finalized above).
#pragma omp parallel for num_threads(nth) schedule(static)
    for (int ii = 0; ii < nn; ii++){ int i=ilist[ii]; if (mask[i]&groupbit) { int m = cmol[i]; v[i] -= molsum[m]*molinv[m]; } }
  } else
#endif
  { for (int ii = 0; ii < nn; ii++){ int i=ilist[ii]; if (mask[i]&groupbit) { int m = cmol[i]; v[i] -= molsum[m]*molinv[m]; } } }
  if (th_on == 1) { th_proj += platform::walltime() - th_t0; th_projcalls++; }
}

/* OMP : lazily (re)allocate the P2 per-thread matvec accumulator (nth*atom->nmax doubles), the same
   headroom the other N-dim scratch (q_r, qb, ...) carries. Regrown on either an nthreads change (rare --
   `package omp N` reissued mid-run) or an atom->nmax growth (the normal reallocate_storage trigger); never
   shrunk. nthreads==1 never calls this (see csr_matvec_add's else-branch) -> zero extra memory there.*/
void FixQEqSam::ensure_mv_out(int nth)
{
  int cap = atom->nmax > mv_out_cap ? atom->nmax : mv_out_cap;
  if (mv_out_nth == nth && mv_out_cap >= atom->nmax) return;
  memory->destroy(mv_out_t);
  memory->create(mv_out_t, nth * cap, "samqeq:mv_out_t");   // nth (few) * cap (atom->nmax) fits int in practice
  mv_out_nth = nth; mv_out_cap = cap;
}

/* P2 (, THE hot loop): out[i] += H.val*x[j]; out[j] += H.val*x[i] over ii in nn. Shared by qeq_matvec and
   coulomb_field (identical CSR pattern). Races on out[j] under a plain ii-partitioned parallel-for (j is not
   co-partitioned with i -- it can be any local or ghost atom in i's cutoff sphere). Fix: each thread gets its
   own private nall-length accumulator slice (mv_out_t, ensure_mv_out above); a thread's static ii-chunk is
   race-free INSIDE its own slice (all writes there are exclusively this thread's), then a deterministic
   thread-ORDERED sweep t=0..nthreads-1 (fixed order, NOT dependent on completion timing) folds every slice's
   contribution into the caller's out with a plain += (out already holds the diagonal/zero-fill from the
   caller -- this only ADDS the pairwise part, exactly like the original `out[i]+=`/`out[j]+=`). Memory:
   nthreads*atom->nmax*8 bytes (e.g. 4 threads * 100k atoms = 3.2 MB). No atomics needed -> no runtime-order
   nondeterminism. nthreads==1 OR nn<OMP_GRAIN (the grain gate, fix_qeq_sam.h) takes the ORIGINAL
   untouched serial loop below (byte-identical, no allocation).*/
void FixQEqSam::csr_matvec_add(double *x, double *out)
{
  host_mv_calls++;   // (#30) probe — see fix_qeq_sam.h
  int *mask = atom->mask;
  int nall = atom->nlocal + atom->nghost;
  int nth = comm->nthreads;
#if defined(_OPENMP)
  if (omp_go(nth, nn)) {              // grain gate on nn = the partitioned CSR-row loop (the actual work)
    ensure_mv_out(nth);
    int cap = mv_out_cap;
#pragma omp parallel num_threads(nth)
    {
      int tid = omp_get_thread_num();
      double *ot = mv_out_t + (size_t)tid*cap;
      for (int i = 0; i < nall; i++) ot[i] = 0.0;                  // this thread's private slice only
#pragma omp for schedule(static)
      for (int ii = 0; ii < nn; ii++) { int i=ilist[ii]; if (!(mask[i]&groupbit)) continue;
        for (int itr=H.firstnbr[i]; itr<H.firstnbr[i]+H.numnbrs[i]; itr++) {
          int j=H.jlist[itr]; ot[i] += H.val[itr]*x[j]; ot[j] += H.val[itr]*x[i]; } }
    }
    for (int t = 0; t < nth; t++) { double *ot = mv_out_t + (size_t)t*cap;   // deterministic thread-ORDERED fold
      for (int i = 0; i < nall; i++) out[i] += ot[i]; }
  } else
#endif
  {
    for (int ii = 0; ii < nn; ii++) { int i=ilist[ii]; if (!(mask[i]&groupbit)) continue;
      for (int itr=H.firstnbr[i]; itr<H.firstnbr[i]+H.numnbrs[i]; itr++) {
        int j=H.jlist[itr]; out[i] += H.val[itr]*x[j]; out[j] += H.val[itr]*x[i]; } }
  }
}

/* projected SPD matvec: out = P(η·x + erfc-Coulomb·x + reciprocal·x). N-dim; comm_v/pack_flag=6 forward-
   comm x to ghosts, reverse-comm assembles the pairwise part, add_reciprocal completes 1/r, then project.*/
/* ----------------------------------------------------------------------*/

void FixQEqSam::qeq_matvec(double *x, double *out)
{
  int *mask = atom->mask;
  int nth = comm->nthreads;
  comm_v = x; pack_flag = 6; comm->forward_comm(this);              // distribute x to ghosts
  // P1 : diagonal fill is purely elementwise (each ii writes only its own out[i]) -- embarrassingly
  // parallel, no race, no reduction.
#if defined(_OPENMP)
  if (omp_go(nth, nn)) {
#pragma omp parallel for num_threads(nth) schedule(static)
    for (int ii = 0; ii < nn; ii++) { int i=ilist[ii]; if (mask[i]&groupbit)   // #M-3: per-atom ridge in mode 3
      out[i] = ((solve_diag_of(i))
               + (ridge_local ? ridge_atom[i] : ridge_cur))*x[i]; }  // #25/(B) diagonal
  } else
#endif
  { for (int ii = 0; ii < nn; ii++) { int i=ilist[ii]; if (mask[i]&groupbit)   // #M-3: per-atom ridge in mode 3
      out[i] = ((solve_diag_of(i))
               + (ridge_local ? ridge_atom[i] : ridge_cur))*x[i]; } }  // #25/(B) diagonal
  int nall = atom->nlocal + atom->nghost;
  for (int i = atom->nlocal; i < nall; i++) out[i] = 0.0;
  // TRIPWIRE 4 (2026-08-22): tripwires 1-3 localised the metal-slab Ewald NaN to A.x with b, the
  // diagonal and the reciprocal all finite. Check the three matvec stages in order so the run names
  // the piece: diagonal -> real-space CSR -> reciprocal.
  auto tw = [&](const char *stage) {
    for (int ii = 0; ii < nn; ii++) { int i=ilist[ii]; if (!(mask[i]&groupbit)) continue;
      if (!std::isfinite(out[i]))
        error->one(FLERR, "samqeq qeq_matvec TRIPWIRE4: non-finite out[{}] (tag {}) after stage '{}' ; x={}"
                          "diag={} nn={} H.m={}", i, atom->tag[i], stage, x[i], solve_diag_of(i), nn, H.m); } };
  tw("diagonal");
  csr_matvec_add(x, out);                                           // P2: out[i]+=H.val*x[j]; out[j]+=H.val*x[i]
  comm_v = out; pack_flag = 6; comm->reverse_comm(this);            // assemble ghost contributions into out
  tw("csr_real_space");
  if (!lr_nrecip) add_reciprocal(x, out);                           // + reciprocal − grid self (-> full 1/r); SKIP in taper-cutoff mode (H is already the full tapered shielded Coulomb)
  tw("reciprocal");
  project_neutral(out);                                             // P -> restrict to per-molecule-neutral
  tw("projected");
}

/* off-diagonal full 1/r Coulomb operator: out = P(erfc-shielded·x + reciprocal·x), i.e. qeq_matvec WITHOUT
   the η (+ridge) on-site diagonal. Used to build the q0 reference-charge field that the lr RHS must
   carry for force↔solve energy consistency: the physical pair/PPPM forces act on qa = qs+q0, so the energy the
   solve stationarizes must include the Coulomb field [J·q0] from the FIXED q0 reference charges. Computing it
   directly (no diagonal) avoids any entanglement with the quartic diagonal/ridge (which are charge-dependent under the quartic
   path); the eta-diagonal belongs to the on-site self-energy ½η q² (gradient η·qs), already on the matvec.*/
void FixQEqSam::coulomb_field(double *x, double *out, bool project)
{
  int nth = comm->nthreads;
  comm_v = x; pack_flag = 6; comm->forward_comm(this);              // distribute x (q0) to ghosts
  int nall = atom->nlocal + atom->nghost;
  // P1: zero-fill is embarrassingly parallel (independent per-index writes).
#if defined(_OPENMP)
  if (omp_go(nth, nall)) {
#pragma omp parallel for num_threads(nth) schedule(static)
    for (int i = 0; i < nall; i++) out[i] = 0.0;
  } else
#endif
  { for (int i = 0; i < nall; i++) out[i] = 0.0; }
  csr_matvec_add(x, out);                                           // P2: out[i]+=H.val*x[j]; out[j]+=H.val*x[i]
  comm_v = out; pack_flag = 6; comm->reverse_comm(this);            // assemble ghost contributions
  if (!lr_nrecip) add_reciprocal(x, out);                           // + reciprocal (− grid self) -> full off-diagonal 1/r
  if (project) project_neutral(out);                                // P -> per-molecule-neutral (all current callers pass project=true, incl. the quartic gate)
}

/* ----------------------------------------------------------------------
   (B') FIELD GATE for the quartic. The quartic must harden ONLY where the field is strong (the ion's shell),
   NOT the bulk equilibrium charge — for SPC-FQ O the bulk charge is already ~-0.85, so an un-gated quartic
   over-hardens bulk water (inflates its entropy: S_rot/D up). Gate the per-atom quartic coefficient on the
   PROJECTED (per-molecule-neutral) reference-charge Coulomb field φ0_i = |P(J·q0)_i| (the ion field at i; q0 is
   fixed ⇒ geometry-only ⇒ no added solve nonlinearity): c4_eff(i) = c4 · φ0²/(φ0² + fld0²). Bulk (φ0≈0) ⇒ gate≈0 ⇒ untouched; shell
   (φ0 large) ⇒ gate≈1 ⇒ bounded. fld0<=0 or no q0 reference ⇒ gate≡1 ⇒ byte-identical un-gated quartic.
   Uses q_d/q_q as scratch (free here: called in pre_force BEFORE the solve uses them); result persists in
   quartic_gate for apply_quartic_eta / the energy / the XL force.
-------------------------------------------------------------------------*/
void FixQEqSam::compute_quartic_gate()
{
  int *type = atom->type, *mask = atom->mask;
  int nth = comm->nthreads;
  if (quartic_fld0 <= 0.0 || !has_q0ref) {
#if defined(_OPENMP)
    if (omp_go(nth, atom->nmax)) {
#pragma omp parallel for num_threads(nth) schedule(static)
      for (int i = 0; i < atom->nmax; i++) quartic_gate[i] = 1.0;   // no gate -> byte-identical
    } else
#endif
    { for (int i = 0; i < atom->nmax; i++) quartic_gate[i] = 1.0; }
    return;
  }
    build_molinv();                                                  // project_neutral needs molinv/nmol_ (qeq_solve hasn't run yet)
#if defined(_OPENMP)
  if (omp_go(nth, atom->nmax)) {
#pragma omp parallel for num_threads(nth) schedule(static)
    for (int i = 0; i < atom->nmax; i++) { q_d[i] = 0.0; quartic_gate[i] = 1.0; }
  } else
#endif
  { for (int i = 0; i < atom->nmax; i++) { q_d[i] = 0.0; quartic_gate[i] = 1.0; } }
#if defined(_OPENMP)
  if (omp_go(nth, nn)) {
#pragma omp parallel for num_threads(nth) schedule(static)
    for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit) q_d[i] = q0[type[i]]; }
  } else
#endif
  { for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit) q_d[i] = q0[type[i]]; } }
  coulomb_field(q_d, q_q, true);                                   // q_q = P(J·q0) = the WITHIN-MOLECULE reference-field
                                                                   // gradient (1/r², sharply localized at the ion's shell —
                                                                   // the asymmetric field that drives intramolecular over-pol;
                                                                   // far better shell/bulk contrast than the 1/r raw potential)
  const double f0sq = quartic_fld0*quartic_fld0;
  double gmax = 0.0, fmax = 0.0;
#if defined(_OPENMP)
  if (omp_go(nth, nn)) {
    // P1: single pass fills quartic_gate[i] (embarrassingly parallel, disjoint writes) AND tracks two
    // diagnostic maxima (gmax/fmax) -- max is order-independent (exact regardless of accumulation order,
    // unlike a sum), so a per-thread partial max + a fixed-order combine is both race-free and deterministic.
    std::vector<double> gmax_t(nth, 0.0), fmax_t(nth, 0.0);
#pragma omp parallel num_threads(nth)
    {
      int tid = omp_get_thread_num();
      double gl = 0.0, fl = 0.0;
#pragma omp for schedule(static)
      for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(!(mask[i]&groupbit)) continue;
        double f = q_q[i]; double g = f*f/(f*f + f0sq); quartic_gate[i] = g;
        if (g>gl) gl=g; if (fabs(f)>fl) fl=fabs(f); }
      gmax_t[tid]=gl; fmax_t[tid]=fl;
    }
    for (int t = 0; t < nth; t++) { if (gmax_t[t]>gmax) gmax=gmax_t[t]; if (fmax_t[t]>fmax) fmax=fmax_t[t]; }
  } else
#endif
  { for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(!(mask[i]&groupbit)) continue;
    double f = q_q[i]; double g = f*f/(f*f + f0sq); quartic_gate[i] = g;
    if (g>gmax) gmax=g; if (fabs(f)>fmax) fmax=fabs(f); } }
  if (update->ntimestep == 0 || update->ntimestep % 1000 == 0) {
    MPI_Allreduce(MPI_IN_PLACE,&fmax,1,MPI_DOUBLE,MPI_MAX,world);   // ALL ranks (collective) — never gate on comm->me
    MPI_Allreduce(MPI_IN_PLACE,&gmax,1,MPI_DOUBLE,MPI_MAX,world);
    if (comm->me == 0)   // A7 diagnostic-print: phi0 is deck-unit -> /ev_scale keeps "eV/e" true; fld0 is a
                         // user fix_modify value (R1) so it is echoed RAW in deck units, label-free.
      utils::logmesg(lmp, "samqeq QUARTIC-GATE step {}: max|phi0|={:.4g} eV/e, max gate={:.3f} (fld0={:.4g})\n",
                     update->ntimestep, fmax/ev_scale, gmax, quartic_fld0);
  }
}

/* S2 (scaling audit): the LOCAL partial of the in-group dot (no MPI). Split out of qeq_dot so
   project_neutral's fused-dot tail slot accumulates the EXACT same local value (same loop, same
   deterministic OMP thread-ordered combine) the standalone qeq_dot would have reduced.*/
double FixQEqSam::qeq_dot_local(double *a, double *b)
{
  int *mask = atom->mask; double s = 0.0;
  int nth = comm->nthreads;
#if defined(_OPENMP)
  if (omp_go(nth, nn)) {
    // P1: deterministic thread-ordered SUM reduction (omp_reduce, see the anonymous-namespace helper above) --
    // NOT `#pragma omp reduction(+:s)`, whose combine order across threads is unspecified by the standard.
    s = omp_reduce(nn, nth, 0.0, omp_sum,
                   [&](int ii)->double{ int i=ilist[ii]; return (mask[i]&groupbit) ? a[i]*b[i] : 0.0; });
  } else
#endif
  { for (int ii = 0; ii < nn; ii++) { int i=ilist[ii]; if (mask[i]&groupbit) s += a[i]*b[i]; } }
  return s;
}

/* MPI-reduced in-group dot over local atoms (N-dim).*/
double FixQEqSam::qeq_dot(double *a, double *b)
{
  double s = qeq_dot_local(a, b);
  double g; MPI_Allreduce(&s, &g, 1, MPI_DOUBLE, MPI_SUM, world); return g;
}

/* projected, Jacobi-preconditioned CG. Ã=P·H is SPD on the per-molecule-neutral subspace (the unconstrained
   H is indefinite — inter-molecular CT mode). RHS b and all iterates stay neutral; the preconditioned
   direction is re-projected (D⁻¹ mixes O/H eta and would otherwise leak out of the subspace).*/
/* ----------------------------------------------------------------------
   (#30 item 5): build the per-molecule Cholesky blocks of (eta + J_intra).

   Built directly from POSITIONS, not from the CSR H: for each molecule we need only its own n_m^2
   pairs (~10 per atom), which is ~80x less work than filtering the full neighbour list, and it keeps
   working when the host H is not built at all (the device-solve path). Minimum-image distances: a
   molecule may straddle the periodic boundary, and for a PRECONDITIONER an approximation there is
   harmless -- it cannot change the converged charges, only how fast we reach them.
   Any block that fails Cholesky (not positive definite) falls back to Jacobi for its atoms, so a
   pathological molecule degrades locally instead of poisoning the solve.
-------------------------------------------------------------------------*/
void FixQEqSam::build_mol_blocks()
{
  precond_mol_ok = 0;
  if (precond_mode != 2 || nmol_ <= 0 || !cmol) return;
  if (lr_ewald < 2 || lr_alpha <= 0.0) return;          // only the Ewald-split operator is mirrored here

  int *mask = atom->mask, *type = atom->type;
  double **x = atom->x;
  const double pref = force->qqrd2e, a = lr_alpha;

  // bucket the local in-group atoms by compact molecule slot
  mb_n.assign(nmol_, 0);
  for (int ii = 0; ii < nn; ii++) { const int i = ilist[ii];
    if ((mask[i] & groupbit) && i < atom->nlocal) mb_n[cmol[i]]++; }
  mb_off.assign(nmol_ + 1, 0);
  size_t tot = 0, totL = 0;
  for (int m = 0; m < nmol_; m++) { mb_off[m] = (int) tot; tot += mb_n[m];
    totL += (size_t) mb_n[m] * (mb_n[m] + 1) / 2; }
  mb_off[nmol_] = (int) tot;
  mb_idx.assign(tot, 0);
  mb_L.assign(totL, 0.0);
  mb_ok.assign(nmol_, 0);
  { std::vector<int> fill(nmol_, 0);
    for (int ii = 0; ii < nn; ii++) { const int i = ilist[ii];
      if ((mask[i] & groupbit) && i < atom->nlocal) { const int m = cmol[i];
        mb_idx[mb_off[m] + fill[m]++] = i; } } }

  // factorize each block in place (packed lower triangle, column-major by row)
  std::vector<double> B;
  size_t Lpos = 0;
  for (int m = 0; m < nmol_; m++) {
    const int n = mb_n[m];
    if (n <= 0) continue;
    B.assign((size_t) n * n, 0.0);
    for (int p = 0; p < n; p++) {
      const int i = mb_idx[mb_off[m] + p];
      B[(size_t) p * n + p] = solve_diag_of(i) + ridge_cur;      // same diagonal the operator uses
      for (int q = p + 1; q < n; q++) {
        const int j = mb_idx[mb_off[m] + q];
        double dx = x[j][0]-x[i][0], dy = x[j][1]-x[i][1], dz = x[j][2]-x[i][2];
        domain->minimum_image(FLERR, dx, dy, dz);
        const double r = sqrt(dx*dx + dy*dy + dz*dz);
        if (r < 1.0e-6 || r > swb) continue;
        const double v = pref * (shielded_coulomb(type[i], type[j], r) - (1.0 - erfc(a*r))/r);
        B[(size_t) p * n + q] = v; B[(size_t) q * n + p] = v;
      }
    }
    // dense Cholesky, bailing out to Jacobi for this molecule if it is not PD
    bool ok = true;
    for (int p = 0; p < n && ok; p++) {
      for (int q = 0; q <= p; q++) {
        double sum = B[(size_t) p * n + q];
        for (int k = 0; k < q; k++) sum -= B[(size_t) p * n + k] * B[(size_t) q * n + k];
        if (p == q) { if (sum <= 1.0e-12) { ok = false; break; } B[(size_t) p * n + p] = sqrt(sum); }
        else B[(size_t) p * n + q] = sum / B[(size_t) q * n + q];
      }
    }
    if (ok) { mb_ok[m] = 1;
      for (int p = 0; p < n; p++) for (int q = 0; q <= p; q++) mb_L[Lpos + (size_t) p*(p+1)/2 + q] = B[(size_t) p * n + q]; }
    Lpos += (size_t) n * (n + 1) / 2;
  }
  precond_mol_ok = 1;
}

/* z = M^-1 r : molecular blocks where they factorized, Jacobi everywhere else (and always when
   `precond mol` is off, which is the default and is byte-identical to the old inline expression).*/
void FixQEqSam::apply_precond(double *r, double *z)
{
  int *mask = atom->mask;
  if (precond_mode != 2 || !precond_mol_ok) {
    // the default Jacobi apply, keeping the OpenMP parallel-for the inline expression used to have
    // (elementwise, no reduction) so threaded CPU runs do not regress by this refactor
#if defined(_OPENMP)
    const int nth = comm->nthreads;
    if (omp_go(nth, nn)) {
#pragma omp parallel for num_threads(nth) schedule(static)
      for (int ii = 0; ii < nn; ii++) { const int i = ilist[ii];
        if (mask[i] & groupbit) z[i] = r[i] * Hdia_inv[i]; }
    } else
#endif
    { for (int ii = 0; ii < nn; ii++) { const int i = ilist[ii];
        if (mask[i] & groupbit) z[i] = r[i] * Hdia_inv[i]; } }
    return;
  }
  size_t Lpos = 0;
  std::vector<double> y;
  for (int m = 0; m < nmol_; m++) {
    const int n = mb_n[m];
    if (n <= 0) continue;
    if (!mb_ok[m]) {
      for (int p = 0; p < n; p++) { const int i = mb_idx[mb_off[m] + p]; z[i] = r[i] * Hdia_inv[i]; }
    } else {
      y.assign(n, 0.0);
      for (int p = 0; p < n; p++) {                     // forward solve L y = r
        double sum = r[mb_idx[mb_off[m] + p]];
        for (int q = 0; q < p; q++) sum -= mb_L[Lpos + (size_t) p*(p+1)/2 + q] * y[q];
        y[p] = sum / mb_L[Lpos + (size_t) p*(p+1)/2 + p];
      }
      for (int p = n - 1; p >= 0; p--) {                // back solve L^T z = y
        double sum = y[p];
        for (int q = p + 1; q < n; q++) sum -= mb_L[Lpos + (size_t) q*(q+1)/2 + p] * z[mb_idx[mb_off[m] + q]];
        z[mb_idx[mb_off[m] + p]] = sum / mb_L[Lpos + (size_t) p*(p+1)/2 + p];
      }
    }
    Lpos += (size_t) n * (n + 1) / 2;
  }
}

int FixQEqSam::qeq_cg(double *b, double *x)
{
  int *mask = atom->mask;
  int nth = comm->nthreads;
  // TRIPWIRE 2 (2026-08-22): the metal-slab Ewald NaN was not in add_reciprocal (tripwire 1 silent), and
  // gdb showed Hdia_inv == 0 at the first iteration -> Jacobi z=0 -> alpha = 0/0. Report it with the
  // diagonal inputs so the un-filled path can be named, and stop instead of returning zero charges.
  { int nbad = 0, ibad = -1;
    for (int ii = 0; ii < nn; ii++) { int i = ilist[ii]; if (!(mask[i] & groupbit)) continue;
      if (!(Hdia_inv[i] > 0.0) || !std::isfinite(Hdia_inv[i])) { if (ibad < 0) ibad = i; nbad++; } }
    if (nbad)
      error->one(FLERR, "samqeq qeq_cg: {} in-group atoms have a non-positive/non-finite Jacobi diagonal"
                        "(first: local {} tag {} Hdia_inv={} eta[type]={} solve_diag_of={}"
                        "lr_quartic={} ridge_cur={} nn={}) -- the preconditioner was never filled on this path",
                 nbad, ibad, atom->tag[ibad], Hdia_inv[ibad], eta[atom->type[ibad]], solve_diag_of(ibad),
                 lr_quartic, ridge_cur, nn); }
  qeq_matvec(x, q_q);                                               // q_q = Ã·x
  // P1: every loop below is a pure elementwise update (each ii writes only its own index i across q_r/q_d/
  // q_p/x) -- embarrassingly parallel, no race, no reduction; project_neutral/qeq_dot/qeq_matvec are threaded
  // internally (see their own definitions above).
#if defined(_OPENMP)
  if (omp_go(nth, nn)) {
#pragma omp parallel for num_threads(nth) schedule(static)
    for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit) q_r[i]=b[i]-q_q[i]; }
  } else
#endif
  { for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit) q_r[i]=b[i]-q_q[i]; } }
  apply_precond(q_r, q_d);   // : Jacobi, or the molecular blocks under `precond mol`
  // S2: <b,b> is fused onto the q_d-projection reduce (LEGAL: b is not touched by projecting q_d, so its
  // local partial -- computed by the same qeq_dot_local -- is byte-identical to a standalone qeq_dot(b,b);
  // at np=1 the fused elementwise reduce returns the same value). The PER-ITERATION dots below (q_d.q_q at
  // alpha, q_r.q_p at signew) CANNOT be fused the same way: each reads the PROJECTED vector, whose values
  // need the reduced molsum first (and alpha/beta feed the NEXT projection's local sums) -- fusing them
  // requires the algebraic reconstruction (audit S2 fix #1, first bullet), which reorders FP summation and
  // is out of scope ([re-golden]). They keep their own qeq_dot calls.
  double bnorm_sq = 0.0;
  project_neutral(q_d, b, b, &bnorm_sq);                            // keep search direction in the subspace
  double bnorm = sqrt(bnorm_sq); if (bnorm==0.0) bnorm = 1.0*ev_scale;  // A7: fallback stands in for an eV/e-normed |b| (inert: b==0 => resid 0)
  double signew = qeq_dot(q_r, q_d);
  // TRIPWIRE 3 (2026-08-22): tripwires 1-2 proved prec/x/recip_self and the Jacobi diagonal are all finite
  // and positive on the metal-slab Ewald NaN case, yet resid/b is NaN after 1 iteration. So it is born in
  // the CG scalars. Report them at the first iteration.
  if (comm->me == 0 && update->ntimestep == 0 && (!std::isfinite(signew) || !std::isfinite(bnorm_sq) || signew < 0.0)) {
    double rr = qeq_dot(q_r, q_r), dd = qeq_dot(q_d, q_d), bb = 0.0;
    for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit) bb += b[i]*b[i]; }
    utils::logmesg(lmp, "samqeq TRIPWIRE3 it=0: bnorm_sq={} signew=<r,d>={} <r,r>={} <d,d>={} raw<b,b>={} nn={}\n",
                   bnorm_sq, signew, rr, dd, bb, nn);
  }
  // A7/R3: criterion sqrt(sig)/bnorm has units e/sqrt(E) (sig = r.Dinv.r with Dinv ~ e^2/E), NOT
  // dimensionless -> the same tol number is sqrt(ev_scale) ~ 4.8x LOOSER in units real. tol_eff keeps the
  // PHYSICAL convergence unit-independent; `tolerance` stays the user's raw number (logs/other criteria).
  // Exact /1.0 at ev_scale==1 (metal) -> byte-identical.
  const double tol_eff = tolerance / sqrt(ev_scale);
  int it = 1;
  for (; it < imax && sqrt(fabs(signew))/bnorm > tol_eff; it++) {
    qeq_matvec(q_d, q_q);
    double alpha = signew / qeq_dot(q_d, q_q);
#if defined(_OPENMP)
    if (omp_go(nth, nn)) {
#pragma omp parallel for num_threads(nth) schedule(static)
      for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit){ x[i]+=alpha*q_d[i]; q_r[i]-=alpha*q_q[i]; } }
    } else
#endif
    { for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit){ x[i]+=alpha*q_d[i]; q_r[i]-=alpha*q_q[i]; } } }
    apply_precond(q_r, q_p);   // : Jacobi, or the molecular blocks under `precond mol`
                               // (its own loop; the OpenMP wrapper that used to be here belonged to
                               //  the inline Jacobi expression and cannot wrap a function call)
    project_neutral(q_p);
    double sigold = signew; signew = qeq_dot(q_r, q_p);
    double beta = signew/sigold;
#if defined(_OPENMP)
    if (omp_go(nth, nn)) {
#pragma omp parallel for num_threads(nth) schedule(static)
      for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit) q_d[i]=q_p[i]+beta*q_d[i]; }
    } else
#endif
    { for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit) q_d[i]=q_p[i]+beta*q_d[i]; } }
  }
  cg_relresid = sqrt(fabs(signew))/bnorm;            // expose for the ASPC quality gate (#16)
  cg_bnorm = bnorm;                                  // : the ASPC committed-residual report uses the same |b|
  // : warn only on a GENUINE non-convergence -- residual still above tol_eff at the cap --
  // and never from the ASPC corrector, whose imax is deliberately n_corr (its accept gate is aspc_rtol; its accuracy
  // is reported by the ASPC-DIAG line). `it >= imax` alone fired on every ASPC step and spent LAMMPS' 100-warning
  // budget by step ~103, silencing every later warning of the run. Budgeted: 10 per run, then counted.
  if (!aspc_corr_call && it >= imax && cg_relresid > tol_eff) {
    cg_nonconv++;
    if (comm->me == 0 && cg_nonconv <= 10)
      error->warning(FLERR, "samqeq projected CG did not converge ({} iters, resid/b={:.2e} > tol {:.1e}) at step {}{}",
                     it, cg_relresid, tol_eff, update->ntimestep,
                     cg_nonconv == 10 ? " -- further occurrences are only counted (SOLVER-DIAG nonconv=)" : "");
  }
  if (comm->me == 0 && (update->ntimestep == 0 || update->ntimestep % 200 == 0)) {  // #28 SOLVER-DIAG (periodic)
    if (aspc_on)   // S1b: append the cumulative corrector accept/reject counters -- a dead ASPC (permanent
                   // reject, the S1 pathology) is now visible in the log. aspc off => line byte-identical.
      utils::logmesg(lmp, "samqeq SOLVER-DIAG step {}: CG {} iters (={} matvecs){}, resid/b={:.2e}, ASPC acc/rej={}/{}{}{}\n",
                     update->ntimestep, it, it, warmstart ? " [warm]" : "", sqrt(fabs(signew))/bnorm,
                     aspc_naccept, aspc_nreject,
                     cg_nonconv > 0 ? fmt::format(", nonconv={}", cg_nonconv) : std::string(),
                     nwarn_ridge > 0 ? fmt::format(", ridge={:.3g} (engaged {} solves)", ridge_cur/ev_scale, nwarn_ridge)
                                     : std::string());
    else
      utils::logmesg(lmp, "samqeq SOLVER-DIAG step {}: CG {} iters (={} matvecs){}, resid/b={:.2e}{}{}\n",
                     update->ntimestep, it, it, warmstart ? " [warm]" : "", sqrt(fabs(signew))/bnorm,
                     cg_nonconv > 0 ? fmt::format(", nonconv={}", cg_nonconv) : std::string(),
                     nwarn_ridge > 0 ? fmt::format(", ridge={:.3g} (engaged {} solves)", ridge_cur/ev_scale, nwarn_ridge)
                                     : std::string());
  }
  return it;
}

/* #M-A: projected MINRES (Paige–Saunders), unpreconditioned. Solves Ã·x=b on the per-molecule-neutral
   subspace, where Ã=P·H is SYMMETRIC. MINRES minimizes ||b−Ã·x|| via the Lanczos tridiagonalization +
   Givens QR, so it stays BOUNDED on indefinite/singular operators and returns the minimal-residual
   (least-squares) solution — no ridge, no model change, bulk water byte-identical.

   CORRECTION (2026-08-02, commit 6e11dc7). This routine was introduced on the belief that close pairs
   near a strong solute drove a LOCALIZED NEGATIVE EIGENVALUE of Ã which broke CG. That was measured and
   REFUTED: dense diagonalization over 29 configurations found no negative eigenvalue (λ_min ∈ [7.4111,
   7.4252] eV, κ = 2.15), and for this shielding kernel indefiniteness is analytically unreachable, since
   J_shield ≤ J_shield(0) = 7.20 eV < η_min = 13.364 eV at every separation. The real defect was in
   compute_H, which group-tested the matrix ROW but not the COLUMN, so a group-restricted solve multiplied
   ~half of H into never-written solver workspace; CG then converged honestly onto a corrupted right-hand
   side. Fixed in 6e11dc7. MINRES appeared to cure it only because it sets r = b and never forms Ã·x₀, so
   it skipped the one contaminated matrix-vector product. Do NOT reintroduce the indefiniteness rationale.
   MINRES remains genuinely indefinite-safe as an algorithm and is kept for the ACKS2 saddles, where
   the operator IS structurally indefinite.

   Vectors: v=q_d, r1=q_r, r2=q_q, y=m_t, w=q_p, w2=m_w2, x=qs.
   The matvec (qeq_matvec) projects its output ⇒ all Lanczos vectors stay in the neutral subspace.*/
int FixQEqSam::qeq_minres(double *b, double *x, double bref)
{
  int *mask = atom->mask, *type = atom->type;
  int nth = comm->nthreads;
  double beta1 = sqrt(qeq_dot(b, b));
  if (beta1 == 0.0) return 0;                                       // x already 0 (T1: or the seed is exact)
  // T1 (ΔSCF trial seed): when the caller solves the SEEDED DELTA system Ã·δ = b − Ã·x0 (x0 accumulated in x),
  // the stopping criterion must stay referenced to the FULL-BO RHS norm (passed as bref) — phibar ≤ tol·bref is
  // then the SAME absolute residual bound the cold full solve satisfies, so both converge to the same solution
  // to the same tolerance (fewer digits needed from the small delta RHS = the whole speedup). bref<=0 (every
  // pre-existing call site) keeps the original ||b||-relative criterion — byte-identical.
  const double bnorm_ref = (bref > 0.0) ? bref : beta1;
  // P1: every per-iteration loop below is elementwise (each ii touches only its own index i) -- embarrassingly
  // parallel, no race, no reduction; qeq_matvec/qeq_dot are threaded internally.
#if defined(_OPENMP)
  if (omp_go(nth, nn)) {
#pragma omp parallel for num_threads(nth) schedule(static)
    for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit){
      q_r[i]=b[i]; q_q[i]=b[i]; q_p[i]=0.0; m_w2[i]=0.0; } }
  } else
#endif
  { for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit){
      q_r[i]=b[i]; q_q[i]=b[i]; q_p[i]=0.0; m_w2[i]=0.0; } } }           // r1=r2=b, w=w2=0
  double beta=beta1, oldb=0.0, dbar=0.0, epsln=0.0, phibar=beta1, cs=-1.0, sn=0.0;
  int it = 1;
  for (; it <= imax; it++) {
    double s = 1.0/beta;
#if defined(_OPENMP)
    if (omp_go(nth, nn)) {
#pragma omp parallel for num_threads(nth) schedule(static)
      for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit) q_d[i]=s*q_q[i]; }
    } else
#endif
    { for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit) q_d[i]=s*q_q[i]; } }   // v = r2/beta
    qeq_matvec(q_d, m_t);                                           // y = Ã·v (projected)
    if (it >= 2) {
#if defined(_OPENMP)
      if (omp_go(nth, nn)) {
#pragma omp parallel for num_threads(nth) schedule(static)
        for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit) m_t[i]-=(beta/oldb)*q_r[i]; }
      } else
#endif
      { for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit) m_t[i]-=(beta/oldb)*q_r[i]; } }
    }
    double alfa = qeq_dot(q_d, m_t);
#if defined(_OPENMP)
    if (omp_go(nth, nn)) {
#pragma omp parallel for num_threads(nth) schedule(static)
      for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit) m_t[i]-=(alfa/beta)*q_q[i]; }
    } else
#endif
    { for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit) m_t[i]-=(alfa/beta)*q_q[i]; } }   // y -= (alfa/beta) r2
#if defined(_OPENMP)
    if (omp_go(nth, nn)) {
#pragma omp parallel for num_threads(nth) schedule(static)
      for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit){ q_r[i]=q_q[i]; q_q[i]=m_t[i]; } }
    } else
#endif
    { for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit){ q_r[i]=q_q[i]; q_q[i]=m_t[i]; } } }  // r1=r2; r2=y
    oldb = beta;
    beta = sqrt(qeq_dot(q_q, q_q));                                 // beta_{k+1} = ||r2||
    // apply previous Givens, then form the next rotation (eliminate beta)
    double oldeps = epsln;
    double delta = cs*dbar + sn*alfa;
    double gbar  = sn*dbar - cs*alfa;
    epsln = sn*beta;
    dbar  = -cs*beta;
    double gamma = sqrt(gbar*gbar + beta*beta); if (gamma < 1.0e-300) gamma = 1.0e-300;
    cs = gbar/gamma; sn = beta/gamma;
    double phi = cs*phibar; phibar = sn*phibar;
    // x += phi*w ; w-recurrence (w2=w_{k-2}=m_w2, w=w_{k-1}=q_p)
    double denom = 1.0/gamma;
#if defined(_OPENMP)
    if (omp_go(nth, nn)) {
#pragma omp parallel for num_threads(nth) schedule(static)
      for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(!(mask[i]&groupbit)) continue;
        double wk = (q_d[i] - oldeps*m_w2[i] - delta*q_p[i])*denom;
        m_w2[i] = q_p[i]; q_p[i] = wk; x[i] += phi*wk; }
    } else
#endif
    { for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(!(mask[i]&groupbit)) continue;
      double wk = (q_d[i] - oldeps*m_w2[i] - delta*q_p[i])*denom;
      m_w2[i] = q_p[i]; q_p[i] = wk; x[i] += phi*wk; } }
    if (minres_qcap > 0.0 && sn > 0.995) {                          // iterative regularization, but ONLY when the
      // residual has STALLED: phibar_new = sn*phibar_old, so sn~1 ⇒ <0.5%/iter reduction = MINRES is growing x in
      // the near-null (catastrophe) direction without reducing the residual. A converging solve (sn<0.995) is NOT
      // truncated even if its x transiently overshoots |q|>cap (MINRES is non-monotonic in x) — this was the np=2
      // water NaN: a transient overshoot of a converging solve was truncated prematurely. Gate on the stall.
      double mq = 0.0;
#if defined(_OPENMP)
      if (omp_go(nth, nn)) {
        // P1: max is order-independent (exact regardless of accumulation order) -- per-thread partial max +
        // fixed-order combine is race-free and deterministic (see omp_reduce banner above).
        mq = omp_reduce(nn, nth, 0.0, omp_max, [&](int ii)->double{ int i=ilist[ii];
          if(!(mask[i]&groupbit)) return 0.0; double qi=x[i]+q0[type[i]]; return fabs(qi); });
      } else
#endif
      { for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit){ double qi=x[i]+q0[type[i]]; if(fabs(qi)>mq)mq=fabs(qi); } } }
      MPI_Allreduce(MPI_IN_PLACE, &mq, 1, MPI_DOUBLE, MPI_MAX, world);
      if (mq > minres_qcap) break;                                 // stalled AND blown up -> bounded regularized solution
    }
    // A7/R3 dimension note: phibar/bnorm_ref = ||resid||/||b|| UNpreconditioned -- numerator and denominator
    // share one dimension (E/e), a truly relative ratio -> unit-invariant, NO ev_scale factor (unlike the
    // Jacobi-preconditioned qeq_cg criterion above). 1e-300 = IEEE underflow guard, not a physical tol.
    // (T1: bnorm_ref == beta1 unless a seeded caller passed the full-BO reference norm -- see the banner.)
    if (phibar/bnorm_ref <= tolerance || beta <= 1.0e-300) break;   // residual small (or Lanczos breakdown)
  }
  if (it > imax && comm->me == 0)
    error->warning(FLERR, "samqeq projected MINRES did not converge ({} iters, resid/b={:.2e}) at step {}",
                   it, phibar/bnorm_ref, update->ntimestep);
  if (comm->me == 0 && (update->ntimestep == 0 || update->ntimestep % 200 == 0))  // #28 SOLVER-DIAG (periodic)
    utils::logmesg(lmp, "samqeq SOLVER-DIAG step {}: MINRES {} iters (={} matvecs){}, resid/b={:.2e}\n",
                   update->ntimestep, it, it, warmstart ? " [warm]" : "", phibar/bnorm_ref);
  return it;
}

/* (b) smallest eigenvalue of a symmetric tridiagonal (diag d[0..n-1], off-diag e[0..n-2]) by Sturm-sequence
   bisection: sturm(σ) = # negative LDL pivots of (T−σI) = # eigenvalues < σ. λ_min = smallest σ with ≥1.*/
static int sturm_neg_count(const double *d, const double *e, int n, double sigma)
{
  double q = d[0] - sigma; int c = (q < 0.0) ? 1 : 0;
  for (int i = 1; i < n; i++) {
    if (fabs(q) < 1.0e-300) q = (q < 0.0 ? -1.0 : 1.0)*1.0e-300;
    q = (d[i] - sigma) - e[i-1]*e[i-1]/q;
    if (q < 0.0) c++;
  }
  return c;
}
double FixQEqSam::tridiag_lambda_min(const double *d, const double *e, int n)
{
  if (n <= 0) return 0.0;
  if (n == 1) return d[0];
  double lo = d[0]-fabs(e[0]), hi = d[0]+fabs(e[0]);             // Gershgorin bounds
  for (int i = 1; i < n; i++) { double r = fabs(e[i-1]) + (i < n-1 ? fabs(e[i]) : 0.0);
    if (d[i]-r < lo) lo = d[i]-r;  if (d[i]+r > hi) hi = d[i]+r; }
  // A7: lo/hi are eigenvalue bounds of the eta+Coulomb operator (eV/e^2 family); the 1e-10 factor is a
  // relative tolerance (dimensionless, untouched) but the additive +1.0 floor is eV/e^2-anchored -> *ev_scale.
  for (int it = 0; it < 200 && (hi-lo) > 1.0e-10*(fabs(lo)+fabs(hi)+1.0*ev_scale); it++) {
    double mid = 0.5*(lo+hi);
    if (sturm_neg_count(d, e, n, mid) >= 1) hi = mid; else lo = mid;
  }
  return 0.5*(lo+hi);
}

/* (b) Lanczos estimate of the smallest eigenvalue of the BARE projected operator Ã₀ = P(η + Coulomb) (ridge=0)
   on the per-molecule-neutral subspace -- the operator the projected CG sees. m steps of the 3-term recurrence
   build a tridiagonal whose λ_min approximates Ã₀'s (extreme eigenvalues converge fast; no reorthogonalization
   needed -- Paige). Deterministic start vector ⇒ reproducible λ_min ⇒ deterministic ridge & charges. Reuses the
   CG scratch (q_p/q_d/q_q) -- called before the CG solve. ~m extra matvecs (incl. reciprocal).*/
double FixQEqSam::estimate_lambda_min(int m)
{
  int *mask = atom->mask; tagint *tag = atom->tag;
  double *vprev = q_p, *vcur = q_d, *w = q_q;
  double save_ridge = ridge_cur; ridge_cur = 0.0;                // λ_min of the BARE operator (matvec adds ridge_cur)
  if (m < 4) m = 4; if (m > 200) m = 200;
  for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; vcur[i] = (mask[i]&groupbit) ? sin(0.7*(double)tag[i]+0.3) : 0.0; }
  project_neutral(vcur);
  double nrm = sqrt(qeq_dot(vcur, vcur));
  if (nrm < 1.0e-300) { ridge_cur = save_ridge; return 0.0; }
  double inv = 1.0/nrm;
  for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit){ vcur[i]*=inv; vprev[i]=0.0; } }
  double *ad = new double[m]; double *bo = new double[m];        // ad[j]=diag; bo[j]=off-diag after row j
  double bprev = 0.0; int mm = 0;
  for (int j = 0; j < m; j++) {
    qeq_matvec(vcur, w);                                         // w = Ã₀·vcur (projected)
    double aj = qeq_dot(vcur, w);
    ad[j] = aj; mm = j+1;
    for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit) w[i] -= aj*vcur[i] + bprev*vprev[i]; }
    project_neutral(w);                                          // keep in the subspace
    double bj = sqrt(qeq_dot(w, w));
    bo[j] = bj;
    if (bj < 1.0e-10*ev_scale) break;                            // invariant subspace reached. A7: bj = ||w||
                                                                 // carries the operator's eV/e^2 scale (vcur is
                                                                 // normalized) -> eV-anchored threshold *ev_scale
    double ib = 1.0/bj;
    for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit){ vprev[i]=vcur[i]; vcur[i]=w[i]*ib; } }
    bprev = bj;
  }
  ridge_cur = save_ridge;
  double lmin = tridiag_lambda_min(ad, bo, mm);                  // bo[0..mm-2] are the off-diagonals
  // A Ritz value is a Rayleigh quotient on the Krylov space, so
  // it is an UPPER bound on lambda_min -- an unconverged m can only MISS a negative mode (m=15 read +2.17 eV on an
  // interface cell whose m>=60 value is -0.31). The convergence readout is free: the (mm-10)-step Ritz value from
  // the same tridiagonal. Logged for the first estimate; one warning if a later estimate is still moving.
  // Diagnostic only -- lmin itself is unchanged.
  if (mm >= 14 && comm->me == 0 && lmin_logged < 2) {
    const double dl = tridiag_lambda_min(ad, bo, mm - 10) - lmin;   // >= 0 (Cauchy interlacing on the tridiagonal)
    const bool moving = dl > 0.05*ev_scale;
    if (!lmin_logged)
      utils::logmesg(lmp, "samqeq: ridge eig lambda_min {:.4f} after {} Lanczos steps (it moved {:.4f} over the last 10"
                          " steps{}); engage target floor+delta = {:.4f}\n", lmin, mm, dl, moving ? " -- NOT converged:"
                          "an upper bound that may hide a negative mode; raise m or set delta" : "",
                     lam_floor + ridge_delta);
    else if (moving)
      error->warning(FLERR, "samqeq: ridge eig lambda_min {:.4f} still moved {:.4f} over the last 10 of {} Lanczos"
                            "steps at step {} -- the estimate is an UPPER bound and may hide a negative mode; raise m"
                            "(`fix_modify {} ridge eig <floor> <m>`, m >= 60 for interface cells). Warned once.",
                     lmin, dl, mm, update->ntimestep, id);
    lmin_logged = (lmin_logged == 0) ? 1 : (moving ? 2 : 1);
  }
  delete[] ad; delete[] bo;
  return lmin;
}

/* (re)size per-molecule scratch and cache molinv[m] = 1/(in-group atom count of molecule m).
   A5: mol == nullptr (atom_style charge) ⇒ maxmol 0 ⇒ nmol_ = 1 ⇒ one GLOBAL fragment.
   B9 (audit): COMPACT reindexing. Raw molecule IDs can be arbitrarily sparse (e.g. an ion tagged `mol 9001`
   on a 216-water box ⇒ raw nmol_ = 9003 for ~219 actual fragments), so every per-ITERATION projector
   allreduce/array (molsum here) used to scale with the MAX raw
   ID, not the fragment COUNT — ~40x wasted bandwidth on the hottest path in the code. Fix: build a dense
   compact map ONCE per solve (here, NOT per CG/MINRES iteration) and route every per-iteration read through
   it. nmol_ now MEANS nactive (the compact fragment count), not raw max(mol)+1 — every other reader in the
   package (project_neutral, qeq_solve, the kokkos S7/S8/S9 device path)
   was swept and updated to index by the per-atom compact slot cmol[i] (or, where a raw id is needed as an
   array subscript rather than an atom's own slot, by translating through mol2c/c2mol below).
   Determinism: the compact map is built from active (all-reduced BEFORE compaction) in ASCENDING raw-ID
   order, so mol2c/c2mol are IDENTICAL on every rank and independent of atom/proc order/np — reproducible.
   Byte-identity: dense-ID decks (all 24 regression cases) get a slot-count identical to before (mol2c is a
   near-identity relabeling); more importantly, MPI_Allreduce sums an array ELEMENTWISE, so a given molecule's
   accumulated value does not depend on which slot it lives in or how many other slots exist — same atoms,
   same FP accumulation order, same reduction ⇒ BYTE-IDENTICAL regardless of the raw-vs-compact array length.*/
void FixQEqSam::build_molinv()
{
  int *mask = atom->mask; tagint *mol = atom->molecule;
  auto PM = [&](int i) -> tagint { return mol[i]; };
  const bool have_pin = (mol != nullptr);

  // ★ S4 (scaling audit): CACHE HIT. Everything this function reduces/derives globally (the active raw-ID
  // set -> mol2c/c2mol/nmol_, the per-fragment in-group counts -> molsum/molinv) is INVARIANT under atom
  // migration: it depends only on the global multiset of (raw mol id, in-group membership), which does not
  // change between invalidation events. Under a `set group metal mol 900000` convention the pass-1 rebuild is
  // a ~3.6 MB zeroed alloc + ~3.6 MB MPI_Allreduce EVERY solve (and every Picard iteration) — pure waste.
  // Invalidation (molinv_valid=0) sites: ctor/init() (run boundary — covers deck-level `set`/group edits
  // between runs), deallocate_storage (arrays freed) and restart(). Atom insertion/deletion is caught by the global atom->natoms snapshot;
  // dynamic groups change membership without any hook ⇒ never cached.
  // ★ DEADLOCK SAFETY (— this REPLACES a false claim). The previous comment asserted every gate term
  // below is rank-uniform, and gated the branch on the bare local flags. It is not: reallocate_storage()
  // -> deallocate_storage() clears molinv_valid AND nulls molinv/mol2c, and its trigger is
  // `atom->nmax > nmax` — a PER-RANK condition that fires on whichever rank's atom arrays regrow at an
  // exchange event. The ranks then split at this gate: the regrown subset entered the collective rebuild
  // below (MPI_Allreduce) while the rest served the cache and marched on to the solve's own collectives —
  // mismatched collectives, permanent silent hang. Observed: np4 Drude MD, both diagnostics hung at the
  // first reneighbor-with-growth step, two ranks in build_molinv's Allreduce and two in coulomb_field's
  // forward_comm. The branch decision is therefore made COLLECTIVE: each rank forms a local verdict and an
  // Allreduce(MIN) makes the branch unanimous BY CONSTRUCTION — a rank that desynchronises for ANY reason
  // (this one or a future one) costs one forced rebuild, never a split branch. min!=max additionally trips
  // a one-shot warning so a recurrence is LOUD (the failure it replaces was 14+ min of silence at 92% CPU).
  // Cost: one 2-int Allreduce per BUILD (per solve entry, not per CG/MINRES iteration), next to the
  // projector's own per-iteration allreduce — noise. Serial (np1): the reduce is an identity and the
  // decision equals the old gate's, so the bit-suite path is untouched.
  // BIT-PRESERVATION: a full rebuild would recompute byte-identical values (pass 1 is a {0,1} MAX; the
  // molsum counts are integer-valued doubles, summed exactly in FP regardless of rank distribution), so
  // serving the cache changes nothing — and, equally, a forced rebuild on a rank whose cache was still
  // valid changes nothing. Only the per-LOCAL-atom cmol cache must be refreshed every call
  // (atoms migrate/reorder on reneighbor) — a local O(nlocal) pass, no comm.
  int mvok = (molinv_valid && molinv && mol2c && !group->dynamic[igroup]
              && molinv_natoms == atom->natoms) ? 1 : 0;
  int mvmm[2] = {mvok, -mvok};                       // one call reduces both: MIN(ok) and -MAX(ok)
  MPI_Allreduce(MPI_IN_PLACE, mvmm, 2, MPI_INT, MPI_MIN, world);
  if (mvmm[0] != -mvmm[1]) {                         // verdicts DIVERGED — the earlier deadlock condition
    if (!molinv_desync_note && comm->me == 0)
      error->warning(FLERR, "samqeq: build_molinv cache validity diverged across ranks at step {}"
                            "(per-rank atom-storage regrow; expected under migration). Recovered by a"
                            "collective rebuild — before this fix this was a permanent MPI deadlock",
                     update->ntimestep);
    molinv_desync_note = 1;                          // one-shot; set on ALL ranks (stays rank-uniform)
  }
  if (mvmm[0]) {
    if (have_pin) {
      for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(!(mask[i]&groupbit)) continue;
        tagint mr = PM(i);
        int c = (mr >= 0 && mr < (tagint)maxmol_) ? mol2c[mr] : -1;
        if (c < 0)   // invariant violated (an in-group mol id outside the cached active set): abort loudly
                     // (LAMMPS error->one convention) rather than solve with a corrupted projector.
          error->one(FLERR, "samqeq: cached molecule map is stale (raw mol id {} not in the cached active"
                     "set) at step {}", (long)mr, update->ntimestep);
        cmol[i] = c; }
    } else {
      for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit) cmol[i] = 0; }   // A5: one global fragment
    }
    return;
  }

  tagint maxmol_l = 0;
  if (have_pin) for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if((mask[i]&groupbit) && PM(i)>maxmol_l) maxmol_l=PM(i); }
  tagint maxmol; MPI_Allreduce(&maxmol_l, &maxmol, 1, MPI_LMP_TAGINT, MPI_MAX, world);
  int nraw = (int)maxmol + 1;                        // raw ID space (0..maxmol); A5 (mol==nullptr) ⇒ nraw=1

  // pass 1 (replaces the OLD nmol_-length molsum count-reduce with an equivalent-cost single allreduce, still
  // ONE per BUILD i.e. per solve — never per CG/MINRES iteration): mark which raw IDs have ≥1 in-group atom
  // on ANY rank. int reduce buffer is still 2x smaller than the double molsum it replaces here (and MPI_INT
  // is portable under the Kokkos/nvcc host-compile path, where MPI_SIGNED_CHAR is not always declared).
  if (nraw != maxmol_) { memory->destroy(mol2c); memory->create(mol2c, nraw, "samqeq:mol2c"); maxmol_ = nraw; }
  // S4: persistent pass-1 scratch, grown monotonically (was `new int[nraw]` — a 3.6 MB heap-churn per
  // solve under the mol-900000 convention even before the caching above).
  if (nraw > molinv_active_cap) {
    memory->destroy(molinv_active);
    memory->create(molinv_active, nraw, "samqeq:molinv_active");
    molinv_active_cap = nraw;
  }
  int *active = molinv_active;
  memset(active, 0, (size_t)nraw*sizeof(int));
  if (have_pin) { for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit) active[PM(i)] = 1; } }
  else     { for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit){ active[0] = 1; break; } } }
  MPI_Allreduce(MPI_IN_PLACE, active, nraw, MPI_INT, MPI_MAX, world);

  // deterministic ascending-raw-ID compaction (see the function banner above for the determinism + byte-id
  // argument): mol2c[raw] = compact index, or -1 for a raw ID with no current in-group atom anywhere.
  int nactive = 0;
  for (int m = 0; m < nraw; m++) mol2c[m] = active[m] ? nactive++ : -1;
  if (nactive < 1) nactive = 1;                      // floor (mirrors the old maxmol+1 ≥ 1 invariant)

  if (nactive != nmol_) { memory->destroy(molinv); memory->destroy(molsum);
    memory->destroy(c2mol);
    // S2: molsum carries PN_TAIL_MAX extra slots so project_neutral can append fused-reduction tails
    // (the dot partial) to the same Allreduce message; slots [0,nmol_) semantics unchanged.
    memory->create(molinv, nactive, "samqeq:molinv"); memory->create(molsum, nactive+PN_TAIL_MAX, "samqeq:molsum");
    memory->create(c2mol, nactive, "samqeq:c2mol");
    nmol_ = nactive; }
  for (int c = 0; c < nmol_; c++) c2mol[c] = 0;        // defensive default; only ever read for a c that DOES
                                                       // come from an active raw id via the fill below
  for (int m = 0; m < nraw; m++) if (mol2c[m] >= 0) c2mol[mol2c[m]] = m;

  // per-LOCAL-atom compact index cache (nmax array; project_neutral/qeq_solve only ever read
  // cmol[i] for i in ilist — local owned atoms — so ghosts are neither filled nor read).
  for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit) cmol[i] = have_pin ? mol2c[PM(i)] : 0; }

  for (int m=0;m<nmol_;m++) molsum[m]=0.0;
  for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit) molsum[cmol[i]] += 1.0; }
  MPI_Allreduce(MPI_IN_PLACE, molsum, nmol_, MPI_DOUBLE, MPI_SUM, world);
  for (int m=0;m<nmol_;m++) molinv[m] = (molsum[m]>0.0) ? 1.0/molsum[m] : 0.0;
  molinv_valid = 1; molinv_natoms = atom->natoms;    // S4: cache is now current (see the cache-hit banner above)
}

/* ionfield: fixq_field[i] (i in the solve group) = shielded Coulomb field from the FIXED non-group charges, in the
   SAME split the matvec uses between solved atoms (net = J_shield): real-space (FULL list, J_shield(γ) − erf(αr)/r)
   + reciprocal (compute_vector with invert_source=true = potential at the solved atoms from the non-group
   charges). Sources are fixed, so it is computed once per solve and enters the RHS.*/
void FixQEqSam::add_fixed_charge_field()
{
  double **x = atom->x; double *q = atom->q; int *type = atom->type, *mask = atom->mask;
  double a = lr_alpha, pref = force->qqrd2e;
  for (int i = 0; i < atom->nmax; i++) fixq_field[i] = 0.0;

  // (perf_s90/ionfield_s90): SKIP the pass when there is no SOURCE anywhere -- no atom OUTSIDE the solve
  // group with a nonzero charge among this rank's locals or ghosts, on any rank. `ionfield` is on by default
  //, so every deck whose group covers all charged atoms paid a full-list sweep plus a full HOST
  // compute_vector per solve for a field that is identically zero: 55 % of host samples and 2.2x the loop on
  // the A100 (99k-atom litpfa_fm, nsys, 2026-09-26). On CPU it is only ~2 % of the loop (callgrind): with no
  // source the sweep is one mask test per neighbour -- the A100 cost is that sweep on ONE host thread over the
  // whole system while everything else runs on device, plus a host<->device compute_vector round trip per solve.
  // Skipping is BIT-IDENTICAL by construction: with every source charge zero the real-space sum adds only
  // (finite)*0.0 and make_rho deposits an all-zero density (FFT -> greensfn -> FFT -> stencil of exact zeros), so
  // the full pass leaves fixq_field at the +0.0 written above (proof: ionfield_s90/REPORT.md s2). Locals AND
  // ghosts are scanned because ghosts are what the full-list sweep reads: a ghost recharged since the last border
  // comm errs on the side of running the pass. One MPI_MAX Allreduce per call; every call site is collective
  // (pre_force / init_matvec / XL), so every rank takes the same branch.
  // NOT cached on purpose: fix gcmc/deposit/evaporate, `set` and fix
  // adapt would all have to invalidate it, and the key costs the same O(nall) scan.
  // Golden `ionfield_cl_water` keeps the source branch under test.
  {
    const int nall = atom->nlocal + atom->nghost;
    int src_local = 0;
    for (int j = 0; j < nall; j++)
      if (!(mask[j] & groupbit) && q[j] != 0.0) { src_local = 1; break; }
    int src_any = 0;
    MPI_Allreduce(&src_local, &src_any, 1, MPI_INT, MPI_MAX, world);
    if (!src_any) return;                                    // field == +0.0 exactly; nothing to add
  }

  // real-space: each solved atom's non-group neighbors via the FULL list (so it sees ALL of them)
  NeighList *fl = list_full ? list_full : list;
  int inum = fl->inum; int *ilst = fl->ilist; int *nnb = fl->numneigh; int **fnb = fl->firstneigh;
  for (int ii = 0; ii < inum; ii++) {
    int i = ilst[ii];
    if (!(mask[i] & groupbit)) continue;                       // field only ON solved atoms
    int *jl = fnb[i]; int jn = nnb[i]; double fi = 0.0;
    for (int jj = 0; jj < jn; jj++) {
      int j = jl[jj] & NEIGHMASK;
      if (mask[j] & groupbit) continue;                        // only FIXED non-group charges as sources
      double dx=x[j][0]-x[i][0], dy=x[j][1]-x[i][1], dz=x[j][2]-x[i][2];
      double rsq = dx*dx+dy*dy+dz*dz;
      if (rsq > swb*swb || rsq < 1e-10) continue;
      double r = sqrt(rsq);
      double shielded = shielded_coulomb(type[i], type[j], r);  // cbrt J_shield or Gaussian (consistent w/ compute_H)
      if (lr_nrecip) fi += shielded * taper_poly(r) * q[j];    // taper-cutoff: full tapered shielded field, no reciprocal
      else fi += (shielded - (1.0 - erfc(a*r))/r) * q[j];      // Ewald real-space (net with the reciprocal = shielded)
    }
    fixq_field[i] += pref * fi;
  }

  if (lr_nrecip) return;                                       // taper-cutoff: field is real-space only, no reciprocal
  // reciprocal: potential at the solved atoms from the non-group charges (invert_source flips the source group)
  for (int i = 0; i < atom->nlocal; i++) prec[i] = 0.0;        // E2: compute_vector writes/reads locals only
  eksp->compute_vector(prec, groupbit, groupbit, true);        // sensors = solved atoms, sources = non-group
  for (int ii = 0; ii < nn; ii++) { int i=ilist[ii]; if(!(mask[i]&groupbit)) continue;
    fixq_field[i] += pref * prec[i]; }
}

/* the MINRES solve path: bare projected operator (no ridge), symmetric-INDEFINITE-safe. Primary solver when
   `solver minres`, AND the auto-fallback when CG diverges. Assumes build_molinv() has run.*/
/* FREEZE-AND-RIDE-THROUGH: when a solve over-polarizes past lr_qfreeze even after the adaptive ridge / MINRES,
   committing it injects energy into the dynamics (the RexPoN-Drude 300 K T-spike chaos). Return true (and leave
   atom->q at the PREVIOUS step's values — the caller then skips its commit AND its ASPC-history advance) so the
   frozen-charge dynamics rides through the transient near-singular config. A genuinely stuck config (frozen for
   freeze_max consecutive steps) is a hard error. lr_qfreeze<=0 disables (byte-identical).*/
bool FixQEqSam::freeze_overpolarized(double maxq)
{
  if (lr_qfreeze <= 0.0 || maxq <= lr_qfreeze) { freeze_nconsec = 0; return false; }
  freeze_nconsec++;
  if (comm->me == 0 && (freeze_nconsec <= 3 || (freeze_nconsec % 100) == 0))
    error->warning(FLERR, "samqeq: over-polarized solve (max|q-q0|={:.3g} > qfreeze={:.3g}) at step {} -> FREEZING"
                   "charges at the previous step ({} consecutive)", maxq, lr_qfreeze, update->ntimestep,
                   (long)freeze_nconsec);
  if (freeze_max > 0 && freeze_nconsec > freeze_max)
    error->all(FLERR, "samqeq: charges frozen {} consecutive steps (config stuck near-singular) at step {} --"
               "raise the `freeze` cap, reduce the timestep, or fix the model", (long)freeze_nconsec,
               update->ntimestep);
  comm_v = atom->q; pack_flag = 6; comm->forward_comm(this);   // refresh ghosts with the (unchanged) charges
  return true;
}

void FixQEqSam::qeq_solve_minres()
{
  int *type = atom->type, *mask = atom->mask; double *qa = atom->q;
  int nth = comm->nthreads;
  ridge_cur = lr_ridge; ridge_local = false;         // bare operator (MINRES handles the negative mode itself)
  // #20 fix: a SEEDED MINRES must solve the DELTA system. qeq_minres never forms
  // b − Ã·x0 (it initializes r = b and ACCUMULATES corrections onto the passed x), so the historical
  // `warmstart` seed here silently produced x0 + Ã⁻¹b (seed + FULL solution) — the opt-in warmstart+minres
  // combo was latently broken and can never have been used in anger. Correct seeded form: shift the RHS to
  // qb ← qb − Ã·x0 and let qeq_minres accumulate δ = Ã⁻¹(b − Ã·x0) onto x0, stopping at the SAME absolute
  // residual bound as the cold solve (phibar ≤ tol·||b_full||, via bref). want_seed = the #20 opt-in
  // (`solver warmstart on`).
  // pass 0 = seeded; pass 1 = COLD canonical full-BO (the only pass when !want_seed => byte-identical, and
  // the FALLBACK if the seeded pass fails to converge — the answer is never approximate).
  const int want_seed = warmstart;
  for (int pass = (want_seed ? 0 : 1); pass < 2; pass++) {
    const int seed0 = (pass == 0);
    // P1: RHS/Hdia_inv/qs fill is elementwise (each ii writes only its own index i) -- embarrassingly parallel.
#if defined(_OPENMP)
    if (omp_go(nth, nn)) {
#pragma omp parallel for num_threads(nth) schedule(static)
      for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(!(mask[i]&groupbit)) continue;
        Hdia_inv[i] = 1.0/((solve_diag_of(i))
                           + ridge_cur);
        qb[i] = -(chi_b(i) + (ionfield_flag ? fixq_field[i] : 0.0) + q0field[i]);
        if (efield) qb[i] -= chi_field[i];
        qs[i] = seed0 ? (qa[i] - q0[type[i]]) : 0.0; }
    } else
#endif
    { for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(!(mask[i]&groupbit)) continue;
      Hdia_inv[i] = 1.0/((solve_diag_of(i))
                         + ridge_cur);    // (unused by MINRES; kept consistent)
      qb[i] = -(chi_b(i) + (ionfield_flag ? fixq_field[i] : 0.0) + q0field[i]);   // + q0 reference-charge field (conservation)
      if (efield) qb[i] -= chi_field[i];                                        // B2: external-field coupling (fix efield)
      qs[i] = seed0 ? (qa[i] - q0[type[i]]) : 0.0; } }   // #20/T1 warm-start from the previous charges
    if (seed0) project_neutral(qs);                                  // keep the seed in the neutral subspace
    project_neutral(qb);
    double bref = -1.0;
    if (seed0) {
      bref = sqrt(qeq_dot(qb, qb));      // FULL-BO RHS norm -> same absolute stopping bound as the cold solve
      qeq_matvec(qs, m_t);               // delta shift: qb ← b − Ã·x0 (qeq_minres accumulates δ onto x0 = qs)
#if defined(_OPENMP)
      if (omp_go(nth, nn)) {
#pragma omp parallel for num_threads(nth) schedule(static)
        for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit) qb[i] -= m_t[i]; }
      } else
#endif
      { for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit) qb[i] -= m_t[i]; } }
    }
    matvecs = qeq_minres(qb, qs, bref);
    if (!(seed0 && matvecs > imax)) break;   // converged (or the cold pass) -> done
    if (comm->me == 0)                       // seeded pass failed -> FULL-BO cold fallback (never approximate)
      error->warning(FLERR, "samqeq: seeded (DeltaSCF) MINRES did not converge at step {} -> cold full-BO re-solve",
                     update->ntimestep);
  }
  if (lr_qfreeze > 0.0) {                                             // FREEZE guard (skip if disabled => byte-id)
    double mq = 0.0;
#if defined(_OPENMP)
    if (omp_go(nth, nn)) {
      mq = omp_reduce(nn, nth, 0.0, omp_max, [&](int ii)->double{ int i=ilist[ii];
        if(!(mask[i]&groupbit)) return 0.0; double qi = qs[i];
        return std::isfinite(qi) ? fabs(qi) : 1.0e30; });             // A6: DEVIATION from q0; non-finite -> sentinel
    } else
#endif
    { for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(!(mask[i]&groupbit)) continue;
      double qi = qs[i];                               // A6: DEVIATION from q0 (formal-charge ions
      if (!std::isfinite(qi)) { mq = 1.0e30; break; } if (fabs(qi) > mq) mq = fabs(qi); } }   // must not trip on |q|)
    MPI_Allreduce(MPI_IN_PLACE, &mq, 1, MPI_DOUBLE, MPI_MAX, world);
    if (freeze_overpolarized(mq)) return;                            // over-polarized -> keep previous charges
  }
#if defined(_OPENMP)
  if (omp_go(nth, nn)) {
#pragma omp parallel for num_threads(nth) schedule(static)
    for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit) qa[i] = qs[i] + q0[type[i]]; }
  } else
#endif
  { for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit) qa[i] = qs[i] + q0[type[i]]; } }
  comm_v = qa; pack_flag = 6; comm->forward_comm(this);
}

/* plain-QEq solve in the per-molecule-neutral subspace: build molinv (per-mol 1/count), RHS b=P(−χ),
   projected CG -> qs (already per-molecule neutral), atom->q = qs + q0.*/
void FixQEqSam::qeq_solve()
{

  int *type = atom->type, *mask = atom->mask; double *qa = atom->q;
  int nth = comm->nthreads;
  { const double t0 = (th_on == 1) ? platform::walltime() : 0.0;
    build_molinv();
    build_mol_blocks();   // (#30 item 5): molecular block-Jacobi factors (no-op unless `precond mol`)
    if (th_on == 1) th_molinv += platform::walltime() - t0; }

  // P1 : every fill/zero loop in this function is elementwise (disjoint per-i writes) -- embarrassingly
  // parallel, no race, no reduction -- unless noted otherwise (the mq/any_over reductions below are called out).

  // q0 reference-charge Coulomb field (energy-conservation): the physical forces use qa = qs+q0, so the RHS
  // must carry [J·q0] (the field from the fixed q0 reference charges). Compute it ONCE per solve and reuse
  // across all RHS builds (ASPC / adaptive-ridge CG / MINRES / auto-fallback). q0=0 models skip it (q0field
  // stays 0 -> byte-identical, no extra matvec). Mirrors the base ACKS2 path's `reffield` (fix_qeq_sam.cpp).
  if (has_q0ref) {
#if defined(_OPENMP)
    if (omp_go(nth, atom->nmax)) {
#pragma omp parallel for num_threads(nth) schedule(static)
      for (int i = 0; i < atom->nmax; i++) m_t[i] = 0.0;
    } else
#endif
    { for (int i = 0; i < atom->nmax; i++) m_t[i] = 0.0; }           // m_t = q0 vector (free scratch here)
#if defined(_OPENMP)
    if (omp_go(nth, nn)) {
#pragma omp parallel for num_threads(nth) schedule(static)
      for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit) m_t[i] = q0[type[i]]; }
    } else
#endif
    { for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit) m_t[i] = q0[type[i]]; } }
    { const double t0 = (th_on == 1) ? platform::walltime() : 0.0;
      coulomb_field(m_t, q0field);                                 // q0field = P(J_offdiag · q0)
      if (th_on == 1) th_q0field += platform::walltime() - t0; }
    q0field_step = update->ntimestep;                              // T1: stamp the step this array is valid for
  } else {
#if defined(_OPENMP)
    if (omp_go(nth, atom->nmax)) {
#pragma omp parallel for num_threads(nth) schedule(static)
      for (int i = 0; i < atom->nmax; i++) q0field[i] = 0.0;
    } else
#endif
    { for (int i = 0; i < atom->nmax; i++) q0field[i] = 0.0; }
  }

  // ★ ASPC (#16) — predictor-corrector on the PRODUCTION projected-CG path (the lr_ewald=2 solver). qeq_solve
  // is already q-direct (qs = neutral charge perturbation, qa = qs + q0). ASPC: build a multi-step
  // time-reversible predictor from q_hist (vs the existing 1-step warm-start), cap qeq_cg at n_corr, omega-mix.
  // Reset history on neighbor rebuild. Falls through to the full BO solve below on runaway. UNTESTED draft (#16).
  bool aspc_reject = false;   // B5(c): set iff an ASPC corrector actually ran and failed the accept gate below --
                              // used to suppress the q_hist push in the BO fallback (an SCF-inconsistent sample).
  if (aspc_on) {
    // q_hist now MIGRATES with atoms (copy_arrays/pack_exchange in FixACKS2Sam) -> no per-reneighbor reset; the
    // warmup (aspc_have climbing to nhist via the BO finalize) happens once at the start of dynamics.
    if (aspc_nhist != aspc_korder + 2 || ngroup_fq <= 0) aspc_setup();
    if (aspc_have >= aspc_nhist) {
      // P1: ASPC predictor is embarrassingly parallel (each ii's inner j-loop over aspc_nhist~4-6 history
      // slots touches only q_hist[i][*] and writes only qs[i] -- no cross-atom dependency).
#if defined(_OPENMP)
      if (omp_go(nth, nn)) {
#pragma omp parallel for num_threads(nth) schedule(static)
        for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(!(mask[i]&groupbit)) continue;
          double acc=0.0; for (int j=0;j<aspc_nhist;j++) acc += aspc_B[j]*q_hist[i][j];
          qs[i]=acc; }
      } else
#endif
      { for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(!(mask[i]&groupbit)) continue;
        double acc=0.0; for (int j=0;j<aspc_nhist;j++) acc += aspc_B[j]*q_hist[i][j];
        qs[i]=acc; } }                                  // ASPC predictor (per-molecule-neutral perturbation)
      if (lr_quartic) {   // ASPC-quartic: set the charge-dependent eta_eff AT the predictor (≈ q*) so this ONE
        // B5(a): snapshot atom->q (qsave = free CG-path scratch here) before overwriting it with the predictor
        // charge -- apply_quartic_eta() only needs atom->q to EVALUATE eta_eff(q*); restore right after so a
        // rejected corrector's BO fallback below warm-starts from the real previous-step charges (not the
        // predictor), and freeze_overpolarized (if it fires) freezes the real previous charges too.
#if defined(_OPENMP)
        if (omp_go(nth, nn)) {
#pragma omp parallel for num_threads(nth) schedule(static)
          for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit) { qsave[i]=qa[i]; qa[i]=qs[i]+q0[type[i]]; } }
        } else
#endif
        { for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit) { qsave[i]=qa[i]; qa[i]=qs[i]+q0[type[i]]; } } }
        apply_quartic_eta();                          // capped-CG corrector then refines q* at eta_eff(predictor)
#if defined(_OPENMP)
        if (omp_go(nth, nn)) {
#pragma omp parallel for num_threads(nth) schedule(static)
          for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit) qa[i]=qsave[i]; }
        } else
#endif
        { for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit) qa[i]=qsave[i]; } }   // restore pre-predictor q
      }
      // ★ S1a (scaling audit): the corrector must solve the SAME (ridged) operator the accepted BO path
      // solves. The BO path below presets ridge_cur from the Lanczos λ_min when ridge_mode==2 (`ridge eig`);
      // the corrector used to hardwire ridge_cur=lr_ridge, so on a near-critical operator (λ_min far below
      // lam_floor) a capped ~2-iter CG could never reach aspc_rtol -> permanent reject -> ASPC dead
      // (~29 matvecs/step instead of ~2-8). Same policy incl. the ridge_every cache; a refresh here also
      // serves the BO fallback on a reject (lmin_step==ntimestep -> it reuses lmin_cache, no double Lanczos).
      // Placed BEFORE the Hdia_inv fill so preconditioner and corrector CG see one consistent ridge.
      // Byte-identical when ridge_mode!=2 (ridge_cur = lr_ridge, exactly the old hardwire) or aspc off.
      ridge_cur = lr_ridge; ridge_local = false;
      if (ridge_mode == 2) {
        if (ridge_every <= 1 || lmin_step < 0 || (update->ntimestep - lmin_step) >= ridge_every) {
          lmin_cache = estimate_lambda_min(nlanczos);
          lmin_step = update->ntimestep;
        }
        const double lam_target = lam_floor + ridge_delta;   // == lam_floor bitwise when delta = 0
        if (lam_target - lmin_cache > ridge_cur) ridge_cur = lam_target - lmin_cache;   // λ_min(A+ridge) ≥ floor if err ≤ δ
      }
#if defined(_OPENMP)
      if (omp_go(nth, nn)) {
#pragma omp parallel for num_threads(nth) schedule(static)
        for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(!(mask[i]&groupbit)) continue;
          Hdia_inv[i]=1.0/((solve_diag_of(i))+ridge_cur);   // S1a: match the ridged operator
          qb[i]=-(chi_b(i)+(ionfield_flag?fixq_field[i]:0.0)+q0field[i]); }
      } else
#endif
      { for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(!(mask[i]&groupbit)) continue;
        Hdia_inv[i]=1.0/((solve_diag_of(i))+ridge_cur);     // S1a: match the ridged operator
        qb[i]=-(chi_b(i)+(ionfield_flag?fixq_field[i]:0.0)+q0field[i]); } }   // + q0 reference-charge field (conservation)
      if (efield) {
#if defined(_OPENMP)
        if (omp_go(nth, nn)) {
#pragma omp parallel for num_threads(nth) schedule(static)
          for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit) qb[i] -= chi_field[i]; }
        } else
#endif
        { for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit) qb[i] -= chi_field[i]; } }   // B2
      }
      project_neutral(qs); project_neutral(qb);
      // (S1a: ridge_cur/ridge_local are set ABOVE, before the Hdia_inv fill)
#if defined(_OPENMP)
      if (omp_go(nth, nn)) {
#pragma omp parallel for num_threads(nth) schedule(static)
        for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit) m_t[i]=qs[i]; }
      } else
#endif
      { for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit) m_t[i]=qs[i]; } }   // save predictor (m_t = free CG-path scratch)
      int imax_save=imax; imax=aspc_ncorr; aspc_corr_call = true; int it;   // D1: the capped corrector never warns
      { const double t0 = (th_on == 1) ? platform::walltime() : 0.0; it=qeq_cg(qb, qs); imax=imax_save; if (th_on == 1) th_cg += platform::walltime() - t0; }            // capped corrector
      aspc_corr_call = false;
      double rresid=cg_relresid;   // corrector solve quality (globally reduced in qeq_cg) -> accept gate below
      double w=aspc_omega, mq=0.0;
#if defined(_OPENMP)
      if (omp_go(nth, nn)) {
#pragma omp parallel for num_threads(nth) schedule(static)
        for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit) qs[i]=w*qs[i]+(1.0-w)*m_t[i]; }
      } else
#endif
      { for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit) qs[i]=w*qs[i]+(1.0-w)*m_t[i]; } }  // omega-damped mix
      // : on SOLVER-DIAG steps also measure the residual of the COMMITTED (omega-mixed,
      // projected) charge. The corrector's own resid/b is taken BEFORE the mix, and the mix keeps (1-omega) of the
      // predictor error whatever n_corr is -- so resid/b alone overstates the accuracy. Same preconditioned,
      // projected norm as qeq_cg; one extra matvec per 200 steps. Rank-uniform condition (qeq_matvec communicates).
      if (update->ntimestep == 0 || update->ntimestep % 200 == 0) {
        qeq_matvec(qs, q_q);
        for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit) q_r[i]=qb[i]-q_q[i]; }
        apply_precond(q_r, q_d);
        project_neutral(q_d);
        const double cres = sqrt(fabs(qeq_dot(q_r, q_d)))/cg_bnorm;
        if (comm->me == 0)
          utils::logmesg(lmp, "samqeq ASPC-DIAG step {}: resid/b corrector={:.2e} committed={:.2e} (omega={:.4g});"
                              "accepted steps above tolerance {}/{}\n", update->ntimestep, rresid, cres, w,
                         aspc_nabove, aspc_naccept);
      }
#if defined(_OPENMP)
      if (omp_go(nth, nn)) {
        mq = omp_reduce(nn, nth, 0.0, omp_max, [&](int ii)->double{ int i=ilist[ii];
          if(!(mask[i]&groupbit)) return 0.0; double qi=qs[i];
          return std::isfinite(qi) ? fabs(qi) : 1.0e30; });
      } else
#endif
      { for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(!(mask[i]&groupbit)) continue;
        double qi=qs[i]; if(!std::isfinite(qi)){mq=1e30;break;} if(fabs(qi)>mq)mq=fabs(qi); } }   // A6: |q-q0|
      MPI_Allreduce(MPI_IN_PLACE,&mq,1,MPI_DOUBLE,MPI_MAX,world);
      // ACCEPT only if the corrector is BOTH non-runaway (max|q| < the same over-polarization threshold the BO
      // freeze guard uses, lr_qfreeze if set else 4.0) AND actually converged (rel-resid < aspc_rtol). The
      // max|q| gate alone let high-bias-metal predictions through with resid/b~2 (charges bounded but WRONG,
      // contaminating q(V)); the residual gate falls those through to the exact BO solve below. SCOPE 2c (#16).
      // A7/R3: rresid = cg_relresid = sqrt(sig)/bnorm carries e/sqrt(E) units (same criterion as qeq_cg),
      // so aspc_rtol gets the same /sqrt(ev_scale) correction (exact /1.0 in metal). The 4.0 is charge (e).
      if (mq < (lr_qfreeze > 0.0 ? lr_qfreeze : 4.0) && rresid < aspc_rtol/sqrt(ev_scale)) {   // accept (converged, not runaway)
        freeze_nconsec = 0;                                                                   // a good commit ends any freeze run
        aspc_naccept++;                                     // S1b diag: cumulative accept count (SOLVER-DIAG acc/rej)
        if (rresid > tolerance/sqrt(ev_scale)) {            // D3: accepted, but above the user's tolerance
          aspc_nabove++;
          if (++aspc_nabove_run >= 100 && !aspc_above_warned) {
            aspc_above_warned = 1;
            if (comm->me == 0)
              error->warning(FLERR, "samqeq ASPC: the capped corrector (n_corr={}) has been above `tolerance` on 100"
                                    "consecutive accepted steps (last resid/b {:.2e}): the committed charge keeps"
                                    "(1-omega)={:.2f} of the predictor error whatever n_corr is, so this is a property of"
                                    "the predictor (how smoothly q(t) moves), not of the"
                                    "corrector. If Born-Oppenheimer accuracy is required use `aspc off` (~3.5x the"
                                    "matvecs). Warned once.", aspc_ncorr, rresid, 1.0 - w);
          }
        } else aspc_nabove_run = 0;
        // P1: commit + q_hist shift is embarrassingly parallel (each ii owns its own q_hist[i][*] row).
#if defined(_OPENMP)
        if (omp_go(nth, nn)) {
#pragma omp parallel for num_threads(nth) schedule(static)
          for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(!(mask[i]&groupbit)) continue;
            qa[i]=qs[i]+q0[type[i]];
            for (int j=aspc_nhist-1;j>0;j--) q_hist[i][j]=q_hist[i][j-1];
            q_hist[i][0]=qs[i]; }
        } else
#endif
        { for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(!(mask[i]&groupbit)) continue;
          qa[i]=qs[i]+q0[type[i]];
          for (int j=aspc_nhist-1;j>0;j--) q_hist[i][j]=q_hist[i][j-1];
          q_hist[i][0]=qs[i]; } }                                                             // store neutral perturbation
        matvecs=it; comm_v=qa; pack_flag=6; comm->forward_comm(this);
        return;
      }
      aspc_reject = true;   // runaway/non-converged corrector -> fall through to the full BO solve below
                            // (S1b: the BO solution IS pushed into q_hist there and the history depth is
                            // reset -- see the revised B5(c) rationale at the q_hist push)
      aspc_nreject++;       // S1b diag: cumulative reject count (SOLVER-DIAG acc/rej; a dead ASPC is visible)
    }
  }

  if (use_minres) { qeq_solve_minres(); return; }    // #M-A primary MINRES path (indefinite-safe; no ridge)

  // ADAPTIVE-RIDGE solve. The SPC-FQ+full-Ewald operator is near-critical: at rare configs the collective
  // polarization mode goes soft and the (converged) charge runs away. A FIXED ridge can't help (the average
  // is near-critical too, so any ridge big enough to bound the worst config over-hardens the average and
  // kills the dipole). Instead: solve at the base ridge (usually 0, preserving the dipole); only if a config
  // produces a runaway charge, re-solve THAT step with an escalating Tikhonov ridge until bounded. Good
  // configs (the vast majority) are untouched; rare bad configs are bounded instead of crashing.
  const double QGUARD = 4.0;            // |q-q0| above this = runaway (deviation metric, A6: formal-charge
                                        // ions with q0=+2..+4 must not sit permanently above the guard)
  int it = 0; double maxq = 0.0; int nridge = 0;
  ridge_cur = lr_ridge;
  double lmin_est = 0.0;
  if (ridge_mode == 2) {                              // (b) Lanczos: preset the ridge from the smallest eigenvalue
    // λ_min varies SMOOTHLY with config -> recompute every ridge_every steps (default 1 = every step, byte-id);
    // the estimate is nlanczos matvecs (each a PPPM FFT), so caching is a big win for ridge>1. A stale ridge that
    // under-bounds a sudden near-critical config is caught by the adaptive re-solve + CG->MINRES auto-fallback.
    if (ridge_every <= 1 || lmin_step < 0 || (update->ntimestep - lmin_step) >= ridge_every) {
      lmin_cache = estimate_lambda_min(nlanczos);     //     of the BARE projected operator (smooth through crit)
      lmin_step = update->ntimestep;
    }
    lmin_est = lmin_cache;
    const double lam_target = lam_floor + ridge_delta;   // == lam_floor bitwise when delta = 0
    if (lam_target - lmin_est > ridge_cur) ridge_cur = lam_target - lmin_est;   // shift so λ_min(A+ridge) ≥ floor if err ≤ δ
  }
  if (ridge_mode == 3) {                              // (#M-3) LOCAL per-atom ridge: start at the base everywhere
    ridge_local = true;
#if defined(_OPENMP)
    if (omp_go(nth, nn)) {
#pragma omp parallel for num_threads(nth) schedule(static)
      for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit) ridge_atom[i] = lr_ridge; }
    } else
#endif
    { for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit) ridge_atom[i] = lr_ridge; } }
  }
  double rmax = lr_ridge;                             // (#M-3) max per-atom ridge applied (for the warning)
  // #20 warm start (`solver warmstart on`): attempt 0 is seeded from the previous charges in atom->q. qeq_cg
  // forms the true residual b − Ã·x0, and its stopping criterion sqrt(rᵀD⁻¹r)/||b_FULL|| < tol is referenced to
  // the full RHS norm, so the seeded solve converges to the SAME solution/operator/tolerance as the cold one --
  // only the iteration count drops. Attempts >0 (ridge escalation) cold-start, and divergence still takes the
  // MINRES auto-fallback / freeze guards below. warmstart==0 => seed0==0 => byte-identical.
  const int seed0 = warmstart;
  for (int attempt = 0; attempt < 16; attempt++) {
#if defined(_OPENMP)
    if (omp_go(nth, nn)) {
#pragma omp parallel for num_threads(nth) schedule(static)
      for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(!(mask[i]&groupbit)) continue;
        Hdia_inv[i] = 1.0/((solve_diag_of(i)) + (ridge_local ? ridge_atom[i] : ridge_cur));
        qb[i] = -(chi_b(i) + (ionfield_flag ? fixq_field[i] : 0.0) + q0field[i]);
        if (efield) qb[i] -= chi_field[i];
        qs[i] = (seed0 && attempt==0) ? (qa[i] - q0[type[i]]) : 0.0; }
    } else
#endif
    { for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(!(mask[i]&groupbit)) continue;
      Hdia_inv[i] = 1.0/((solve_diag_of(i)) + (ridge_local ? ridge_atom[i] : ridge_cur));
      qb[i] = -(chi_b(i) + (ionfield_flag ? fixq_field[i] : 0.0) + q0field[i]);   // RexPoN: + fixed-shell field + q0 reference field (conservation)
      if (efield) qb[i] -= chi_field[i];                                       // B2: external-field coupling (fix efield)
      qs[i] = (seed0 && attempt==0) ? (qa[i] - q0[type[i]]) : 0.0; } }  // #20/T1 warm-start (attempt 0)
    if (seed0 && attempt==0) project_neutral(qs);                  // keep the seed in the neutral subspace
    project_neutral(qb);                                            // RHS into the per-molecule-neutral subspace
    { const double t0 = (th_on == 1) ? platform::walltime() : 0.0; it = qeq_cg(qb, qs); if (th_on == 1) th_cg += platform::walltime() - t0; }                                            // projected CG: Ã qs = P(−χ) − P·H·Δ_p
    double mq = 0.0;
#if defined(_OPENMP)
    if (omp_go(nth, nn)) {
      // P1: max is order-independent -- per-thread partial max + fixed-order combine (see omp_reduce banner).
      mq = omp_reduce(nn, nth, 0.0, omp_max, [&](int ii)->double{ int i=ilist[ii];
        if(!(mask[i]&groupbit)) return 0.0; double qi = qs[i];
        return std::isfinite(qi) ? fabs(qi) : 1.0e30; });
    } else
#endif
    { for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(!(mask[i]&groupbit)) continue;
      double qi = qs[i];                              // A6: guard on the DEVIATION |q - q0| — the
      if (!std::isfinite(qi)) { mq = 1.0e30; break; }               // runaway metric; q0=0 decks byte-identical
      if (fabs(qi) > mq) mq = fabs(qi); } }
    MPI_Allreduce(MPI_IN_PLACE, &mq, 1, MPI_DOUBLE, MPI_MAX, world);
    maxq = mq;
    if (ridge_mode == 2) {
      // (b) Lanczos: the ridge preset from λ_min before the loop normally sizes the operator PD -> one
      // solve, done. Any solve that stays bounded (|q-q0| < QGUARD: all alloy/confinement/regression
      // decks) takes the break here -> byte-identical.
      //
      // CORRECTION (2026-08-05). This test read `(!trial_solve || maxq < QGUARD)`: a COMMITTED solve broke
      // out UNCONDITIONALLY, while only a TRIAL was gated on QGUARD. A committed solve whose charges landed
      // between QGUARD and the |q|>50 divergence guard therefore got neither the local-ridge escalation
      // below nor an error, and committed SILENTLY — observed at max|q| = 14.79 e with pe = +17288 eV.
      // Every other ridge mode tests maxq before breaking; mode 2 was the outlier. Committed solves now
      // take the same escalation path as trials, still bounded by the 16-attempt cap and still backed by
      // the auto-fallback / freeze / |q|>50 guards after the loop. Byte-identical wherever maxq < QGUARD,
      // which is every regression case.
      if (maxq < QGUARD) {
        if (ridge_cur > lr_ridge + 1.0e-30) nridge = 1;
        break;
      }
      if (!ridge_local && comm->me == 0 && warn_budget(nwarn_overpol))   // never silent; budgeted
        error->warning(FLERR, "samqeq: committed solve over-polarized at step {} (max|q-q0| = {:.4g} e > {:.4g})"
                              "— escalating the local ridge and re-solving; inspect this result [{} so far]",
                       update->ntimestep, maxq, QGUARD, nwarn_overpol);
      // A solve on a NEAR-CRITICAL cell can run past QGUARD (the soft solvent collective mode; the single preset
      // solve runs to |q|~20e, below the |q|>50 MINRES/freeze rescues). Condition it with a LOCAL per-atom ridge
      // (reuse the #M-3 ridge_atom machinery -- always allocated): escalate ONLY the runaway atoms' diagonal,
      // leaving the BULK solvent at the physical Lanczos ridge, then re-solve.
      //   ★ WHY LOCAL, NOT global ridge_cur*=4: a global escalation de-polarizes the WHOLE cell and shifts the
      //   bulk ½·η·q² self-energy by an ion-independent offset; escalating only the runaway atoms keeps the bulk
      //   energy physical. A DELOCALIZED collective runaway is only partially bounded by local escalation.
      if (!ridge_local) {                                          // first over-polarization this solve: go local,
        ridge_local = true;                                        //   seed every group atom at the physical
        for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit) ridge_atom[i] = ridge_cur; }  // Lanczos ridge
      }
      { const double FAC = (ridge_gain > 1.0 ? ridge_gain : 4.0);  // ratchet only the atoms above the onset
        for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(!(mask[i]&groupbit)) continue;
          if (fabs(qs[i]) > QGUARD) ridge_atom[i] *= FAC; } }
      nridge++;
    } else if (ridge_mode == 1) {
      // (a) SMOOTH max|q| ridge: damped fixed point on a C1 quadratic ramp
      //   ridge = lr_ridge + ridge_gain*(max|q| - ridge_qonset)^2 (0 below onset).
      double over = maxq - ridge_qonset;
      double ridge_tgt = lr_ridge + (over > 0.0 ? ridge_gain*over*over : 0.0);
      // A7: 1e-3 = relative tolerance (dimensionless); the +1.0 additive floor is eV/e^2-anchored -> *ev_scale
      if (fabs(ridge_tgt - ridge_cur) <= 1.0e-3*(ridge_cur + 1.0*ev_scale)) break;   // fixed point reached
      ridge_cur += 0.6*(ridge_tgt - ridge_cur);                            // under-relaxed update
      nridge++;
    } else if (ridge_mode == 3) {
      // (#M-3) LOCAL per-atom RATCHET escalation: where |q_i| exceeds ridge_qonset (runaway), multiply that
      // atom's diagonal ridge by ridge_gain (×4 default) and re-solve. MONOTONIC (never released) ⇒ no
      // oscillation, converges to the minimal ridge that bounds |q| (the smooth |q|-ramp RELEASES when |q|
      // drops below onset → blows up again → never converges). Lifts the runaway-charge block
      // ONLY on the offending atoms ⇒ bulk dipole untouched (unlike the global ridge that over-hardens all).
      int any_over = 0;
      const double FAC = (ridge_gain > 1.0 ? ridge_gain : 4.0);
#if defined(_OPENMP)
      if (omp_go(nth, nn)) {
        // P1: each ii only ever writes its OWN ridge_atom[i] (no cross-atom write) -- the parallel-for is
        // race-free on the write side; any_over (OR) and rmax (MAX) are both order-independent reductions,
        // so a per-thread partial + fixed-order combine is exact and deterministic (see omp_reduce banner).
        std::vector<int> over_t(nth, 0);
        std::vector<double> rmax_t(nth, 0.0);
#pragma omp parallel num_threads(nth)
        {
          int tid = omp_get_thread_num();
          int ol = 0; double rl = 0.0;
#pragma omp for schedule(static)
          for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(!(mask[i]&groupbit)) continue;
            double qi = qs[i] + q0[type[i]];
            if (fabs(qi) > ridge_qonset) {
              ridge_atom[i] = (ridge_atom[i] > 0.0 ? ridge_atom[i]*FAC : 0.5*ev_scale);
              if (ridge_atom[i] > rl) rl = ridge_atom[i];
              ol = 1; } }
          over_t[tid]=ol; rmax_t[tid]=rl;
        }
        for (int t=0;t<nth;t++){ if(over_t[t]) any_over=1; if(rmax_t[t]>rmax) rmax=rmax_t[t]; }
      } else
#endif
      { for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(!(mask[i]&groupbit)) continue;
        double qi = qs[i] + q0[type[i]];
        if (fabs(qi) > ridge_qonset) {                            // this atom is running away -> escalate ITS ridge
          ridge_atom[i] = (ridge_atom[i] > 0.0 ? ridge_atom[i]*FAC : 0.5*ev_scale);   // A7: eV/e^2 seed -> deck units (FAC = dimensionless ratchet)
          if (ridge_atom[i] > rmax) rmax = ridge_atom[i];
          any_over = 1; } } }
      MPI_Allreduce(MPI_IN_PLACE, &any_over, 1, MPI_INT, MPI_MAX, world);
      if (!any_over) { if (rmax > lr_ridge + 1.0e-30) nridge = 1; break; }   // all atoms bounded -> done
      nridge++;
    } else {
      // LEGACY discrete escalation (mode 0, default -> byte-identical): 0 below QGUARD, x4 above.
      if (maxq < QGUARD) break;                                     // bounded -> accept (QGUARD = 4 e: charge, unit-invariant, A7)
      ridge_cur = (ridge_cur > 0.0) ? ridge_cur*4.0 : 0.5*ev_scale; // escalate and re-solve this step
                                                                    // (A7: 0.5 eV/e^2 seed -> deck units; x4 = dimensionless)
      nridge++;
    }
  }
  ridge_local = false;                                              // (#M-3) reset: modes 0/1/2 + Lanczos use scalar
  matvecs = it;
  if (nridge > 0 && comm->me == 0 && warn_budget(nwarn_ridge)) {   // : budgeted; SOLVER-DIAG carries the count
    // A7 diagnostic-print: deck-unit energies /ev_scale so the "eV" labels stay TRUE eV in units real
    // (exact /1.0 in metal -> byte-identical warnings). max|q| is charge (e), unconverted.
    if (ridge_mode == 2)
      error->warning(FLERR, "samqeq: Lanczos lambda_min={:.4g} eV at step {} -> ridge={:.4g} eV (max|q|={:.3f})"
                            "[{} engaged solves so far]",
                     lmin_est/ev_scale, update->ntimestep, ridge_cur/ev_scale, maxq, nwarn_ridge);
    else if (ridge_mode == 3)
      error->warning(FLERR, "samqeq: near-critical config at step {} bounded with LOCAL ridge (max ridge_atom="
                            "{:.3g} eV, max|q|={:.3f}) [{} engaged solves so far]", update->ntimestep, rmax/ev_scale,
                     maxq, nwarn_ridge);
    else
      error->warning(FLERR, "samqeq: near-critical config at step {} bounded with adaptive ridge={:.3g} eV"
                            "(max|q|={:.3f}) [{} engaged solves so far]", update->ntimestep, ridge_cur/ev_scale, maxq,
                     nwarn_ridge);
  }
  if (maxq > 50.0 && lr_autofb) {                                  // AUTO-FALLBACK: rescue the divergent step with MINRES
    if (comm->me == 0)
      error->warning(FLERR, "samqeq: CG diverged (|q|>50) at step {} -> MINRES auto-fallback", update->ntimestep);
    double save_qcap = minres_qcap;                                // bound the trial states on the rescue solve
    if (minres_qcap <= 0.0) minres_qcap = 4.0;
    qeq_solve_minres();                                            // re-solve this step (build_molinv valid;
    minres_qcap = save_qcap;                                       //   its own freeze guard applies if still bad)
    return;
  }
  if (freeze_overpolarized(maxq)) return;                          // FREEZE-AND-RIDE-THROUGH: over-polarized past
                                                                   // lr_qfreeze -> keep previous charges, no commit
  if (maxq > 50.0)                                                 // freeze + autofb both off -> original hard error
    error->all(FLERR, "samqeq: charge solve diverged (|q|>50) even with adaptive ridge at step {}", update->ntimestep);

#if defined(_OPENMP)
  if (omp_go(nth, nn)) {
#pragma omp parallel for num_threads(nth) schedule(static)
    for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(!(mask[i]&groupbit)) continue;
      qa[i] = qs[i] + q0[type[i]]; }
  } else
#endif
  { for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(!(mask[i]&groupbit)) continue;
    qa[i] = qs[i] + q0[type[i]]; } }                    // qs neutral + Δ_p net offset (#Ω M2)
  if (aspc_on) {                                                    // ASPC (#16): seed/advance predictor history (BO path).
    // B5(c) REVISED (S1b, scaling audit): the ORIGINAL rationale skipped the q_hist push on a rejected
    // corrector (its fallback solve here is SCF-inconsistent with q_hist's accepted samples, and skipping
    // preserved uniform spacing). But under CONSECUTIVE rejects that policy FREEZES q_hist permanently:
    // the predictor extrapolates from an ever-staler history, the corrector's start point only degrades,
    // and the reject becomes SELF-LOCKING (measured: permanent reject, ~29 matvecs/step). A frozen history
    // is strictly worse than a restarted one -- so on reject we now PUSH the converged BO solution (with
    // the usual shift) and RESET the history depth to 1: the predictor stays off while aspc_nhist
    // uniformly-spaced BO samples re-accumulate (exactly the initial-warmup semantics), then resumes.
    // P1: embarrassingly parallel (each ii owns its own q_hist[i][*] row).
#if defined(_OPENMP)
    if (omp_go(nth, nn)) {
#pragma omp parallel for num_threads(nth) schedule(static)
      for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(!(mask[i]&groupbit)) continue;
        for (int j=aspc_nhist-1;j>0;j--) q_hist[i][j]=q_hist[i][j-1];
        q_hist[i][0]=qs[i]; }
    } else
#endif
    { for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(!(mask[i]&groupbit)) continue;
      for (int j=aspc_nhist-1;j>0;j--) q_hist[i][j]=q_hist[i][j-1];
      q_hist[i][0]=qs[i]; } }                                         // store the neutral perturbation
    if (aspc_reject) aspc_have = 1;                                   // S1b: restart the history from this BO sample
    else if (aspc_have < aspc_nhist) aspc_have++;
  }
  comm_v = qa; pack_flag = 6; comm->forward_comm(this);             // update ghost q for the pair force compute
  field_diag();                                                     // DEBUG (env SAMQEQ_FIELDDIAG): matvec field on tags 4/5
}

/* ----------------------------------------------------------------------
   DEBUG (gated on env SAMQEQ_FIELDDIAG, whose value = how many invocations to dump; default 1):
   dump the matvec's electrostatic field on the
   O/H cores of molecule 2 (tags 4,5), broken into core-core real-space (the H.val formula), core-core
   reciprocal (add_reciprocal), and the Drude shell field. Compare V_O-V_H to pe's FD field (14.16) and
   the BO-implied mu_self_H-mu_self_O (18.34) to pin the force<->solve non-conservation.
-------------------------------------------------------------------------*/
void FixQEqSam::field_diag()
{
  // (#29): the env value is now how many INVOCATIONS to dump (default 1 = the previous
  // one-shot behaviour). Needed to compare the per-atom reciprocal at step 1, where the
  // pppm/samqeq/kk defect shows, against step 0, where it does not.
  const char *fdenv = getenv("SAMQEQ_FIELDDIAG");
  if (!fdenv) return;
  static int fd_left = -1;
  if (fd_left < 0) { fd_left = atoi(fdenv); if (fd_left <= 0) fd_left = 1; }
  if (fd_left == 0) return;
  fd_left--;
  double **x = atom->x; double *qa = atom->q; int *type = atom->type, *mask = atom->mask;
  tagint *tag = atom->tag;
  int nall = atom->nlocal + atom->nghost;
  double pref = force->qqrd2e, a = lr_alpha;
  double *rec = q_p;                                                // scratch (free post-solve)
  for (int i = 0; i < atom->nmax; i++) rec[i] = 0.0;
  add_reciprocal(qa, rec);                                          // core-core reciprocal field (recip_self removed)
  const char *e1 = getenv("SAMQEQ_FD_TAG1"), *e2 = getenv("SAMQEQ_FD_TAG2");
  tagint tgts[2] = { (tagint)(e1 ? atoi(e1) : 4), (tagint)(e2 ? atoi(e2) : 5) };
  for (int t = 0; t < 2; t++) {
    int li = -1; for (int i = 0; i < atom->nlocal; i++) if (tag[i] == tgts[t]) { li = i; break; }
    if (li < 0) continue;
    int ti = type[li]; double xi = x[li][0], yi = x[li][1], zi = x[li][2];
    double mreal = 0.0;
    for (int j = 0; j < nall; j++) {
      if (j == li || !(mask[j] & groupbit)) continue;              // CORE neighbors only (compute_H is core-core)
      double dx = x[j][0]-xi, dy = x[j][1]-yi, dz = x[j][2]-zi, rsq = dx*dx+dy*dy+dz*dz;
      if (rsq > swb*swb || rsq < 1e-10) continue;
      double r = sqrt(rsq);
      mreal += pref * (shielded_coulomb(ti, type[j], r) - (1.0 - erfc(a*r))/r) * qa[j];
    }
    double df = ionfield_flag ? fixq_field[li] : 0.0;
    utils::logmesg(lmp, "FIELDDIAG tag={} q={:.5f} mreal={:.5f} mrecip={:.5f} drude={:.5f} matvec_total={:.5f}\n",
                   (long)tgts[t], qa[li], mreal, rec[li], df, mreal + rec[li] + df);
  }
  // PER-ATOM recip_self spread: measure recip_self at several atoms (neutral +1/-1 vs a fixed reference R,
  // remove the A-R cross-term analytically) -> if S_A varies across atoms, the homogeneous lr_self_meas is
  // the ~0.3 eV inconsistency source. (serial diag only: eksp->compute_vector is COLLECTIVE -- calling it from
  // just rank 0 deadlocks at np>1, so this probe is gated to np==1.)
  if (comm->me == 0 && comm->nprocs == 1) {
    int R = ilist[0];                                           // fixed reference core
    double xr = x[R][0], yr = x[R][1], zr = x[R][2];
    for (int probe = 0; probe < 12; probe++) {
      int A = ilist[probe % nn];
      if (A == R) continue;
      for (int i = 0; i < atom->nmax; i++) qsave[i] = qa[i];
      for (int i = 0; i < atom->nmax; i++) qa[i] = 0.0;
      qa[A] = 1.0; qa[R] = -1.0;
      for (int i = 0; i < atom->nmax; i++) prec[i] = 0.0;
      eksp->compute_vector(prec, groupbit, groupbit, false);
      double dxr = x[A][0]-xr, dyr = x[A][1]-yr, dzr = x[A][2]-zr;
      double rAR = sqrt(dxr*dxr+dyr*dyr+dzr*dzr);
      double S_A = prec[A] + erf(lr_alpha*rAR)/rAR;             // remove the A-R cross-term -> A's grid self
      if (force->kspace && force->kspace->slabflag == 1)        // EW3DC: remove the analytic slab pair term too (see calibrate_recip_self)
        S_A -= MathConst::MY_2PI/(domain->xprd*domain->yprd*domain->zprd*force->kspace->slab_volfactor) * dzr*dzr;
      for (int i = 0; i < atom->nmax; i++) qa[i] = qsave[i];
      double cA = 0.0;                                            // R4: the exact per-atom coefficient, for comparison
      { std::vector<double> tmp(atom->nmax, 0.0); if (eksp->compute_self_peratom(tmp.data(), groupbit)) cA = tmp[A]; }
      const double xiA = eksp->self_image_term();
      utils::logmesg(lmp, "SELFDIAG atomtag={} recip_self_A={:.6f} (homogeneous lr_self_meas={:.6f}, diff={:.6f} = {:.4f} eV/e;"
                          "exact c_A=K_A−ξ={:.6f} [K_A={:.6f}, ξ={:.6f}], probe−exact={:+.6f})\n",
                     (long)atom->tag[A], S_A, lr_self_meas, S_A - lr_self_meas,
                     (S_A - lr_self_meas)*force->qqrd2e/ev_scale, cA, cA + xiA, xiA, S_A - cA);   // A7: /ev_scale keeps the eV/e label true in units real
    }
  }
}


/* ----------------------------------------------------------------------
   R4: fill lr_self_atom with the exact per-atom grid self-coefficient for this solve (positions
   moved since the last one), and print min/mean/max over the group once per run so the operator's
   diagonal is disclosed the way the scalar route's calibration line discloses it.
-------------------------------------------------------------------------*/
void FixQEqSam::compute_self_peratom_now()
{
  if (atom->nmax > lr_self_atom_nmax) {
    memory->grow(lr_self_atom, atom->nmax, "samqeq:lr_self_atom");
    lr_self_atom_nmax = atom->nmax;
  }
  for (int i = 0; i < atom->nmax; i++) lr_self_atom[i] = 0.0;
  if (!eksp->compute_self_peratom(lr_self_atom, groupbit))
    error->all(FLERR, "samqeq: fix_modify recip_self peratom needs kspace_style pppm/samqeq (the Kokkos backend"
                      "does not implement compute_self_peratom)");
  for (int ii = 0; ii < nn; ii++) { int i = ilist[ii]; if (!(atom->mask[i] & groupbit)) continue;
    if (!std::isfinite(lr_self_atom[i]))
      error->one(FLERR, "samqeq: per-atom grid self-coefficient is non-finite at atom {} (tag {})", i, atom->tag[i]); }
  if (!lr_self_atom_logged) {
    lr_self_atom_logged = 1;
    double lo = 1.0e300, hi = -1.0e300, sum = 0.0; bigint cnt = 0;
    for (int ii = 0; ii < nn; ii++) { int i = ilist[ii]; if (!(atom->mask[i] & groupbit)) continue;
      const double c = lr_self_atom[i]; if (c < lo) lo = c; if (c > hi) hi = c; sum += c; cnt++; }
    double g[3] = {lo, -hi, 0.0}; MPI_Allreduce(MPI_IN_PLACE, g, 2, MPI_DOUBLE, MPI_MIN, world);
    MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, world);
    MPI_Allreduce(MPI_IN_PLACE, &cnt, 1, MPI_LMP_BIGINT, MPI_SUM, world);
    if (comm->me == 0 && cnt > 0) {
      const double f = force->qqrd2e/ev_scale, mean = sum/(double)cnt;
      const double xi = eksp->self_image_term();
      utils::logmesg(lmp, "samqeq: exact per-atom grid self-coefficient at step {}: mean {:.5f}, min {:.5f}, max {:.5f} raw"
                          "(= {:.4f} eV/e mean, spread {:.4f} eV/e) over {} atoms = K_ii − ξ with image term ξ = {:.5f} raw"
                          "({:.4f} eV/e; the diagonal keeps η+ξ, as the Ewald energy does); refreshed every solve, no calibration\n",
                     update->ntimestep, mean, g[0], -g[1], mean*f, (-g[1]-g[0])*f, cnt, xi, xi*f);
    }
  }
}
