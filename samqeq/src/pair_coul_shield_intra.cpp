// clang-format off
/* ----------------------------------------------------------------------
   coul/shield/intra — intra-molecular Coulomb shielding correction for samQEq.
   See pair_coul_shield_intra.h. Adds qqrd2e q_i q_j (J_shield(r) − 1/r) for
   same-molecule pairs so the pair forces/energy match FixQEqSam's solve
   (which uses J_shield intra, bare 1/r inter). Short-range, →0 at large r.
-------------------------------------------------------------------------*/

#include "pair_coul_shield_intra.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "fix.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using MathConst::MY_PIS;   // sqrt(pi)

/* ----------------------------------------------------------------------*/

PairCoulShieldIntra::PairCoulShieldIntra(LAMMPS *lmp) : Pair(lmp)
{
  cut = nullptr; gamma = nullptr; intra_only = 1;   // default: intra-only (lr_ewald=1)
  shield_gauss = 0; shield_lambda = 0.462770;       // default cbrt J_shield (byte-identical); gauss = PQEq erf
  is2s = nullptr;                                   // slater per-type 2s/1s flag (allocated in settings()/read_restart())
  iondamp_on = 0; iondamp_b = nullptr; iondamp_n = nullptr;   // #5 damped interionic kernel (settings `iondamp`)
}

/* ----------------------------------------------------------------------*/

PairCoulShieldIntra::~PairCoulShieldIntra()
{
  if (copymode) return;
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cut);
    memory->destroy(gamma);
  }
  memory->destroy(is2s);
  memory->destroy(shield_rpair);   //
  memory->destroy(iondamp_b); memory->destroy(iondamp_n);   // #5 (settings-time arrays, like is2s)
}

/* ----------------------------------------------------------------------*/

void PairCoulShieldIntra::compute(int eflag, int vflag)
{
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double qtmp, xtmp, ytmp, ztmp, delx, dely, delz, ecoul, fpair;
  double rsq, r, r2inv, rinv, factor_coul;
  int *ilist, *jlist, *numneigh, **firstneigh;

  ecoul = 0.0;
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  int *type = atom->type;
  tagint *molecule = atom->molecule;
  int nlocal = atom->nlocal;
  double *special_coul = force->special_coul;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;


  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    qtmp = q[i];
    xtmp = x[i][0]; ytmp = x[i][1]; ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      // INTRA-molecular pairs only (same nonzero molecule ID)
      // intra_only=1 (lr_ewald=1): correct same-molecule pairs only. intra_only=0 (lr_ewald=2): all pairs.
      if (intra_only && (!molecule || molecule[i] == 0 || molecule[i] != molecule[j])) continue;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r2inv = 1.0/rsq;
        rinv = sqrt(r2inv);
        r = 1.0/rinv;
        // ROBUST special_bonds: the SHIELDED (J_shield) part is the FULL intramolecular Coulomb the FQ solve
        // equilibrates against (always full strength, factor 1); ONLY the −1/r compensation tracks what
        // coul/long actually nets for this pair (= factor_coul·1/r). ⇒ net (coul/long + this) = J_shield for
        // ANY in-list weight, and force↔solve stay consistent. Byte-identical at factor_coul=1. (Pairs DROPPED
        // from the neighborlist -- special_lj==special_coul==0 WITHOUT a kspace style -- are refused in init_list (s88b).)
        double K = qqrd2e * qtmp * q[j];
        if (iondamp_on && iondamp_b[itype][jtype] > 0.0) {
          // #5 DAMPED interionic kernel: net = f_n(b r)·J(r) for this designated (ion) type pair, J from
          // whichever shield kernel is active -- mirrors the fix's damped shielded_coulomb() (solve side).
          // E = K(f·J − factor_coul/r); dE/dr = K(b·f′·J + f·J′ + factor_coul/r²); fpair = −(1/r)dE/dr.
          double J, dJdr;
          if (shield_gauss == SHIELD_SLATER) {
            int idx = slater_tri_index(itype, jtype, atom->ntypes);
            J = slater_tabs[idx].eval(r, dJdr);
          } else if (shield_gauss == SHIELD_GAUSSIAN) {
            double aij = pqeq_aij(itype, jtype);   // : per-pair override or the combination rule
            J = erf(aij*r)*rinv;
            dJdr = (2.0*aij/MY_PIS)*exp(-aij*aij*rsq)*rinv - J*rinv;   // d(erf(ar)/r)/dr
          } else {
            double g = gamma[itype][jtype];
            double sh = 1.0/cbrt(rsq*r + 1.0/(g*g*g));
            J = sh;
            dJdr = -rsq*sh*sh*sh*sh;                                    // d(r³+1/γ³)^(−1/3)/dr = −r²·J⁴
          }
          const double bd = iondamp_b[itype][jtype];
          double dfdx;
          const double fd = samqeq_tt_damp(bd*r, iondamp_n[itype][jtype], dfdx);
          fpair = -K * (bd*dfdx*J + fd*dJdr + factor_coul*r2inv) * rinv;
          ecoul = K * (fd*J - factor_coul*rinv);
        } else if (shield_gauss == SHIELD_SLATER) {
          // Rick Slater-overlap J(r): net = J(r); correction E = K(J(r) − factor_coul/r). Table gives J AND
          // dJ/dr directly (force<->solve consistency IS the point of this kernel -- the table is the SAME
          // one shielded_coulomb() uses on the solve side, transcribed once in slater_jtable.h).
          int idx = slater_tri_index(itype, jtype, atom->ntypes);
          double dJdr;
          double J = slater_tabs[idx].eval(r, dJdr);
          // E = K(J − factor_coul/r); dE/dr = K(dJ/dr + factor_coul/r²); fpair = −(1/r)dE/dr
          fpair = -K * (dJdr + factor_coul*r2inv) * rinv;
          ecoul = K * (J - factor_coul*rinv);
        } else if (shield_gauss == SHIELD_GAUSSIAN) {
          // PQEq Gaussian: net = erf(α_ij r)/r; correction E = K(erf(α_ij r)/r − factor_coul/r). α_i = λ/(2 Rc_i²),
          // Rc_i = diagonal gamma[i][i], α_ij = √(α_i α_j/(α_i+α_j)) (same as the solve's shielded_coulomb).
          double aij = pqeq_aij(itype, jtype);   // : per-pair override or the combination rule
          double erfar = erf(aij*r), expt = exp(-aij*aij*rsq);
          double r3inv = rinv*r2inv;
          // fpair = −(1/r)dE/dr = K[(erf−factor_coul)/r³ − (2α_ij/√π)exp(−α²r²)/r²] (the erf/expt parts are full)
          fpair = K * ((erfar - factor_coul)*r3inv - (2.0*aij/MY_PIS)*expt*r2inv);
          ecoul = K * (erfar - factor_coul) * rinv;
        } else {
          double g = gamma[itype][jtype];
          double sh = 1.0/cbrt(rsq*r + 1.0/(g*g*g));     // J_shield = 1/∛(r³+1/γ³)
          double sh4 = sh*sh*sh*sh;
          // E = K (sh − factor_coul/r); fpair = −(1/r)dE/dr = K (r·sh⁴ − factor_coul/r³)
          fpair = K * (r*sh4 - factor_coul*rinv*r2inv);
          ecoul = K * (sh - factor_coul*rinv);
        }

        f[i][0] += delx*fpair; f[i][1] += dely*fpair; f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair; f[j][1] -= dely*fpair; f[j][2] -= delz*fpair;
        }
        if (evflag) ev_tally(i, j, nlocal, newton_pair, 0.0, ecoul, fpair, delx, dely, delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------*/

void PairCoulShieldIntra::allocate()
{
  allocated = 1;
  int np1 = atom->ntypes + 1;
  memory->create(setflag, np1, np1, "pair:setflag");
  for (int i = 1; i < np1; i++)
    for (int j = i; j < np1; j++) setflag[i][j] = 0;
  memory->create(cutsq, np1, np1, "pair:cutsq");
  memory->create(cut, np1, np1, "pair:cut");
  memory->create(gamma, np1, np1, "pair:gamma");
}

/* ----------------------------------------------------------------------*/

void PairCoulShieldIntra::settings(int narg, char **arg)
{
  if (narg < 1) error->all(FLERR, "Illegal pair_style coul/shield/intra command");
  cut_global = utils::numeric(FLERR, arg[0], false, lmp);
  intra_only = 1;                                    // default: intra-molecular pairs only (lr_ewald=1)
  shield_gauss = 0;                                  // default: cbrt J_shield (byte-identical)
  // slater per-type 2s/1s flag: allocate HERE (independent of allocate()/gamma/cut/setflag, which aren't
  // sized until the first pair_coeff -- see the header) so `2s <type>...` below can set it immediately, and
  // so it round-trips through write_restart/read_restart even without settings() being reissued (LAMMPS
  // restores a pair style's full state from the binary restart, not by replaying the input script).
  memory->destroy(is2s);
  memory->create(is2s, atom->ntypes+1, "pair:is2s");
  for (int i = 0; i <= atom->ntypes; i++) is2s[i] = 0;
  // #5 iondamp: same settings-time reset semantics as is2s/intra_only/shield_gauss above -- re-issuing
  // pair_style clears the damping unless the keyword is repeated.
  memory->destroy(iondamp_b); memory->destroy(iondamp_n);
  iondamp_b = nullptr; iondamp_n = nullptr; iondamp_on = 0;

  shield_lambda = 0.462770;          // PQEq λ: same settings-time reset semantics as shield_gauss above
                                     // (re-issuing pair_style restores the default unless `lambda` repeats)

  for (int k = 1; k < narg; k++) {   // optional keywords: intra|all (scope), cbrt|gaussian|slater (kernel),
                                     // lambda <val> (PQEq λ),
                                     // iondamp <I> <J> <b> [<n>] (repeatable) [+2s... LAST]
    if (strcmp(arg[k], "all") == 0)          intra_only = 0;
    else if (strcmp(arg[k], "intra") == 0)   intra_only = 1;
    else if (strcmp(arg[k], "gaussian") == 0)  shield_gauss = SHIELD_GAUSSIAN;   // Gaussian erf(α_ij r)/r (PQEq form)
    else if (strcmp(arg[k], "pqeq") == 0)     // : renamed `gaussian` (it named a force field, not the kernel); alias removed
      error->all(FLERR, "pair coul/shield/intra:" "the Gaussian shielding kernel keyword `pqeq` was renamed `gaussian` in samQEq (same kernel, same numbers); replace `pqeq` with `gaussian` in the deck");
    else if (strcmp(arg[k], "cbrt") == 0)    shield_gauss = SHIELD_CBRT;   // (not "gauss" — collides with pair_style gauss)
    else if (strcmp(arg[k], "slater") == 0)  shield_gauss = SHIELD_SLATER; // Rick Slater-overlap J(r), JCP 101,6141
    else if (strcmp(arg[k], "lambda") == 0) {
      // (Change A, channel 3): PQEq λ in α_i = λ/(2 Rc_i²). MIRRORS `fix_modify <id> shield pqeq <lambda>`
      // on the solve side. Until this was hardcoded to 0.462770 here with NO keyword at all, so any deck
      // that retuned λ on the fix was GUARANTEED force<->solve inconsistent with no way to fix it — the
      // channel had no legal configuration. Only meaningful for `pqeq`; harmless (unread) otherwise.
      // Put `lambda <val>` BEFORE any `2s ...` (which consumes to the end of the command).
      if (k + 1 >= narg) error->all(FLERR, "pair_style coul/shield/intra lambda: need <value>");
      shield_lambda = utils::numeric(FLERR, arg[k+1], false, lmp);
      if (shield_lambda <= 0.0)
        error->all(FLERR, "pair_style coul/shield/intra lambda: must be > 0 (PQEq default 0.462770)");
      k++;
    }
    else if (strcmp(arg[k], "iondamp") == 0) {
      // #5 damped interionic kernel: `iondamp <typeI> <typeJ> <b> [<n>]` -- TT-damp the shielded kernel for
      // the designated type pairs, MIRRORING `fix_modify <id> iondamp` (which owns the solve side; this owns
      // forces/energy -- both must carry the same pairs/b/n)..
      if (kokkosable)   // device pair hardcodes the undamped kernels (host-first precedent, like slater's guard)
        error->all(FLERR, "pair coul/shield/intra iondamp is host-only (not ported to the /kk device pair)");
      if (k + 3 >= narg)
        error->all(FLERR, "pair_style coul/shield/intra iondamp: need <typeI> <typeJ> <b> [<n>]");
      const int nt = atom->ntypes;
      if (!iondamp_b) {
        memory->create(iondamp_b, nt+1, nt+1, "pair:iondamp_b");
        memory->create(iondamp_n, nt+1, nt+1, "pair:iondamp_n");
        for (int i = 0; i <= nt; i++)
          for (int j = 0; j <= nt; j++) { iondamp_b[i][j] = 0.0; iondamp_n[i][j] = 4; }
      }
      int ilo, ihi, jlo, jhi;
      utils::bounds(FLERR, arg[k+1], 1, nt, ilo, ihi, error);
      utils::bounds(FLERR, arg[k+2], 1, nt, jlo, jhi, error);
      double bdamp = utils::numeric(FLERR, arg[k+3], false, lmp);
      if (bdamp <= 0.0) error->all(FLERR, "pair_style coul/shield/intra iondamp: b must be > 0");
      int ttn = 4;                                  // default TT order n=4 (the CL&Pol convention)
      k += 3;
      if (k + 1 < narg && utils::is_integer(arg[k+1])) { ttn = utils::inumeric(FLERR, arg[k+1], false, lmp); k++; }
      if (ttn < 1 || ttn > 8) error->all(FLERR, "pair_style coul/shield/intra iondamp: TT order n must be 1..8");
      for (int i = ilo; i <= ihi; i++)
        for (int j = jlo; j <= jhi; j++) {          // both triangles -> symmetric
          iondamp_b[i][j] = iondamp_b[j][i] = bdamp;
          iondamp_n[i][j] = iondamp_n[j][i] = ttn;
        }
      iondamp_on = 1;
    }
    else if (strcmp(arg[k], "2s") == 0) {
      // per-type 2s(O/M)-vs-1s(H) selector for the slater kernel; consumes to the END of narg (no closing
      // keyword) -- put "2s ..." LAST in the pair_style command.
      for (int m = k+1; m < narg; m++) {
        int t = utils::inumeric(FLERR, arg[m], false, lmp);
        if (t < 1 || t > atom->ntypes)
          error->all(FLERR, "pair_style coul/shield/intra: bad 2s type {} (ntypes={})", t, atom->ntypes);
        is2s[t] = 1;
      }
      break;
    }
    else error->all(FLERR, "pair_style coul/shield/intra: unknown keyword {}"
                          "(use intra|all cbrt|gaussian|slater [2s ...])", arg[k]);
  }
  if (allocated) {
    for (int i = 1; i <= atom->ntypes; i++)
      for (int j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------*/

void PairCoulShieldIntra::coeff(int narg, char **arg)
{
  if (narg < 3 || narg > 4)
    error->all(FLERR, "Incorrect args for pair coul/shield/intra coefficients");
  if (!allocated) allocate();

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  double gamma_one = utils::numeric(FLERR, arg[2], false, lmp);   // shielding γ
  double cut_one = cut_global;
  if (narg == 4) cut_one = utils::numeric(FLERR, arg[3], false, lmp);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      gamma[i][j] = gamma_one;
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }
  if (count == 0) error->all(FLERR, "Incorrect args for pair coul/shield/intra coefficients");
}

/* ----------------------------------------------------------------------*/

void PairCoulShieldIntra::init_style()
{
  for (int ifx = 0; ifx < modify->nfix; ifx++)
    if (strcmp(modify->fix[ifx]->style, "drude") == 0)
      error->all(FLERR, "Pair coul/shield/intra does not support Drude shells (`fix drude`)");
  if (!atom->q_flag) error->all(FLERR, "Pair coul/shield/intra requires atom attribute q");
  if (!atom->molecule_flag) error->all(FLERR, "Pair coul/shield/intra requires atom attribute molecule");
  // s88b: the special_bonds guard moved to init_list(): LAMMPS decides whether a 0/0-weighted bonded pair is DROPPED
  // from the neighbor list (neighbor->special_flag[k] == 0) only in Neighbor::init(), which runs after this.
  if (shield_gauss == SHIELD_SLATER) build_slater_tables();   // lazily (re)built every init_style() call; see below
  neighbor->add_request(this);
}

/* ----------------------------------------------------------------------
   build_slater_tables(): (re)build the per-(ti<=tj)-type-pair Slater J(r) tables from is2s/gamma
   (reinterpreted as zeta) and cut_global. Called from init_style() -- UNCONDITIONALLY every call while
   slater is active (init_style() runs once per run-setup, not per force call); the simplest correct policy
   for "rebuild if the cutoff changes" since cut_global can only change between runs, via settings(). Mirrors
   FixQEqSam::build_slater_tables() exactly (same slater_jtable.h) so the pair's force and the fix's solve
   see the IDENTICAL table.
-------------------------------------------------------------------------*/
void PairCoulShieldIntra::build_slater_tables()
{
  const int nt = atom->ntypes;
  const int npairs = nt * (nt + 1) / 2;
  slater_tabs.assign(npairs, SlaterJTable());
  for (int ti = 1; ti <= nt; ti++)
    for (int tj = ti; tj <= nt; tj++) {
      int idx = slater_tri_index(ti, tj, nt);
      // gamma[i][i] is REINTERPRETED as the Slater exponent zeta (1/Å) when shield_gauss==SHIELD_SLATER (only
      // the per-type DIAGONAL is used -- same convention as the pqeq Rc reinterpretation above).
      slater_tabs[idx].build(is2s[ti] != 0, is2s[tj] != 0, gamma[ti][ti], gamma[tj][tj], cut_global);
    }
  if (comm->me == 0)
    utils::logmesg(lmp, "samqeq: pair coul/shield/intra Slater-overlap J(r) tables built ({} unique type"
                        "pairs, {} pts each, r in [{:.3g},{:.3g}])\n", npairs,
                        slater_tabs.empty() ? 0 : slater_tabs[0].npts,
                        slater_tabs.empty() ? 0.0 : slater_tabs[0].rmin, cut_global);
}

/* ----------------------------------------------------------------------
   s88b : the special_bonds and exclusion guards, checked where the
   truth is known. LAMMPS removes a bonded 1-(k+1) pair from every neighbor list only when
   neighbor->special_flag[k] == 0: both weights exactly 0 AND no kspace style AND no special-keeping pair style
   (neighbor.cpp ~519-576; npair.h find_special). Under a kspace style the pairs are KEPT with factor_coul = 0 and this
   style nets the full J_shield (coul/long nets 0), so nothing is missing -- the old init_style() test on the weights
   refused such valid decks (every fix qeq/sam deck has kspace). special_flag is final only after Neighbor::init(),
   which runs after init_style() and before this callback. Kokkos inherits this (no override there).
-------------------------------------------------------------------------*/

void PairCoulShieldIntra::init_list(int id, NeighList *ptr)
{
  Pair::init_list(id, ptr);
  if (atom->nbonds > 0)
    for (int k = 1; k <= 3; k++)
      if (neighbor->special_flag[k] == 0)
        error->all(FLERR, "Pair coul/shield/intra: 1-{} bonded pairs are dropped from the neighbor list"
                   "(special_bonds lj 0 coul 0 for that hop and no kspace_style), so the shielded intramolecular"
                   "correction never sees them and the intramolecular Coulomb silently vanishes. Add a kspace_style"
                   "(the pairs are then kept with factor_coul 0 and the physics is complete), or give the hop a tiny"
                   "nonzero lj weight, e.g. `special_bonds lj 1e-8 1e-8 1e-8` with your coul weights", k+1);
  // s88b (refuse `exclude molecule/intra` only): that exclusion removes exactly the intramolecular pairs
  // this style exists for, while kspace still sums them -- the shielded correction, coul/long's compensation and the
  // qeq/sam solve coupling all vanish silently (measured 0.6 e charge errors, s88b B4). Other exclusions are left alone:
  // fix gcmc / widom / charge/regulation add an internal group exclusion to switch trial molecules off, legitimately.
  for (int i = 0; i < neighbor->nex_mol; i++)
    if (neighbor->ex_mol_intra[i])
      error->all(FLERR, "Pair coul/shield/intra: neigh_modify exclude molecule/intra is active. It removes the"
                 "intramolecular pairs this style corrects while kspace still sums them, so the shielded correction,"
                 "coul/long's compensation and the qeq/sam solve coupling vanish (measured 0.6 e charge errors,"
                 "samQEq s88b). Remove it; keep molecules rigid with fix rigid/shake instead");
}

/* ----------------------------------------------------------------------*/

double PairCoulShieldIntra::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    gamma[i][j] = sqrt(gamma[i][i] * gamma[j][j]);     // samQEq convention: γ_ij = √(γ_ii γ_jj)
    cut[i][j] = mix_distance(cut[i][i], cut[j][j]);
  }
  gamma[j][i] = gamma[i][j];
  // #5 iondamp truncation hygiene: this pair truncates HARD at cut[i][j], and the damped correction
  // f·J − 1/r ~ −(1−f(b r))/r has an exponential TT tail -- if 1−f(b·cut) is not tiny the truncation is a
  // kcal-scale energy step at the cutoff. Budget = the Ewald-tolerance scale (1e-3). Deck fix: raise the
  // damped pairs' per-pair cutoff to the fix's swb (e.g. `pair_coeff 3 4 coul/shield/intra 0.5 8.0`).
  if (iondamp_on && iondamp_b[i][j] > 0.0 && comm->me == 0) {
    double dfdx; double fres = samqeq_tt_damp(iondamp_b[i][j]*cut[i][j], iondamp_n[i][j], dfdx);
    if (1.0 - fres > 1.0e-3)
      error->warning(FLERR, "pair coul/shield/intra iondamp {}-{}: TT damping not converged at the pair"
                            "cutoff (1-f(b*cut)={:.2e} at cut={:.2f}) -- raise this pair's cutoff (to the"
                            "fix's swb) or b", i, j, 1.0 - fres, cut[i][j]);
  }
  return cut[i][j];
}

/* ----------------------------------------------------------------------*/

void PairCoulShieldIntra::write_restart(FILE *fp)
{
  write_restart_settings(fp);
  // is2s (slater per-type 2s/1s flag): written ONLY when shield_gauss==SHIELD_SLATER, so an OLD-format
  // restart file (shield_gauss 0 or 1, written before this mode existed) is read back byte-for-byte
  // identically -- read_restart mirrors this same conditional, keyed off the shield_gauss value it JUST
  // read from the (still-shared) settings block above.
  if (shield_gauss == SHIELD_SLATER)
    for (int i = 1; i <= atom->ntypes; i++) fwrite(&is2s[i], sizeof(int), 1, fp);
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j], sizeof(int), 1, fp);
      if (setflag[i][j]) { fwrite(&gamma[i][j], sizeof(double), 1, fp); fwrite(&cut[i][j], sizeof(double), 1, fp); }
    }
}

/* ----------------------------------------------------------------------*/

void PairCoulShieldIntra::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();
  int me = comm->me;
  // is2s is NOT part of allocate() (it's independent of gamma/cut/setflag -- see the header); (re)allocate it
  // here so a restart-restored pair (no settings() reissued) still has a valid array, then read it back ONLY
  // if shield_gauss==SHIELD_SLATER (matches write_restart's same-conditional write, so old-format files with
  // shield_gauss in {0,1} are read unchanged -- see the comment there).
  memory->destroy(is2s);
  memory->create(is2s, atom->ntypes+1, "pair:is2s");
  for (int i = 0; i <= atom->ntypes; i++) is2s[i] = 0;
  if (shield_gauss == SHIELD_SLATER)
    for (int i = 1; i <= atom->ntypes; i++) {
      if (me == 0) utils::sfread(FLERR, &is2s[i], sizeof(int), 1, fp, nullptr, error);
      MPI_Bcast(&is2s[i], 1, MPI_INT, 0, world);
    }
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++) {
      if (me == 0) utils::sfread(FLERR, &setflag[i][j], sizeof(int), 1, fp, nullptr, error);
      MPI_Bcast(&setflag[i][j], 1, MPI_INT, 0, world);
      if (setflag[i][j]) {
        if (me == 0) {
          utils::sfread(FLERR, &gamma[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &cut[i][j], sizeof(double), 1, fp, nullptr, error);
        }
        MPI_Bcast(&gamma[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&cut[i][j], 1, MPI_DOUBLE, 0, world);
      }
    }
}

/* ----------------------------------------------------------------------*/

void PairCoulShieldIntra::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global, sizeof(double), 1, fp);
  fwrite(&intra_only, sizeof(int), 1, fp);
  fwrite(&shield_gauss, sizeof(int), 1, fp);
  fwrite(&shield_lambda, sizeof(double), 1, fp);
  fwrite(&mix_flag, sizeof(int), 1, fp);
}

/* ----------------------------------------------------------------------*/

void PairCoulShieldIntra::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    utils::sfread(FLERR, &cut_global, sizeof(double), 1, fp, nullptr, error);
    utils::sfread(FLERR, &intra_only, sizeof(int), 1, fp, nullptr, error);
    utils::sfread(FLERR, &shield_gauss, sizeof(int), 1, fp, nullptr, error);
    utils::sfread(FLERR, &shield_lambda, sizeof(double), 1, fp, nullptr, error);
    utils::sfread(FLERR, &mix_flag, sizeof(int), 1, fp, nullptr, error);
  }
  MPI_Bcast(&cut_global, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&intra_only, 1, MPI_INT, 0, world);
  MPI_Bcast(&shield_gauss, 1, MPI_INT, 0, world);
  MPI_Bcast(&shield_lambda, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&mix_flag, 1, MPI_INT, 0, world);
}

/* ----------------------------------------------------------------------*/

double PairCoulShieldIntra::single(int i, int j, int itype, int jtype, double rsq,
                                   double factor_coul, double /*factor_lj*/, double &fforce)
{
  if (intra_only && (!atom->molecule || atom->molecule[i] == 0 || atom->molecule[i] != atom->molecule[j])) {
    fforce = 0.0; return 0.0;
  }
  double r2inv = 1.0/rsq, rinv = sqrt(r2inv), r = 1.0/rinv;
  double K = force->qqrd2e * atom->q[i] * atom->q[j];   // full J_shield; only the −1/r tracks factor_coul (see compute)
  if (iondamp_on && iondamp_b[itype][jtype] > 0.0) {    // #5 damped interionic kernel (mirrors compute())
    double J, dJdr;
    if (shield_gauss == SHIELD_SLATER) {
      int idx = slater_tri_index(itype, jtype, atom->ntypes);
      J = slater_tabs[idx].eval(r, dJdr);
    } else if (shield_gauss == SHIELD_GAUSSIAN) {
      double aij = pqeq_aij(itype, jtype);   // : per-pair override or the combination rule
      J = erf(aij*r)*rinv;
      dJdr = (2.0*aij/MY_PIS)*exp(-aij*aij*rsq)*rinv - J*rinv;
    } else {
      double g = gamma[itype][jtype];
      double sh = 1.0/cbrt(rsq*r + 1.0/(g*g*g));
      J = sh;
      dJdr = -rsq*sh*sh*sh*sh;
    }
    const double bd = iondamp_b[itype][jtype];
    double dfdx;
    const double fd = samqeq_tt_damp(bd*r, iondamp_n[itype][jtype], dfdx);
    fforce = -K * (bd*dfdx*J + fd*dJdr + factor_coul*r2inv) * rinv;
    return K * (fd*J - factor_coul*rinv);
  }
  if (shield_gauss == SHIELD_SLATER) {
    int idx = slater_tri_index(itype, jtype, atom->ntypes);
    double dJdr;
    double J = slater_tabs[idx].eval(r, dJdr);
    fforce = -K * (dJdr + factor_coul*r2inv) * rinv;
    return K * (J - factor_coul*rinv);
  }
  if (shield_gauss == SHIELD_GAUSSIAN) {
    double aij = pqeq_aij(itype, jtype);   // : per-pair override or the combination rule
    double erfar = erf(aij*r), expt = exp(-aij*aij*rsq), r3inv = rinv*r2inv;
    fforce = K * ((erfar - factor_coul)*r3inv - (2.0*aij/MY_PIS)*expt*r2inv);
    return K * (erfar - factor_coul) * rinv;
  }
  double g = gamma[itype][jtype];
  double sh = 1.0/cbrt(rsq*r + 1.0/(g*g*g));
  double sh4 = sh*sh*sh*sh;
  fforce = K * (r*sh4 - factor_coul*rinv*r2inv);
  return K * (sh - factor_coul*rinv);
}

/* option A: per-type-pair Gaussian radius override; pushed by FixQEqSam::push_shield_pairs() at init.*/
void PairCoulShieldIntra::shield_rpair_set(int i, int j, double r)
{
  const int np1 = atom->ntypes + 1;
  if (!shield_rpair) {
    memory->create(shield_rpair, np1, np1, "pair:shield_rpair");
    for (int a = 0; a < np1; a++) for (int b = 0; b < np1; b++) shield_rpair[a][b] = 0.0;
  }
  shield_rpair[i][j] = shield_rpair[j][i] = r;
}
