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
   Contributing author: Hasan Metin Aktulga, Purdue University
   (now at Lawrence Berkeley National Laboratory, hmaktulga@lbl.gov)

     Hybrid and sub-group capabilities: Ray Shan (Sandia)
-------------------------------------------------------------------------*/

#include "fix_qeq_base_sam.h"

#include "atom.h"
#include "citeme.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix_efield.h"   // core fix (always built); requires friend class FixQEqBaseSam in fix_efield.h
#include "force.h"
#include "group.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "pair.h"
#include "region.h"
#include "respa.h"
#include "text_file_reader.h"
#include "update.h"

#include "math_special.h"   // core: square()/cube() (replaces reaxff_defs.h SQR/CUBE)

// SQR/CUBE map to the core MathSpecial helpers so this base needs no REAXFF package
// while keeping the upstream call sites verbatim. The sizing knobs below are likewise inlined defaults
// (samQEq is file-mode only — no reaxff pair).
#define SQR(x)  MathSpecial::square(x)
#define CUBE(x) MathSpecial::cube(x)
static constexpr int    REAX_MIN_CAP   = 50;
static constexpr int    REAX_MIN_NBRS  = 100;
static constexpr double REAX_SAFE_ZONE = 1.2;

#include <cmath>
#include <cstring>
#include <exception>

using namespace LAMMPS_NS;
using namespace FixConst;

// NOTE: the Coulomb prefactor is force->qqrd2e (units-agnostic: 14.399645 eV·Å in metal,
// 332.06 kcal/mol·Å in real), matching the DSF/Ewald paths and the LAMMPS QEQ convention
// (cf. fix qeq/slater).
static constexpr double SMALL = 1.0e-14;
static constexpr double QSUMSMALL = 0.00001;

static const char cite_fix_qeq_reaxff[] =
  "fix qeq/reaxff command: https://doi.org/10.1016/j.parco.2011.08.005\n\n"
  "@Article{Aktulga12,\n"
  " author = {H. M. Aktulga and J. C. Fogarty and S. A. Pandit and A. Y. Grama},\n"
  " title = {Parallel Reactive Molecular Dynamics: {N}umerical Methods and Algorithmic Techniques},\n"
  " journal = {Parallel Computing},\n"
  " year = 2012,\n"
  " volume = 38,\n"
  " pages = {245--259}\n"
  "}\n\n";

/* ----------------------------------------------------------------------*/

FixQEqBaseSam::FixQEqBaseSam(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), matvecs(0), pertype_option(nullptr)
{
  scalar_flag = 1;
  extscalar = 0;
  imax = 1000;   // above the qeq/reaxff default of 200: the near-singular METAL saddle (BiCGStab,
                 // diagonal precond) needs ~hundreds–1000 iters, and a lower cap truncates it to a wrong
                 // charge. Converging solves stop at convergence regardless of the cap (well-conditioned
                 // cases finish in <50), so only the metal case is affected; override with `maxiter <N>`.
  maxwarn = 1;

  if ((narg < 8) || (narg > 12)) error->all(FLERR,"Illegal fix qeq/reaxff command");

  nevery = utils::inumeric(FLERR,arg[3],false,lmp);
  if (nevery <= 0) error->all(FLERR,"Illegal fix qeq/reaxff command");

  swa = utils::numeric(FLERR,arg[4],false,lmp);
  swb = utils::numeric(FLERR,arg[5],false,lmp);
  tolerance = utils::numeric(FLERR,arg[6],false,lmp);
  pertype_option = utils::strdup(arg[7]);

  // Units policy: internal hardcoded eV-anchored constants scale by this at
  // point of use; charge/geometry/time constants never do. update->unit_style is valid here (units
  // precedes all fix commands). Runs first in the ctor chain (FixQEqBaseSam is the base of
  // FixACKS2Sam/FixQEqSam), so every derived-class ctor default that wants ev_scale can use it.
  if (strcmp(update->unit_style,"metal") == 0) ev_scale = 1.0;
  else if (strcmp(update->unit_style,"real") == 0) ev_scale = 23.060549;
  else error->all(FLERR, "Fix {} supports only units metal or real", style);

  int iarg = 8;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"nowarn") == 0) maxwarn = 0;
    else if (strcmp(arg[iarg],"maxiter") == 0) {
      if (iarg+1 > narg-1)
        error->all(FLERR, iarg, "Illegal fix {} command", style);
      imax = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      iarg++;
    } else error->all(FLERR, iarg, "Illegal fix {} command", style);
    iarg++;
  }
  shld = nullptr;

  nn = n_cap = 0;
  nmax = 0;
  m_fill = m_cap = 0;
  pack_flag = 0;
  s = nullptr;
  t = nullptr;
  nprev = 4;
  // DECLARE the per-atom exchange payload (LAMMPS contract: any fix whose pack_exchange
  // appends data must set Fix::maxexchange, else Comm/Irregular size their exchange buffers
  // without it — an undersized-slack heap overrun that corrupts adjacent packed atoms). Base
  // packs s_hist only (nprev); FixACKS2Sam/FixQEqSam override both pack_exchange AND this value.
  maxexchange = nprev;

  Hdia_inv = nullptr;
  b_s = nullptr;
  chi_field = nullptr;
  b_t = nullptr;
  b_prc = nullptr;
  b_prm = nullptr;

  // CG

  p = nullptr;
  q = nullptr;
  r = nullptr;
  d = nullptr;

  // H matrix

  H.firstnbr = nullptr;
  H.numnbrs = nullptr;
  H.jlist = nullptr;
  H.val = nullptr;

  // comm sizes for this fix
  comm_forward = comm_reverse = 1;

  // perform initial allocation of atom-based arrays
  // register with Atom class
  // (samQEq: file-mode only — no reaxff pair coupling)

  s_hist = t_hist = nullptr;

  // ASPC: off by default => exact BO path; q_hist allocated in grow_arrays, reset on neighbor rebuild
  aspc_on = 0; aspc_ncorr = 1; aspc_korder = 2; aspc_nhist = 4;
  aspc_have = 0; aspc_build = -1; aspc_omega = 4.0/7.0; q_hist = nullptr; ngroup_fq = 0;
  aspc_rtol = 1.0e-2; cg_relresid = 0.0;   // quality gate: accept the corrector only if rel-resid < rtol
  aspc_naccept = 0; aspc_nreject = 0;      // diag: cumulative corrector accept/reject counters
  for (int j = 0; j < 8; ++j) aspc_B[j] = 0.0;

  atom->add_callback(Atom::GROW);
}

/* ----------------------------------------------------------------------*/

FixQEqBaseSam::~FixQEqBaseSam()
{
  if (copymode) return;

  delete[] pertype_option;

  // unregister callbacks to this fix from Atom class

  atom->delete_callback(id,Atom::GROW);

  memory->destroy(s_hist);
  memory->destroy(t_hist);
  memory->destroy(q_hist);   // ASPC

  FixQEqBaseSam::deallocate_storage();
  FixQEqBaseSam::deallocate_matrix();

  memory->destroy(shld);

  if (!reaxflag) {
    memory->destroy(chi);
    memory->destroy(eta);
    memory->destroy(gamma);
  }
}

/* ----------------------------------------------------------------------*/

void FixQEqBaseSam::post_constructor()
{
  if (lmp->citeme) lmp->citeme->add(cite_fix_qeq_reaxff);

  grow_arrays(atom->nmax);
  for (int i = 0; i < atom->nmax; i++)
    for (int j = 0; j < nprev; ++j)
      s_hist[i][j] = t_hist[i][j] = 0;

  pertype_parameters(pertype_option);
}

/* ----------------------------------------------------------------------*/

int FixQEqBaseSam::setmask()
{
  int mask = 0;
  mask |= PRE_FORCE;
  mask |= PRE_FORCE_RESPA;
  mask |= MIN_PRE_FORCE;
  return mask;
}

/* ----------------------------------------------------------------------*/


/* ----------------------------------------------------------------------*/

void FixQEqBaseSam::allocate_storage()
{
  nmax = atom->nmax;

  memory->create(s,nmax,"qeq:s");
  memory->create(t,nmax,"qeq:t");

  memory->create(Hdia_inv,nmax,"qeq:Hdia_inv");
  memory->create(b_s,nmax,"qeq:b_s");
  memory->create(chi_field,nmax,"qeq:chi_field");
  memory->create(b_t,nmax,"qeq:b_t");
  memory->create(b_prc,nmax,"qeq:b_prc");
  memory->create(b_prm,nmax,"qeq:b_prm");

  // dual CG support
  int size = nmax;

  memory->create(p,size,"qeq:p");
  memory->create(q,size,"qeq:q");
  memory->create(r,size,"qeq:r");
  memory->create(d,size,"qeq:d");
}

/* ----------------------------------------------------------------------*/

void FixQEqBaseSam::deallocate_storage()
{
  memory->destroy(s);
  memory->destroy(t);

  memory->destroy(Hdia_inv);
  memory->destroy(b_s);
  memory->destroy(b_t);
  memory->destroy(b_prc);
  memory->destroy(b_prm);
  memory->destroy(chi_field);

  memory->destroy(p);
  memory->destroy(q);
  memory->destroy(r);
  memory->destroy(d);
}

/* ----------------------------------------------------------------------*/

void FixQEqBaseSam::reallocate_storage()
{
  deallocate_storage();
  allocate_storage();
  init_storage();
}

/* ----------------------------------------------------------------------*/

void FixQEqBaseSam::allocate_matrix()
{
  int i,ii;
  bigint m;

  int mincap = REAX_MIN_CAP;
  double safezone = REAX_SAFE_ZONE;

  n_cap = MAX((int)(atom->nlocal * safezone), mincap);

  // determine the total space for the H matrix

  m = 0;
  for (ii = 0; ii < nn; ii++) {
    i = ilist[ii];
    m += numneigh[i];
  }
  auto m_cap_big = (bigint)MAX(m * safezone, mincap * REAX_MIN_NBRS);
  if (m_cap_big > MAXSMALLINT)
    error->one(FLERR, Error::NOLASTLINE, "Too many neighbors in fix {}",style);
  m_cap = m_cap_big;

  H.n = n_cap;
  H.m = m_cap;
  memory->create(H.firstnbr,n_cap,"qeq:H.firstnbr");
  memory->create(H.numnbrs,n_cap,"qeq:H.numnbrs");
  memory->create(H.jlist,m_cap,"qeq:H.jlist");
  memory->create(H.val,m_cap,"qeq:H.val");
}

/* ----------------------------------------------------------------------*/

void FixQEqBaseSam::deallocate_matrix()
{
  memory->destroy(H.firstnbr);
  memory->destroy(H.numnbrs);
  memory->destroy(H.jlist);
  memory->destroy(H.val);
}

/* ----------------------------------------------------------------------*/

void FixQEqBaseSam::reallocate_matrix()
{
  deallocate_matrix();
  allocate_matrix();
}

/* ----------------------------------------------------------------------
   Guarantee the H CSR arrays can hold THIS step's fill.

   ★ Why: allocate_matrix() sizes m_cap once from the neighbour counts of the moment, and
   pre_force() only regrows when the PREVIOUS step's m_fill crossed DANGER_ZONE (0.90). At high
   rank counts a rank owns few atoms, so a single migrating atom can lift the fill >10 % in ONE
   step and overflow m_cap before any guard observes it (compute_H then stops with
   "samqeq: H matrix overflow"). Widening the margin only moves the cliff; sizing from the
   CURRENT list removes it.

   sum(numneigh) over ilist is a strict UPPER BOUND on m_fill (compute_H additionally drops
   non-group columns, applies the swb cutoff and half-list dedup), so m_cap >= that sum makes
   an overflow impossible rather than merely unlikely. Cost is one O(N_local) pass per step.
-------------------------------------------------------------------------*/

void FixQEqBaseSam::ensure_matrix_capacity()
{
  if (!ilist || !numneigh) return;
  bigint need = 0;
  for (int ii = 0; ii < nn; ii++) need += numneigh[ilist[ii]];
  if ((bigint) m_cap < need) reallocate_matrix();   // recomputes m_cap from the CURRENT list
}

/* ----------------------------------------------------------------------*/

void FixQEqBaseSam::init()
{
  if (!atom->q_flag)
    error->all(FLERR, Error::NOLASTLINE, "Fix {} requires atom attribute q", style);

  if (group->count(igroup) == 0)
    error->all(FLERR, Error::NOLASTLINE, "Fix {} group has no atoms", style);

  // compute net charge and print warning if too large

  double qsum_local = 0.0, qsum = 0.0;
  for (int i = 0; i < atom->nlocal; i++) {
    if (atom->mask[i] & groupbit)
      qsum_local += atom->q[i];
  }
  MPI_Allreduce(&qsum_local,&qsum,1,MPI_DOUBLE,MPI_SUM,world);

  if ((comm->me == 0) && (fabs(qsum) > QSUMSMALL))
    error->warning(FLERR, "Fix {} group is not charge neutral, net charge = {:.8}" + utils::errorurl(29), style, qsum);

  // get pointer to fix efield if present. there may be at most one instance of fix efield in use.
  // (samQEq accesses FixEfield's per-atom field via friendship, exactly as qeq/reaxff does;
  //  fix_efield.h is CORE — always built — and lists `friend class FixQEqBaseSam`.)

  efield = nullptr;
  auto fixes = modify->get_fix_by_style("^efield");
  if (fixes.size() == 1) efield = dynamic_cast<FixEfield *>(fixes.front());
  else if (fixes.size() > 1)
    error->all(FLERR, Error::NOLASTLINE, "There may be only one fix efield instance used with fix {}", style);

  // ensure that fix efield is properly initialized before accessing its data and check some settings
  if (efield) {
    efield->init();
    // Unlike qeq/reaxff (whose chi/eta are always eV), samQEq runs chi/eta/H in NATIVE DECK UNITS
    // (force->qqrd2e), so get_chi_field uses factor = -1.0 (chi_field = deck energy/e, see the
    // unit-chain comment there) and fix efield works in BOTH metal and real. (Unit-style gating overall
    // is handled in the ctor: ev_scale errors on anything but metal/real.)

    if (efield->varflag == FixEfield::ATOM && efield->pstyle != FixEfield::ATOM)
      error->all(FLERR, Error::NOLASTLINE, "Atom-style external electric field requires atom-style"
                 "potential variable when used with fix {}", style);
    // PERIODIC-DIRECTION GUARD. Only a CONSTANT field component along a periodic axis is ill-defined: its potential -E.x is not
    // periodic, so get_chi_field's unmapped coordinate makes the RHS depend on image flags. An
    // ATOM-STYLE potential is different -- it is a user function of the WRAPPED coordinates and is
    // therefore periodic by construction. That is exactly how a field is applied across a slab in a
    // periodic cell (a sawtooth: -E.z over the slab, compensated in the vacuum), i.e. what VASP's
    // EFIELD/LDIPOL does. Forbidding it would force every samQEq field calculation onto cutoff
    // electrostatics, which a metal slab cannot tolerate (the truncated lattice sum gives the
    // interlayer coupling the wrong sign and a staggered lowest mode -> alternating layer
    // charges). The user owns the physics of the potential they
    // supply; the code only has to refuse the case it genuinely cannot represent.
    const bool atom_pot = (efield->varflag == FixEfield::ATOM && efield->pstyle == FixEfield::ATOM);
    if (!atom_pot) {
      if (((efield->xstyle != FixEfield::CONSTANT) && domain->xperiodic) ||
           ((efield->ystyle != FixEfield::CONSTANT) && domain->yperiodic) ||
           ((efield->zstyle != FixEfield::CONSTANT) && domain->zperiodic))
        error->all(FLERR, Error::NOLASTLINE, "Must not have electric field component in direction of periodic"
                         "boundary when using charge equilibration with samQEq (use an atom-style"
                         "potential variable for a periodic sawtooth field).");
      if (((fabs(efield->ex) > SMALL) && domain->xperiodic) ||
           ((fabs(efield->ey) > SMALL) && domain->yperiodic) ||
           ((fabs(efield->ez) > SMALL) && domain->zperiodic))
        error->all(FLERR, Error::NOLASTLINE, "Must not have electric field component in direction of periodic"
                         "boundary when using charge equilibration with samQEq (use an atom-style"
                         "potential variable for a periodic sawtooth field).");
    }
  }

  // we need a half neighbor list w/ Newton off
  // built whenever re-neighboring occurs

  neighbor->add_request(this, NeighConst::REQ_NEWTON_OFF);

  init_shielding();
  init_taper();

  if (utils::strmatch(update->integrate_style,"^respa"))
    nlevels_respa = (dynamic_cast<Respa *>(update->integrate))->nlevels;
}

/* ----------------------------------------------------------------------*/

double FixQEqBaseSam::compute_scalar()
{
  return matvecs/2.0;
}

/* ----------------------------------------------------------------------*/

void FixQEqBaseSam::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ----------------------------------------------------------------------*/

void FixQEqBaseSam::init_shielding()
{
  int i,j;
  int ntypes;

  ntypes = atom->ntypes;
  if (shld == nullptr)
    memory->create(shld,ntypes+1,ntypes+1,"qeq:shielding");

  for (i = 1; i <= ntypes; ++i)
    for (j = 1; j <= ntypes; ++j)
      shld[i][j] = pow(gamma[i] * gamma[j], -1.5);
}

/* ----------------------------------------------------------------------*/

void FixQEqBaseSam::init_taper()
{
  double d7, swa2, swa3, swb2, swb3;

  if (fabs(swa) > 0.01 && comm->me == 0)
    error->warning(FLERR,"Fix qeq/reaxff has non-zero lower Taper radius cutoff");
  if (swb < 0)
    error->all(FLERR, Error::NOLASTLINE, "Fix qeq/reaxff has negative upper Taper radius cutoff");
  else if (swb < 5 && comm->me == 0)
    error->warning(FLERR,"Fix qeq/reaxff has very low Taper radius cutoff");
  if (swb <= swa)
    error->all(FLERR, "fix qeq/sam taper: swb must be > swa");

  d7 = pow(swb - swa, 7);
  swa2 = SQR(swa);
  swa3 = CUBE(swa);
  swb2 = SQR(swb);
  swb3 = CUBE(swb);

  Tap[7] =  20.0 / d7;
  Tap[6] = -70.0 * (swa + swb) / d7;
  Tap[5] =  84.0 * (swa2 + 3.0*swa*swb + swb2) / d7;
  Tap[4] = -35.0 * (swa3 + 9.0*swa2*swb + 9.0*swa*swb2 + swb3) / d7;
  Tap[3] = 140.0 * (swa3*swb + 3.0*swa2*swb2 + swa*swb3) / d7;
  Tap[2] =-210.0 * (swa3*swb2 + swa2*swb3) / d7;
  Tap[1] = 140.0 * swa3 * swb3 / d7;
  Tap[0] = (-35.0*swa3*swb2*swb2 + 21.0*swa2*swb3*swb2 -
            7.0*swa*swb3*swb3 + swb3*swb3*swb) / d7;
}

/* ----------------------------------------------------------------------*/

void FixQEqBaseSam::setup_pre_force(int vflag)
{
  {
    nn = list->inum;
    ilist = list->ilist;
    numneigh = list->numneigh;
    firstneigh = list->firstneigh;
  }

  deallocate_storage();
  allocate_storage();

  init_storage();

  deallocate_matrix();
  allocate_matrix();

  pre_force(vflag);
}

/* ----------------------------------------------------------------------*/

void FixQEqBaseSam::setup_pre_force_respa(int vflag, int ilevel)
{
  if (ilevel < nlevels_respa-1) return;
  setup_pre_force(vflag);
}

/* ----------------------------------------------------------------------*/

void FixQEqBaseSam::min_setup_pre_force(int vflag)
{
  setup_pre_force(vflag);
}

/* ----------------------------------------------------------------------*/

void FixQEqBaseSam::init_storage()
{
  if (efield) get_chi_field();

  for (int ii = 0; ii < nn; ii++) {
    int i = ilist[ii];
    if (atom->mask[i] & groupbit) {
      Hdia_inv[i] = 1. / eta[atom->type[i]];
      b_s[i] = -chi[atom->type[i]];
      if (efield) b_s[i] -= chi_field[i];
      b_t[i] = -1.0;
      b_prc[i] = 0;
      b_prm[i] = 0;
      s[i] = t[i] = 0;
    }
  }
}

/* ----------------------------------------------------------------------*/


/* ----------------------------------------------------------------------
   ASPC — q-direct predictor-corrector: coefficient setup.
   Targets the QEq base (global neutrality, sum q = 0). NOT engaged for the ACKS2 saddle (it keeps
   aspc_on=0 -> BO).
-------------------------------------------------------------------------*/

void FixQEqBaseSam::aspc_setup()
{
  // time-reversible Kolafa coefficients (sum B = 1); k = aspc_korder, history = k+2, omega = (k+2)/(2k+3)
  int k = aspc_korder;
  aspc_nhist = k + 2;
  aspc_omega = (double)(k + 2) / (double)(2*k + 3);
  for (int j = 0; j < 8; ++j) aspc_B[j] = 0.0;
  if (k == 1)      { double b[] = { 2.5, -2.0, 0.5 };                              for (int j=0;j<3;j++) aspc_B[j]=b[j]; }
  else if (k == 2) { double b[] = { 2.8, -2.8, 1.2, -0.2 };                        for (int j=0;j<4;j++) aspc_B[j]=b[j]; }
  else if (k == 3) { double b[] = { 3.0, -24.0/7, 27.0/14, -4.0/7, 1.0/14 };       for (int j=0;j<5;j++) aspc_B[j]=b[j]; }
  else             { aspc_korder = 2; aspc_setup(); return; }   // default/clamp to k=2
  aspc_have = 0; aspc_build = -1;
  ngroup_fq = group->count(igroup);
}


/* ----------------------------------------------------------------------*/

void FixQEqBaseSam::pre_force_respa(int vflag, int ilevel, int /*iloop*/)
{
  if (ilevel == nlevels_respa-1) pre_force(vflag);
}

/* ----------------------------------------------------------------------*/

void FixQEqBaseSam::min_pre_force(int vflag)
{
  pre_force(vflag);
}

/* ----------------------------------------------------------------------*/


/* ----------------------------------------------------------------------*/

void FixQEqBaseSam::compute_H()
{
  ensure_matrix_capacity();
  int jnum;
  int i, j, ii, jj, flag;
  double dx, dy, dz, r_sqr;
  constexpr double EPSILON = 0.0001;

  int *type = atom->type;
  tagint *tag = atom->tag;
  double **x = atom->x;
  int *mask = atom->mask;

  // fill in the H matrix
  m_fill = 0;
  r_sqr = 0;
  for (ii = 0; ii < nn; ii++) {
    i = ilist[ii];
    if (mask[i] & groupbit) {
      jlist = firstneigh[i];
      jnum = numneigh[i];
      H.firstnbr[i] = m_fill;

      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        j &= NEIGHMASK;

        // COLUMN group test -- see the full rationale in FixQEqSam::compute_H
        // (fix_qeq_sam.cpp). In brief: H multiplies the solve vector, so a column for a non-group
        // atom multiplies never-written workspace (silent garbage) or zero (missing physics); the
        // field of fixed non-group charges belongs in the RHS. Whole-system solves are unaffected.
        if (!(mask[j] & groupbit)) continue;

        dx = x[j][0] - x[i][0];
        dy = x[j][1] - x[i][1];
        dz = x[j][2] - x[i][2];
        r_sqr = SQR(dx) + SQR(dy) + SQR(dz);

        flag = 0;
        if (r_sqr <= SQR(swb)) {
          if (j < atom->nlocal) flag = 1;
          else if (tag[i] < tag[j]) flag = 1;
          else if (tag[i] == tag[j]) {
            if (dz > EPSILON) flag = 1;
            else if (fabs(dz) < EPSILON) {
              if (dy > EPSILON) flag = 1;
              else if (fabs(dy) < EPSILON && dx > EPSILON)
                flag = 1;
            }
          }
        }

        if (flag) {
          if (m_fill >= H.m)
            error->one(FLERR, "samqeq: H matrix overflow at atom {} (m_fill {} >= {})", i, m_fill, H.m);
          H.jlist[m_fill] = j;
          // calc_Hval (virtual): default = the shld/calculate_H lookup; FixQEqSam overrides it to
          // route through the Slater J(r) table when shield_gauss==SHIELD_SLATER, so this gas path uses
          // the same kernel as shielded_coulomb().
          H.val[m_fill] = calc_Hval(sqrt(r_sqr), type[i], type[j]);
          m_fill++;
        }
      }
      H.numnbrs[i] = m_fill - H.firstnbr[i];
    }
  }

  if (m_fill >= H.m)
    error->all(FLERR, Error::NOLASTLINE, "Fix qeq/reaxff H matrix size has been exceeded: m_fill={} H.m={}\n",
               m_fill, H.m);
}

/* ----------------------------------------------------------------------*/

double FixQEqBaseSam::calculate_H(double r, double gamma)
{
  double Taper, denom;

  Taper = Tap[7] * r + Tap[6];
  Taper = Taper * r + Tap[5];
  Taper = Taper * r + Tap[4];
  Taper = Taper * r + Tap[3];
  Taper = Taper * r + Tap[2];
  Taper = Taper * r + Tap[1];
  Taper = Taper * r + Tap[0];

  denom = r * r * r + gamma;
  denom = pow(denom,1.0/3.0);

  return Taper * force->qqrd2e / denom;
}

/* ----------------------------------------------------------------------*/

/* calc_Hval(r,ti,tj): default = the inline shld-via-calculate_H lookup, used by every class that
   doesn't override this (and by FixQEqSam itself whenever shield_gauss != SHIELD_SLATER -- see
   FixQEqSam::calc_Hval, fix_qeq_sam.cpp).*/
double FixQEqBaseSam::calc_Hval(double r, int ti, int tj)
{
  return calculate_H(r, shld[ti][tj]);
}

/* ----------------------------------------------------------------------*/


/* ----------------------------------------------------------------------*/


/* ----------------------------------------------------------------------*/


/* ----------------------------------------------------------------------*/

int FixQEqBaseSam::pack_forward_comm(int n, int *list, double *buf,
                                  int /*pbc_flag*/, int * /*pbc*/)
{
  int m;

  if (pack_flag == 1)
    for (m = 0; m < n; m++) buf[m] = d[list[m]];
  else if (pack_flag == 2)
    for (m = 0; m < n; m++) buf[m] = s[list[m]];
  else if (pack_flag == 3)
    for (m = 0; m < n; m++) buf[m] = t[list[m]];
  else if (pack_flag == 4)
    for (m = 0; m < n; m++) buf[m] = atom->q[list[m]];
  return n;
}

/* ----------------------------------------------------------------------*/

void FixQEqBaseSam::unpack_forward_comm(int n, int first, double *buf)
{
  int i, m;

  if (pack_flag == 1)
    for (m = 0, i = first; m < n; m++, i++) d[i] = buf[m];
  else if (pack_flag == 2)
    for (m = 0, i = first; m < n; m++, i++) s[i] = buf[m];
  else if (pack_flag == 3)
    for (m = 0, i = first; m < n; m++, i++) t[i] = buf[m];
  else if (pack_flag == 4)
    for (m = 0, i = first; m < n; m++, i++) atom->q[i] = buf[m];
}

/* ----------------------------------------------------------------------*/

int FixQEqBaseSam::pack_reverse_comm(int n, int first, double *buf)
{
  int i, m;
  for (m = 0, i = first; m < n; m++, i++) buf[m] = q[i];
  return n;
}

/* ----------------------------------------------------------------------*/

void FixQEqBaseSam::unpack_reverse_comm(int n, int *list, double *buf)
{
  for (int m = 0; m < n; m++) q[list[m]] += buf[m];
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
-------------------------------------------------------------------------*/

double FixQEqBaseSam::memory_usage()
{
  double bytes;

  bytes = (double)atom->nmax*nprev*2 * sizeof(double); // s_hist & t_hist
  bytes += (double)atom->nmax*11 * sizeof(double); // storage
  bytes += (double)n_cap*2 * sizeof(int); // matrix...
  bytes += (double)m_cap * sizeof(int);
  bytes += (double)m_cap * sizeof(double);

  return bytes;
}

/* ----------------------------------------------------------------------
   allocate fictitious charge arrays
-------------------------------------------------------------------------*/

void FixQEqBaseSam::grow_arrays(int nmax)
{
  memory->grow(s_hist,nmax,nprev,"qeq:s_hist");
  memory->grow(t_hist,nmax,nprev,"qeq:t_hist");
  // ASPC: q-history; sized to the max supported order (k<=4 -> nhist<=6). This BASE-class
  // copy_arrays/pack_exchange (below) do NOT migrate it, but FixQEqBaseSam has no FixStyle (base only);
  // the concrete FixACKS2Sam (used by qeq/sam) OVERRIDES grow_arrays/copy_arrays/pack_exchange and DOES
  // migrate q_hist (see fix_acks2_sam.cpp), so in practice it always travels with the atom.
  memory->grow(q_hist,nmax,6,"qeq:q_hist");
}

/* ----------------------------------------------------------------------
   copy values within fictitious charge arrays
-------------------------------------------------------------------------*/

void FixQEqBaseSam::copy_arrays(int i, int j, int /*delflag*/)
{
  for (int m = 0; m < nprev; m++) {
    s_hist[j][m] = s_hist[i][m];
    t_hist[j][m] = t_hist[i][m];
  }
}

/* ----------------------------------------------------------------------
   pack values in local atom-based array for exchange with another proc
-------------------------------------------------------------------------*/

int FixQEqBaseSam::pack_exchange(int i, double *buf)
{
  for (int m = 0; m < nprev; m++) buf[m] = s_hist[i][m];
  for (int m = 0; m < nprev; m++) buf[nprev+m] = t_hist[i][m];
  return nprev*2;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based array from exchange with another proc
-------------------------------------------------------------------------*/

int FixQEqBaseSam::unpack_exchange(int nlocal, double *buf)
{
  for (int m = 0; m < nprev; m++) s_hist[nlocal][m] = buf[m];
  for (int m = 0; m < nprev; m++) t_hist[nlocal][m] = buf[nprev+m];
  return nprev*2;
}

/* ----------------------------------------------------------------------*/

double FixQEqBaseSam::parallel_norm(double *v, int n)
{
  int  i;
  double my_sum, norm_sqr;

  int ii;

  my_sum = 0.0;
  norm_sqr = 0.0;
  for (ii = 0; ii < n; ++ii) {
    i = ilist[ii];
    if (atom->mask[i] & groupbit)
      my_sum += SQR(v[i]);
  }

  MPI_Allreduce(&my_sum, &norm_sqr, 1, MPI_DOUBLE, MPI_SUM, world);

  return sqrt(norm_sqr);
}

/* ----------------------------------------------------------------------*/

double FixQEqBaseSam::parallel_dot(double *v1, double *v2, int n)
{
  int  i;
  double my_dot, res;

  int ii;

  my_dot = 0.0;
  res = 0.0;
  for (ii = 0; ii < n; ++ii) {
    i = ilist[ii];
    if (atom->mask[i] & groupbit)
      my_dot += v1[i] * v2[i];
  }

  MPI_Allreduce(&my_dot, &res, 1, MPI_DOUBLE, MPI_SUM, world);

  return res;
}

/* ----------------------------------------------------------------------*/

double FixQEqBaseSam::parallel_vector_acc(double *v, int n)
{
  int  i;
  double my_acc, res;

  int ii;

  my_acc = 0.0;
  res = 0.0;
  for (ii = 0; ii < n; ++ii) {
    i = ilist[ii];
    if (atom->mask[i] & groupbit)
      my_acc += v[i];
  }

  MPI_Allreduce(&my_acc, &res, 1, MPI_DOUBLE, MPI_SUM, world);

  return res;
}

/* ----------------------------------------------------------------------*/

void FixQEqBaseSam::vector_sum(double* dest, double c, double* v,
                                double d, double* y, int k)
{
  int kk;

  for (--k; k>=0; --k) {
    kk = ilist[k];
    if (atom->mask[kk] & groupbit)
      dest[kk] = c * v[kk] + d * y[kk];
  }
}

/* ----------------------------------------------------------------------*/

void FixQEqBaseSam::vector_add(double* dest, double c, double* v, int k)
{
  int kk;

  for (--k; k>=0; --k) {
    kk = ilist[k];
    if (atom->mask[kk] & groupbit)
      dest[kk] += c * v[kk];
  }
}

/* ----------------------------------------------------------------------*/

void FixQEqBaseSam::get_chi_field()
{
  memset(&chi_field[0],0,atom->nmax*sizeof(double));
  if (!efield) return;

  const auto *const x = (const double * const *)atom->x;
  const int *mask = atom->mask;
  const imageint *image = atom->image;
  const int nlocal = atom->nlocal;


  // update electric field region if necessary

  Region *region = efield->region;
  if (region) region->prematch();

  // UNIT CHAIN: fix efield stores ex = qe2f*E_input (E_input in V/Ang in BOTH metal and real; qe2f =
  // 1.0 metal / 23.060549 real), i.e. deck-FORCE per charge = deck-ENERGY/(e*Ang). factor = -1.0
  // therefore leaves chi_field = -(stored ex)*x in NATIVE DECK ENERGY UNITS per e (eV/e metal,
  // kcal/mol/e real) -- exactly what the RHS wants, since it is summed with the native-deck-unit chi
  // (b_s = -chi - chi_field). qeq/reaxff's factor -1/qe2f gives TRUE eV/e instead, which suits its
  // always-eV chi but would be 23.06x too weak against samQEq's native-unit chi in units real.
  // In units metal the two agree (qe2f == 1.0).

  const double factor = -1.0;


  if (efield->varflag != FixEfield::CONSTANT)
    efield->update_efield_variables();

  // atom selection is for the group of fix efield

  double unwrap[3];
  const double ex = efield->ex;
  const double ey = efield->ey;
  const double ez = efield->ez;
  const int efgroupbit = efield->groupbit;

    // charge interactions
    // force = qE, potential energy = F dot x in unwrapped coords
  if (efield->varflag != FixEfield::ATOM) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & efgroupbit) {
        if (region && !region->match(x[i][0],x[i][1],x[i][2])) continue;
        domain->unmap(x[i],image[i],unwrap);
        chi_field[i] = factor*(ex*unwrap[0] + ey*unwrap[1] + ez*unwrap[2]);
      }
    }
  } else { // must use atom-style potential from FixEfield
    // efield[i][3] holds the potential-variable value in VOLTS (= eV/e); fix efield
    // itself converts it to deck energy as qe2f*q*phi (fix_efield.cpp fsum). Mirror that here so
    // chi_field = dU/dq = qe2f*phi is in native deck energy units per e, consistent with the
    // constant-field branch above. qe2f == 1.0 in units metal.
    const double qe2f = force->qe2f;
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & efgroupbit) {
        if (region && !region->match(x[i][0],x[i][1],x[i][2])) continue;
        chi_field[i] = qe2f*efield->efield[i][3];
      }
    }
  }
}
