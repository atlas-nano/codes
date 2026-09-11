// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors:
     Tod A Pascal (UCSD)
------------------------------------------------------------------------- */

#include "fix_xpt.h"

#include "atom.h"
#include "citeme.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "memory.h"
#include "modify.h"
#include "input.h"
#include "pair.h"
#include "update.h"
#include "variable.h"

#include "fft3d_wrap.h"       // FFT3d + FFT_SCALAR (backend-agnostic: KISS/FFTW3/MKL)
#ifdef FFT_FFTW3
#  include <fftw3.h>
#endif
#include "math_eigen.h"       // MathEigen::jacobi3 (3x3 inertia-tensor diagonalization)
#include "math_eigen_impl.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <complex>       // std::complex for kernel_volterra Laplace transform
#include <cstdio>
#include <cstring>
#include <limits>
#include <strings.h>     // strcasecmp (POSIX)
#include <map>
#include <set>
#include <string>
#include <vector>

// LAMMPS mask bits
#include "fix.h"

// Shared physical constants + FIXXPT_LOG / FIXXPT_TIME diagnostic macros,
// split out so the sibling fix_xpt_*.cpp translation units share one
// definition (defines FIX_XPT_DEBUG; include after the LAMMPS headers).
#include "fix_xpt_const.h"

using namespace LAMMPS_NS;
using namespace FixConst;

static const char cite_fix_xpt[] =
  "fix xpt command: https://doi.org/xxxx\n\n"
  "@Article{Pascal,11,\n"
  " author = {T.A. Pascal},\n"
  " title = {FixXPT: On-the-fly Estimation of Quantum-Corrected Thermodynamics in LAMMPS},\n"
  " journal = {Comput.\\ Phys.\\ Commun.},\n"
  " year =    2026,\n"
  " volume =  xx,\n"
  " pages =   {xxx--xxx}\n"
  "}\n\n";

/* ======================================================================
   Constructor / destructor
====================================================================== */

FixXPT::FixXPT(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
{

  if (lmp->citeme) lmp->citeme->add(cite_fix_xpt);

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  dynamic_group_allow = 1;

  // ── Initialize defaults BEFORE parsing args so set_option() operates on a
  //    fully-initialized object ───────────────────────────────────────────────
  nevery         = 0;
  nframes        = 0;
  prefix         = nullptr;
  do_classical   = 0;
  do_normalize   = 0;
  units_lj       = 0;
  vac_to_jmol    = 0.0;
  dt_to_ps       = 0.0;
  ke_to_pe_fac   = 0.0;
  pe_to_kjmol    = 0.0;
  ke_to_kjmol    = 0.0;
  mass_to_gmol   = 1.0;
  vol_to_angst3  = 1.0;
  e_out_scale    = 1.0;
  s_out_scale    = 1.0;
  lj_eps         = 0.0;
  lj_sig         = 0.0;
  lj_mass        = 0.0;
  lj_params_user = false;
  hbar_star      = 1.0;
  volume_style   = VOL_BOX;
  user_volume    = -1.0;
  volume_varstr  = nullptr;
  volume_varindex = -1;
  hs_entropy_mode  = 0;
  use_sim_z_mode   = 0;
  req_mode            = REQ_UNSET;          // `mode` string; resolved in init()
  refinement          = REF_RIGOROUS;       // sub-method; meaning depends on mode
  refinement_set      = 0;                  // user gave `refinement`?  (init default)
  r2pt_delta          = 1.5;                // Sun 2017 δ (refinement=r2pt)
  cage_entropy        = 0;                  // 3PT translational cage
  cage_entropy_rot    = 0;                  // 3PT rotational cage (molecular)
  s_cage_trans_last   = 0.0;
  s_cage_rot_last     = 0.0;

  // Multi-tau correlator defaults: opt-in opportunistic memory saver.
  correlator          = CORR_FFT;
  mt_M                = 32;
  mt_P                = 16;
  mt_S                = 2;
  mt_L                = 0;     // 0 = auto from nframes / mt_M
  nsamples            = 1;     // 1 = no sub-window snapshots (default)
  maxfreq             = 0.0;   // 0.0 = no truncation of .pwr/.vac output
  dnu_out             = 0.0;   // 0.0 = write at native dnu

  // Per-window timing buckets (FIX_XPT_DEBUG-only meaning).
  reset_window_timings();
  mt_n_levels         = 0;
  mt_MP               = 0;
  mt_natom_ring       = 0;
  mt_vel_buf_single_frame = false;
  mt_restart_warn_emitted = false;
  mt_molecular_active     = false;
  mt_trans.n_units        = 0;
  mt_rot.n_units          = 0;
  mt_vib.n_units          = 0;
  mt_trans.is_distributed = mt_rot.is_distributed = mt_vib.is_distributed = false;
  do_molecule  = 0;
  allow_mixed_molecules = 0;
  symmetry             = "C1";


  mol_mode     = 2;
  is_linear    = 0;
  rotsym       = 1;
  do_show_split = 0;
  nmol_group   = 0;
  com_vel_buf  = nullptr;
  omega_buf    = nullptr;
  angmom_buf   = nullptr;
  vib_vel_buf  = nullptr;
  vib_vel_buf_f = nullptr;
  vel_buf_f     = nullptr;
  buffer_precision = BUFFER_FP64;
  buffer_layout    = LAYOUT_DISTRIBUTED;
  buffer_layout_explicit = 0;
  ng_window    = 0;
  n_home       = 0;
  nm_home      = 0;
  nmol_buf     = 0;
  home_xu      = nullptr;
  mol_inertia  = nullptr;
  mol_inertia_count = 0;
  mol_topology_generation = 0;
  dof_last_window         = 0.0;
  max_memory_gb           = 0.0;   // 0 = no auto-resize

  // ── Dispatch on syntax mode ───────────────────────────────────────────────
  //
  // Legacy:    fix ID grp xpt <Nevery> <Nframes> <prefix> [keyword [val] ...]
  // new (INI): fix ID grp xpt output <prefix> input <ini-file>
  // new (cfg): fix ID grp xpt output <prefix> config """key=val\n..."""
  //
  // Detection: if arg[3] starts with a digit (Nevery), legacy syntax.
  // If arg[3] is "output", new syntax.
  if (narg < 4) error->all(FLERR, "fix xpt: too few arguments");

  bool new_syntax = (strcmp(arg[3], "output") == 0);

  if (new_syntax) {
    if (narg < 7) error->all(FLERR,
        "fix xpt: new syntax requires 'output <prefix> input <file>' "
        "or 'output <prefix> config \"\"\"...\"\"\"'");
    prefix = utils::strdup(arg[4]);

    if (strcmp(arg[5], "input") == 0) {
      parse_ini_file(arg[6]);
    } else if (strcmp(arg[5], "config") == 0) {
      parse_kv_lines(std::string(arg[6]));
    } else {
      error->all(FLERR, "fix xpt: expected 'input <file>' or 'config "
                        "\"\"\"...\"\"\"' after 'output <prefix>'");
    }
    if (nevery <= 0) error->all(FLERR, "fix xpt: ini/config must set "
                                       "'nevery' to a positive integer");
    if (nframes <= 0) error->all(FLERR, "fix xpt: ini/config must set "
                                        "'nframes' to a positive integer");

  } else {
    // Legacy: fix ID grp xpt <Nevery> <Nframes> <prefix> [keyword [val] ...]
    if (narg < 6) error->all(FLERR, "fix xpt: too few arguments");
    nevery  = utils::inumeric(FLERR, arg[3], false, lmp);
    nframes = utils::inumeric(FLERR, arg[4], false, lmp);
    if (nevery  <= 0) error->all(FLERR, "fix xpt: Nevery must be > 0");
    if (nframes <= 0) error->all(FLERR, "fix xpt: Nframes must be > 0");
    prefix = utils::strdup(arg[5]);

    int iarg = 6;
    while (iarg < narg) {
      const char *key = arg[iarg];
      // Boolean-flag keywords (presence-based, no value in legacy syntax)
      if (strcmp(key, "classical") == 0  || strcmp(key, "normalize")   == 0 ||
          strcmp(key, "molecule")  == 0  || strcmp(key, "linear")      == 0 ||
          strcmp(key, "show_split") == 0 || strcmp(key, "allow_mixed_molecules") == 0) {
        set_option(key, "");      // empty value → enable
        iarg++;
      } else {
        // Keyword with required value
        if (iarg + 1 >= narg)
          error->all(FLERR, "fix xpt: keyword '{}' requires a value", key);
        set_option(key, arg[iarg + 1]);
        iarg += 2;
      }
    }
  }

  // Auto-detect unit system from LAMMPS and set conversion factors.
  //
  // vac_to_jmol  : native (mass × vel²) → J/mol
  // dt_to_ps     : native timestep → ps  (for DOS freq axis in cm⁻¹)
  // ke_to_pe_fac : native (mass × vel²)/atom → native PE units/atom
  // pe_to_kjmol  : native PE total → kJ/mol-equivalent (for E_inst / Cv_fluct)
  // ke_to_kjmol  : native (mass × vel²) total → kJ/mol-equivalent
  // mass_to_gmol : native mass/atom → g/mol  (for K and D formulas)
  // vol_to_angst3: native volume → Å³        (for K, hs_entropy, rho)
  //
  // Physical constants used here:
  //   NA = 6.02214076e23 mol⁻¹
  //   1 Hartree = 4.35974e-18 J;  1 atu = 2.41888e-17 s
  //   1 Bohr = 5.29177e-11 m = 0.529177 Å
  const char *us = update->unit_style;
  if (strcmp(us, "lj") == 0) {
    units_lj = 1;
    // LJ factors not needed (handled by the units_lj branch everywhere)
    vac_to_jmol  = 1.0; dt_to_ps = 1.0; ke_to_pe_fac = 1.0;
    pe_to_kjmol  = 1.0; ke_to_kjmol = 1.0;
    mass_to_gmol = 1.0; vol_to_angst3 = 1.0;

  } else if (strcmp(us, "real") == 0) {
    // Mass=g/mol, vel=Å/fs, time=fs, energy=kcal/mol, vol=Å³
    vac_to_jmol  = 1e7;                  // (g/mol)(Å/fs)² = 1e7 J/mol
    dt_to_ps     = 1e-3;                 // fs → ps
    ke_to_pe_fac = 1e7 / 4184.0;        // (g/mol)(Å/fs)² → kcal/mol
    pe_to_kjmol  = 4.184;               // kcal/mol → kJ/mol
    ke_to_kjmol  = 1e7 * 1e-3;          // (g/mol)(Å/fs)² → kJ/mol (= 1e4)
    mass_to_gmol = 1.0;
    vol_to_angst3 = 1.0;

  } else if (strcmp(us, "metal") == 0) {
    // Mass=g/mol, vel=Å/ps, time=ps, energy=eV, vol=Å³
    vac_to_jmol  = VAC_TO_JMOL;         // (g/mol)(Å/ps)² = 10 J/mol
    dt_to_ps     = 1.0;
    ke_to_pe_fac = VAC_TO_JMOL / 96485.0; // (g/mol)(Å/ps)² → eV/atom
    pe_to_kjmol  = 96.485;              // eV → kJ/mol
    ke_to_kjmol  = VAC_TO_JMOL * 1e-3;  // (g/mol)(Å/ps)² → kJ/mol (= 0.01)
    mass_to_gmol = 1.0;
    vol_to_angst3 = 1.0;
    e_out_scale  = 1.0 / 96.485;           // kJ/mol → eV for output
    s_out_scale  = 1.0 / 96485.0;          // J/mol/K → eV/K for output

  } else if (strcmp(us, "si") == 0) {
    // Mass=kg/atom, vel=m/s, time=s, energy=J/atom, vol=m³
    vac_to_jmol  = NA;                   // kg*(m/s)² = J/atom → J/mol: ×NA
    dt_to_ps     = 1e12;                 // s → ps
    ke_to_pe_fac = 1.0;                  // kg*(m/s)² = J = pe units
    pe_to_kjmol  = NA * 1e-3;           // J/atom → kJ/mol
    ke_to_kjmol  = NA * 1e-3;
    mass_to_gmol = NA * 1e3;             // kg/atom × NA = kg/mol × 1e3 = g/mol
    vol_to_angst3 = 1e30;               // m³ → Å³  (1 m = 1e10 Å)

  } else if (strcmp(us, "cgs") == 0) {
    // Mass=g/atom, vel=cm/s, time=s, energy=erg/atom (g·cm²/s²), vol=cm³
    vac_to_jmol  = NA * 1e-7;           // erg/atom = 1e-7 J/atom → J/mol: ×NA×1e-7
    dt_to_ps     = 1e12;
    ke_to_pe_fac = 1.0;                  // g*(cm/s)² = erg = pe units
    pe_to_kjmol  = NA * 1e-10;          // erg/atom → kJ/mol: ×NA×1e-7×1e-3
    ke_to_kjmol  = NA * 1e-10;
    mass_to_gmol = NA;                   // g/atom × NA = g/mol
    vol_to_angst3 = 1e24;               // cm³ → Å³  (1 cm = 1e8 Å)

  } else if (strcmp(us, "micro") == 0) {
    // Mass=pg/atom (1e-12 g), vel=μm/μs (=m/s), time=μs, energy=pg·(μm/μs)², vol=μm³
    // 1 pg·(μm/μs)² = 1e-12 g · (m/s)² = 1e-15 J/atom
    vac_to_jmol  = NA * 1e-15;
    dt_to_ps     = 1e6;                  // μs → ps
    ke_to_pe_fac = 1.0;
    pe_to_kjmol  = NA * 1e-18;          // 1e-15 J/atom × NA × 1e-3 = NA×1e-18 kJ/mol
    ke_to_kjmol  = NA * 1e-18;
    mass_to_gmol = NA * 1e-12;           // pg/atom × NA = pg/mol × 1e-12 = g/mol
    vol_to_angst3 = 1e12;               // μm³ → Å³  (1 μm = 1e4 Å)

  } else if (strcmp(us, "nano") == 0) {
    // Mass=ag/atom (1e-18 g), vel=nm/ns (=m/s), time=ns, energy=ag·(nm/ns)², vol=nm³
    // 1 ag·(nm/ns)² = 1e-18 g · (m/s)² = 1e-21 J/atom
    vac_to_jmol  = NA * 1e-21;
    dt_to_ps     = 1e3;                  // ns → ps
    ke_to_pe_fac = 1.0;
    pe_to_kjmol  = NA * 1e-24;          // 1e-21 J/atom × NA × 1e-3 = NA×1e-24 kJ/mol
    ke_to_kjmol  = NA * 1e-24;
    mass_to_gmol = NA * 1e-18;           // ag/atom × NA = ag/mol × 1e-18 = g/mol
    vol_to_angst3 = 1e3;                // nm³ → Å³  (1 nm = 10 Å)

  } else if (strcmp(us, "electron") == 0) {
    // Mass=amu=g/mol, vel=Bohr/fs, time=fs, energy=Hartree/atom, vol=Bohr³.
    //
    // The time unit is the femtosecond, not the atomic time unit: LAMMPS sets
    // force->femtosecond = 1.0 and dt = 0.001 for this style.  The velocity
    // unit follows from force->mvv2e = 1.06657236, which states that
    // 1 amu*(velocity unit)^2 = 1.06657236 Hartree and solves to
    // 5.291772e4 m/s = 1 Bohr/fs.  (The units documentation gives
    // "Bohr/atu, 1 atu = 1.03275e-15 s", i.e. 5.1240e4 m/s, which does not
    // agree with mvv2e; mvv2e scales the velocities this fix reads.)
    // The factors are derived from mvv2e so they stay tied to it.
    static constexpr double HARTREE_J  = 4.3597447222e-18;  // J/Hartree
    static constexpr double BOHR_M     = 5.29177210903e-11; // m/Bohr
    static constexpr double MVV2E_ELEC = 1.06657236;        // (g/mol)*(Bohr/fs)² → Hartree
    const double VAC_ELEC = MVV2E_ELEC * HARTREE_J * NA;    // → J/mol (2.800285e6)
    vac_to_jmol  = VAC_ELEC;
    dt_to_ps     = 1e-3;                // fs → ps
    ke_to_pe_fac = MVV2E_ELEC;          // (g/mol)*(Bohr/fs)² → Hartree/atom
    pe_to_kjmol  = NA * HARTREE_J * 1e-3; // Hartree/atom → kJ/mol
    ke_to_kjmol  = VAC_ELEC * 1e-3;     // (g/mol)*(Bohr/fs)² → kJ/mol
    mass_to_gmol = 1.0;                 // amu = g/mol already
    vol_to_angst3 = BOHR_M > 0 ?
      (BOHR_M / 1e-10) * (BOHR_M / 1e-10) * (BOHR_M / 1e-10) : 0.14818;
    //   1 Bohr = 0.529177 Å → 1 Bohr³ = 0.529177³ Å³ = 0.14818 Å³

  } else {
    error->all(FLERR, "fix xpt: unsupported unit style '{}' "
               "(supported: real, metal, lj, si, cgs, micro, nano, electron)", us);
  }

  // Fix flags
  vector_flag    = 1;
  // Layout: standard slots + ZPE + μ_q + μ_c at the end.
  //   Monoatomic           : 8 standard + 2 (μ_q, μ_c) = 10
  //   molecular            : 12 standard + 2 (μ_q, μ_c) = 14
  //   molecular+show_split : 16 standard + 2 (μ_q, μ_c) = 18
  if (do_molecule)
    nvec = (do_show_split ? 16 : 12) + 2;
  else
    nvec = 8 + 2;
  size_vector    = nvec;
  global_freq    = nevery;
  extvector      = 0;    // all intensive (J/mol/K etc.)
  restart_global = 1;    // persist multi-tau accumulators

  // Internal state
  me     = comm->me;
  nprocs = comm->nprocs;

  iframe          = 0;
  window_idle     = false;
  nwindow         = 0;
  step_start      = 0;
  mt_run0_dump    = false;
  mt_restart_pending = false;
  fix_dof_window  = 0;
  natom_buf    = 0;
  vel_buf      = nullptr;
  mass_buf     = nullptr;

  // Shared velocity buffer: default to owner role.  init() may
  // demote this instance to consumer if an earlier fix_xpt with matching
  // (group, nevery, nframes, correlator, do_molecule) is found.
  buffer_owner = this;
  owns_buffer  = true;
  fft_vac      = nullptr;     // allocated in init() — persistent across windows
  fft_buf      = nullptr;
  result_ready = false;
  for (int i = 0; i < NVEC_MAX; i++) result_vec[i] = 0.0;

  press_compute    = nullptr;
  own_press_compute = false;
  press_sum        = 0.0;
  press_count      = 0;

  pe_compute   = nullptr;
  own_pe_compute = false;
  pe_peratom   = false;
  pe_available = false;
  E_sum        = 0.0;
  E_sq_sum     = 0.0;
  E_count      = 0;

  fp     = nullptr;
  fp_pwr = nullptr;
  fp_vac = nullptr;

  // ──────────────────────────────────────────────────────────────────────
  // Eager pe_compute creation — MUST happen in the constructor, NOT in
  // init().  LAMMPS::init() runs update->init() (Integrate::ev_setup()
  // snapshots elist_atom from modify->compute_list) BEFORE modify->init()
  // runs each fix's init().  A pe/atom compute created in FixXPT::init()
  // therefore lands in compute_list AFTER elist_atom is frozen — matchstep()
  // is never called, eflag_atom is never set on our sample step, and E_md
  // stays 0 every window with no error.  Creating it here puts it in
  // compute_list before ev_setup() snapshots elist_atom.
  //
  // pe_available is still gated on pair-style support in init() (force->pair
  // isn't set up yet here); the compute is created unconditionally and simply
  // never queried if pair doesn't support single.
  // ──────────────────────────────────────────────────────────────────────
  {
    const bool ctor_group_is_all = (igroup == 0);
    if (ctor_group_is_all) {
      int ipe = modify->find_compute("thermo_pe");
      if (ipe >= 0 && strcmp(modify->compute[ipe]->style, "pe") == 0) {
        pe_compute     = modify->compute[ipe];
        own_pe_compute = false;
      } else {
        std::string cid = std::string("_fixxpt_pe_") + id;
        modify->add_compute(cid + " all pe");
        ipe = modify->find_compute(cid.c_str());
        pe_compute     = modify->compute[ipe];
        own_pe_compute = true;
      }
      pe_peratom = false;
    } else {
      std::string cid = std::string("_fixxpt_pe_") + id;
      std::string cmd = cid + " " + group->names[igroup] + " pe/atom";
      modify->add_compute(cmd);
      int ipe        = modify->find_compute(cid.c_str());
      pe_compute     = modify->compute[ipe];
      own_pe_compute = true;
      pe_peratom     = true;
    }
    // pe_available finalised in init() once force->pair is set up.
  }
}

/* ======================================================================
   emit_block_header — write a per-block header line to an output file.
     nsamples == 1 (default) or iframe == nframes (end-of-window):
       "# Step <step>"  (nsamples == 1) / "# cumulative, timestep <step>".
     nsamples > 1 with iframe < nframes (intermediate snapshot):
       "# sample <iframe / sub> of <nsamples - 1>, timestep <step>",
       where sub = nframes / nsamples (sub-window size in frames).
====================================================================== */
void FixXPT::emit_block_header(FILE *f, bigint step)
{
  if (!f) return;
  if (nsamples > 1 && iframe < nframes) {
    const int sub        = nframes / nsamples;
    const int sample_idx = iframe / sub;
    fprintf(f, "# sample %d of %d, timestep %ld\n",
            sample_idx, nsamples - 1, (long)step);
  } else if (nsamples > 1) {
    fprintf(f, "# cumulative, timestep %ld\n", (long)step);
  } else {
    fprintf(f, "# Step %ld\n", (long)step);
  }
}

/* ======================================================================
   reset_window_timings — zero the per-window timing accumulators
   (FIX_XPT_DEBUG-only meaning; harmless no-cost reset otherwise).
====================================================================== */
void FixXPT::reset_window_timings()
{
  t_vac_atom = t_vac_mol = t_vac_vib = t_vac_multitau = 0.0;
  t_dos = t_thermo = 0.0;
  t_multitau_push = 0.0;
}

/* ======================================================================
   pg_rotsym_for_label — proper-rotation-subgroup order σ_rot for a
   Schoenflies point-group symbol.  Drives the rotational symmetry number
   in the molecular partition function (the `symmetry` keyword).  Returns
   -1 for unknown symbols.  Match is case-insensitive.
====================================================================== */
int FixXPT::pg_rotsym_for_label(const std::string &label)
{
  std::string lbl;
  lbl.reserve(label.size());
  for (char c : label) lbl.push_back((char)std::tolower((unsigned char)c));

  // Trivial / improper-only.
  if (lbl == "c1" || lbl == "ci" || lbl == "cs") return 1;

  // Linear: C∞v / D∞h (multiple spellings tolerated).
  if (lbl == "cinfv" || lbl == "c_infv" || lbl == "c-infv" ||
      lbl == "coov"  || lbl == "c_oov"  || lbl == "c-oov")  return 1;
  if (lbl == "dinfh" || lbl == "d_infh" || lbl == "d-infh" ||
      lbl == "dooh"  || lbl == "d_ooh"  || lbl == "d-ooh")  return 2;

  // Cubic / icosahedral high-symmetry.
  if (lbl == "t"  || lbl == "td" || lbl == "th") return 12;
  if (lbl == "o"  || lbl == "oh")                return 24;
  if (lbl == "i"  || lbl == "ih")                return 60;

  // Sn: σ_rot = n/2 (even n only).
  if (lbl == "s4")   return 2;
  if (lbl == "s6")   return 3;
  if (lbl == "s8")   return 4;

  // Cn, Cnv, Cnh: σ_rot = n.   Dn, Dnh, Dnd: σ_rot = 2n.
  // Parse a single digit n in [2, 6] from positions [1..].
  if (lbl.size() < 2) return -1;
  const char c0 = lbl[0];
  const char d  = lbl[1];
  if (d < '2' || d > '6') return -1;
  const int n = d - '0';
  const std::string tail = lbl.substr(2);   // suffix after Cn/Dn
  if (c0 == 'c') {
    if (tail == "" || tail == "v" || tail == "h") return n;
  } else if (c0 == 'd') {
    if (tail == "" || tail == "h" || tail == "d") return 2 * n;
  }
  return -1;
}

/* ======================================================================
   set_option — apply one (key, value) pair to fix_xpt's internal state.
   Used by both the legacy CLI parser and the new INI/config-block path.
   `value` may be "" for boolean-flag keywords (legacy CLI presence form).
====================================================================== */
void FixXPT::set_option(const char *key, const char *value)
{
  // Helper: parse value as boolean.  "" / "1" / "true" / "yes" / "on"  → 1
  //                                  "0" / "false" / "no" / "off"      → 0
  auto as_bool = [&](const char *v) -> int {
    if (v == nullptr || v[0] == '\0')           return 1;  // legacy CLI presence
    if (!strcasecmp(v, "1")  || !strcasecmp(v, "true")  ||
        !strcasecmp(v, "yes")|| !strcasecmp(v, "on"))     return 1;
    if (!strcasecmp(v, "0")  || !strcasecmp(v, "false") ||
        !strcasecmp(v, "no") || !strcasecmp(v, "off"))    return 0;
    error->all(FLERR, "fix xpt: '{}' expects boolean (0/1/true/false), got '{}'",
               key, v);
    return 0;
  };
  // Helper: require non-empty value
  auto need_val = [&](const char *v) {
    if (v == nullptr || v[0] == '\0')
      error->all(FLERR, "fix xpt: '{}' requires a value", key);
  };

  // ── positional keys (only meaningful in INI/config path) ─────────────────
  if (!strcasecmp(key, "nevery")) {
    need_val(value);
    nevery = utils::inumeric(FLERR, value, false, lmp);
    if (nevery <= 0) error->all(FLERR, "fix xpt: nevery must be > 0");
  }
  else if (!strcasecmp(key, "nframes")) {
    need_val(value);
    nframes = utils::inumeric(FLERR, value, false, lmp);
    if (nframes <= 0) error->all(FLERR, "fix xpt: nframes must be > 0");
  }
  // ── boolean flags ────────────────────────────────────────────────────────
  else if (!strcasecmp(key, "classical"))   do_classical    = as_bool(value);
  else if (!strcasecmp(key, "normalize"))   do_normalize    = as_bool(value);
  else if (!strcasecmp(key, "molecule"))    do_molecule     = as_bool(value);
  else if (!strcasecmp(key, "allow_mixed_molecules"))
                                            allow_mixed_molecules = as_bool(value);
  else if (!strcasecmp(key, "symmetry")) {
    need_val(value);
    symmetry = std::string(value);
    // Derive rotsym from the point-group lookup table.
    const int s = pg_rotsym_for_label(symmetry);
    if (s < 0) {
      error->all(FLERR,
        "fix xpt: unknown point group `{}` for `symmetry` keyword.  "
        "Supported: C1, Ci, Cs; Cn / Cnv / Cnh (n=2-6); Dn / Dnh / Dnd "
        "(n=2-6); T, Td, Th; O, Oh; I, Ih; S4, S6, S8; Cinfv, Dinfh.",
        symmetry);
    }
    rotsym = s;
  }
  else if (!strcasecmp(key, "linear"))      is_linear       = as_bool(value);
  else if (!strcasecmp(key, "show_split"))  do_show_split   = as_bool(value);
  // ── LJ parameter keys ────────────────────────────────────────────────────
  else if (!strcasecmp(key, "epsilon")) {
    need_val(value);
    lj_eps = utils::numeric(FLERR, value, false, lmp);
    lj_params_user = true;
  }
  else if (!strcasecmp(key, "sigma")) {
    need_val(value);
    lj_sig = utils::numeric(FLERR, value, false, lmp);
    lj_params_user = true;
  }
  else if (!strcasecmp(key, "mass")) {
    need_val(value);
    lj_mass = utils::numeric(FLERR, value, false, lmp);
    lj_params_user = true;
  }
  // ── volume override ──────────────────────────────────────────────────────
  else if (!strcasecmp(key, "volume")) {
    need_val(value);
    if (utils::strmatch(value, "^v_")) {
      volume_style  = VOL_VARIABLE;
      volume_varstr = utils::strdup(value + 2);
    } else {
      volume_style = VOL_CONSTANT;
      user_volume  = utils::numeric(FLERR, value, false, lmp);
      if (user_volume <= 0.0) error->all(FLERR, "fix xpt: volume must be > 0");
    }
  }
  // ── mode & molecular options ─────────────────────────────────────────────
  else if (!strcasecmp(key, "mode")) {
    need_val(value);
    // mode taxonomy: 1PT | 2PT | 3PT (case-insensitive).  The 2PT sub-method is
    // chosen by `refinement`; 3PT implies rigorous + cage.  Resolved to internal
    // mol_mode in init().
    if      (!strcasecmp(value, "1pt")) req_mode = REQ_1PT;
    else if (!strcasecmp(value, "2pt")) req_mode = REQ_2PT;
    else if (!strcasecmp(value, "3pt")) req_mode = REQ_3PT;
    else error->all(FLERR, "fix xpt: mode must be 1PT, 2PT, or 3PT");
  }
  else if (!strcasecmp(key, "refinement")) {
    need_val(value);
    // mode-aware: 2PT values (rigorous..r2pt) | 3PT value (none).  The
    // value is validated against req_mode in init() (keyword order is free).
    refinement_set = 1;
    if      (!strcasecmp(value, "rigorous"))   refinement = REF_RIGOROUS;
    else if (!strcasecmp(value, "lin2003"))    refinement = REF_LIN2003;
    else if (!strcasecmp(value, "desjarlais")) refinement = REF_DESJARLAIS;
    else if (!strcasecmp(value, "r2pt"))       refinement = REF_R2PT;
    else if (!strcasecmp(value, "none"))       refinement = REF_NONE;
    else error->all(FLERR, "fix xpt: refinement must be rigorous, lin2003, "
                           "desjarlais, r2pt (mode 2PT) or none (mode 3PT)");
  }
  else if (!strcasecmp(key, "r2pt_delta")) {
    need_val(value);
    r2pt_delta = utils::numeric(FLERR, value, false, lmp);
    if (r2pt_delta <= 0.0) error->all(FLERR, "fix xpt: r2pt_delta must be > 0");
  }
  else if (!strcasecmp(key, "rotsym")) {
    error->all(FLERR,
      "fix xpt: `rotsym` is no longer a user-set keyword — it is auto-"
      "derived from `symmetry` via the point-group lookup table.  Specify "
      "`symmetry <pg>` (e.g. C1, C2v, D6h, Td, Oh) instead.  See the "
      "fix_xpt header for the full list.");
  }
  else if (!strcasecmp(key, "use_sim_z")) {
    need_val(value);
    use_sim_z_mode = as_bool(value);
  }
  // Per-fix RAM budget in GB.  Auto-reduces nframes
  // to largest power-of-2 that fits within this budget (see init()).
  else if (!strcasecmp(key, "max_memory_gb")) {
    need_val(value);
    max_memory_gb = utils::numeric(FLERR, value, false, lmp);
    if (max_memory_gb < 0.0)
      error->all(FLERR, "fix xpt: max_memory_gb must be >= 0 "
                        "(0 disables auto-resize)");
  }
  // Optional FP32 vel_buf / vib_vel_buf storage.
  // Halves the largest per-fix buffers; loses ~5-7 sig figs on those
  // velocity samples → state vars (D, f, S(0), μ, T_vac) typically
  // unaffected (<0.5%); Cv may be precision-sensitive — warn loudly.
  else if (!strcasecmp(key, "buffer_precision")) {
    need_val(value);
    if      (!strcasecmp(value, "fp64") || !strcasecmp(value, "double"))
      buffer_precision = BUFFER_FP64;
    else if (!strcasecmp(value, "fp32") || !strcasecmp(value, "float"))
      buffer_precision = BUFFER_FP32;
    else
      error->all(FLERR, "fix xpt: buffer_precision must be 'fp64' or 'fp32'");
  }
  // Frame-buffer layout: distributed = O(N/P) per rank; replicated = every
  // rank holds the whole history (validation reference).
  else if (!strcasecmp(key, "buffer_layout")) {
    need_val(value);
    if      (!strcasecmp(value, "distributed")) buffer_layout = LAYOUT_DISTRIBUTED;
    else if (!strcasecmp(value, "replicated"))  buffer_layout = LAYOUT_REPLICATED;
    else
      error->all(FLERR, "fix xpt: buffer_layout must be 'distributed' or 'replicated'");
    buffer_layout_explicit = 1;
  }
  else if (!strcasecmp(key, "correlator")) {
    need_val(value);
    if (!strcasecmp(value, "fft"))            correlator = CORR_FFT;
    else if (!strcasecmp(value, "multitau"))  correlator = CORR_MULTITAU;
    else error->all(FLERR, "fix xpt: correlator must be 'fft' or 'multitau'");
  }
  else if (!strcasecmp(key, "mt_m")) {
    need_val(value);
    mt_M = utils::inumeric(FLERR, value, false, lmp);
    if (mt_M < 4)
      error->all(FLERR, "fix xpt: mt_M must be >= 4");
  }
  else if (!strcasecmp(key, "mt_p")) {
    need_val(value);
    mt_P = utils::inumeric(FLERR, value, false, lmp);
    if (mt_P < 0)
      error->all(FLERR, "fix xpt: mt_P must be >= 0");
  }
  else if (!strcasecmp(key, "mt_s")) {
    need_val(value);
    mt_S = utils::inumeric(FLERR, value, false, lmp);
    if (mt_S != 2 && mt_S != 4 && mt_S != 8)
      error->all(FLERR, "fix xpt: mt_S must be 2, 4, or 8");
  }
  else if (!strcasecmp(key, "mt_l")) {
    need_val(value);
    mt_L = utils::inumeric(FLERR, value, false, lmp);
    if (mt_L < 0)
      error->all(FLERR, "fix xpt: mt_L must be >= 0 (0 = auto)");
  }
  else if (!strcasecmp(key, "nsamples")) {
    need_val(value);
    nsamples = utils::inumeric(FLERR, value, false, lmp);
    if (nsamples < 1)
      error->all(FLERR, "fix xpt: nsamples must be >= 1 (1 = no sub-window snapshots)");
  }
  else if (!strcasecmp(key, "maxfreq")) {
    need_val(value);
    maxfreq = utils::numeric(FLERR, value, false, lmp);
    if (maxfreq < 0.0)
      error->all(FLERR, "fix xpt: maxfreq must be >= 0.0 (0 = no .pwr/.vac truncation)");
  }
  else if (!strcasecmp(key, "dnu")) {
    need_val(value);
    dnu_out = utils::numeric(FLERR, value, false, lmp);
    if (dnu_out < 0.0)
      error->all(FLERR, "fix xpt: dnu must be >= 0.0 (0 = write at native spacing)");
  }
  else {
    error->all(FLERR, "fix xpt: unknown option '{}'", key);
  }
}

/* ======================================================================
   parse_kv_lines — split text into key=value lines, call set_option() on each
   Format:   key = value     (= and whitespace flexible)
             key   value     (also accepted; first whitespace-separated token = key)
   Comments: '#' or ';' to end of line, inline or full-line (quote-aware, so a
             quoted value may legitimately contain '#'/';').
   Blank lines are skipped.
====================================================================== */
void FixXPT::parse_kv_lines(const std::string &text)
{
  std::string line;
  size_t pos = 0;
  while (pos < text.size()) {
    size_t eol = text.find('\n', pos);
    if (eol == std::string::npos) eol = text.size();
    line = text.substr(pos, eol - pos);
    pos = eol + 1;

    // Strip trailing CR (for Windows line-endings)
    while (!line.empty() && (line.back() == '\r' || line.back() == ' ' ||
                              line.back() == '\t')) line.pop_back();
    size_t first = line.find_first_not_of(" \t");
    if (first == std::string::npos) continue;          // blank line
    line.erase(0, first);

    // Strip comments: the first unquoted '#' or ';' begins a comment and is
    // truncated (inline or full-line — a full-line comment collapses to empty
    // here).  Quote-aware so a quoted value may legitimately contain '#'/';'.
    {
      char q = '\0';
      for (size_t i = 0; i < line.size(); i++) {
        char c = line[i];
        if (q)                            { if (c == q) q = '\0'; }
        else if (c == '"' || c == '\'')   q = c;
        else if (c == '#' || c == ';')    { line.erase(i); break; }
      }
      while (!line.empty() && (line.back() == ' ' || line.back() == '\t'))
        line.pop_back();                  // re-trim ws exposed by the cut
    }
    if (line.empty()) continue;           // was a full-line comment

    // Find delimiter (= or whitespace)
    size_t eq = line.find('=');
    size_t ws = line.find_first_of(" \t");
    size_t split;
    bool   has_eq = (eq != std::string::npos);
    if (has_eq && (ws == std::string::npos || eq < ws)) split = eq;
    else if (ws != std::string::npos)                   split = ws;
    else { set_option(line.c_str(), ""); continue; }   // bare keyword (boolean)

    std::string key = line.substr(0, split);
    while (!key.empty() && (key.back() == ' ' || key.back() == '\t')) key.pop_back();

    std::string val = line.substr(split + 1);
    // Trim leading whitespace + leading '=' from value
    size_t vstart = val.find_first_not_of(" \t=");
    if (vstart == std::string::npos) val.clear();
    else val.erase(0, vstart);
    while (!val.empty() && (val.back() == ' ' || val.back() == '\t')) val.pop_back();
    // Strip surrounding quotes
    if (val.size() >= 2 && ((val.front() == '"'  && val.back() == '"') ||
                              (val.front() == '\'' && val.back() == '\''))) {
      val = val.substr(1, val.size() - 2);
    }

    set_option(key.c_str(), val.c_str());
  }
}

/* ======================================================================
   parse_ini_file — read INI file (rank 0) + broadcast content to all ranks.
   Then dispatch to parse_kv_lines.
====================================================================== */
void FixXPT::parse_ini_file(const char *path)
{
  std::string text;
  int len = 0;
  if (me == 0) {
    FILE *fp = fopen(path, "r");
    if (!fp) error->one(FLERR, "fix xpt: cannot open INI file '{}'", path);
    fseek(fp, 0, SEEK_END);
    len = (int)ftell(fp);
    fseek(fp, 0, SEEK_SET);
    text.resize(len);
    if (len > 0 && fread(&text[0], 1, len, fp) != (size_t)len)
      error->one(FLERR, "fix xpt: short read on INI file '{}'", path);
    fclose(fp);
  }
  MPI_Bcast(&len, 1, MPI_INT, 0, world);
  text.resize(len);
  MPI_Bcast(&text[0], len, MPI_CHAR, 0, world);
  parse_kv_lines(text);
}

FixXPT::~FixXPT()
{
  delete[] prefix;
  delete[] volume_varstr;
  // Buffer-sharing: only the owner frees the shared buffers.
  // Consumers point at owner's storage; freeing here would double-free.
  // CAVEAT: if the user `unfix`es the owner while consumers still exist,
  // consumers will end up with dangling pointers.  The expected workflow
  // is to unfix consumers first (LAMMPS destroys fixes in reverse order
  // of registration by default, which already gives the right ordering
  // when fixes are added in their natural order).
  //
  // Warn loudly (but don't crash) if we're an owner being destroyed with
  // live consumers — typically a user-ordering mistake at `unfix` time.
  if (owns_buffer && !buffer_consumers.empty()) {
    // Actively detach consumers BEFORE the owner frees its shared buffers.
    // LAMMPS does NOT destroy fixes in strict reverse-add order in all
    // configurations, so an owner can be destructed before a consumer that
    // still aliases its buffers; without active detach the consumer's later
    // destructor would dereference a freed owner.
    //
    // Use `id` (not group->names[igroup]) in the message: `group` may be null
    // at LAMMPS-shutdown destruction time (Pointers' group_* nulled before
    // fixes destruct in some configurations).
    if (me == 0 && lmp) {
      utils::logmesg(lmp, "FixXPT::{}: NOTE — destroying buffer owner with "
                          "{} consumer(s) still attached.  Auto-detaching "
                          "consumers from the shared buffer so their "
                          "destructors don't dereference freed owner state.\n",
                     id ? id : "(null)", (int)buffer_consumers.size());
    }
    for (FixXPT *c : buffer_consumers) {
      if (!c) continue;
      c->buffer_owner = nullptr;
      // Null the consumer's host pointer aliases so its destructor won't touch freed storage.
      c->vel_buf = nullptr;
      c->vel_buf_f = nullptr;
      c->mass_buf = nullptr;
      c->com_vel_buf = nullptr;
      c->omega_buf = nullptr;
      c->angmom_buf = nullptr;
      c->vib_vel_buf = nullptr;
      c->vib_vel_buf_f = nullptr;
      c->home_xu = nullptr;
      if (c->distributed()) c->mol_inertia = nullptr;   // aliased from the owner
    }
    buffer_consumers.clear();
  }
  // Consumer: deregister from owner's consumer list so a subsequent owner
  // destruction doesn't see stale entries.  No-op if buffer_owner is gone
  // already (defensive — should not happen with correct unfix ordering).
  if (!owns_buffer && buffer_owner && buffer_owner != this) {
    auto &v = buffer_owner->buffer_consumers;
    v.erase(std::remove(v.begin(), v.end(), this), v.end());
  }
  if (owns_buffer) {
    memory->destroy(vel_buf);
    memory->destroy(vel_buf_f);
    memory->destroy(mass_buf);
    memory->destroy(com_vel_buf);
    memory->destroy(omega_buf);
    memory->destroy(angmom_buf);
    memory->destroy(vib_vel_buf);
    memory->destroy(vib_vel_buf_f);
    memory->destroy(home_xu);
  } else {
    vel_buf = nullptr;
    vel_buf_f = nullptr;
    mass_buf = nullptr;
    com_vel_buf = nullptr;
    omega_buf = nullptr;
    angmom_buf = nullptr;
    vib_vel_buf = nullptr;
    vib_vel_buf_f = nullptr;
    home_xu = nullptr;
    // A distributed consumer aliases the owner's mol_inertia (it is sized to
    // the owner's home molecules); a replicated consumer owns its own.
    if (distributed()) mol_inertia = nullptr;
  }
  memory->destroy(mol_inertia);  // per-fix unless aliased (nulled above)
  // Persistent FFT scratch (created in init(), valid through to fix teardown).
  // Destroy BEFORE LAMMPS's KSpace cleanup runs (Force::~Force) so FFTW3's
  // plan cache is in a consistent state when PPPM destroys its own FFT3d.
  if (fft_vac) { delete fft_vac; fft_vac = nullptr; }
  if (fft_buf) memory->destroy(fft_buf);
  if (fp       && me == 0) fclose(fp);
  if (fp_pwr   && me == 0) fclose(fp_pwr);
  if (fp_vac   && me == 0) fclose(fp_vac);
  if (own_press_compute && press_compute)
    modify->delete_compute(press_compute->id);
  if (own_pe_compute && pe_compute)
    modify->delete_compute(pe_compute->id);
}

/* ======================================================================
   setmask
====================================================================== */

int FixXPT::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ----------------------------------------------------------------------
   FFT3d's constructor arity is LAMMPS-version dependent and both forms are
   in use: releases before 2025-11-26 take 21 arguments, later ones take 22
   (upstream 8ca245b3 added a trailing `nonblocking` option in the KSPACE
   comms rework).  LAMMPS exposes only LAMMPS_VERSION, a string such as
   "30 Mar 2026", which the preprocessor cannot compare, so the arity is
   detected instead of hard-coded.

   Overload resolution does the work: the 22-argument overload is preferred
   (int beats long for the literal 0 tag) and drops out of the candidate set
   by SFINAE when its trailing argument does not fit, leaving the 21-argument
   one.  Neither form is hard-coded, so this file builds on either LAMMPS.
------------------------------------------------------------------------- */
namespace {

template <typename F = FFT3d>
auto make_fft3d_vac_impl(LAMMPS *lmp, int n, int *mysize, int)
    -> decltype(new F(lmp, MPI_COMM_SELF, 1, 1, n, 0, 0, 0, 0, 0, n - 1,
                      0, 0, 0, 0, 0, n - 1, 0, 0, mysize, 0, 0))
{
  return new F(lmp, MPI_COMM_SELF, 1, 1, n, 0, 0, 0, 0, 0, n - 1,
               0, 0, 0, 0, 0, n - 1, 0, 0, mysize, 0, 0);
}

template <typename F = FFT3d>
auto make_fft3d_vac_impl(LAMMPS *lmp, int n, int *mysize, long)
    -> decltype(new F(lmp, MPI_COMM_SELF, 1, 1, n, 0, 0, 0, 0, 0, n - 1,
                      0, 0, 0, 0, 0, n - 1, 0, 0, mysize, 0))
{
  return new F(lmp, MPI_COMM_SELF, 1, 1, n, 0, 0, 0, 0, 0, n - 1,
               0, 0, 0, 0, 0, n - 1, 0, 0, mysize, 0);
}

FFT3d *make_fft3d_vac(LAMMPS *lmp, int n, int *mysize, int tag)
{
  return make_fft3d_vac_impl(lmp, n, mysize, tag);
}

}   // anonymous namespace

/* ======================================================================
   init
====================================================================== */

void FixXPT::init()
{

  // ── Resolve (mode, refinement) → internal dispatch ─────────────────────
  // Idempotent (init may run once per `run` command).  REQ_UNSET keeps the
  // constructor default (mol_mode = 2 = HS, refinement = rigorous = 2PT).
  // hs_entropy_mode is DERIVED from refinement (no standalone keyword):
  //   rigorous→0, lin2003→1, desjarlais→1 (+lnZ convention), r2pt→0 (lnz-free).
  // 3PT reuses the Volterra kernel for its parameter-free cage.
  //
  // `refinement` is mode-aware: 2PT takes rigorous|lin2003|desjarlais|r2pt,
  // 3PT takes none (default none).  Validate the value against the mode and
  // default the 3PT refinement when the user left it unset.
  const bool ref_is_3pt = (refinement == REF_NONE);
  if (req_mode == REQ_3PT) {
    if (!refinement_set) refinement = REF_NONE;
    else if (!ref_is_3pt)
      error->all(FLERR, "fix xpt: for mode 3PT, refinement must be none");
  } else if (refinement_set && ref_is_3pt) {
    error->all(FLERR, "fix xpt: for mode 2PT, refinement must be rigorous, "
                      "lin2003, desjarlais, or r2pt");
  }

  switch (req_mode) {
    case REQ_1PT: mol_mode = 1; break;
    case REQ_2PT:
      switch (refinement) {
        case REF_RIGOROUS:   mol_mode = 2; hs_entropy_mode = 0; break;
        case REF_LIN2003:    mol_mode = 2; hs_entropy_mode = 1; break;
        case REF_DESJARLAIS: mol_mode = 3; hs_entropy_mode = 1; break;
        case REF_R2PT:       mol_mode = 2; hs_entropy_mode = 0; break;  // r2pt delta applied post-assembly
      }
      break;
    case REQ_3PT:                         // rigorous-HS base + parameter-free cage
      mol_mode = 2; hs_entropy_mode = 0;
      // The 3PT cage (cage_memory_entropy) is a physical-units construction —
      // its gas/solid quantum weights use cm^-1 frequencies, Kelvin, and the
      // de Broglie thermal wavelength (PLANCK, H_SI, KB).  It was never ported
      // to the LJ reduced-unit (hbar*=1 / ST*) entropy convention, so in LJ
      // units it is skipped (3PT degrades to its 2PT-rigorous base) and warned
      // about once, rather than producing a meaningless S_cage.
      if (units_lj) {
        if (me == 0)
          error->warning(FLERR, "fix xpt: mode 3PT cage is unsupported in LJ "
              "reduced units (physical-units construction); skipping the cage "
              "— results equal mode 2PT refinement rigorous");
      } else {
        cage_entropy = 1;
        if (do_molecule) cage_entropy_rot = 1;   // cage applied post-assembly
      }
      break;
    default: break;                       // REQ_UNSET → ctor default (2PT/rigorous)
  }

  // ── FP32 buffer-precision compatibility checks ──────
  if (buffer_precision == BUFFER_FP32) {
    if (correlator == CORR_MULTITAU)
      error->all(FLERR, "fix xpt: `buffer_precision fp32` is not supported "
                        "with `correlator multitau` in V1 — multi-tau push "
                        "expects a double* ring source.  Use `correlator "
                        "fft` or switch to fp64.");
    if (comm->me == 0) {
      // Loud warning per spec.  D/f/S(0)/μ/T_vac typically robust at FP32;
      // cv (σ²(E)/RT² fluctuation) is precision-sensitive — validate
      // before trusting in production.
      utils::logmesg(lmp,
        "FixXPT::{}-{}: WARNING — `buffer_precision fp32` enabled.  "
        "Halves vel_buf / vib_vel_buf memory (~8 GB saved per fix at "
        "0p1M nframes=16384) at the cost of FP32 velocity-sample "
        "precision.  D, f, S(0), μ, T_vac normally unaffected (<0.5%); "
        "Cv may be precision-sensitive (σ²(E)/RT² fluctuation term).  "
        "Validate before trusting Cv in production.\n",
        id, group->names[igroup]);
    }
  }

  // ── auto-resize nframes to fit max_memory_gb budget ─
  // Must run BEFORE the FFT plan + per-frame buffer allocations later in
  // init() so the chosen N_fft = 2 * nframes is locked in for the entire
  // window-stream.  Estimate at init time using group->count(igroup) for
  // the group atom count (exact for static groups; conservative upper
  // bound for dynamic groups that grow during the run).
  // Atoms are routed and indexed by ID; without IDs they cannot be followed.
  if (!atom->tag_enable)
    error->all(FLERR, "fix xpt requires atom IDs (atom_modify id yes)");
  // Masses are read per type (atom->mass); an atom style with per-atom
  // masses (rmass) leaves that table null.
  if (atom->rmass_flag || !atom->mass)
    error->all(FLERR, "fix xpt requires per-type masses; atom styles with per-atom "
                      "masses (rmass) are not supported");

  // On one rank the two layouts hold the same history, so the choice is only
  // a speed one and the measured winner differs by mode: an atomic group is
  // faster replicated (a slot is tag - 1, with no home-rank exchange and no
  // binary search, and xpt/kk keeps the frames on the device), a molecular
  // group faster distributed (the home rank assembles each molecule from data
  // it already holds).  An explicit buffer_layout always wins.
  if (nprocs == 1 && !buffer_layout_explicit && !do_molecule && distributed()) {
    buffer_layout = LAYOUT_REPLICATED;
    if (comm->me == 0)
      utils::logmesg(lmp, "FixXPT::{}-{}: one rank, atomic group; "
                     "using buffer_layout replicated\n", id, group->names[igroup]);
  }

  if (max_memory_gb > 0.0 && nframes > 0) {
    // The budget is per rank: in the distributed layout a rank holds about
    // 1/nprocs of the group's history.
    const bigint n_in_group = group->count(igroup);
    const bigint n_per_rank = distributed() ? (n_in_group + nprocs - 1) / nprocs : n_in_group;
    const int    n_atom_est = std::max<int>(1, (int)n_per_rank);
    // Conservative per-atom-per-frame footprint (in bytes) — covers
    // vel_buf + vib_vel_buf + 3 mol channels indexed by nmol ≤ natom:
    //   24 (vel) + 24 (vib if molecular) + 3·24 (3 mol bufs assuming
    //                                            ≤1 mol per atom)
    // For non-molecular runs (do_molecule=0) the second + third terms
    // vanish; using the upper bound keeps the calculation conservative.
    const double per_atom_per_frame =
        do_molecule ? (24.0 + 24.0 + 3.0 * 24.0) : 24.0;
    const double per_frame_bytes    = n_atom_est * per_atom_per_frame;
    const double budget_bytes       = max_memory_gb * 1.0e9;
    int max_pow2 = 1;
    while (((bigint)(max_pow2 << 1)) * per_frame_bytes <= budget_bytes
           && (max_pow2 << 1) > 0) max_pow2 <<= 1;
    if (max_pow2 < nframes) {
      const int old_nframes = nframes;
      nframes = max_pow2;
      if (comm->me == 0)
        FIXXPT_LOG("FixXPT::{}-{}: WARNING — auto-resizing nframes "
                   "from {} to {} to fit max_memory_gb={:.3g} budget "
                   "(estimated {:.2f} GB for {} group atoms).\n",
                   id, group->names[igroup], old_nframes, nframes,
                   max_memory_gb,
                   per_frame_bytes * nframes / 1.0e9, n_atom_est);
    }
  }

  // ── Shared velocity buffer detection ───────────────────────
  // Scan modify->fix[] for an earlier fix_xpt with matching equivalence
  // key.  If found, become a consumer pointing at owner's buffers and
  // register on owner's consumer list (so owner can re-alias us at every
  // grow_buf event).  init() is called once per `run` command; clear and
  // re-detect each time so dynamic fix add/delete is handled cleanly.
  //
  // V1 restriction: sharing is only enabled for correlator == FFT.  The
  // multi-tau ring buffers (mt_*) are per-fix and currently un-shared, so
  // a multi-tau consumer would have no VAC at end-of-window.  Multi-tau
  // fixes remain owners-of-themselves (no sharing).  Lifting this is
  // tracked (GPU multi-tau push) which would also need
  // to make the ring buffer addressable across fixes.
  buffer_owner = this;
  owns_buffer  = true;
  buffer_consumers.clear();
  if (correlator == CORR_FFT) {
    for (int i = 0; i < modify->nfix; ++i) {
      Fix *f = modify->fix[i];
      if (f == this) break;  // only consider earlier fixes
      // Also match "xpt/kk", "xpt/kk/device", "xpt/kk/host"
      // — the strncmp("xpt", ..., 3) check catches every FixXPT-derived
      // style.  dynamic_cast as a belt-and-braces guard against any
      // future 3rd-party fix style that starts with "xpt" but isn't
      // actually a FixXPT.
      if (strncmp(f->style, "xpt", 3) != 0) continue;
      FixXPT *other = dynamic_cast<FixXPT *>(f);
      if (!other) continue;
      if (other->correlator == CORR_FFT     &&
          other->igroup      == this->igroup     &&
          other->nframes     == this->nframes    &&
          other->nevery      == this->nevery     &&
          other->do_molecule == this->do_molecule &&
          other->buffer_precision == this->buffer_precision &&
          other->buffer_layout    == this->buffer_layout) {
        buffer_owner = other;
        owns_buffer  = false;
        other->buffer_consumers.push_back(this);
        FIXXPT_LOG("FixXPT::{}-{}: sharing velocity buffer with FixXPT::{}-{} "
                   "(nevery={}, nframes={}, FFT, do_molecule={})\n",
                   id, group->names[igroup],
                   other->id, group->names[other->igroup],
                   nevery, nframes, do_molecule);
        break;
      }
    }
  }

  // ── Multi-tau restart-persistence warning ────────────────────────────
  // The multi-tau correlator (mt_v_ring + per-level accumulators) is NOT
  // currently serialised into the LAMMPS restart file.  After a `read_restart`, the
  // correlator silently re-initialises from scratch — fine for runs that
  // complete in a single LAMMPS invocation, but corrupts statistics for
  // production replicas that span multiple SLURM auto-resume cycles.
  //
  // Emit a clear, production-visible (NOT FIXXPT_LOG-gated) warning once
  // per fix construction so users see it before the first analysis window.
  if (correlator == CORR_MULTITAU && !mt_restart_warn_emitted) {
    if (me == 0) {
      // write_restart/restart now
      // persists multi-tau accumulators (c_sum, count_seen, head, down_n)
      // For both the scalar and 3 molecular streams.  v_ring + down_acc
      // are reset on restart — loses ≤ MP frames of correlation per
      // boundary; accumulated c_sum is bit-exact.
      utils::logmesg(lmp,
        "FixXPT::{}-{}: NOTE — `correlator multitau` state IS persisted "
        "across LAMMPS restart-file cycles.  v_ring + "
        "down_acc reset to zero on restart, losing up to mt_MP frames "
        "of correlation per boundary; accumulated c_sum is preserved "
        "exactly.\n",
        id, group->names[igroup]);
    }
    mt_restart_warn_emitted = true;
  }

  // ── Volume variable resolution ────────────────────────────────────────
  if (volume_style == VOL_VARIABLE) {
    volume_varindex = input->variable->find(volume_varstr);
    if (volume_varindex < 0)
      error->all(FLERR, "fix xpt: volume variable {} does not exist", volume_varstr);
    if (!input->variable->equalstyle(volume_varindex))
      error->all(FLERR, "fix xpt: volume variable {} must be equal-style", volume_varstr);
  }

  // ── Pressure compute ─────────────────────────────────────────────────
  // Prefer an existing pressure compute for this group; fall back to
  // thermo_press (global); if neither exists, create one for all atoms.
  // For subgroups the kinetic contribution is group-local; the virial
  // contribution is always global (LAMMPS limitation).
  if (own_press_compute && press_compute)
    modify->delete_compute(press_compute->id);
  press_compute    = nullptr;
  own_press_compute = false;

  // Look for an existing compute pressure on our group
  for (int i = 0; i < modify->ncompute; i++) {
    if (strcmp(modify->compute[i]->style, "pressure") == 0 &&
        modify->compute[i]->igroup == igroup) {
      press_compute = modify->compute[i];
      break;
    }
  }
  // Fall back: use thermo_press (group all)
  if (!press_compute) {
    int ipress = modify->find_compute("thermo_press");
    if (ipress >= 0) press_compute = modify->compute[ipress];
  }
  // Last resort: create our own global pressure compute
  if (!press_compute) {
    std::string cid = std::string("_fixxpt_press_") + id;
    std::string cmd = cid + " all pressure thermo_temp";
    modify->add_compute(cmd);
    int ipress = modify->find_compute(cid.c_str());
    press_compute    = modify->compute[ipress];
    own_press_compute = true;
  }

  // pe_available finalisation only — the pe (or pe/atom) compute itself
  // was created eagerly in the constructor (see the long comment there
  // for why init-time creation is wrong: Integrate::ev_setup() snapshots
  // elist_atom in update->init() BEFORE modify->init() runs each fix's
  // init(), so a compute added here lands in compute_list too late;
  // matchstep() is never called, eflag_atom never set, E_md silently 0).
  //
  // Here we just gate pe_available on pair-style support (which IS
  // available now that force->pair is initialised):
  //   * group "all":   always available (the global "pe" compute works
  //                    with any pair style).
  //   * subgroup:      requires pair->single_enable for the per-atom
  //                    PE accumulation; warn + disable if not supported.
  // Idempotent across repeated init() calls — only the gating flag flips,
  // the compute itself stays in place.
  bool group_is_all = (igroup == 0);
  if (group_is_all) {
    pe_available = (pe_compute != nullptr);
  } else if (pe_compute && pe_peratom) {
    const bool can_peratom = (force->pair != nullptr && force->pair->single_enable);
    if (!can_peratom) {
      FIXXPT_LOG( "FixXPT::{}-{}: WARNING — pair style does not support "
                     "per-atom energies for subgroup; E_md and Cv fluctuation will "
                     "be zero.\n", id, group->names[igroup]);
      pe_available = false;
    } else {
      pe_available = true;
    }
  } else {
    pe_available = false;
  }

  // ── LJ reference parameters (ε, σ, m) ───────────────────────────────────
  // Auto-detect from pair_coeff 1 1 and atom mass if not user-specified.
  // These are used only for documentation (printed in the output header).
  if (units_lj && !lj_params_user) {
    if (force->pair) {
      int dim = 2;
      auto *eps_a = (double **)force->pair->extract("epsilon", dim);
      auto *sig_a = (double **)force->pair->extract("sigma", dim);
      if (eps_a && atom->ntypes >= 1) lj_eps = eps_a[1][1];
      if (sig_a && atom->ntypes >= 1) lj_sig = sig_a[1][1];
    }
    if (atom->mass && atom->ntypes >= 1) lj_mass = atom->mass[1];
  }

  // ── Compute reduced ħ* from physical LJ parameters ─────────────────────────
  // Requires all three: lj_eps [kcal/mol], lj_sig [Angstrom], lj_mass [g/mol].
  // Only applied when the user explicitly provided these values (lj_params_user),
  // because auto-detected values (eps=sig=mass=1 in LJ reduced units) carry no
  // physical scale information.
  //   ħ* = ħ / (σ[m] · sqrt(m[kg/atom] · ε[J/atom]))
  hbar_star = 1.0;
  if (units_lj && lj_params_user && lj_eps > 0.0 && lj_sig > 0.0 && lj_mass > 0.0) {
    double sig_m = lj_sig * 1e-10;               // Å → m
    double m_kg  = lj_mass * 1e-3 / NA;          // g/mol → kg/atom
    double eps_J = lj_eps * 4184.0 / NA;         // kcal/mol → J/atom
    hbar_star = (H_SI / (2.0 * PI)) / (sig_m * sqrt(m_kg * eps_J));
  }

  // ── Molecular topology ────────────────────────────────────────────────────
  if (do_molecule) {
    if (atom->molecular == Atom::ATOMIC)
      error->all(FLERR, "fix xpt molecule: requires molecular atom_style");
    // Re-entrant init() guard: `rerun` calls modify->init() before EVERY
    // snapshot, so a mid-window re-init must NOT rebuild the molecular
    // topology or destroy the frame buffers — window frame data lives in
    // them, and vib_vel_buf is only re-grown at accumulate_frame's
    // iframe==0 prologue, so a mid-window destroy left it null and
    // segfaulted accumulate_mol_frame on the very next snapshot.  Topology and
    // buffers are rebuilt at every window start anyway (the iframe==0
    // prologue), so skipping here is safe for dynamic groups too.
    if (iframe == 0 && distributed()) {
      // Per-molecule frame buffers are sized to this rank's home molecules,
      // which the window-start prologue decides; nothing to allocate here.
      build_mol_topology();
      if (!owns_buffer) sync_buffer_pointers_from_owner();
    } else if (iframe == 0) {
    const int init_old_nmol = nmol_group;
    build_mol_topology();
    // Reallocate ONLY on first allocation or a genuine topology change:
    // `rerun` re-enters init() at every window boundary too (iframe==0
    // there), and an unconditional destroy left vib_vel_buf null —
    // grow_buf() early-outs when natom_buf is unchanged, so nothing
    // recreated it → segfault at window 2, frame 1.
    const bool init_mol_realloc =
        (nmol_group != init_old_nmol) ||
        (owns_buffer ? (com_vel_buf == nullptr) : (mol_inertia == nullptr));

    // (Re)allocate per-molecule frame buffers.  Buffer-sharing:
    // For consumers, skip the destroy/create and alias to owner's storage.
    // Owner's init() ran first (we're scanning fixes in modify-list order),
    // so owner->com_vel_buf etc. are already populated at this point.
    if (owns_buffer && init_mol_realloc) {
      memory->destroy(com_vel_buf);
      memory->destroy(omega_buf);
      memory->destroy(angmom_buf);
      memory->destroy(vib_vel_buf);
      memory->destroy(mol_inertia);
      if (nmol_group > 0) {
        // Multi-tau path: only the CURRENT frame is needed (the multi-tau
        // ring buffer holds the history); allocate single-frame to retire
        // the O(nframes · nmol · 3) per-channel buffer.
        const int mol_frames =
            (correlator == CORR_MULTITAU) ? 1 : nframes;
        memory->create(com_vel_buf, mol_frames, nmol_group, 3, "fix_xpt:com_vel_buf");
        memory->create(omega_buf,   mol_frames, nmol_group, 3, "fix_xpt:omega_buf");
        memory->create(angmom_buf,  mol_frames, nmol_group, 3, "fix_xpt:angmom_buf");
        // vib_vel_buf created/resized in grow_buf() once natom_buf is known
        memory->create(mol_inertia, nmol_group, 3,          "fix_xpt:mol_inertia");
      }
    } else if (!owns_buffer) {
      // Consumer: leave per-fix mol_inertia null until first end_of_step
      // rebuilds it (per-fix; small ~10 KB even at 0p1M); alias the shared
      // per-frame buffers from owner.  vib_vel_buf is null until owner's
      // first grow_buf call populates it (we re-sync there).
      if (init_mol_realloc) {
        memory->destroy(mol_inertia);
        if (nmol_group > 0) {
          memory->create(mol_inertia, nmol_group, 3, "fix_xpt:mol_inertia");
        }
      }
      // Always re-alias: the owner may have just reallocated its buffers.
      sync_buffer_pointers_from_owner();
    }
    }  // iframe == 0 re-entrant init() guard (see comment above)
  }

  // ── Persistent FFT scratch (one-time alloc, reused all windows) ─────────
  // The FFT3d plan and scratch buffer depend ONLY on N_fft = 2·nframes,
  // which is set at fix construction (the `Nframes` constructor arg) and
  // never changes during the run.  Atom-indexed and molecule-indexed
  // buffers (vel_buf, com_vel_buf, group_slots) are legitimately rebuilt
  // each window for dynamic-group correctness via grow_buf() and
  // build_mol_topology() — but the FFT scratch and plan are invariant
  // and safely persistent.
  //
  // Per-window alloc/free crashes PPPM at LAMMPS shutdown via FFTW3
  // plan-cache corruption (verified: fftw_forget_wisdom() does NOT fix
  // it).  A single create-destroy cycle in the fix's lifetime keeps
  // FFTW state clean.
  if (!fft_vac) {
    const int N_fft = 2 * nframes;
    int mysize = 0;
    fft_vac = make_fft3d_vac(lmp, N_fft, &mysize, 0);
    memory->create(fft_buf, 2 * N_fft, "fix_xpt:fft_buf");
  }

  // Open output files after all computes are configured (write_header uses
  // pe_available).  Only once per fix lifetime: `rerun` calls modify->init()
  // for EVERY snapshot, and consecutive `run` commands call it once per run —
  // an unguarded fopen("w") here re-truncated .thermo/.pwr/.vac each time
  // (the old "header-only rerun output" quirk).
  if (me == 0 && !fp) open_file();
}

/* ======================================================================
   setup — called once before the run; schedule first pressure sample
   so that virial is tallied at the first end_of_step call.
====================================================================== */

void FixXPT::setup(int /*vflag*/)
{
  bigint first = ((update->ntimestep / nevery) + 1) * (bigint)nevery;
  if (press_compute) press_compute->addstep(first);
  if (pe_compute && pe_available) pe_compute->addstep(first);

  // Early-termination guard.  If the upcoming run is too
  // short to complete at least one analysis window, the partial-state
  // teardown can crash on certain code paths (uninitialised FFT scratch,
  // empty multi-tau ring buffers, etc.).  ERROR out cleanly with a
  // diagnostic instead of letting the user hit an obscure segfault.
  //
  // Update->beginstep / endstep are valid here (LAMMPS sets them before
  // calling setup).  We need (endstep − ntimestep) ≥ nevery·nframes
  // sample steps for end_of_step to be called nframes times.
  //
  // Evaluate ONLY at the true start of the (pseudo-)run: `rerun` calls
  // setup_minimal() — and hence this setup() — for EVERY snapshot with a
  // shrinking (endstep − ntimestep), so an unguarded check aborted any rerun
  // whose trailing partial window was shorter than nframes (after all the
  // full windows had already been analysed).  Mid-run/mid-rerun re-entries
  // skip the guard; a partial trailing window is simply never analysed —
  // the same behaviour as a normal run whose length is not window-aligned.
  const bigint steps_remaining = update->endstep - update->ntimestep;
  const bigint steps_per_window = (bigint)nevery * (bigint)nframes;
  const bool run_start = (update->ntimestep == update->beginstep);
  if (run_start && steps_remaining < steps_per_window) {
    // Exemption: a bare `run 0` with correlator=multitau emits a thermo +
    // .pwr/.vac block from the accumulated/restart multi-tau correlator
    // state (post_run handles it).  This is the ONLY short-run case that is
    // allowed; every other short run (FFT any length, multitau 0 < N < one
    // window) still errors verbatim below.
    if (correlator == CORR_MULTITAU && steps_remaining == 0) {
      mt_run0_dump = true;
    } else {
      error->all(FLERR,
        "fix xpt: run length ({} steps) is shorter than one analysis "
        "window (nevery × nframes = {} × {} = {} steps).  Either lengthen "
        "the run or reduce nevery / nframes so at least one window "
        "completes.  Partial-window output is not supported.",
        (long)steps_remaining, nevery, nframes, (long)steps_per_window);
    }
  }
  // Nsamples sub-window snapshots: multi-tau only, and nframes must be
  // an integer multiple of nsamples so the snapshot boundaries land on
  // integer frame counts.
  if (nsamples > 1) {
    if (correlator != CORR_MULTITAU) {
      error->all(FLERR,
        "fix xpt: nsamples > 1 requires correlator = multitau.  The FFT "
        "path overwrites vel_buf each frame and cannot produce partial-"
        "window spectra.  Either set `correlator multitau` or drop "
        "`nsamples`.");
    }
    if (nframes % nsamples != 0) {
      error->all(FLERR,
        "fix xpt: nframes ({}) must be an integer multiple of nsamples "
        "({}).  Got nframes / nsamples = {} with remainder {}.",
        nframes, nsamples, nframes / nsamples, nframes % nsamples);
    }
    if (me == 0) {
      FIXXPT_LOG(
        "FixXPT::{}-{}: nsamples = {} — emitting {} intermediate analysis "
        "blocks per window (every {} frames) plus the usual end-of-window "
        "block.\n",
        id, group->names[igroup], nsamples, nsamples - 1,
        nframes / nsamples);
    }
  }

  // Nframes sanity.  A meaningful VAC needs at least a
  // handful of lags; an FFT plan of length 2·nframes is most efficient
  // when nframes is a power of 2.
  if (nframes < 4) {
    error->all(FLERR,
      "fix xpt: nframes = {} is too small for a meaningful VAC "
      "(minimum 4).  Increase nframes; a typical liquid-MD setting is "
      "nframes = 512..4096 with nevery = 2..10.", nframes);
  }
  if (me == 0 && nframes < 32) {
    FIXXPT_LOG(
      "FixXPT::{}-{}: WARNING — nframes = {} gives a coarse DoS "
      "(< 32 frames).  Larger nframes (e.g. 512+) recommended for "
      "production thermo.\n", id, group->names[igroup], nframes);
  }
  if (me == 0 && correlator == CORR_FFT
      && (nframes & (nframes - 1)) != 0) {
    int nf_pow2 = 1;
    while (nf_pow2 < nframes) nf_pow2 <<= 1;
    FIXXPT_LOG(
      "FixXPT::{}-{}: WARNING — nframes = {} is not a power of 2; "
      "the FFT3d backend falls back to Bluestein, costing ~3-5x more "
      "wall time.  Consider nframes = {} (next power of 2).\n",
      id, group->names[igroup], nframes, nf_pow2);
  }

  // Restart-safe velocity-buffer allocation. grow_buf() is otherwise only reached at window-start
  // (iframe==0 in accumulate_frame); a mid-window restart RESUME skips it -> null vel_buf -> SEGV in
  // push_velocity_frame. setup() runs before any end_of_step on EVERY run, so (re)allocate here:
  // owner allocates (grow_buf reallocs when the pointer is null even if natom_buf was restored),
  // consumer re-points to the owner (whose setup ran first — owner is earlier in modify->fix[]).
  if (distributed()) {
    // Build the home layout here as well as at each window start, so the
    // buffers exist (and are reported) from the start of the run.  A run that
    // continues a window keeps the layout it has.
    if (owns_buffer && iframe == 0) {
      int *amask = atom->mask; tagint *atag = atom->tag; int an = atom->nlocal;
      int maxTag_local = 0;
      for (int i = 0; i < an; i++)
        if ((amask[i] & groupbit) && (int)atag[i] > maxTag_local) maxTag_local = (int)atag[i];
      int maxTag_global = 0;
      MPI_Allreduce(&maxTag_local, &maxTag_global, 1, MPI_INT, MPI_MAX, world);
      if (maxTag_global > 0) build_home_layout();
    } else if (!owns_buffer) {
      sync_buffer_pointers_from_owner();
    }
  } else if (owns_buffer) {
    int *amask = atom->mask; tagint *atag = atom->tag; int an = atom->nlocal;
    int maxTag_local = 0;
    for (int i = 0; i < an; i++)
      if ((amask[i] & groupbit) && (int)atag[i] > maxTag_local) maxTag_local = (int)atag[i];
    int maxTag_global = 0;
    MPI_Allreduce(&maxTag_local, &maxTag_global, 1, MPI_INT, MPI_MAX, world);
    if (maxTag_global > 0) grow_buf(maxTag_global);
  } else {
    sync_buffer_pointers_from_owner();
  }
}

/* ======================================================================
   end_of_step — called every nevery steps
====================================================================== */

void FixXPT::end_of_step()
{
  accumulate_frame();

  // Window never started this frame: the group was globally empty at iframe==0
  // (e.g. an unpopulated dynamic shell, or a stale mask right after a restart
  // resume).  accumulate_frame left the per-window buffers/topology unbuilt and
  // held iframe at 0, so skip all accumulation and the iframe++ below — touching
  // vel_buf / com_vel_buf here would deref null.  We DO keep the compute addstep
  // chain alive so the next sampled step still tallies virial / per-atom energy.
  // The empty-group condition is post-Allreduce global, so every rank takes this
  // branch together (no MPI collective skew).
  if (window_idle) {
    if (press_compute) press_compute->addstep(update->ntimestep + nevery);
    if (pe_compute && pe_available) pe_compute->addstep(update->ntimestep + nevery);
    return;
  }

  // Accumulate instantaneous pressure for window average.
  // Addstep() was called before this step's force eval (via setup or previous end_of_step),
  // so the virial was tallied and compute_scalar() is safe to call.
  if (press_compute) {
    press_compute->compute_scalar();
    press_sum += press_compute->scalar;
    press_count++;
    // Schedule virial tallying for the next sample step
    press_compute->addstep(update->ntimestep + nevery);
  }

  // ── Accumulate KE + PE for E_md and Cv fluctuation ───────────────────
  // IMPORTANT: MPI_Allreduce calls here must be called by ALL ranks collectively,
  // even those with no local group atoms.  The outer guard must use a globally-
  // consistent condition.  Using !group_slots.empty() (per-rank) as the sole
  // guard would deadlock: ranks with no local group atoms skip the Allreduce
  // while other ranks call it — a classic MPI collective mismatch.
  //
  // For group "all" (pe_peratom=false):
  //   compute_scalar() is collective; it is called unconditionally because
  //   group_slots can be empty on a rank.
  //
  // For subgroups (pe_peratom=true):
  //   compute_peratom() is local-only (fine to skip on empty-slot ranks).
  //   MPI_Allreduce MUST be called by all ranks; ranks with no local atoms
  //   contribute total_local=0 correctly to the global sum.
  if (pe_available) {
    double E_inst = 0.0;

    if (!pe_peratom) {
      // Ibuf selects vel_buf row 0 when grow_buf allocated a single-
      // frame buffer (multitau with no matrix-VAC dependency) and the
      // live iframe otherwise.
      const int ibuf = mt_vel_buf_single_frame ? 0 : iframe;
      // Every rank must reach the Allreduce and the collective
      // compute_scalar() below; a rank holding no group atoms (possible in
      // the distributed layout, or a vacuum region) contributes zero.
      {
        double ke_raw_local = 0.0;
        for (int slot : group_slots) {
          double m = mass_buf[slot];
          for (int d = 0; d < 3; d++) {
            double v = vread(ibuf, slot, d);
            ke_raw_local += 0.5 * m * v * v;
          }
        }
        double ke_raw_global = 0.0;
        MPI_Allreduce(&ke_raw_local, &ke_raw_global, 1, MPI_DOUBLE, MPI_SUM, world);
        double ke_val;
        if (units_lj) ke_val = ke_raw_global;
        else          ke_val = ke_raw_global * ke_to_kjmol;

        pe_compute->compute_scalar();
        double pe_raw = pe_compute->scalar;
        double pe_kjmol;
        if (units_lj) pe_kjmol = pe_raw;
        else          pe_kjmol = pe_raw * pe_to_kjmol;
        E_inst = ke_val + pe_kjmol;

        if (std::isfinite(E_inst)) {
          E_sum    += E_inst;
          E_sq_sum += E_inst * E_inst;
          E_count++;
        }
        pe_compute->addstep(update->ntimestep + nevery);
      }

    } else {
      // Subgroup: sum (PE + KE) per atom from current atom->v, matching the reference
      // v_atomEng = atomPE + atomKE convention.  Using atom->v directly (same
      // timestep as pe/atom) avoids dependence on the stale group_slots list.
      // KE conversion: native (mass×vel²)/atom → native PE units/atom
      //   real:     (g/mol)(Å/fs)²  → kcal/mol    × ke_to_pe_fac = 1e7/4184
      //   metal:    (g/mol)(Å/ps)²  → eV           × ke_to_pe_fac = VAC_TO_JMOL/96485
      //   si:       kg·(m/s)²       → J            × 1.0
      //   cgs:      g·(cm/s)²       → erg          × 1.0
      //   micro:    pg·(μm/μs)²     → pg·μm²/μs²   × 1.0
      //   nano:     ag·(nm/ns)²     → ag·nm²/ns²    × 1.0
      //   electron: (g/mol)(Bohr/atu)² → Hartree    × ke_to_pe_fac
      //   lj:       m*(σ/τ)²        → ε             × 1.0
      double ke_fac = units_lj ? 1.0 : ke_to_pe_fac;

      // compute_peratom() MUST be called on ALL ranks regardless of whether
      // this rank has local group atoms.  The implementation calls
      // comm->reverse_comm() (an MPI collective for Newton ghost contributions)
      // And also calls error->all() (another MPI collective) when energies
      // were not tallied.  Calling it on only a subset of ranks deadlocks.
      //
      // Guard only on eflag_atom so all ranks either call it or all skip it.
      // The accumulation loop uses groupbit so ranks with no group atoms
      // contribute total_local=0 to the MPI_Allreduce correctly.
      double total_local = 0.0;
      if (update->eflag_atom == update->ntimestep) {
        pe_compute->compute_peratom();  // collective: must be called by ALL ranks
        int    nlocal_     = atom->nlocal;
        int   *mask_       = atom->mask;
        double **v_        = atom->v;
        double *amass      = atom->mass;
        int    *atype      = atom->type;
        double *peatom     = pe_compute->vector_atom;
        for (int i = 0; i < nlocal_; i++) {
          if (!(mask_[i] & groupbit)) continue;
          double ke_i = 0.5 * amass[atype[i]]
                        * (v_[i][0]*v_[i][0] + v_[i][1]*v_[i][1] + v_[i][2]*v_[i][2])
                        * ke_fac;
          total_local += peatom[i] + ke_i;
        }
      }
      // Addstep() MUST be called by ALL ranks regardless of local group atoms.
      // It tells LAMMPS to tally per-atom energies (including PPPM kspace FFT)
      // Before the next force evaluation.  That FFT is a distributed collective;
      // if only some ranks call addstep(), eflag_atom differs between ranks and
      // the PPPM per-atom FFT deadlocks — identical to the MPI collective
      // mismatch pattern described above for the Allreduce.
      pe_compute->addstep(update->ntimestep + nevery);
      // Allreduce called by ALL ranks (total_local=0 on empty-slot ranks or
      // when eflag_atom was not set — both contribute zero correctly).
      double total_global = 0.0;
      MPI_Allreduce(&total_local, &total_global, 1, MPI_DOUBLE, MPI_SUM, world);
      // Convert total (pe+ke in native PE units) → kJ/mol-equiv (or ε for lj)
      if (units_lj) E_inst = total_global;
      else          E_inst = total_global * pe_to_kjmol;

      if (std::isfinite(E_inst)) {
        E_sum    += E_inst;
        E_sq_sum += E_inst * E_inst;
        E_count++;
      }
    }
  }

  iframe++;

  // Intermediate sub-window snapshot triggers (multi-tau + nsamples > 1
  // only).  At iframe = k · (nframes / nsamples) for k = 1..nsamples-1
  // fire a snapshot of the CURRENT cumulative state — run_analysis()
  // Works correctly because press_sum / E_sum / mt_c_sum are all
  // running cumulative totals that get sampled here without reset.
  if (nsamples > 1 && iframe > 0 && iframe < nframes
      && (iframe % (nframes / nsamples)) == 0) {
    run_analysis();
  }

  if (iframe == nframes) {
    run_analysis();

#ifdef FIX_XPT_DEBUG
    // ── End-of-window timing report ─────────────────────────────────────
    // Format mirrors LAMMPS's fix amoeba: one parent breakdown line per
    // grouping with cumulative %, then indented sub-items each carrying
    // absolute seconds + their individual %.
    if (me == 0) {
      // FFT-path VAC buckets (zero in multi-tau mode); multi-tau merge
      // is reported separately so the breakdown is meaningful in either
      // correlator.
      const double t_vac_fft = t_vac_atom + t_vac_mol + t_vac_vib;
      const double t_vac     = t_vac_fft + t_vac_multitau;
      const double t_thermo_core = t_thermo;
      // t_multitau_push is per-step accumulation —
      // include in totals so the % column reflects the full window cost.
      const double t_total   = t_vac + t_dos + t_thermo + t_multitau_push;
      const double inv_T     = (t_total > 0.0) ? 1.0 / t_total : 0.0;
      FIXXPT_LOG(
        "FixXPT::{}-{}: window {} analysis timing (Nframes={}):\n"
        "    VAC computation       :              {:.3g}%\n"
        "      Atom-VAC (FFT) time : {:.6g} {:.3g}%\n"
        "      Trans+rot-VAC time  : {:.6g} {:.3g}%\n"
        "      Vib-VAC       time  : {:.6g} {:.3g}%\n"
        "      Multi-tau merge time: {:.6g} {:.3g}%\n"
        "    Multi-tau push (cum.) : {:.6g} {:.3g}%\n"
        "    DOS computation       : {:.6g} {:.3g}%\n"
        "    2PT thermodynamics    :              {:.3g}%\n"
        "      2PT core      time  : {:.6g} {:.3g}%\n"
        "    Total window          : {:.6g}\n",
        id, group->names[igroup],
        nwindow, nframes,
        100.0 * t_vac * inv_T,
        t_vac_atom,     100.0 * t_vac_atom     * inv_T,
        t_vac_mol,      100.0 * t_vac_mol      * inv_T,
        t_vac_vib,      100.0 * t_vac_vib      * inv_T,
        t_vac_multitau, 100.0 * t_vac_multitau * inv_T,
        t_multitau_push,100.0 * t_multitau_push* inv_T,
        t_dos,          100.0 * t_dos          * inv_T,
        100.0 * t_thermo * inv_T,
        t_thermo_core,    100.0 * t_thermo_core    * inv_T,
        t_total);
    }
    reset_window_timings();
#endif

    // End-of-window state resets (moved out of run_analysis so that
    // intermediate-snapshot calls don't truncate the cumulative within-
    // window averages).
    press_sum   = 0.0;
    press_count = 0;
    E_sum       = 0.0;
    E_sq_sum    = 0.0;
    E_count     = 0;

    iframe    = 0;
    nwindow++;
    step_start = update->ntimestep;
  }
}

/* ======================================================================
   post_run — analyse whatever is accumulated if partial window
====================================================================== */

void FixXPT::post_run()
{
  // Multi-tau `run 0`: emit a thermo + .pwr/.vac block straight from the
  // accumulated / restart multi-tau correlator state (setup() set the flag
  // and skipped the short-run error).  run_analysis() is normally entered at
  // the end of a completed window with per-window inputs freshly built by
  // accumulate_frame()'s iframe==0 prologue; at a bare run 0 those are stale
  // or (after restart) never built — group_slots is NOT persisted.  So we
  // reconstruct exactly what accumulate_frame()'s iframe==0 branch and the
  // run_analysis() prologue rely on: group_slots, natom_buf, mass_buf (via
  // grow_buf), fix_dof_window, and the molecular topology when do_molecule.
  //
  // If the multi-tau accumulators were never restored (fresh run 0, no
  // restart) mt_n_levels is 0; we allocate zeroed accumulators so the VAC
  // build sees all-zero c_sum.  run_analysis() then derives vac[0]≈0 →
  // T_vac≤0 and returns at its `if (T <= 0.0) return;` guard before touching
  // V_box / mass / molecular / write paths — a valid trivial (no-op) block.
  // When the accumulators WERE restored, mass_buf rebuilt here matches the
  // restored group, so the full analysis + file writers fire normally.
  if (mt_run0_dump) {
    mt_run0_dump = false;

    int *mask  = atom->mask;
    int *tag   = atom->tag;
    int nlocal = atom->nlocal;

    if (do_molecule) build_mol_topology();

    int maxTag_local = 0;
    for (int i = 0; i < nlocal; i++)
      if ((mask[i] & groupbit) && tag[i] > maxTag_local) maxTag_local = tag[i];
    int maxTag_global = 0;
    MPI_Allreduce(&maxTag_local, &maxTag_global, 1, MPI_INT, MPI_MAX, world);
    if (distributed()) {
      // Keep a layout that a previous run in this process built (its window
      // accumulators and inertia belong to it); build one if none exists.
      // The pushed frame fills mass_buf, which the analysis normalises by;
      // it lands in row 0, the single multi-tau frame, which no analysis reads.
      const bool have_layout = (natom_buf > 0) && (vel_buf || vel_buf_f);
      if (maxTag_global > 0 && owns_buffer) {
        if (!have_layout) build_home_layout();
        push_velocity_frame(0);
      }
    } else {
      group_slots.clear();
      for (int i = 0; i < nlocal; i++)
        if (mask[i] & groupbit) group_slots.push_back(tag[i] - 1);
      if (maxTag_global > 0 && owns_buffer) {
        grow_buf(maxTag_global);
        // A freshly allocated mass_buf holds no masses (no frame was pushed
        // after read_restart); the analysis normalises by them.
        double *amass = atom->mass;
        int    *atype = atom->type;
        for (int i = 0; i < nlocal; i++)
          if ((mask[i] & groupbit) && tag[i] >= 1 && tag[i] <= natom_buf)
            mass_buf[tag[i] - 1] = amass[atype[i]];
      }
      int ng_local = (int)group_slots.size();
      MPI_Allreduce(&ng_local, &ng_window, 1, MPI_INT, MPI_SUM, world);
    }

    fix_dof_window = 0;
    for (const auto &ifix : modify->get_fix_list())
      if (ifix->dof_flag) fix_dof_window += ifix->dof(igroup);

    // Establish multi-tau buffer shapes, then re-apply the stashed restart
    // state.  restart() ran at fix-construction (before any buffers existed)
    // and only stashed the payload; now that the shapes are known we can land
    // it.  Molecular streams are allocated lazily in accumulate_mol_frame,
    // which never runs at `run 0`, so allocate them here too (mirroring that
    // path) before applying — otherwise the saved trans/rot/vib state has no
    // destination and run_analysis sees an all-zero VAC.
    if (correlator == CORR_MULTITAU && owns_buffer) {
      if (mt_n_levels == 0) multitau_alloc(natom_buf);
      if (do_molecule && nmol_group > 0) {
        // Live widths, as accumulate_mol_frame allocates them.  A v1 restart
        // (pre-1.0.1, single rank) is restored stream by stream at its SAVED
        // widths: grow_buf rounds natom_buf up (max(n, natom_buf+32)), so the
        // live vib width (e.g. 2624) need not equal the saved one (2592), and
        // a mismatch would skip the vib stream → all-zero vac_vib → inf in
        // the 2PT solve.  v2 stores global sums, so any width takes them.
        const bool dist = distributed();
        int nu_trans = dist ? nmol_buf : nmol_group, nu_rot = nu_trans, nu_vib = natom_buf;
        if (mt_restart_pending && (int)mt_restart_stash.size() >= 9
            && (int)mt_restart_stash[5] /*mol_in*/ && mt_restart_stash[0] < 200.0) {
          nu_trans = (int)mt_restart_stash[6];
          nu_rot   = (int)mt_restart_stash[7];
          nu_vib   = (int)mt_restart_stash[8];
        }
        if (mt_trans.n_units != nu_trans || mt_trans.is_distributed != dist)
          multitau_stream_alloc(mt_trans, nu_trans, dist);
        if (mt_rot.n_units != nu_rot || mt_rot.is_distributed != dist)
          multitau_stream_alloc(mt_rot,   nu_rot,   dist);
        if (mt_vib.n_units != nu_vib)
          multitau_stream_alloc(mt_vib,   nu_vib,   /*distributed=*/true);
        mt_molecular_active = true;
      }
      multitau_apply_restart_stash();

      // mol_inertia feeds the rotational-gas entropy (I_avg).  Option 1
      // restores it from the restart when present (mol_inertia_count > 0
      // after the apply above).  For a restart written BEFORE Option-1
      // persistence it is still 0 → the molecular analysis block would be
      // skipped (S_trans/rot/vib = 0, S_q = ±inf).  Option-2 bridge (TEMP):
      // reconstruct it from the current restart geometry.  Remove this block
      // with compute_mol_inertia_oneframe() once all restarts carry Option-1
      // inertia.
      if (do_molecule && nmol_group > 0 && mol_inertia_count == 0) {
        if (!mol_inertia)
          memory->create(mol_inertia, nmol_group, 3, "fix_xpt:mol_inertia");
        compute_mol_inertia_oneframe();
      }
    }

    run_analysis();
    return;
  }

  // Trailing partial window (≥ 25%): only the multi-tau correlator can
  // analyse it — the log-lag accumulators are frame-count agnostic (same
  // machinery as the nsamples sub-window snapshots and the run-0 dump).
  // The FFT path's plan/scratch are sized for the full window
  // (N_fft = 2·nframes); analysing a shorter stream through them emits a
  // garbage (zero/NaN) row, so the tail is discarded there instead —
  // consistent with the "partial-window output is not supported" stance
  // of the setup() short-run guard.  iframe is left in place on the FFT
  // path so a follow-up `run` continues filling the window.
  if (correlator == CORR_MULTITAU && iframe > nframes / 4) {
    // Analysed as a sub-window snapshot (the nsamples path): the multi-tau
    // sums are per-lag averages, so the window length enters only through
    // the DOS grid, and the persistent FFT plan is sized for nframes.
    // Substituting the tail length for nframes ran that plan on a shorter
    // mirrored VAC and produced a wrong DOS.
    run_analysis();
    iframe = 0;
  }
}

/* ======================================================================
   compute_vector — thermo access (1-based index)
====================================================================== */

double FixXPT::compute_vector(int n)
{
  // LAMMPS thermo passes a 0-based index (f_xpt[1] → n=0, f_xpt[7] → n=6)
  if (n < 0 || n >= nvec) return 0.0;
  return result_ready ? result_vec[n] : 0.0;
}

/* ======================================================================
   memory_usage
====================================================================== */

double FixXPT::memory_usage()
{
  // Account for ALL fix_xpt-owned buffers, surfaced via LAMMPS's `info`
  // command + the "Per MPI rank memory" line at run start.
  //
  // Buffer-sharing accounting: the shared buffers (vel_buf / mass_buf /
  // com_vel_buf / omega_buf / angmom_buf / vib_vel_buf) are physically owned
  // by `buffer_owner` and aliased by consumers, so counting them on every fix
  // would N× over-report — tally them only when this fix owns the buffer.
  // Per-fix buffers (mol_inertia, fft_buf, multi-tau rings) are always counted.
  double bytes = 0.0;
  const double d = (double)sizeof(double);
  const int    vel_buf_frames = mt_vel_buf_single_frame ? 1 : nframes;

  // Per-atom velocity buffer (FFT or multi-tau single-frame mode).
  // Account for FP32 storage when enabled.
  if (owns_buffer && natom_buf > 0) {
    const double per_elem = (buffer_precision == BUFFER_FP32)
                          ? (double)sizeof(float)
                          : d;
    bytes += (double)vel_buf_frames * natom_buf * 3 * per_elem;  // vel_buf
    bytes += (double)natom_buf * d;                              // mass_buf (always FP64)
  }

  // Molecular mode buffers: single-frame under multi-tau (the rings hold the
  // history); molecule extent nmol_group (replicated) or nmol_buf (distributed).
  if (do_molecule && nmol_group > 0) {
    const int    mol_frames = (correlator == CORR_MULTITAU) ? 1 : nframes;
    const double nmol_ext   = distributed() ? (double)nmol_buf : (double)nmol_group;
    if (owns_buffer) {
      bytes += 3.0 * mol_frames * nmol_ext * 3 * d;          // com_vel / omega / angmom
      const double per_elem = (buffer_precision == BUFFER_FP32)
                            ? (double)sizeof(float)
                            : d;
      bytes += (double)mol_frames * natom_buf * 3 * per_elem; // vib_vel_buf
      if (home_xu) bytes += (double)natom_buf * 3 * d;        // home_xu
    }
    if (owns_buffer || !distributed())
      bytes += nmol_ext * 3 * d;                              // mol_inertia
  }

  // Distributed-layout bookkeeping and per-frame exchange scratch.
  if (distributed() && owns_buffer) {
    bytes += (double)home_tag.capacity() * sizeof(tagint);
    bytes += (double)(home_mol_start.capacity() + mol_lo.capacity()) * sizeof(int);
    bytes += (double)home_tag_lo.capacity() * sizeof(tagint);
    bytes += (double)(molmass_home.capacity() + molunit_home.capacity()) * d;
    bytes += (double)home_recv.capacity();
    bytes += (double)(xc_sendbuf.capacity() + xc_recvbuf.capacity()) * d;
  }

  // FFT scratch (always present once init() ran).
  if (fft_buf != nullptr) {
    bytes += (double)(2 * 2 * nframes) * d;                // fft_buf (complex)
  }

  // Multi-tau rings (when correlator==CORR_MULTITAU).
  if (correlator == CORR_MULTITAU && mt_n_levels > 0) {
    // Scalar all-atom path
    bytes += (double)mt_n_levels * mt_MP * mt_natom_ring * 3 * d;  // mt_v_ring
    bytes += (double)mt_n_levels * mt_MP * d;                      // mt_c_sum
    if (mt_n_levels > 1) {
      bytes += (double)mt_n_levels * mt_natom_ring * 3 * d;        // mt_down_acc
    }
    // Molecular streams (trans, rot, vib)
    for (const MultiTauStream *s : {&mt_trans, &mt_rot, &mt_vib}) {
      if (s->n_units <= 0) continue;
      const int L = (int)s->v_ring.size();
      bytes += (double)L * mt_MP * s->n_units * 3 * d;
      bytes += (double)L * mt_MP * d;
      if (L > 1) bytes += (double)L * s->n_units * 3 * d;
    }
  }

  return bytes;
}

/* ======================================================================
   write_restart / restart

   Persist the multi-tau correlator ACCUMULATORS across LAMMPS
   restart-file cycles.  Saved: c_sum, c_cnt, count_seen, head, down_n
   for scalar all-atom + each of the 3 molecular streams (trans/rot/vib).
   NOT saved: v_ring, down_acc.  Those reset to zero on restart; the
   correlator loses up to MP frames of correlation per restart boundary.
   The accumulated c_sum IS restored for `run 0` reanalysis via
   multitau_apply_restart_stash() (a normal run re-zeroes the correlator at
   the first window boundary, so restart state is for reanalysis, not for
   resuming accumulation).

   Restart payload size:  ~few KB even at production scale.

   Wire format v2 (flat double array; every c_sum is the global sum):
     header  : [correlator (+100 inertia block, +200 = v2), mt_n_levels,
                mt_MP, N_g (group atoms),
                matrix_flag (always 0; reader skips legacy matrix payloads),
                mol_active, nmol (trans), nmol (rot), N_g (vib); 0 = stream absent]
     scalar  : c_sum[L*MP], c_cnt[L*MP], count_seen[L], head[L], down_n[L]
     mol_tr  : c_sum[L*MP], count_seen[L], head[L], down_n[L]
     mol_rot : (same shape)
     mol_vib : (same shape)
     inertia : mol_inertia_count, mol_inertia[nmol*3] (global molecule order)
   v1 (pre-1.0.1, no +200) stored rank 0's partial sums and per-rank
   widths in fields 3 and 6-8; it is restored on a single rank only.
   All ints/longs encoded as doubles (matches fix_ave_correlate_long.cpp).
====================================================================== */

void FixXPT::write_restart(FILE *fp)
{
  // Modify::write_restart calls this on every rank, so the per-rank partial
  // accumulators are reduced here, collectively, and rank 0 writes.  Format
  // v2 (+200 on field 0): every c_sum holds the global sum, fields 3 and 6-8
  // hold global counts, and the inertia block covers all molecules in global
  // molecule order.  Every branch below is taken identically on all ranks.
  const bool has_scalar = (correlator == CORR_MULTITAU && mt_n_levels > 0);
  const bool has_mol    = mt_molecular_active;
  const bool write_inertia = (has_mol && mol_inertia
                              && mol_inertia_count > 0 && nmol_group > 0);

  int ng_local = (int)group_slots.size(), ng = 0;
  MPI_Allreduce(&ng_local, &ng, 1, MPI_INT, MPI_SUM, world);

  // The scalar accumulator is a per-rank partial in both layouts; a stream
  // is one when its is_distributed flag says so.
  auto global_sum = [&](const std::vector<double> &v, bool partial) {
    std::vector<double> g(v);
    if (partial)
      MPI_Allreduce(v.data(), g.data(), (int)v.size(), MPI_DOUBLE, MPI_SUM, world);
    return g;
  };
  std::vector<std::vector<double>> sc_sum;
  if (has_scalar)
    for (int l = 0; l < mt_n_levels; l++) sc_sum.push_back(global_sum(mt_c_sum[l], true));
  std::vector<std::vector<double>> st_sum[3];
  const MultiTauStream *streams[3] = {&mt_trans, &mt_rot, &mt_vib};
  if (has_mol)
    for (int c = 0; c < 3; c++)
      if (streams[c]->n_units > 0)
        for (const auto &cs : streams[c]->c_sum)
          st_sum[c].push_back(global_sum(cs, streams[c]->is_distributed));

  std::vector<double> inertia_all;
  if (write_inertia) {
    if (distributed()) {
      std::vector<int> cnt(nprocs), dsp(nprocs);
      for (int r = 0; r < nprocs; r++) {
        cnt[r] = 3 * (mol_lo[r + 1] - mol_lo[r]);
        dsp[r] = 3 * mol_lo[r];
      }
      inertia_all.assign((size_t)nmol_group * 3, 0.0);
      MPI_Gatherv(&mol_inertia[0][0], 3 * nm_home, MPI_DOUBLE, inertia_all.data(),
                  cnt.data(), dsp.data(), MPI_DOUBLE, 0, world);
    } else {
      inertia_all.assign(&mol_inertia[0][0], &mol_inertia[0][0] + (size_t)nmol_group * 3);
    }
  }

  if (comm->me != 0) return;

  // Compute total double count.
  const int hdr = 9;
  long n = hdr;
  if (has_scalar) {
    const long L  = mt_n_levels;
    const long MP = mt_MP;
    n += L * MP * 2;     // c_sum + c_cnt
    n += 3 * L;          // count_seen + head + down_n
  }
  auto add_stream = [&](const MultiTauStream &s) {
    if (s.n_units <= 0) return;
    const long L  = (long)s.v_ring.size();
    n += L * (long)mt_MP;  // c_sum
    n += 3 * L;            // count_seen + head + down_n
  };
  if (has_mol) {
    add_stream(mt_trans); add_stream(mt_rot); add_stream(mt_vib);
  }
  // Option 1: persist the window-summed mol_inertia + its frame count so the
  // rotational-gas I_avg restores exactly on a `run 0` reanalysis.  Appended
  // after the streams; signalled by +100 on the header correlator field so
  // older restarts (which omit it) still parse.  Length: 1 + nmol_group*3.
  if (write_inertia) n += 1 + (long)nmol_group * 3;

  std::vector<double> list((size_t)n, 0.0);
  long m = 0;
  list[m++] = (double)(correlator + (write_inertia ? 100 : 0) + 200);
  list[m++] = (double)(has_scalar ? mt_n_levels : 0);
  list[m++] = (double)mt_MP;
  list[m++] = (double)ng;
  list[m++] = 0.0;   // matrix-stream flag (streams removed; always 0)
  list[m++] = has_mol ? 1.0 : 0.0;
  list[m++] = (mt_trans.n_units > 0) ? (double)nmol_group : 0.0;
  list[m++] = (mt_rot.n_units   > 0) ? (double)nmol_group : 0.0;
  list[m++] = (mt_vib.n_units   > 0) ? (double)ng         : 0.0;

  if (has_scalar) {
    const int L  = mt_n_levels;
    const int MP = mt_MP;
    for (int l = 0; l < L; l++)
      for (int k = 0; k < MP; k++) list[m++] = sc_sum[l][k];
    for (int l = 0; l < L; l++)
      for (int k = 0; k < MP; k++) list[m++] = (double)mt_c_cnt[l][k];
    for (int l = 0; l < L; l++) list[m++] = (double)mt_count_seen[l];
    for (int l = 0; l < L; l++) list[m++] = (double)mt_head[l];
    for (int l = 0; l < L; l++) list[m++] = (double)mt_down_n[l];
  }

  auto write_stream = [&](const MultiTauStream &s, const std::vector<std::vector<double>> &cs) {
    if (s.n_units <= 0) return;
    const int L  = (int)s.v_ring.size();
    const int MP = mt_MP;
    for (int l = 0; l < L; l++)
      for (int k = 0; k < MP; k++) list[m++] = cs[l][k];
    for (int l = 0; l < L; l++) list[m++] = (double)s.count_seen[l];
    for (int l = 0; l < L; l++) list[m++] = (double)s.head[l];
    for (int l = 0; l < L; l++) list[m++] = (double)s.down_n[l];
  };
  if (has_mol) {
    write_stream(mt_trans, st_sum[0]);
    write_stream(mt_rot,   st_sum[1]);
    write_stream(mt_vib,   st_sum[2]);
  }
  if (write_inertia) {
    list[m++] = (double)mol_inertia_count;
    for (double I : inertia_all) list[m++] = I;
  }

  const int size = (int)(n * sizeof(double));
  fwrite(&size, sizeof(int),    1,        fp);
  fwrite(list.data(), sizeof(double), n,  fp);
}

void FixXPT::restart(char *buf)
{
  // Stash-and-defer (Option A).  LAMMPS delivers restart state at fix-
  // declaration time (Modify::add_fix), BEFORE multitau_alloc /
  // multitau_stream_alloc have established the ring/stream shapes — so here
  // mt_n_levels / mt_MP / mt_natom_ring are still 0 and the molecular streams
  // have n_units = 0.  Restoring now would discard the scalar block (shape 0
  // != saved) and silently skip every molecular stream.  Instead copy the raw
  // payload aside and re-apply it from multitau_apply_restart_stash() once the
  // buffers exist (the multitau `run 0` reanalysis path in post_run()).  The
  // payload length is recoverable from the fixed 9-double header.
  auto *list = (double *) buf;
  // header[0] = correlator (+100 when an Option-1 mol_inertia block follows,
  // +200 for the v2 format).  The length rule is the same for v1 and v2.
  const int raw0          = (int)list[0];
  const int raw_v1        = (raw0 >= 200) ? (raw0 - 200) : raw0;
  const bool inertia_in   = (raw_v1 >= 100);
  const int correlator_in = inertia_in ? (raw_v1 - 100) : raw_v1;
  if (correlator_in != correlator) return;   // not our correlator kind

  const int Lin        = (int)list[1];
  const int MPin       = (int)list[2];
  const int matrix_in  = (int)list[4];
  const int mol_in     = (int)list[5];
  const int n_trans_in = (int)list[6];
  const int n_rot_in   = (int)list[7];
  const int n_vib_in   = (int)list[8];

  long len = 9;
  if (Lin > 0) {
    len += (long)Lin * MPin * 2 + 3L * Lin;     // c_sum + c_cnt + count/head/down_n
    if (matrix_in) len += (long)Lin * MPin * 6; // 6 per-axis scalar accumulators
  }
  if (mol_in) {
    auto molpart = [&](int nu) { return nu > 0 ? (long)Lin * MPin + 3L * Lin : 0L; };
    len += molpart(n_trans_in) + molpart(n_rot_in) + molpart(n_vib_in);
  }
  // Option-1 mol_inertia block: 1 (count) + n_trans_in*3 (nmol == trans units).
  if (inertia_in && mol_in && n_trans_in > 0)
    len += 1 + (long)n_trans_in * 3;

  mt_restart_stash.assign(list, list + len);
  mt_restart_pending = true;
}

