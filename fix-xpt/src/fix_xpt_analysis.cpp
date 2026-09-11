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
#  include <fftw3.h>           // fftw_forget_wisdom cleanup at shutdown
#endif
#include "math_eigen.h"       // MathEigen::jacobi3 (molecular inertia-tensor diagonalization)
#include "math_eigen_impl.h"  // MathEigen::Jacobi<>

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
#include <vector>

// LAMMPS mask bits
#include "fix.h"

// Shared physical constants + FIXXPT_LOG / FIXXPT_TIME diagnostic macros,
// split out so the sibling fix_xpt_*.cpp translation units share one
// definition (defines FIX_XPT_DEBUG; include after the LAMMPS headers).
#include "fix_xpt_const.h"

using namespace LAMMPS_NS;
using namespace FixConst;


// ============================================================================
// fix_xpt_analysis.cpp — the per-window analysis orchestrator (run_analysis)
// plus the DoS/VAC power-spectrum builders and the .thermo/.pwr/.vac writers.
// ============================================================================
/* ======================================================================
   run_analysis — compute 2PT after nframes are collected
====================================================================== */

void FixXPT::run_analysis()
{
  const int ns = nframes;

  // ── 1. Count group atoms locally and across MPI ──────────────────────
  int ng_local  = (int)group_slots.size();
  int ng_global = 0;
  MPI_Allreduce(&ng_local, &ng_global, 1, MPI_INT, MPI_SUM, world);
  if (ng_global == 0) return;

  // Window-averaged pressure, computed once.  Used by use_sim_z_mode, where
  // Z_sim = P V / (N kB T) overrides hs_compressibility in the mu_q gas-PV term.
  double P_avg = (press_count > 0) ? press_sum / press_count : 0.0;

  // ── 2. FFT setup ──────────────────────────────────────────────────────
  // FFT3d plan and scratch buffer are allocated once in init() and reused
  // across all windows; dimensions depend only on N_fft = 2·nframes, so they
  // remain valid even for a dynamic group.
  const int N_fft = 2 * ns;
  // init() must have allocated the persistent plan/buffer.  Re-creating one
  // here would corrupt the FFTW3 plan cache at LAMMPS shutdown; error instead.
  if (!fft_vac || !fft_buf) {
    error->all(FLERR, "fix xpt: internal error — FFT scratch is not "
                      "allocated.  init() should have run before any "
                      "analysis window; if you reached this from a "
                      "non-standard call path, please report it.");
  }

  // ── 3. Compute local mass-weighted VAC ───────────────────────────────
  // Vac[k] = (1/ns) * sum_{atoms, dims} mass_i * v_i(t) * v_i(t+k)
  // Via Wiener-Khinchin: FFT(v_pad) → |FFT|² → IFFT/scale
  double mass_local = 0.0;
  for (int slot : group_slots) mass_local += mass_buf[slot];

#ifdef FIX_XPT_DEBUG
  double t0 = 0.0;
#endif

  std::vector<double> vac;
  double mass_global = 0.0;
  // 3PT cage DoS (extensive, ∫=3N) captured for the .pwr cage column; empty
  // unless mode 3PT.  mono = whole-group/translational; trans/rot = molecular.
  std::vector<double> cage_mono_w, cage_trans_w, cage_rot_w;
  // S_cage .thermo columns: re-captured (or left 0) by this analysis pass.
  s_cage_trans_last = 0.0;
  s_cage_rot_last   = 0.0;

  if (correlator == CORR_MULTITAU) {
    FIXXPT_TIME(t_vac_multitau);
    // ── Multi-tau: finalise per-step ring-buffer accumulation ────────
    // Each rank holds the partial Σ over its local group_slots for every
    // (level, lag) pair; MPI-reduce once, normalise by per-lag sample count,
    // then feed the c_per_level / stride_per_level vectors the phase-4
    // interp-merge expects.
    const int MP   = mt_MP;
    const int Lreq = mt_n_levels;
    std::vector<std::vector<double>> c_per_level(Lreq);
    std::vector<long>                stride_per_level(Lreq, 1);
    long stride = 1;
    for (int Lev = 0; Lev < Lreq; Lev++) {
      stride_per_level[Lev] = stride;
      std::vector<double> c_local = mt_c_sum[Lev];      // copy partial sum
      std::vector<double> c_global(MP, 0.0);
      MPI_Allreduce(c_local.data(), c_global.data(), MP,
                    MPI_DOUBLE, MPI_SUM, world);
      // Biased estimator (matches FFT Wiener-Khinchin): divide by the total
      // sample count at this level (ns_ℓ = mt_count_seen[Lev]), not the
      // per-lag count.  L=1 with M+P = ns then bit-matches FFT.
      const long ns_L = mt_count_seen[Lev];
      const double inv_nsL = (ns_L > 0) ? 1.0 / (double)ns_L : 0.0;
      c_per_level[Lev].assign(MP, 0.0);
      for (int k = 0; k < MP; k++) {
        c_per_level[Lev][k] = c_global[k] * inv_nsL;
      }
      stride *= mt_S;
    }

    MPI_Allreduce(&mass_local, &mass_global, 1, MPI_DOUBLE, MPI_SUM, world);

    // ── Phase 4: interp-merge all levels onto the uniform grid ────────
    // Build a piecewise (t, c) sequence: level 0 gives lags 0..M+P−1 at dt
    // spacing; level ℓ≥1 gives lags k_min_emit..M+P−1 at S^ℓ·dt spacing
    // (P-overlap dropped).  Linear interp onto the uniform t_j = j·dt grid.
    // For j ≤ M+P−1 the interp is exact (level-0 sample at t_j); past that
    // cap it bridges log-spaced samples — the only multi-tau approximation
    // in the DoS pipeline.
    const int k_min_emit_phase4 = (MP - 1) / mt_S + 1;
    std::vector<double> t_merged, c_merged;
    t_merged.reserve(MP + (Lreq - 1) * (MP - k_min_emit_phase4));
    c_merged.reserve(t_merged.capacity());
    for (int Lev = 0; Lev < Lreq; Lev++) {
      const int K_L  = (int)c_per_level[Lev].size();
      const int k_lo = (Lev == 0) ? 0 : k_min_emit_phase4;
      for (int k = k_lo; k < K_L; k++) {
        t_merged.push_back((double)k * (double)stride_per_level[Lev]);
        c_merged.push_back(c_per_level[Lev][k]);
      }
    }
    vac.assign(ns, 0.0);
    if (!t_merged.empty()) {
      const double t_max = t_merged.back();
      size_t idx = 0;
      for (int j = 0; j < ns; j++) {
        const double tj = (double)j;
        if (tj > t_max) break;     // remainder stays zero
        while (idx + 1 < t_merged.size() && t_merged[idx + 1] < tj) idx++;
        if (idx + 1 >= t_merged.size()) {
          vac[j] = c_merged.back();
        } else {
          const double t0 = t_merged[idx], t1 = t_merged[idx + 1];
          const double c0 = c_merged[idx], c1 = c_merged[idx + 1];
          const double w  = (t1 > t0) ? (tj - t0) / (t1 - t0) : 0.0;
          vac[j] = c0 * (1.0 - w) + c1 * w;
        }
      }
    }

    // Diagnostic: dump per-level (time, c) pairs (level ℓ ≥ 1 emitted only
    // for k > (M+P−1)/S, the P-overlap drop).
    if (comm->me == 0) {
      // dt_vac is computed later in this function; derive it locally here to
      // match the definition used downstream.
      const double dt_vac_local = (units_lj ? nevery * update->dt
                                            : nevery * update->dt * dt_to_ps);
      const std::string fname = std::string(prefix) + ".vac.multitau";
      FILE *fmt = fopen(fname.c_str(), "w");
      if (fmt) {
        fprintf(fmt, "# multi-tau VAC (per-level diagnostic) — "
                     "M=%d P=%d S=%d L=%d\n", mt_M, mt_P, mt_S, Lreq);
        fprintf(fmt, "# columns: level  stride  k  time_ps  vac\n");
        const int k_min_emit = (MP - 1) / mt_S + 1;
        // Nyquist time threshold from `maxfreq` (when set): lags past it
        // carry only frequencies above maxfreq, so drop them.  Units match
        // t_ps: ps for non-LJ, tau for LJ.
        double t_max_keep = std::numeric_limits<double>::infinity();
        if (maxfreq > 0.0) {
          const double c_cm_ps = VLIGHT * 100.0 * 1e-12;       // cm/ps
          t_max_keep = units_lj ? 1.0 / (2.0 * maxfreq)
                                : 1.0 / (2.0 * maxfreq * c_cm_ps);
        }
        for (int Lev = 0; Lev < Lreq; Lev++) {
          const int K_L = (int)c_per_level[Lev].size();
          const int k_lo = (Lev == 0) ? 0 : k_min_emit;
          for (int k = k_lo; k < K_L; k++) {
            const double t_ps = (double)k * stride_per_level[Lev] * dt_vac_local;
            if (t_ps > t_max_keep) break;   // t_ps monotonic in k at fixed Lev
            fprintf(fmt, "%5d  %8ld  %5d  %14.6e  %14.6e\n",
                    Lev, stride_per_level[Lev], k, t_ps, c_per_level[Lev][k]);
          }
        }
        fclose(fmt);
      }
      FIXXPT_LOG(
        "FixXPT::{}-{}: multi-tau correlator (L={}, M+P={}, S={}, "
        "per-step ring-buffer) — {} merged lags interp-linearised onto "
        "uniform vac[ns={}]; diagnostic written to "
        "{}.vac.multitau\n",
        id, group->names[igroup],
        Lreq, MP, mt_S, (int)t_merged.size(), ns, prefix);
    }
#ifdef FIX_XPT_DEBUG_VERIFY
    if (me == 0)
      utils::logmesg(lmp, "FixXPT::{}-{} verify: count_seen L0 scalar {} (ns {})\n",
                     id, group->names[igroup], mt_count_seen.empty() ? -1L : mt_count_seen[0], ns);
#endif
  } else {
    FIXXPT_TIME(t_vac_atom);
    // ── Default Wiener-Khinchin path ─────────────────────────────────
    std::vector<double> pwr_local(N_fft, 0.0);
    if (buffer_precision == BUFFER_FP32) {
      pwr_from_atoms_f(vel_buf_f, group_slots, mass_buf, pwr_local,
                       fft_buf, fft_vac, ns);
    } else {
      pwr_from_atoms(vel_buf, group_slots, mass_buf, pwr_local,
                     fft_buf, fft_vac, ns);
    }
    std::vector<double> pwr_global(N_fft, 0.0);
    MPI_Allreduce(pwr_local.data(), pwr_global.data(), N_fft,
                  MPI_DOUBLE, MPI_SUM, world);
    MPI_Allreduce(&mass_local, &mass_global, 1, MPI_DOUBLE, MPI_SUM, world);
    vac_from_pwr(pwr_global, vac, fft_buf, fft_vac, ns);
  }

  // Spatial dimension (cage prefactor 1/d, anharmonic law, 2D-aware HS).
  const int spat_dim = domain->dimension;

  // ── 5. Temperature from VAC(0) ────────────────────────────────────────
  // Equipartition: vac[0] * vac_to_jmol = DOF * R * T  (non-LJ)
  //   Lj: vac[0] = DOF * T  (kB=1)

  // fix_dof saved at window start, consistent with group_slots; recomputing
  // from the current mask would be wrong for dynamic groups.
  bigint fix_dof = fix_dof_window;

  double DOF = (double)(domain->dimension * ng_global) - (double)fix_dof;
  dof_last_window = DOF;   // surfaced in .thermo "DOF" column
  double T_vac;
  if (units_lj)
    T_vac = (DOF > 0) ? vac[0] / DOF : 0.0;
  else
    T_vac = (DOF > 0) ? (vac[0] * vac_to_jmol) / (R * DOF) : 0.0;
  double T = T_vac;
  if (T <= 0.0) return;

  // ── 6. DOS via cosine transform ──────────────────────────────────────
  // dt_vac: effective time step between frames (in ps for non-LJ, τ for LJ)
  double dt_vac;
  if (units_lj) dt_vac = nevery * update->dt;          // τ
  else          dt_vac = nevery * update->dt * dt_to_ps; // native time → ps

  // vac_norm normalizes sig = vac * vac_norm so that sig[0] = 2*DOF.
  //   Non-LJ: vac[0] = DOF*R*T / vac_to_jmol  → vac_norm = 2*vac_to_jmol/(R*T)
  //   Lj:     vac[0] = DOF*T                   → vac_norm = 2/T
  double vac_norm;
  if (units_lj) vac_norm = 2.0 / T;
  else          vac_norm = 2.0 * vac_to_jmol / (R * T);

  // Zero-padded cosine transform (inverse FFT on the 2*ns array):
  //   Dos[j] = sum_k sig[k] * cos(2π·j·k / (2·ns)) * dos_factor
  //   real/metal: dos_factor = dt_ps * VLIGHT * 1e-10,  dnu = 1/(2*ns*dt_ps*VLIGHT*1e-10)
  //   Lj:         dos_factor = dt_lj,                   dnu = 1/(2*ns*dt_lj)
  double dos_factor, dnu;
  if (units_lj) {
    dos_factor = dt_vac;
    dnu        = 1.0 / ((double)(2 * ns) * dt_vac);
  } else {
    dos_factor = dt_vac * VLIGHT * 1e-10;
    dnu        = 1.0 / ((double)(2 * ns) * dt_vac * VLIGHT * 1e-10);
  }

  int nused = ns + 1;   // one-sided spectrum: j = 0..ns inclusive

#ifdef FIX_XPT_DEBUG
  t0 = MPI_Wtime();
#endif
  std::vector<double> dos;
  dos_from_vac(vac, vac_norm, dos, fft_buf, fft_vac, ns, nused, dos_factor);
#ifdef FIX_XPT_DEBUG
  t_dos += MPI_Wtime() - t0;
#endif

  double s0 = dos[0];   // zero-frequency DOS

  // ── 6. 2PT partitioning: K → fluidicity ─────────────────────────────
  double V_box;
  if (volume_style == VOL_VARIABLE)
    V_box = input->variable->compute_equal(volume_varindex);
  else if (volume_style == VOL_CONSTANT)
    V_box = user_volume;
  else
    V_box = domain->xprd * domain->yprd * domain->zprd;  // Å³ or σ³
  double nmol  = (double)ng_global;


  double K = 0.0;
  if (s0 > 0.0 && T > 0.0 && mass_global > 0.0 && V_box > 0.0) {
    if (units_lj) {
      // K in LJ reduced units (kB=1, V in σ³, S(0) in τ, m* = mass/atom):
      //   K = (S(0)/N) * sqrt(π·T/m*) * (2/9) * (N/V)^(1/3) * (6/π)^(2/3)
      double m_star = mass_global / nmol;   // per-atom mass in LJ units (typically 1.0)
      K = (s0 / nmol) * sqrt(PI * T / m_star) * 2.0 / 9.0
          * pow(nmol / V_box, 1.0 / 3.0) * pow(6.0 / PI, 2.0 / 3.0);
    } else {
      // K in SI (all non-LJ):
      //   K = (s0[cm/mol]/VLIGHT*1e-2) / N * sqrt(π*NA*kB*T / (M[kg/mol]/N))
      //       * 2/9 * (N/V[Å³])^(1/3) * 1e10 * (6/π)^(2/3)
      // All non-LJ quantities normalised to g/mol (mass) and Å³ (volume).
      double mass_gmol = mass_global * mass_to_gmol;    // native → g/mol
      double V_angst3  = V_box      * vol_to_angst3;    // native → Å³
      double s0_si     = s0 / VLIGHT * 1e-2;
      double mass_kg   = mass_gmol * 1e-3;              // g/mol → kg/mol
      double sqrt_term = sqrt(PI * NA * KB * T / (mass_kg / nmol));
      double rho_term  = pow(nmol / V_angst3, 1.0 / 3.0) * 1e10 * pow(6.0 / PI, 2.0 / 3.0);
      K = s0_si / nmol * sqrt_term * 2.0 / 9.0 * rho_term;
    }
  }


  // Mode 1 (1PT): all-solid — suppress gas component for monoatomic too.
  double f    = (mol_mode != 1 && K > 0.0) ? search2pt(K) : 0.0;

  double y_ref= (K > 0.0 && f > 0.0) ? pow(f / K, 1.5) : 0.0;

  // Mode 3 (Desjarlais): find Bg for monoatomic DOS.
  double Bg_mono = (mol_mode == 3 && f > 0.0 && !dos.empty())
                   ? refine_Bg_des(dos, s0, dnu, nmol, f) : 0.0;

  // Monoatomic gas DOS array: all-zero for 1PT, Lorentzian / Desjarlais
  // (via Bg_mono) otherwise.
  std::vector<double> gas_mono;
  if (mol_mode == 1) {
    gas_mono.assign(dos.size(), 0.0);
  } else {
    gas_mono = build_gas_arr(dos, dnu, s0, f, Bg_mono, nmol);
  }

#ifdef FIX_XPT_DEBUG
  t0 = MPI_Wtime();
#endif
  // ── 7. HSDF and TTDF ─────────────────────────────────────────────────
  double hsdf = 0.0, ttdf = 0.0;
  for (int j = 0; j < nused; j++) {
    double sg_j = gas_mono[j];
    double w = (j == 0 || j == nused - 1) ? 0.5 : 1.0;
    hsdf += w * sg_j * dnu;
    ttdf += w * dos[j] * dnu;
  }

  double y = (ttdf > 0.0) ? y_ref * (hsdf / ttdf) : 0.0;
  if (y > 0.74) { f = 0.0; y = 0.0; }

  // ── 8. Thermodynamic integration ─────────────────────────────────────
  //
  // LJ units: classical only (quantum weights require ħ, undefined in LJ).
  //   Solid: wE = 1 (kT/mode), wCv = 1 (kB/mode); S undefined → 0.
  //   Gas:   wE = 0.5, wCv = 0.5; S undefined → 0.
  //   Output: E* in ε·N_atoms, Cv* in kB·N_atoms.
  //
  // Real/metal: quantum and (optionally) classical.
  //   Solid: Einstein oscillator with u = PLANCK*ν/T.
  //   Gas:   Hard-sphere + Sackur-Tetrode entropy.
  //   Output: E in kJ/mol, S in J/(mol·K), Cv in J/(mol·K), A in kJ/mol.

  double S_q = 0.0, A_q = 0.0, E_q = 0.0, Cv_q = 0.0;
  double S_c = 0.0, A_c = 0.0, E_c = 0.0, Cv_c = 0.0;
  double mu_q = 0.0, mu_c = 0.0;
  // μ_correction accumulated per 2PT-HS component (only gas adds PV term)
  // Units: same as A_q (kJ/mol for real/metal; ε for lj).
  double mu_pv_correction = 0.0;
  double ZPE_q = 0.0;
  double D   = 0.0;

  if (units_lj) {
    // ── LJ quantum + classical path (ħ*=1, h*=2π, kB*=1) ─────────────
    // Quantum parameter: u = ħ*ω/(kB*T*) = 2π·ν*/T*  (ν* in 1/τ units)
    //
    // Gas entropy uses Sackur-Tetrode with h*=2π:
    //   λ* = h*/√(2π m* T*) = √(2π/(m*T*))
    //   ST* = 2.5 + ln[(V*/N_hs) * (m*T*/(2π))^(3/2)]
    double nmol_hs = hsdf / 3.0;
    double mass_per_atom = mass_global / nmol;
    double ws_hs = hs_entropy_lj(y, mass_per_atom, nmol_hs, T, V_box, hbar_star,
                                  hs_entropy_mode);

    double E_s_q=0.0, S_s_q=0.0, Cv_s_q=0.0, ZPE_s_q=0.0;
    double E_s_c=0.0, S_s_c=0.0, Cv_s_c=0.0;

    for (int j = 0; j < nused; j++) {
      double nu   = j * dnu;
      double sg_j = gas_mono[j];
      double ss_j = dos[j] - sg_j;
      double w = (j == 0 || j == nused - 1) ? 0.5 : 1.0;

      if (ss_j <= 0.0 || nu == 0.0) {
        E_s_c += w * ss_j * dnu * 0.5;   // classical zero-freq: kT/2
        continue;
      }

      double u   = 2.0 * PI * hbar_star * nu / T;   // ħ*ω/(kB*T*)
      double eu  = exp(u);
      double em1 = eu - 1.0;
      double wE_q  = 0.5 * u * (eu + 1.0) / em1;
      double wS_q  = u / em1 - log(1.0 - 1.0 / eu);
      double wCv_q = u * u * eu / (em1 * em1);
      double wE_c  = 1.0;
      double wS_c  = 1.0 - log(u);
      double wCv_c = 1.0;

      E_s_q  += w * ss_j * wE_q  * dnu;
      S_s_q  += w * ss_j * wS_q  * dnu;
      Cv_s_q += w * ss_j * wCv_q * dnu;
      ZPE_s_q += w * ss_j * 0.5 * u * dnu;
      E_s_c  += w * ss_j * wE_c  * dnu;
      S_s_c  += w * ss_j * wS_c  * dnu;
      Cv_s_c += w * ss_j * wCv_c * dnu;
    }

    double E_g  = hsdf * 0.5;
    double Cv_g = hsdf * 0.5;
    double S_g  = hsdf * ws_hs;

    // Units: S* [kB/group], E* [ε/group], Cv* [kB/group], A* [ε/group]
    S_q  = S_s_q + S_g;
    E_q  = T * (E_s_q + E_g);
    Cv_q = Cv_s_q + Cv_g;
    A_q  = E_q - T * S_q;
    // ZPE = 0 for monoatomic: "ZPE" is intramolecular bonded vibrational
    // zero-point energy; the solid DoS here is cage/phonon motion, not bonds.
    ZPE_q = 0.0;
    (void)ZPE_s_q;

    S_c  = S_s_c + S_g;
    E_c  = T * (E_s_c + E_g);
    Cv_c = Cv_s_c + Cv_g;
    A_c  = E_c - T * S_c;

    // Chemical potential: μ = A + (HSDF · T · Z(y) / 3) [LJ: kB=1 so RT→T]
    // Solid component contributes no PV term; only the gas (HSDF) does.
    // When use_sim_z_mode=1, Z_sim from the simulation virial overrides Z_CS.
    {
      double Z_use = (use_sim_z_mode && ng_global > 0)
        ? z_sim_from_pressure(P_avg, V_box, ng_global, T,
                                /*units_lj=*/1, /*vol_to_angst3=*/1.0)
        : hs_compressibility(y);
      mu_pv_correction = hsdf * T * Z_use / 3.0;
      mu_q = A_q + mu_pv_correction;
      mu_c = A_c + mu_pv_correction;
    }

    // Diffusivity: D* [σ²/τ] = S_g(0)[τ] * T / (12 * M_lj)
    if (f > 0.0 && mass_global > 0.0 && T > 0.0)
      D = s0 * T / (12.0 * mass_global);

  } else {
    // ── non-LJ (real/metal/si/cgs/micro/nano/electron) quantum + classical path ──
    double V_angst3 = V_box * vol_to_angst3;          // native → Å³
    double V_m3     = V_angst3 * 1e-30;               // Å³ → m³
    double mass_gmol = mass_global * mass_to_gmol;     // native → g/mol

    // HS gas entropy weight (Sackur-Tetrode + Carnahan-Starling)
    // nmol_hs = gas molecule count = hsdf/3
    double nmol_hs = hsdf / 3.0;
    double ws_hs = hs_entropy(y, mass_gmol / nmol, nmol_hs, T, V_m3,
                                hs_entropy_mode);

    double E_s_q=0, S_s_q=0, Cv_s_q=0, ZPE_s_q=0;
    double E_s_c=0, S_s_c=0, Cv_s_c=0;

    for (int j = 0; j < nused; j++) {
      double nu   = j * dnu;
      double sg_j = gas_mono[j];
      double ss_j = dos[j] - sg_j;

      double w = (j == 0 || j == nused - 1) ? 0.5 : 1.0;

      if (ss_j <= 0.0 || nu == 0.0) {
        E_s_c += w * ss_j * dnu * 0.5;   // classical zero-freq: kT/2
        continue;
      }

      double u   = PLANCK * nu / T;
      double eu  = exp(u);
      double em1 = eu - 1.0;
      double wE_q  = 0.5 * u * (eu + 1.0) / em1;
      double wS_q  = u / em1 - log(1.0 - 1.0 / eu);
      double wCv_q = u * u * eu / (em1 * em1);
      double wE_c  = 1.0;
      double wS_c  = 1.0 - log(u);
      double wCv_c = 1.0;

      E_s_q  += w * ss_j * wE_q  * dnu;
      S_s_q  += w * ss_j * wS_q  * dnu;
      Cv_s_q += w * ss_j * wCv_q * dnu;
      ZPE_s_q += w * ss_j * 0.5 * u * dnu;

      E_s_c  += w * ss_j * wE_c  * dnu;
      S_s_c  += w * ss_j * wS_c  * dnu;
      Cv_s_c += w * ss_j * wCv_c * dnu;
    }

    double E_g  = hsdf * 0.5, Cv_g = hsdf * 0.5;
    double S_g  = hsdf * ws_hs;

    S_q  = R * (S_s_q + S_g);
    E_q  = R * T * (E_s_q + E_g) * 1e-3;
    Cv_q = R * (Cv_s_q + Cv_g);
    A_q  = E_q - T * S_q * 1e-3;
    // ZPE = 0 for monoatomic — see LJ-units branch above.
    ZPE_q = 0.0;
    (void)ZPE_s_q;

    S_c  = R * (S_s_c + S_g);
    E_c  = R * T * (E_s_c + E_g) * 1e-3;
    Cv_c = R * (Cv_s_c + Cv_g);
    A_c  = E_c - T * S_c * 1e-3;

    // Chemical potential: μ = A + (HSDF · R · T · Z(y) / 3)  [kJ/mol]
    // Solid component contributes no PV term; only the gas (HSDF) does.
    // When use_sim_z_mode=1, Z_sim from the simulation virial overrides Z_CS.
    {
      double Z_use = (use_sim_z_mode && ng_global > 0)
        ? z_sim_from_pressure(P_avg, V_box, ng_global, T,
                                /*units_lj=*/0, vol_to_angst3)
        : hs_compressibility(y);
      mu_pv_correction = hsdf * R * T * Z_use / 3.0 * 1e-3;   // J/mol -> kJ/mol
      mu_q = A_q + mu_pv_correction;
      mu_c = A_c + mu_pv_correction;
    }

    // ── R2PT refinement (Sun 2017): override translational gas+solid entropy ──
    // Applied as a delta off the rigorous-HS value (monatomic / whole-group).
    // Real/metal units only (SI Sackur-Tetrode).
    if (refinement == REF_R2PT && !do_molecule && f > 0.0 && nmol > 0
        && !dos.empty()) {
      double S_r2 = r2pt_entropy(dnu, dos, nused, f, T,
                                 mass_gmol / nmol, V_angst3 / nmol,
                                 (double)nmol, r2pt_delta);
      if (!std::isnan(S_r2)) {
        double S_rig  = S_q / (nmol * R);          // current rigorous S* [k_B/atom]
        double dS_ext = (S_r2 - S_rig) * nmol;     // extensive ΔS/R
        double A_ex   = -dS_ext * R * T * 1e-3;    // kJ/mol
        S_q += dS_ext*R;  A_q += A_ex;  mu_q += A_ex;
        S_c += dS_ext*R;  A_c += A_ex;  mu_c += A_ex;
        if (me == 0)
          FIXXPT_LOG("FixXPT::{}-{}: R2PT(d={:.2f}) S*={:.4f} kB/atom "
                     "(rigorous-HS {:.4f}, dS*={:+.4f})\n",
                     id, group->names[igroup], r2pt_delta, S_r2, S_rig, S_r2 - S_rig);
      } else if (me == 0) {
        utils::logmesg(lmp, "FixXPT::{}-{}: WARNING - R2PT Eq A8 has no root "
                            "(f={:.3f}); translational entropy left at "
                            "rigorous-HS.\n", id, group->names[igroup], f);
      }
    }

    // ── 3PT cage-memory entropy: translational cage, monatomic ──
    // Parameter-free ΔS = (1/d)·g(f)·∫cage·(1−w)·(W_g−W_s)dν added to the
    // rigorous-HS entropy.
    if (cage_entropy && !do_molecule && f > 0.0 && nmol > 0 && !vac.empty()
        && !dos.empty()) {
      std::vector<double> dos_pa(nused), gas_pa(nused);
      for (int j = 0; j < nused; j++) { dos_pa[j] = dos[j]/nmol; gas_pa[j] = gas_mono[j]/nmol; }
      double dS = cage_memory_entropy(dt_vac, vac, (int)vac.size(), dnu,
                                      dos_pa, gas_pa, nused, T,
                                      mass_gmol/nmol, V_angst3/nmol,
                                      1.0/spat_dim, spat_dim,
                                      std::numeric_limits<double>::quiet_NaN(),
                                      &cage_mono_w);
      for (double &v : cage_mono_w) v *= nmol;       // per-atom → extensive (.pwr)
      if (dS != 0.0) {
        double dS_ext = dS * nmol;
        double A_ex   = -dS_ext * R * T * 1e-3;
        S_q += dS_ext*R;  A_q += A_ex;  mu_q += A_ex;
        S_c += dS_ext*R;  A_c += A_ex;  mu_c += A_ex;
        s_cage_trans_last = dS_ext * R;   // extensive; ×norm at write time
        if (me == 0)
          FIXXPT_LOG("FixXPT::{}-{}: 3PT cage (trans) dS={:.4g} J/mol/K/atom "
                     "(p=1/{}, d={})\n", id, group->names[igroup],
                     dS*R, spat_dim, spat_dim);
      }
    }


    // Diffusivity [cm²/s]: S_g(0)[cm/mol] * R * T / (12 * VLIGHT * M[g/mol]) * 1e5
    if (f > 0.0 && mass_gmol > 0.0 && T > 0.0)
      D = s0 * R * T / (12.0 * VLIGHT * mass_gmol) * 1e5;
  }

  // ── 9. E_md and Cv from energy fluctuations ──────────────────────────
  // E_md   = <KE+PE> over the window [kJ/mol, full group, or ε for lj]
  // Cv_q   = σ²(E)/(R·T²) + (Cv_q_dos − Cv_c_dos)
  // E_q,Aq shifted by Eo = E_md − E_c  (DOS energy vs actual MD energy)
  double Cv_fluct  = 0.0;  // full-group fluctuation Cv; also used in molecular section
  bool   has_e_corr = false;
  double E_md = 0.0;
  if (pe_available && E_count > 1 && T > 0.0) {
    double E_avg   = E_sum / E_count;
    double Var_E   = E_sq_sum / E_count - E_avg * E_avg;
    E_md = E_avg;   // kJ/mol full group (or ε)
    has_e_corr = true;

    // Cv_fluct: full group, J/mol/K (or kB for lj)
    if (units_lj)
      Cv_fluct = Var_E / (T * T);            // kB, full group
    else
      Cv_fluct = Var_E * 1e6 / (R * T * T); // Var in kJ²/mol² → J/mol/K

    // Energy correction: shift E_q, A_q, μ_q to reproduce the actual MD energy.
    // Classical quantities (E_c, A_c, S_c, μ_c) are intentionally NOT shifted —
    // the classical DOS already reproduces equipartition; only the quantum DOS
    // energy needs reconciling with the MD energy.
    double Eo = E_md - E_c;  // kJ/mol, full group
    E_q += Eo;
    A_q += Eo;
    mu_q += Eo;

    // Cv: fluctuation classical + DOS quantum correction
    Cv_q = Cv_fluct + (Cv_q - Cv_c);
  }
  // E_sum / E_sq_sum / E_count are reset in end_of_step's end-of-window
  // branch, so intermediate snapshots keep accumulating the window average.

#ifdef FIX_XPT_DEBUG
  t_thermo += MPI_Wtime() - t0;
#endif

  // ── 10. Normalize (optional) ───────────────────────────────────────────
  double norm = do_normalize ? 1.0 / (do_molecule ? (double)nmol_group : (double)ng_global) : 1.0;
  S_q  *= norm; E_q  *= norm; Cv_q *= norm; A_q *= norm;
  S_c  *= norm; E_c  *= norm; Cv_c *= norm; A_c *= norm;
  ZPE_q *= norm;
  mu_q *= norm; mu_c *= norm;
  mu_pv_correction *= norm;
  double E_md_out = E_md * norm;

  // ── 11a. Molecular 2PT decomposition (trans + rot + vib) ──────────────
  double S_trans = 0.0, S_rot = 0.0, S_vib = 0.0, f_rot = 0.0, D_rot = 0.0;
  double S_trans_gas = 0.0, S_trans_solid = 0.0;
  double S_rot_gas   = 0.0, S_rot_solid   = 0.0;
  // Molecular energy/Cv accumulators — normalized same as entropy; assembled after all three components
  double Eq_t = 0, Cvq_t = 0, Ec_t = 0, Cvc_t = 0, Sc_t = 0;
  double Eq_r = 0, Cvq_r = 0, Ec_r = 0, Cvc_r = 0, Sc_r = 0;
  double Eq_v = 0, Cvq_v = 0, Ec_v = 0, Cvc_v = 0, Sc_v = 0;
  double ZPE_t = 0.0, ZPE_r = 0.0;  // molecular ZPE components (hoisted for scope)
  // Per-component HS packing fractions and gas-DOF counts (hoisted for μ_q calc)
  double y_trans_w = 0.0;
  double hsdf_trans_w = 0.0, hsdf_rot_w = 0.0;

  // Molecular VAC/DOS/gas vectors, filled via std::move at the end of the
  // molecular block for write_pwr_vac().  The gas arrays carry the partition
  // actually built (Desjarlais or Lorentzian), so the .pwr writer prints the
  // true gas/solid split rather than an analytic Lorentzian refit.
  std::vector<double> vac_trans_w, vac_rot_w, vac_vib_w;
  std::vector<double> dos_trans_w, dos_rot_w, dos_vib_w;
  std::vector<double> gas_trans_w, gas_rot_w;

  // ── Buffer-sharing: consumers read the OWNER's molecular inertia ────────
  // Only the buffer owner runs accumulate_mol_frame(), so a consumer's own
  // mol_inertia stays zero; without borrowing the owner's, the molecular
  // block below would be silently skipped on consumers.  The owner has a
  // lower modify-list index, so its run_analysis ran first this step and
  // owner->mol_inertia is complete and stable here.  We read rather than
  // alias the pointer: aliasing would let the consumer's zeroing / realloc
  // paths clobber the owner's array.
  double **mol_inertia_use       = mol_inertia;
  int      mol_inertia_count_use = mol_inertia_count;
  if (!owns_buffer && buffer_owner && do_molecule) {
    mol_inertia_use       = buffer_owner->mol_inertia;
    mol_inertia_count_use = buffer_owner->mol_inertia_count;
  }

  if (do_molecule && nmol_group > 0 && mol_inertia_count_use > 0) {

    // Average inertia per molecule.  Distributed layout: this rank's home
    // molecules, global index mol_lo[me] + ml.
    const int nm_rank  = distributed() ? nm_home : nmol_group;
    const int m_offset = distributed() ? mol_lo[me] : 0;
    std::vector<double> I_avg(nm_rank * 3, 0.0);
    for (int m = 0; m < nm_rank; m++)
      for (int d = 0; d < 3; d++)
        I_avg[m*3+d] = mol_inertia_use[m][d] / mol_inertia_count_use;

    // Translational VAC: mass-weighted COM velocities (mol buffers are already global)
    // Rotational VAC: anguv = V·(ω_body×√I), unit weight per molecule
    // All via Wiener-Khinchin FFT — O(nmol * N_fft * log N_fft)
#ifdef FIX_XPT_DEBUG
    t0 = MPI_Wtime();
#endif
    std::vector<double> vac_trans, vac_rot, vac_vib;
    if (correlator == CORR_MULTITAU && mt_molecular_active) {
      FIXXPT_TIME(t_vac_multitau);
      // ── Multi-tau molecular per-channel ──────────────────────────────
      // Trans/rot stream c_sum is already global on every rank; vib stream
      // c_sum is partial per rank (MPI-reduce at finalise).  Both go through
      // the standard interp-merge → uniform vac[ns] pipeline.
      auto stream_to_vac = [&](MultiTauStream& s, std::vector<double>& v_out) {
        const int MP = mt_MP;
        std::vector<std::vector<double>> c_per_level(mt_n_levels);
        std::vector<long> stride_per_level(mt_n_levels, 1);
        long stride = 1;
        for (int Lev = 0; Lev < mt_n_levels; Lev++) {
          stride_per_level[Lev] = stride;
          std::vector<double> c_global(MP, 0.0);
          if (s.is_distributed) {
            MPI_Allreduce(s.c_sum[Lev].data(), c_global.data(), MP,
                          MPI_DOUBLE, MPI_SUM, world);
          } else {
            c_global = s.c_sum[Lev];
          }
          const long ns_L = s.count_seen[Lev];
          const double inv_nsL = (ns_L > 0) ? 1.0 / (double)ns_L : 0.0;
          c_per_level[Lev].assign(MP, 0.0);
          for (int k = 0; k < MP; k++) c_per_level[Lev][k] = c_global[k] * inv_nsL;
          stride *= mt_S;
        }
        const int k_min_emit = (MP - 1) / mt_S + 1;
        std::vector<double> t_merged, c_merged;
        for (int Lev = 0; Lev < mt_n_levels; Lev++) {
          const int K_L  = (int)c_per_level[Lev].size();
          const int k_lo = (Lev == 0) ? 0 : k_min_emit;
          for (int k = k_lo; k < K_L; k++) {
            t_merged.push_back((double)k * (double)stride_per_level[Lev]);
            c_merged.push_back(c_per_level[Lev][k]);
          }
        }
        v_out.assign(ns, 0.0);
        if (t_merged.empty()) return;
        const double t_max = t_merged.back();
        size_t idx = 0;
        for (int j = 0; j < ns; j++) {
          const double tj = (double)j;
          if (tj > t_max) break;
          while (idx + 1 < t_merged.size() && t_merged[idx + 1] < tj) idx++;
          if (idx + 1 >= t_merged.size()) { v_out[j] = c_merged.back(); }
          else {
            const double t0 = t_merged[idx], t1 = t_merged[idx + 1];
            const double c0 = c_merged[idx], c1 = c_merged[idx + 1];
            const double w  = (t1 > t0) ? (tj - t0) / (t1 - t0) : 0.0;
            v_out[j] = c0 * (1.0 - w) + c1 * w;
          }
        }
      };
      stream_to_vac(mt_trans, vac_trans);
      stream_to_vac(mt_rot,   vac_rot);
      stream_to_vac(mt_vib,   vac_vib);
#ifdef FIX_XPT_DEBUG_VERIFY
      if (me == 0)
        utils::logmesg(lmp, "FixXPT::{}-{} verify: count_seen L0 scalar {} trans {} rot {} vib {}\n",
                       id, group->names[igroup],
                       mt_count_seen.empty() ? -1L : mt_count_seen[0],
                       mt_trans.count_seen.empty() ? -1L : mt_trans.count_seen[0],
                       mt_rot.count_seen.empty()   ? -1L : mt_rot.count_seen[0],
                       mt_vib.count_seen.empty()   ? -1L : mt_vib.count_seen[0]);
#endif
    } else {
      // ── FFT path (default) ───────────────────────────────────────────
      std::vector<double> pwr_trans(N_fft, 0.0), pwr_rot(N_fft, 0.0);
      if (distributed()) {
        // Each rank transforms its home molecules; one reduction per channel.
        std::vector<double> pwr_local(N_fft, 0.0);
        pwr_from_mols(com_vel_buf, nm_home, molmass_home, pwr_local, fft_buf, fft_vac, ns);
        MPI_Allreduce(pwr_local.data(), pwr_trans.data(), N_fft, MPI_DOUBLE, MPI_SUM, world);
        std::fill(pwr_local.begin(), pwr_local.end(), 0.0);
        pwr_from_mols(omega_buf, nm_home, molunit_home, pwr_local, fft_buf, fft_vac, ns);
        MPI_Allreduce(pwr_local.data(), pwr_rot.data(), N_fft, MPI_DOUBLE, MPI_SUM, world);
      } else {
        pwr_from_mols(com_vel_buf, nmol_group, molmass, pwr_trans, fft_buf, fft_vac, ns);
        pwr_from_mols(omega_buf,   nmol_group, std::vector<double>(nmol_group, 1.0), pwr_rot, fft_buf, fft_vac, ns);
      }
#ifdef FIX_XPT_DEBUG
      t_vac_mol += MPI_Wtime() - t0;
#endif

      vac_from_pwr(pwr_trans, vac_trans, fft_buf, fft_vac, ns);
      vac_from_pwr(pwr_rot,   vac_rot,  fft_buf, fft_vac, ns);

      // vac_vib: mass-weighted VAC of per-atom vibrational residuals (local atoms, needs MPI reduce)
#ifdef FIX_XPT_DEBUG
      t0 = MPI_Wtime();
#endif
      std::vector<double> pwr_vib_local(N_fft, 0.0);
      if (buffer_precision == BUFFER_FP32) {
        pwr_from_atoms_f(vib_vel_buf_f, group_slots, mass_buf,
                         pwr_vib_local, fft_buf, fft_vac, ns);
      } else {
        pwr_from_atoms(vib_vel_buf, group_slots, mass_buf,
                       pwr_vib_local, fft_buf, fft_vac, ns);
      }
#ifdef FIX_XPT_DEBUG
      t_vac_vib += MPI_Wtime() - t0;
#endif

      std::vector<double> pwr_vib_global(N_fft, 0.0);
      MPI_Allreduce(pwr_vib_local.data(), pwr_vib_global.data(), N_fft, MPI_DOUBLE, MPI_SUM, world);
      vac_from_pwr(pwr_vib_global, vac_vib, fft_buf, fft_vac, ns);
    }

    // DOF counts — vib is the residual after subtracting constraints (fix_dof),
    // trans, and rot.  This automatically handles fix shake, fix rigid, etc.
    int dof_rot_total = 0;
    for (int m = 0; m < nmol_group; m++) {
      int nr = mol_is_linear[m] ? 2 : 3;
      dof_rot_total += nr;
    }
    int dof_trans_total = 3 * nmol_group;
    int dof_vib_total   = (int)(domain->dimension * ng_global - fix_dof) - dof_trans_total - dof_rot_total;
    if (dof_vib_total < 0) dof_vib_total = 0;


    // Temperature per component, each from its own DOF (COM and rotational
    // DOF are never constrained by SHAKE, unlike the total T).
    double T_trans_mol = T_from_vac(vac_trans, dof_trans_total, 0.0);
    double T_rot_mol   = T_from_vac(vac_rot,   dof_rot_total,   0.0);
    // T_trans is the best SHAKE-unaffected estimate of physical T.
    double T_mol = (T_trans_mol > 0.0) ? T_trans_mol : T;

    // Cosine transform — each VAC component uses its own T
    double T_vib_mol = T_from_vac(vac_vib, dof_vib_total, 0.0);
    std::vector<double> dos_trans, dos_rot, dos_vib;
#ifdef FIX_XPT_DEBUG
    t0 = MPI_Wtime();
#endif
    auto vnorm = [&](double T_comp) -> double {
      return units_lj ? 2.0 / T_comp : 2.0 * vac_to_jmol / (R * T_comp);
    };
    if (dof_trans_total > 0 && T_trans_mol > 0.0)
      dos_from_vac(vac_trans, vnorm(T_trans_mol), dos_trans, fft_buf, fft_vac, ns, nused, dos_factor);
    if (dof_rot_total   > 0 && T_rot_mol   > 0.0)
      dos_from_vac(vac_rot,   vnorm(T_rot_mol),   dos_rot,   fft_buf, fft_vac, ns, nused, dos_factor);
    if (dof_vib_total   > 0 && T_vib_mol   > 0.0)
      dos_from_vac(vac_vib,   vnorm(T_vib_mol),   dos_vib,   fft_buf, fft_vac, ns, nused, dos_factor);
#ifdef FIX_XPT_DEBUG
    t_dos += MPI_Wtime() - t0;
#endif

    // ── 2PT for translational component ──────────────────────────────────
    double s0t = dos_trans.empty() ? 0.0 : dos_trans[0];
    double f_trans_mol = 0.0, D_trans = 0.0;
    // Trans K uses nmol_group and total molmass
    double mass_mol_total = 0.0;
    for (int m = 0; m < nmol_group; m++) mass_mol_total += molmass[m];



    if (s0t > 0.0 && T_trans_mol > 0.0 && mass_mol_total > 0.0) {
      double nmol_d = (double)nmol_group;
      if (units_lj) {
        double m_star = mass_mol_total / nmol_d;
        K = (s0t / nmol_d) * sqrt(PI*T_trans_mol/m_star) * 2.0/9.0
            * pow(nmol_d/V_box, 1.0/3.0) * pow(6.0/PI, 2.0/3.0);
      } else {
        double mass_mol_gmol = mass_mol_total * mass_to_gmol;  // native → g/mol
        double V_angst3_m    = V_box * vol_to_angst3;           // native → Å³
        double s0t_si  = s0t / VLIGHT * 1e-2;
        double mass_kg = mass_mol_gmol * 1e-3;
        double sqrt_t  = sqrt(PI * NA * KB * T_trans_mol / (mass_kg / nmol_d));
        double rho_t   = pow(nmol_d / V_angst3_m, 1.0/3.0) * 1e10 * pow(6.0/PI, 2.0/3.0);
        K = s0t_si / nmol_d * sqrt_t * 2.0/9.0 * rho_t;
      }

      // Mode 1 (1PT): all-solid — no gas component anywhere; skip search2pt.
      f_trans_mol = (mol_mode >= 2 && K > 0.0) ? search2pt(K) : 0.0;
    }


    // Mode 3: Desjarlais Bg for translational DOS.
    double Bg_trans = (mol_mode == 3 && f_trans_mol > 0.0 && !dos_trans.empty())
                      ? refine_Bg_des(dos_trans, s0t, dnu, (double)nmol_group, f_trans_mol) : 0.0;

    // Translational gas DoS: zero for 1PT, Lorentzian / Desjarlais otherwise.
    std::vector<double> gas_trans;
    if (dos_trans.empty() || mol_mode < 2) {
      gas_trans.assign(dos_trans.size(), 0.0);
    } else {
      gas_trans = build_gas_arr(dos_trans, dnu, s0t, f_trans_mol,
                                  Bg_trans, (double)nmol_group);
    }

    // Translational gas fraction and entropy
    double y_trans = 0.0, hsdf_trans = 0.0, ttdf_trans = 0.0;
    if (f_trans_mol > 0.0 && !dos_trans.empty()) {
      double y_ref_t = pow(f_trans_mol / K, 1.5);
      for (int j = 0; j < nused; j++) {
        double sg = gas_trans[j];
        double w = (j==0||j==nused-1) ? 0.5 : 1.0;
        hsdf_trans += w*sg*dnu;
        ttdf_trans += w*dos_trans[j]*dnu;
      }
      y_trans = (ttdf_trans > 0.0) ? y_ref_t * (hsdf_trans / ttdf_trans) : 0.0;
      if (y_trans > 0.74) { f_trans_mol = 0.0; y_trans = 0.0; }
    }

    // Translational entropy: integrate dos_trans
    {
      double nmol_hs_t = hsdf_trans / 3.0;
      double ws_hs_t;
      if (units_lj)
        ws_hs_t = hs_entropy_lj(y_trans, mass_mol_total/nmol_group,
                                 nmol_hs_t, T_trans_mol, V_box, hbar_star,
                                 hs_entropy_mode);
      else {
        double mass_mol_gmol = mass_mol_total * mass_to_gmol;
        ws_hs_t = hs_entropy(y_trans, mass_mol_gmol/nmol_group,
                              nmol_hs_t, T_trans_mol, V_box*vol_to_angst3*1e-30,
                              hs_entropy_mode);
      }

      double S_t_gas = 0.0, S_t_solid = 0.0;
      double Es_q_t = 0, Es_c_t = 0, Cvs_q_t = 0, Cvs_c_t = 0, Ss_c_t = 0, ZPE_s_t = 0;
      for (int j = 0; j < nused && !dos_trans.empty(); j++) {
        double nu = j * dnu;
        double sg = gas_trans[j];
        if (sg > dos_trans[j]) sg = dos_trans[j];
        double ss = dos_trans[j] - sg;
        double w = (j==0||j==nused-1) ? 0.5 : 1.0;
        if (ss <= 0.0 || nu == 0.0) {
          if (ss > 0.0) Es_c_t += w * ss * 0.5 * dnu;  // classical zero-freq: kT/2
          continue;
        }
        double u = units_lj ? 2.0*PI*hbar_star*nu/T_trans_mol : PLANCK*nu/T_trans_mol;
        double eu = exp(u), em1 = eu - 1.0;
        S_t_solid  += w * ss * (u/em1 - log(1.0-1.0/eu)) * dnu;
        Es_q_t    += w * ss * 0.5*u*(eu+1.0)/em1          * dnu;
        Es_c_t    += w * ss                                * dnu;
        Cvs_q_t   += w * ss * u*u*eu/(em1*em1)             * dnu;
        Cvs_c_t   += w * ss                                * dnu;
        Ss_c_t    += w * ss * (1.0 - log(u))               * dnu;
        ZPE_s_t   += w * ss * 0.5 * u                      * dnu;
      }
      S_t_gas = hsdf_trans * ws_hs_t;
      S_trans_gas   = units_lj ? S_t_gas   : R*S_t_gas;
      S_trans_solid = units_lj ? S_t_solid : R*S_t_solid;
      S_trans = S_trans_gas + S_trans_solid;
      // Hoist for μ_q calculation later
      y_trans_w     = y_trans;
      hsdf_trans_w  = hsdf_trans;

      {
        double Eg_t  = hsdf_trans * 0.5, Cvg_t = hsdf_trans * 0.5;
        Eq_t  = units_lj ? T_trans_mol*(Es_q_t+Eg_t)     : R*T_trans_mol*(Es_q_t+Eg_t)*1e-3;
        Cvq_t = units_lj ? (Cvs_q_t+Cvg_t)               : R*(Cvs_q_t+Cvg_t);
        Ec_t  = units_lj ? T_trans_mol*(Es_c_t+Eg_t)     : R*T_trans_mol*(Es_c_t+Eg_t)*1e-3;
        Cvc_t = units_lj ? (Cvs_c_t+Cvg_t)               : R*(Cvs_c_t+Cvg_t);
        Sc_t  = units_lj ? (Ss_c_t+S_t_gas)              : R*(Ss_c_t+S_t_gas);
      }
      ZPE_t = units_lj ? T_trans_mol*ZPE_s_t : R*T_trans_mol*ZPE_s_t*1e-3;

      // ── 3PT cage-memory entropy: translational cage, molecular ──
      // Added pre-normalization (extensive).
      if (cage_entropy && f_trans_mol > 0.0 && nmol_group > 0
          && !vac_trans.empty() && !dos_trans.empty()) {
        double mass_mol_gmol = mass_mol_total * mass_to_gmol;
        double V_a3 = V_box * vol_to_angst3;
        std::vector<double> dpa(nused), gpa(nused);
        for (int j = 0; j < nused; j++) {
          dpa[j] = dos_trans[j]/nmol_group; gpa[j] = gas_trans[j]/nmol_group;
        }
        double dS_t = cage_memory_entropy(dt_vac, vac_trans, (int)vac_trans.size(),
                          dnu, dpa, gpa, nused, T_trans_mol,
                          mass_mol_gmol/nmol_group, V_a3/nmol_group, 1.0/3.0, 3,
                          std::numeric_limits<double>::quiet_NaN(), &cage_trans_w);
        for (double &v : cage_trans_w) v *= nmol_group;   // per-mol → extensive
        if (dS_t != 0.0) {
          double dS_ext = dS_t * nmol_group;
          S_trans += dS_ext*R;  S_trans_solid += dS_ext*R;  Sc_t += dS_ext*R;
          s_cage_trans_last = dS_ext * R;   // extensive; ×norm at write time
          if (me == 0)
            FIXXPT_LOG("FixXPT::{}-{}: 3PT cage (mol trans) dS={:.4g} J/mol/K/mol\n",
                       id, group->names[igroup], dS_t*R);
        }
      }

      // ── R2PT refinement (Sun 2017): molecular translational channel ──
      // Override the trans gas+solid entropy with the revised-2PT value, as a
      // delta off rigorous-HS (TRANS only).
      if (refinement == REF_R2PT && !units_lj && f_trans_mol > 0.0
          && nmol_group > 0 && !dos_trans.empty()) {
        double mass_mol_gmol = mass_mol_total * mass_to_gmol;
        double V_a3 = V_box * vol_to_angst3;
        double S_r2 = r2pt_entropy(dnu, dos_trans, nused, f_trans_mol, T_trans_mol,
                                   mass_mol_gmol/nmol_group, V_a3/nmol_group,
                                   (double)nmol_group, r2pt_delta);
        if (!std::isnan(S_r2)) {
          double S_rig  = S_trans / (nmol_group * R);   // per-mol rigorous trans S*
          double dS_ext = (S_r2 - S_rig) * nmol_group;
          S_trans += dS_ext*R;  S_trans_solid += dS_ext*R;  Sc_t += dS_ext*R;
          if (me == 0)
            FIXXPT_LOG("FixXPT::{}-{}: R2PT(d={:.2f} mol trans) S*={:.4f} "
                       "(rigorous {:.4f})\n", id, group->names[igroup],
                       r2pt_delta, S_r2, S_rig);
        }
      }

      if (do_normalize) {
        S_trans_gas /= nmol_group; S_trans_solid /= nmol_group; S_trans /= nmol_group;
        Eq_t /= nmol_group; Cvq_t /= nmol_group;
        Ec_t /= nmol_group; Cvc_t /= nmol_group; Sc_t /= nmol_group;
        ZPE_t /= nmol_group;
      }

      // Translational diffusivity
      if (units_lj && mass_mol_total > 0.0)
        D_trans = s0t * T_trans_mol / (12.0 * mass_mol_total);
      else if (!units_lj && mass_mol_total > 0.0)
        D_trans = s0t * R * T_trans_mol / (12.0 * VLIGHT * (mass_mol_total * mass_to_gmol)) * 1e5;
      D = D_trans;   // overwrite monoatomic D with molecular translational D
      f = f_trans_mol;
    }

    // ── Rotational component ──────────────────────────────────────────────
    double s0r = dos_rot.empty() ? 0.0 : dos_rot[0];
    f_rot = 0.0;

    if (mol_mode >= 2 && s0r > 0.0 && T_rot_mol > 0.0 && mass_mol_total > 0.0) {
      double nmol_d = (double)nmol_group;
      // K_rot uses same formula as K_trans (ANGUL mode):
      // Anguv has same units as mass-weighted translational velocity, so molecular mass
      // is the correct inertia scale in the hard-sphere diffusion formula.
      if (units_lj) {
        double m_star = mass_mol_total / nmol_d;
        K = (s0r / nmol_d) * sqrt(PI*T_rot_mol/m_star) * 2.0/9.0
            * pow(nmol_d/V_box, 1.0/3.0) * pow(6.0/PI, 2.0/3.0);
      } else {
        double mass_mol_gmol = mass_mol_total * mass_to_gmol;
        double V_angst3_r    = V_box * vol_to_angst3;
        double s0r_si  = s0r / VLIGHT * 1e-2;
        double mass_kg = mass_mol_gmol * 1e-3;
        double sqrt_r  = sqrt(PI * NA * KB * T_rot_mol / (mass_kg / nmol_d));
        double rho_t   = pow(nmol_d / V_angst3_r, 1.0/3.0) * 1e10 * pow(6.0/PI, 2.0/3.0);
        K = s0r_si / nmol_d * sqrt_r * 2.0/9.0 * rho_t;
      }
      f_rot = (K > 0.0) ? search2pt(K) : 0.0;
    }

    // Mode 3: Desjarlais Bg for rotational DOS.
    double Bg_rot = (mol_mode == 3 && f_rot > 0.0 && !dos_rot.empty())
                    ? refine_Bg_des(dos_rot, s0r, dnu, (double)nmol_group, f_rot) : 0.0;

    // Rotational gas DoS: zero for 1PT, Lorentzian / Desjarlais otherwise.
    std::vector<double> gas_rot;
    if (dos_rot.empty() || mol_mode < 2) {
      gas_rot.assign(dos_rot.size(), 0.0);
    } else {
      gas_rot = build_gas_arr(dos_rot, dnu, s0r, f_rot,
                                Bg_rot, (double)nmol_group);
    }

    // Rotational gas entropy
    {
      double hsdf_rot = 0.0, y_rot = 0.0;
      if (f_rot > 0.0 && !dos_rot.empty()) {
        double y_ref_r = pow(f_rot / K, 1.5);
        double ttdf_rot = 0.0;
        for (int j = 0; j < nused; j++) {
          double sg = gas_rot[j];
          double w = (j==0||j==nused-1) ? 0.5 : 1.0;
          hsdf_rot += w*sg*dnu;
          ttdf_rot += w*dos_rot[j]*dnu;
        }
        y_rot = (ttdf_rot > 0.0) ? y_ref_r * (hsdf_rot / ttdf_rot) : 0.0;
        if (y_rot > 0.74) { f_rot = 0.0; y_rot = 0.0; }
      }

      // Rotational gas entropy weight: rigid-rotor Sackur-Tetrode
      // For each molecule, S_rot_gas per mode = hs_entropy_rot(...)
      // hs_entropy_rot is nonlinear in I, so each molecule is evaluated where
      // its inertia lives and the distributed partial sums are reduced.
      double ws_rot_gas = 0.0;
      for (int ml = 0; ml < nm_rank; ml++) {
        const int m = m_offset + ml;
        int istart = mol_is_linear[m] ? 1 : 0;
        int nrot_m = mol_is_linear[m] ? 2 : 3;
        double I1 = I_avg[ml*3 + istart];
        double I2 = (nrot_m >= 2) ? I_avg[ml*3 + istart+1] : I1;
        double I3 = (nrot_m >= 3) ? I_avg[ml*3 + istart+2] : 0.0;
        if (units_lj) {
          double I_s = (I1*I2 > 0) ? sqrt(I1*I2) : I1;
          ws_rot_gas += hs_entropy_rot_lj(y_rot, I_s, hsdf_rot/nrot_m,
                                          T_rot_mol, rotsym, mol_is_linear[m], hbar_star);
        } else {
          ws_rot_gas += hs_entropy_rot(y_rot, I1, I2, I3,
                                       hsdf_rot/nrot_m, T_rot_mol, rotsym, mol_is_linear[m]);
        }
      }
      if (distributed()) {
        double ws_sum = 0.0;
        MPI_Allreduce(&ws_rot_gas, &ws_sum, 1, MPI_DOUBLE, MPI_SUM, world);
        ws_rot_gas = ws_sum;
      }
      ws_rot_gas /= nmol_group;  // average per-molecule, in same units as dos integration

      double Sr_gas = 0.0, Sr_solid = 0.0;
      double Es_q_r = 0, Es_c_r = 0, Cvs_q_r = 0, Cvs_c_r = 0, Ss_c_r = 0, ZPE_s_r = 0;
      for (int j = 0; j < nused && !dos_rot.empty(); j++) {
        double nu = j * dnu;
        double sg = gas_rot[j];
        if (sg > dos_rot[j]) sg = dos_rot[j];
        double ss = dos_rot[j] - sg;
        double w = (j==0||j==nused-1) ? 0.5 : 1.0;
        if (ss <= 0.0 || nu == 0.0) {
          if (ss > 0.0) Es_c_r += w * ss * 0.5 * dnu;
          continue;
        }
        double u = units_lj ? 2.0*PI*hbar_star*nu/T_rot_mol : PLANCK*nu/T_rot_mol;
        double eu = exp(u), em1 = eu - 1.0;
        Sr_solid  += w * ss * (u/em1 - log(1.0-1.0/eu)) * dnu;
        Es_q_r   += w * ss * 0.5*u*(eu+1.0)/em1          * dnu;
        Es_c_r   += w * ss                                * dnu;
        Cvs_q_r  += w * ss * u*u*eu/(em1*em1)             * dnu;
        Cvs_c_r  += w * ss                                * dnu;
        Ss_c_r   += w * ss * (1.0 - log(u))               * dnu;
        ZPE_s_r  += w * ss * 0.5 * u                      * dnu;
      }
      Sr_gas = hsdf_rot * ws_rot_gas;
      S_rot_gas   = units_lj ? Sr_gas   : R*Sr_gas;
      S_rot_solid = units_lj ? Sr_solid : R*Sr_solid;
      S_rot = S_rot_gas + S_rot_solid;
      // Hoist for μ_q calculation later
      hsdf_rot_w  = hsdf_rot;

      {
        double Eg_r  = hsdf_rot * 0.5, Cvg_r = hsdf_rot * 0.5;
        Eq_r  = units_lj ? T_rot_mol*(Es_q_r+Eg_r)     : R*T_rot_mol*(Es_q_r+Eg_r)*1e-3;
        Cvq_r = units_lj ? (Cvs_q_r+Cvg_r)             : R*(Cvs_q_r+Cvg_r);
        Ec_r  = units_lj ? T_rot_mol*(Es_c_r+Eg_r)     : R*T_rot_mol*(Es_c_r+Eg_r)*1e-3;
        Cvc_r = units_lj ? (Cvs_c_r+Cvg_r)             : R*(Cvs_c_r+Cvg_r);
        Sc_r  = units_lj ? (Ss_c_r+Sr_gas)             : R*(Ss_c_r+Sr_gas);
      }
      ZPE_r = units_lj ? T_rot_mol*ZPE_s_r : R*T_rot_mol*ZPE_s_r*1e-3;

      // ── 3PT rotational cage-memory entropy ─────────────────────────────
      // Wg_override = ws_rot_gas (the free rigid-rotor per-DoF weight);
      // prefactor 1/d_rot.
      if (cage_entropy_rot && f_rot > 0.0 && nmol_group > 0
          && !vac_rot.empty() && !dos_rot.empty() && ws_rot_gas != 0.0) {
        int d_rot = (nmol_group > 0 && mol_is_linear[0]) ? 2 : 3;
        std::vector<double> dpa(nused), gpa(nused);
        for (int j = 0; j < nused; j++) {
          dpa[j] = dos_rot[j]/nmol_group; gpa[j] = gas_rot[j]/nmol_group;
        }
        double dS_r2 = cage_memory_entropy(dt_vac, vac_rot, (int)vac_rot.size(),
                          dnu, dpa, gpa, nused, T_rot_mol, 1.0, 1.0,
                          1.0/(double)d_rot, d_rot, ws_rot_gas, &cage_rot_w);
        for (double &v : cage_rot_w) v *= nmol_group;     // per-mol → extensive
        if (dS_r2 != 0.0) {
          double dS_ext = dS_r2 * nmol_group;
          S_rot += dS_ext*R;  S_rot_solid += dS_ext*R;  Sc_r += dS_ext*R;
          s_cage_rot_last = dS_ext * R;     // extensive; ×norm at write time
          if (me == 0)
            FIXXPT_LOG("FixXPT::{}-{}: 3PT cage (mol rot) dS={:.4g} J/mol/K/mol "
                       "(d_rot={})\n", id, group->names[igroup], dS_r2*R, d_rot);
        }
      }

      if (do_normalize) {
        S_rot_gas /= nmol_group; S_rot_solid /= nmol_group; S_rot /= nmol_group;
        Eq_r /= nmol_group; Cvq_r /= nmol_group;
        Ec_r /= nmol_group; Cvc_r /= nmol_group; Sc_r /= nmol_group;
        ZPE_r /= nmol_group;
      }
    }

    // Rotational diffusivity (same formula as translational; anguv has same unit structure).
    if (f_rot > 0.0 && mass_mol_total > 0.0 && T_rot_mol > 0.0) {
      if (units_lj)
        D_rot = s0r * T_rot_mol / (12.0 * mass_mol_total);
      else
        D_rot = s0r * R * T_rot_mol / (12.0 * VLIGHT * (mass_mol_total * mass_to_gmol)) * 1e5;
    }

    // ── Vibrational component: all solid (1PT for vib) ───────────────────
    double ZPE_v = 0.0;
    if (!dos_vib.empty()) {
      double Sv = 0.0;
      double Es_q_v = 0, Es_c_v = 0, Cvs_q_v = 0, Cvs_c_v = 0, Ss_c_v = 0, ZPE_s_v = 0;
      for (int j = 0; j < nused; j++) {
        double nu = j * dnu;
        double w = (j==0||j==nused-1) ? 0.5 : 1.0;
        if (dos_vib[j] <= 0.0 || nu == 0.0) {
          if (dos_vib[j] > 0.0) Es_c_v += w * dos_vib[j] * 0.5 * dnu;
          continue;
        }
        double u = units_lj ? 2.0*PI*hbar_star*nu/T_vib_mol : PLANCK*nu/T_vib_mol;
        double eu = exp(u), em1 = eu - 1.0;
        Sv       += w * dos_vib[j] * (u/em1 - log(1.0-1.0/eu)) * dnu;
        Es_q_v  += w * dos_vib[j] * 0.5*u*(eu+1.0)/em1          * dnu;
        Es_c_v  += w * dos_vib[j]                                * dnu;
        Cvs_q_v += w * dos_vib[j] * u*u*eu/(em1*em1)             * dnu;
        Cvs_c_v += w * dos_vib[j]                                * dnu;
        Ss_c_v  += w * dos_vib[j] * (1.0 - log(u))               * dnu;
        ZPE_s_v += w * dos_vib[j] * 0.5 * u                      * dnu;
      }
      S_vib = units_lj ? Sv : R*Sv;
      Eq_v  = units_lj ? T_vib_mol*Es_q_v      : R*T_vib_mol*Es_q_v*1e-3;
      Cvq_v = units_lj ? Cvs_q_v               : R*Cvs_q_v;
      Ec_v  = units_lj ? T_vib_mol*Es_c_v      : R*T_vib_mol*Es_c_v*1e-3;
      Cvc_v = units_lj ? Cvs_c_v               : R*Cvs_c_v;
      Sc_v  = units_lj ? Ss_c_v                : R*Ss_c_v;
      ZPE_v = units_lj ? T_vib_mol*ZPE_s_v     : R*T_vib_mol*ZPE_s_v*1e-3;
      if (do_normalize) {
        S_vib /= nmol_group;
        Eq_v /= nmol_group; Cvq_v /= nmol_group;
        Ec_v /= nmol_group; Cvc_v /= nmol_group; Sc_v /= nmol_group;
        ZPE_v /= nmol_group;
      }
    }

    // Total entropy = trans + rot + vib (overwrite monoatomic S_q)
    S_q = S_trans + S_rot + S_vib;


    // ZPE is purely vibrational (trans/rot have no quantum zero-point energy)
    ZPE_q = ZPE_v;

    // ── Overwrite monoatomic E_q, A_q, Cv_q, S_c, E_c, A_c, Cv_c ────────
    // Assemble from the per-component molecular DOS integrals (each with its
    // own T).  Eo = E_md_out − E_c_mol shifts DOS energies to reproduce the
    // actual MD energy (parallel to monoatomic section 9).
    {
      double E_c_mol  = Ec_t + Ec_r + Ec_v;
      double E_q_dos  = Eq_t + Eq_r + Eq_v;
      double Eo_mol   = has_e_corr ? (E_md_out - E_c_mol) : 0.0;

      E_q = E_q_dos + Eo_mol;
      E_c = E_c_mol;
      S_c = Sc_t + Sc_r + Sc_v;
      // Cv_c is the measured classical Cv (MD energy fluctuations), not the
      // classical DOS integral; fall back to the DOS integral when PE is
      // unavailable.
      Cv_c = has_e_corr ? Cv_fluct * norm : Cvc_t + Cvc_r + Cvc_v;

      // A per component (each uses its own T): A = E_dos - T*S, then +Eo shifts A_q
      // real: A [kJ/mol] = E [kJ/mol] − T [K] * S [J/(mol·K)] * 1e-3
      // lj:   A [ε]      = E [ε]      − T [ε/kB] * S [kB]
      double A_q_dos, A_c_new;
      if (units_lj) {
        A_q_dos = (Eq_t - T_trans_mol * S_trans)
                + (Eq_r - T_rot_mol   * S_rot)
                + (Eq_v - T_vib_mol   * S_vib);
        A_c_new = (Ec_t - T_trans_mol * Sc_t)
                + (Ec_r - T_rot_mol   * Sc_r)
                + (Ec_v - T_vib_mol   * Sc_v);
      } else {
        A_q_dos = (Eq_t - T_trans_mol * S_trans * 1e-3)
                + (Eq_r - T_rot_mol   * S_rot   * 1e-3)
                + (Eq_v - T_vib_mol   * S_vib   * 1e-3);
        A_c_new = (Ec_t - T_trans_mol * Sc_t * 1e-3)
                + (Ec_r - T_rot_mol   * Sc_r * 1e-3)
                + (Ec_v - T_vib_mol   * Sc_v * 1e-3);
      }
      A_q = A_q_dos + Eo_mol;
      A_c = A_c_new;

      // Cv_q: fluctuation + quantum-classical DOS correction from molecular components
      // Cv_fluct is full-group; norm = 1/nmol_group if do_normalize
      Cv_q = Cv_fluct * norm + (Cvq_t + Cvq_r + Cvq_v) - (Cvc_t + Cvc_r + Cvc_v);

      // ── Chemical potential (molecular mode), per Lin 2010 ────────────
      //   Trans gas:  μ_trans − A_trans = HSDF_trans · R T_trans · Z_HS(y_trans)/3
      //   Rot gas:    μ_rot   − A_rot   = HSDF_rot · R T_rot / n_rot
      //               (ideal rotor: PV per particle = kT, no volume excess;
      //                n_rot = 2 linear / 3 nonlinear, averaged here)
      //   Vib: no gas component → no PV term.
      // Z_trans is the simulation virial (use_sim_z) or the HS-EOS Z_CS/Z_BMCSL;
      // use_sim_z corrects the trans gas-PV term only (rotational PV stays ideal).
      double Z_trans = (use_sim_z_mode && nmol_group > 0)
        ? z_sim_from_pressure(P_avg, V_box, nmol_group, T_trans_mol,
                                units_lj, vol_to_angst3)
        : hs_compressibility(y_trans_w);
      double n_rot_avg = (nmol_group > 0)
                         ? (double)dof_rot_total / (double)nmol_group : 3.0;
      if (n_rot_avg < 1.0) n_rot_avg = 3.0;   // sanity for degenerate cases
      double mu_pv;
      if (units_lj) {
        mu_pv = hsdf_trans_w * T_trans_mol * Z_trans / 3.0
              + hsdf_rot_w   * T_rot_mol            / n_rot_avg;
      } else {
        mu_pv = (hsdf_trans_w * R * T_trans_mol * Z_trans / 3.0
              +  hsdf_rot_w   * R * T_rot_mol            / n_rot_avg) * 1e-3;
      }
      mu_pv *= norm;     // match A_q normalization
      mu_q = A_q + mu_pv;
      mu_c = A_c + mu_pv;
    }

    // Store molecular results in result_vec[7..10] (and optionally [11..14])
    result_vec[7]  = S_trans * s_out_scale;
    result_vec[8]  = S_rot   * s_out_scale;
    result_vec[9]  = S_vib   * s_out_scale;
    result_vec[10] = f_rot;
    if (do_show_split) {
      result_vec[11] = S_trans_gas   * s_out_scale;
      result_vec[12] = S_trans_solid * s_out_scale;
      result_vec[13] = S_rot_gas   * s_out_scale;
      result_vec[14] = S_rot_solid * s_out_scale;
    }

    // Capture for write_pwr_vac() (called after this block); move the gas DoS
    // arrays as-built so the .pwr writer prints the actual partition used by
    // the entropy, not an analytic Lorentzian refit.
    vac_trans_w = std::move(vac_trans);
    vac_rot_w   = std::move(vac_rot);
    vac_vib_w   = std::move(vac_vib);
    dos_trans_w = std::move(dos_trans);
    dos_rot_w   = std::move(dos_rot);
    dos_vib_w   = std::move(dos_vib);
    gas_trans_w = std::move(gas_trans);
    gas_rot_w   = std::move(gas_rot);
  } // end molecular block

  // ── 11b. Store monoatomic results ─────────────────────────────────────
  result_vec[0] = S_q  * s_out_scale;
  result_vec[1] = A_q  * e_out_scale;
  result_vec[2] = E_q  * e_out_scale;
  result_vec[3] = Cv_q * s_out_scale;
  result_vec[4] = D;
  result_vec[5] = f;
  result_vec[6] = T_vac;
  // ZPE always at slot (nvec - 3); μ_q at (nvec - 2); μ_c at (nvec - 1)
  result_vec[nvec - 3] = ZPE_q * e_out_scale;
  result_vec[nvec - 2] = mu_q  * e_out_scale;
  result_vec[nvec - 1] = mu_c  * e_out_scale;
  result_ready  = true;

  // ── 12. The pressure accumulator is reset in end_of_step's end-of-window
  // branch, so intermediate snapshots keep accumulating; P_avg is recomputed
  // from press_sum/press_count on each entry to run_analysis.

  // S_cage diagnostic columns share the entropy normalisation (captured
  // extensive at the cage sites; the channel sums they mirror are divided
  // by the same count).
  s_cage_trans_last *= norm;
  s_cage_rot_last   *= norm;

  // ── 14. Write output ──────────────────────────────────────────────────
  if (me == 0)
    write_pwr_vac(update->ntimestep, dnu, nused, ns, dt_vac,
                  dos, vac, gas_mono,
                  dos_trans_w, dos_rot_w, dos_vib_w,
                  gas_trans_w, gas_rot_w,
                  vac_trans_w, vac_rot_w, vac_vib_w,
                  cage_mono_w, cage_trans_w, cage_rot_w);

  if (me == 0)
    write_result(update->ntimestep, S_q, A_q, E_q, Cv_q, ZPE_q,
                 S_c, A_c, E_c, Cv_c, D, f, T_vac, P_avg, E_md_out,
                 do_molecule ? nmol_group : ng_global, V_box,
                 S_trans, S_rot, S_vib, f_rot, D_rot,
                 S_trans_gas, S_trans_solid, S_rot_gas, S_rot_solid,
                 mu_q, mu_c);

  // fft_vac and fft_buf are persistent (allocated in init(), freed in
  // ~FixXPT()).  Per-window alloc/free here corrupts FFTW3's internal
  // state, crashing PPPM's KSpace destructor at LAMMPS shutdown — verified
  // that fftw_forget_wisdom() between cycles does NOT prevent the crash.
}

/* ======================================================================
   FFT-based VAC / DOS helper methods
====================================================================== */

/* pwr_from_atoms — accumulate mass-weighted power spectrum from atom velocities.
   buf[frame][slot][dim], weight = mbuf[slot], result added into pwr (size N_fft=2*ns). */
void FixXPT::pwr_from_atoms(double ***buf, const std::vector<int> &slots,
                             const double *mbuf, std::vector<double> &pwr,
                             FFT_SCALAR *fft_buf, FFT3d *fft_vac, int ns)
{
  int N_fft = 2 * ns;
  for (int d = 0; d < 3; d++) {
    for (int s : slots) {
      double m = mbuf[s];
      if (m <= 0.0) continue;
      for (int t = 0; t < ns; t++) {
        fft_buf[2*t]   = (FFT_SCALAR)buf[t][s][d];
        fft_buf[2*t+1] = 0.0;
      }
      for (int t = ns; t < N_fft; t++) {
        fft_buf[2*t] = 0.0; fft_buf[2*t+1] = 0.0;
      }
      fft_vac->compute(fft_buf, fft_buf, FFT3d::FORWARD);
      for (int k = 0; k < N_fft; k++)
        pwr[k] += m * (fft_buf[2*k]*fft_buf[2*k] + fft_buf[2*k+1]*fft_buf[2*k+1]);
    }
  }
}

// FP32 input variant.  Identical to the FP64 overload
// above except `buf` is `float***`; the FFT input cast widens FP32 → FFT_SCALAR.
void FixXPT::pwr_from_atoms_f(float ***buf, const std::vector<int> &slots,
                              const double *mbuf, std::vector<double> &pwr,
                              FFT_SCALAR *fft_buf, FFT3d *fft_vac, int ns)
{
  int N_fft = 2 * ns;
  for (int d = 0; d < 3; d++) {
    for (int s : slots) {
      double m = mbuf[s];
      if (m <= 0.0) continue;
      for (int t = 0; t < ns; t++) {
        fft_buf[2*t]   = (FFT_SCALAR)buf[t][s][d];
        fft_buf[2*t+1] = 0.0;
      }
      for (int t = ns; t < N_fft; t++) {
        fft_buf[2*t] = 0.0; fft_buf[2*t+1] = 0.0;
      }
      fft_vac->compute(fft_buf, fft_buf, FFT3d::FORWARD);
      for (int k = 0; k < N_fft; k++)
        pwr[k] += m * (fft_buf[2*k]*fft_buf[2*k] + fft_buf[2*k+1]*fft_buf[2*k+1]);
    }
  }
}

/* pwr_from_mols — accumulate weighted power spectrum from per-molecule quantity.
   buf[frame][mol][dim], weight = wts[mol], result added into pwr (size N_fft=2*ns). */
void FixXPT::pwr_from_mols(double ***buf, int nmol, const std::vector<double> &wts,
                            std::vector<double> &pwr,
                            FFT_SCALAR *fft_buf, FFT3d *fft_vac, int ns)
{
  int N_fft = 2 * ns;
  for (int d = 0; d < 3; d++) {
    for (int m = 0; m < nmol; m++) {
      double w = wts[m];
      if (w <= 0.0) continue;
      for (int t = 0; t < ns; t++) {
        fft_buf[2*t]   = (FFT_SCALAR)buf[t][m][d];
        fft_buf[2*t+1] = 0.0;
      }
      for (int t = ns; t < N_fft; t++) {
        fft_buf[2*t] = 0.0; fft_buf[2*t+1] = 0.0;
      }
      fft_vac->compute(fft_buf, fft_buf, FFT3d::FORWARD);
      for (int k = 0; k < N_fft; k++)
        pwr[k] += w * (fft_buf[2*k]*fft_buf[2*k] + fft_buf[2*k+1]*fft_buf[2*k+1]);
    }
  }
}

/* vac_from_pwr — inverse FFT of power spectrum → VAC[0..ns-1].
   timing1d BACKWARD is unnormalized; divide by N_fft*ns. */
void FixXPT::vac_from_pwr(const std::vector<double> &pwr, std::vector<double> &vac,
                           FFT_SCALAR *fft_buf, FFT3d *fft_vac, int ns)
{
  int N_fft = 2 * ns;
  for (int k = 0; k < N_fft; k++) {
    fft_buf[2*k]   = (FFT_SCALAR)pwr[k];
    fft_buf[2*k+1] = 0.0;
  }
  fft_vac->compute(fft_buf, fft_buf, FFT3d::BACKWARD);
  double norm = 1.0 / ((double)N_fft * ns);
  vac.resize(ns);
  for (int k = 0; k < ns; k++)
    vac[k] = (double)fft_buf[2*k] * norm;
}

/* dos_from_vac — cosine transform of VAC → DOS via zero-symmetric backward FFT.
   vacc[0..ns-1], normalised by vnorm; result in dos_out[0..nused-1]. */
void FixXPT::dos_from_vac(const std::vector<double> &vacc, double vnorm,
                           std::vector<double> &dos_out,
                           FFT_SCALAR *fft_buf, FFT3d *fft_vac,
                           int ns, int nused, double dos_factor)
{
  int N_fft = 2 * ns;
  for (int k = 0; k < ns; k++) {
    fft_buf[2*k]   = (FFT_SCALAR)(vacc[k] * vnorm);
    fft_buf[2*k+1] = 0.0;
  }
  fft_buf[2*ns] = 0.0; fft_buf[2*ns+1] = 0.0;
  for (int k = 1; k < ns; k++) {
    fft_buf[2*(N_fft-k)]   = fft_buf[2*k];
    fft_buf[2*(N_fft-k)+1] = 0.0;
  }
  fft_vac->compute(fft_buf, fft_buf, FFT3d::BACKWARD);
  dos_out.resize(nused);
  for (int j = 0; j < nused; j++)
    dos_out[j] = (double)fft_buf[2*j] * dos_factor;
}

/* T_from_vac — temperature from VAC(0) via equipartition.
   vacc[0] is the unnormalized VAC at lag 0 in native (mass×vel²) units.
   The vel_to_Angps2 parameter is kept for API compatibility but is unused;
   temperature now uses the pre-computed vac_to_jmol factor directly. */
double FixXPT::T_from_vac(const std::vector<double> &vacc, int dof_comp,
                           double /*vel_to_Angps2*/) const
{
  if (dof_comp <= 0 || vacc.empty()) return 0.0;
  if (units_lj)
    return vacc[0] / dof_comp;
  else
    return (vacc[0] * vac_to_jmol) / (R * dof_comp);
}

/* gas_comp — gas DOS at bin j: Lorentzian (Bg=0) or Desjarlais (Bg>0). */
double FixXPT::gas_comp(int j, double dnu, double dos_j, double s0_c,
                        double f_c, double Bg_c, double nmol_c) const
{
  if (f_c <= 0.0 || s0_c <= 0.0) return 0.0;
  double nu = j * dnu;
  double sg;
  if (Bg_c > 0.0) {
    sg = (j == 0) ? s0_c : sgmf_des(nu, s0_c, f_c, Bg_c, nmol_c);
  } else {
    sg = (nu == 0.0) ? s0_c
                     : s0_c / (1.0 + pow(PI * s0_c * nu / (6.0 * nmol_c * f_c), 2.0));
  }
  return (sg > dos_j) ? dos_j : sg;
}

/* build_gas_arr — evaluate gas_comp for all bins [0..nused) at once. */
std::vector<double> FixXPT::build_gas_arr(const std::vector<double> &dos_c, double dnu,
                                           double s0_c, double f_c, double Bg_c, double nmol_c)
{
  std::vector<double> gas(dos_c.size());
  for (int j = 0; j < (int)dos_c.size(); j++)
    gas[j] = gas_comp(j, dnu, dos_c[j], s0_c, f_c, Bg_c, nmol_c);
  return gas;
}


/* ======================================================================
   I/O helpers
====================================================================== */

void FixXPT::open_file()
{
  std::string base = std::string(prefix);

  std::string fname = base + ".thermo";
  fp = fopen(fname.c_str(), "w");
  if (!fp) error->one(FLERR, "fix xpt: cannot open thermo file {}", fname);

  std::string fpname = base + ".pwr";
  fp_pwr = fopen(fpname.c_str(), "w");
  if (!fp_pwr) error->one(FLERR, "fix xpt: cannot open power spectrum file {}", fpname);

  std::string fvname = base + ".vac";
  fp_vac = fopen(fvname.c_str(), "w");
  if (!fp_vac) error->one(FLERR, "fix xpt: cannot open VAC file {}", fvname);

  write_header();
}

void FixXPT::write_header()
{
  if (!fp) return;
  fprintf(fp, "# xPT on-the-fly analysis  (fix xpt)\n");
  fprintf(fp, "# Nevery = %d  Nframes = %d  prefix = %s\n",
          nevery, nframes, prefix);
  if (units_lj) {
    fprintf(fp, "# Units: LJ reduced  "
                "S*,Cv* [kB]  A*,E*,E_md [epsilon]  D* [sigma^2/tau]  "
                "f []  T* [epsilon/kB]  P* [epsilon/sigma^3]\n");
    if (lj_eps > 0.0 || lj_sig > 0.0 || lj_mass > 0.0)
      fprintf(fp, "# LJ reference: epsilon = %.6g kcal/mol  sigma = %.6g Ang  mass = %.6g g/mol\n",
              lj_eps, lj_sig, lj_mass);
    fprintf(fp, "# hbar* = %.6g%s\n", hbar_star,
            hbar_star == 1.0 ? "  (default; provide epsilon/sigma/mass for physical scaling)" : "");
  } else {
    // Pressure unit label depends on unit style
    const char *p_unit = "atm";  // real default
    { const char *us2 = update->unit_style;
      if      (strcmp(us2,"metal")    == 0) p_unit = "bar";
      else if (strcmp(us2,"si")       == 0) p_unit = "Pa";
      else if (strcmp(us2,"cgs")      == 0) p_unit = "dyne/cm2";
      else if (strcmp(us2,"micro")    == 0) p_unit = "pg/(um*us2)";
      else if (strcmp(us2,"nano")     == 0) p_unit = "ag/(nm*ns2)";
      else if (strcmp(us2,"electron") == 0) p_unit = "Pa";
    }
    const char *e_unit = (e_out_scale != 1.0) ? "eV"   : "kJ/mol";
    const char *s_unit = (s_out_scale != 1.0) ? "eV/K" : "J/mol/K";
    fprintf(fp, "# Units: S,Cv [%s]  A,E,ZPE,E_md [%s]"
                "  D [cm^2/s]  f []  T_vac [K]"
                "  P_avg [%s]\n"
                "# Classical quantities (S_c,A_c,E_c,Cv_c) use same units as quantum equivalents.\n",
                s_unit, e_unit, p_unit);
  }
  if (do_normalize) fprintf(fp, "# Quantities normalized per %s\n", do_molecule ? "molecule" : "atom");
  if (do_molecule)
    fprintf(fp, "# Molecular mode: %d molecules, mode=%d, linear=%s, rotsym=%d\n",
            nmol_group, mol_mode, is_linear ? "yes" : "no", rotsym);
  // Hard-sphere EOS diagnostic — 2PT only: the HS gas/solid
  // partition is the 2PT construction; 3PT reports S_cage instead and 1PT
  // has no HS reference.
  if (req_mode == REQ_2PT || req_mode == REQ_UNSET) {
    const char *hs_ent =
        (refinement == REF_LIN2003 || refinement == REF_DESJARLAIS)
            ? "lin2003" : "rigorous";
    fprintf(fp, "# Hard-sphere EOS: %s  entropy: %s\n",
            "CS", hs_ent);
  }
  if (!pe_available) fprintf(fp, "# WARNING: E_md and Cv(fluct) unavailable for this subgroup/pair style\n");
  const char *v_lbl = units_lj ? "V(sigma3)"  : "V(Ang3)";
  const char *p_lbl;
  if (units_lj) p_lbl = "P*(e/s3)";
  else {
    const char *us2 = update->unit_style;
    if      (strcmp(us2,"metal")    == 0) p_lbl = "P_avg(bar)";
    else if (strcmp(us2,"si")       == 0) p_lbl = "P_avg(Pa)";
    else if (strcmp(us2,"cgs")      == 0) p_lbl = "P_avg(dyn/cm2)";
    else if (strcmp(us2,"micro")    == 0) p_lbl = "P_avg(micro)";
    else if (strcmp(us2,"nano")     == 0) p_lbl = "P_avg(nano)";
    else if (strcmp(us2,"electron") == 0) p_lbl = "P_avg(Pa)";
    else                                   p_lbl = "P_avg(atm)";
  }
  const char *e_lbl  = units_lj ? "E_md(eps)"    : (e_out_scale != 1.0 ? "E_md(eV)"      : "E_md(kJ/mol)");
  const char *sc_lbl  = units_lj ? "S_c(kB)"      : (s_out_scale != 1.0 ? "S_c(eV/K)"      : "S_c(J/molK)");
  const char *ac_lbl  = units_lj ? "A_c(eps)"     : (e_out_scale != 1.0 ? "A_c(eV)"        : "A_c(kJ/mol)");
  const char *ec_lbl  = units_lj ? "E_c(eps)"     : (e_out_scale != 1.0 ? "E_c(eV)"        : "E_c(kJ/mol)");
  const char *cvc_lbl = units_lj ? "Cv_c(kB)"     : (s_out_scale != 1.0 ? "Cv_c(eV/K)"     : "Cv_c(J/molK)");
  const char *zpe_lbl = units_lj ? "ZPE_q(eps)"   : (e_out_scale != 1.0 ? "ZPE_q(eV)"     : "ZPE_q(kJ/mol)");
  const char *muq_lbl = units_lj ? "mu_q(eps)"    : (e_out_scale != 1.0 ? "mu_q(eV)"      : "mu_q(kJ/mol)");
  const char *muc_lbl = units_lj ? "mu_c(eps)"    : (e_out_scale != 1.0 ? "mu_c(eV)"      : "mu_c(kJ/mol)");
  fprintf(fp, "#%9s %12s %12s %12s %12s %12s %12s %12s %10s %10s %12s %12s %14s",
          "Step", "S_q", "A_q", "E_q", "Cv_q", "D", "fluidicity", "T_vac",
          do_molecule ? "Nmol" : "Natom", "DOF",
          v_lbl, p_lbl, e_lbl);
  if (do_classical)
    fprintf(fp, " %12s %12s %12s %12s", sc_lbl, ac_lbl, ec_lbl, cvc_lbl);
  if (do_molecule) {
    fprintf(fp, " %12s %12s %12s %12s %12s", "S_trans", "S_rot", "S_vib", "f_rot", "D_rot");
    if (do_show_split)
      fprintf(fp, " %12s %12s %12s %12s",
              "S_trans_gas", "S_trans_sol", "S_rot_gas", "S_rot_sol");
  }
  fprintf(fp, " %14s", zpe_lbl);
  fprintf(fp, " %14s %14s", muq_lbl, muc_lbl);
  // 3PT S_cage diagnostic (cage-memory + anharmonic ΔS already folded into
  // S_q/S_trans/S_rot above; solid-side, gas contribution is zero).  Always
  // emitted; zero unless mode 3PT.
  fprintf(fp, " %12s %12s", "S_cage_t", "S_cage_r");
  fprintf(fp, "\n");
  fflush(fp);

  // ── PWR file column header ────────────────────────────────────────────
  if (fp_pwr) {
    const char *f_lbl = units_lj ? "freq_1/tau" : "freq_cm-1";
    fprintf(fp_pwr, "# xPT on-the-fly analysis  (fix xpt)\n");
    fprintf(fp_pwr, "# Nevery = %d  Nframes = %d  prefix = %s\n",
            nevery, nframes, prefix);
    if (units_lj) fprintf(fp_pwr, "# Units: LJ reduced  freq [1/tau]  DoS [tau]\n");
    else          fprintf(fp_pwr, "# Units: freq [cm-1]  DoS [cm/mol]\n");
    fprintf(fp_pwr, "# Each block is one analysis window, preceded by '# Step N'\n");
    if (do_molecule) {
      fprintf(fp_pwr, "# %12s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s",
              f_lbl,
              "DoS_trans", "gas_trans", "sol_trans",
              "DoS_rot",   "gas_rot",  "sol_rot",
              "DoS_imvib", "DoS_total", "cage_trans", "cage_rot");
    } else {
      fprintf(fp_pwr, "# %12s %14s %14s %14s %14s",
              f_lbl, "DoS_total", "gas_total", "sol_total", "cage_total");
    }
    fprintf(fp_pwr, "\n");
    fflush(fp_pwr);
  }

  // ── VAC file column header ────────────────────────────────────────────
  if (fp_vac) {
    const char *t_lbl = units_lj ? "time_tau" : "time_ps";
    fprintf(fp_vac, "# xPT on-the-fly analysis  (fix xpt)\n");
    fprintf(fp_vac, "# Nevery = %d  Nframes = %d  prefix = %s\n",
            nevery, nframes, prefix);
    if (units_lj) fprintf(fp_vac, "# Units: LJ reduced  time [tau]  VAC [sigma^2/tau^2 * g/mol]\n");
    else          fprintf(fp_vac, "# Units: time [ps]  VAC [(Ang/ps)^2 * g/mol]\n");
    fprintf(fp_vac, "# Each block is one analysis window, preceded by '# Step N'\n");
    if (do_molecule) {
      fprintf(fp_vac, "# %12s %14s %14s %14s %14s",
              t_lbl, "VAC_trans", "VAC_rot", "VAC_imvib", "VAC_total");
    } else {
      fprintf(fp_vac, "# %12s %14s", t_lbl, "VAC_total");
    }
    fprintf(fp_vac, "\n");
    fflush(fp_vac);
  }
}

void FixXPT::write_result(bigint step,
                           double S_q, double A_q, double E_q, double Cv_q, double ZPE_q,
                           double S_c, double A_c, double E_c, double Cv_c,
                           double D, double f, double T_vac, double P_avg,
                           double E_md, int natom_group, double V_box,
                           double S_trans, double S_rot, double S_vib, double f_rot, double D_rot,
                           double S_trans_gas, double S_trans_solid,
                           double S_rot_gas,   double S_rot_solid,
                           double mu_q, double mu_c)
{
  if (!fp) return;
  // Per-row block-kind header — only emit for nsamples > 1 to preserve
  // the existing .thermo layout (which has no per-row header) for
  // default nsamples = 1 runs.
  if (nsamples > 1) emit_block_header(fp, step);
  fprintf(fp, " %9ld %12.6g %12.6g %12.6g %12.6g %12.5e %12.6f %12.4f %10d %10.0f %12.4f %12.4f %14.6g",
          step, S_q*s_out_scale, A_q*e_out_scale, E_q*e_out_scale, Cv_q*s_out_scale, D, f, T_vac,
          natom_group, dof_last_window, V_box, P_avg, E_md*e_out_scale);
  if (do_classical)
    fprintf(fp, " %12.6g %12.6g %12.6g %12.6g",
            S_c*s_out_scale, A_c*e_out_scale, E_c*e_out_scale, Cv_c*s_out_scale);
  if (do_molecule) {
    fprintf(fp, " %12.6g %12.6g %12.6g %12.6f %12.5e",
            S_trans*s_out_scale, S_rot*s_out_scale, S_vib*s_out_scale, f_rot, D_rot);
    if (do_show_split)
      fprintf(fp, " %12.6g %12.6g %12.6g %12.6g",
              S_trans_gas*s_out_scale, S_trans_solid*s_out_scale,
              S_rot_gas*s_out_scale, S_rot_solid*s_out_scale);
  }
  fprintf(fp, " %14.6g", ZPE_q*e_out_scale);
  fprintf(fp, " %14.6g %14.6g", mu_q*e_out_scale, mu_c*e_out_scale);
  // 3PT S_cage columns (zero unless mode 3PT).
  fprintf(fp, " %12.6g %12.6g",
          s_cage_trans_last*s_out_scale, s_cage_rot_last*s_out_scale);
  fprintf(fp, "\n");
  fflush(fp);
}

/* ======================================================================
   write_pwr_vac — append one window block to the .pwr and .vac files.

   PWR columns (monoatomic):
     freq  DoS_total  gas_total  sol_total
   PWR columns (molecular):
     freq  DoS_trans gas_trans sol_trans  DoS_rot gas_rot sol_rot  DoS_imvib  DoS_total

   VAC columns (monoatomic):
     time  VAC_total
   VAC columns (molecular):
     time  VAC_trans  VAC_rot  VAC_imvib  VAC_total

   Gas component at frequency nu = j*dnu (mode 1/2: Lorentzian, mode 3: Desjarlais):
     mode 1/2: sg[j] = s0 / (1 + (π·s0·nu / (6·N·f))²)  for nu > 0, f > 0
     mode 3:   sg[j] = sgmf_des(nu, s0, f, Bg, N)         for nu > 0, Bg > 0
     sg[0] = s0  (f > 0),  sg[j] = 0  (f == 0),  ss[j] = dos[j] − sg[j]
====================================================================== */

void FixXPT::write_pwr_vac(bigint step,
    double dnu, int nused, int ns_val, double dt_vac,
    const std::vector<double>& dos,
    const std::vector<double>& vac,
    const std::vector<double>& gas_mono,
    const std::vector<double>& dos_trans,
    const std::vector<double>& dos_rot,
    const std::vector<double>& dos_vib,
    const std::vector<double>& gas_trans,
    const std::vector<double>& gas_rot,
    const std::vector<double>& vac_trans,
    const std::vector<double>& vac_rot,
    const std::vector<double>& vac_vib,
    const std::vector<double>& cage_mono,
    const std::vector<double>& cage_trans,
    const std::vector<double>& cage_rot)
{
  // ── PWR / VAC truncation for the `maxfreq` keyword ───────────────────
  // PWR bins to emit: j = 0..M where M = floor(maxfreq / dnu), bounded
  // by [0, nused-1].  VAC is truncated symmetrically to ns_out = M
  // samples (preserves the original "PWR has one more element than VAC"
  // FFT-pair shape; falls back to 1 when M = 0).  When maxfreq <= 0 or
  // dnu <= 0, both loops run to the full original length.
  int nused_out = nused;
  int ns_out    = ns_val;
  if (maxfreq > 0.0 && dnu > 0.0) {
    int M = (int)std::floor(maxfreq / dnu);
    if (M < 0)         M = 0;
    if (M >= nused)    M = nused - 1;
    nused_out = M + 1;
    ns_out    = std::min(ns_val, std::max(1, M));
  }

  // ── Output-spacing stride for the `dnu` keyword ──────────────────────
  // K = max(1, round(dnu_out / dnu_real)).  Both loops emit every K-th
  // bin (PWR: in freq; VAC: in time — keeps the disk pair FFT-shaped).
  // K = 1 (default) reproduces the native spacing exactly.
  int stride_out = 1;
  if (dnu_out > 0.0 && dnu > 0.0) {
    stride_out = (int)std::floor(dnu_out / dnu + 0.5);
    if (stride_out < 1) stride_out = 1;
  }

  // ── PWR file ─────────────────────────────────────────────────────────
  if (fp_pwr && !dos.empty()) {
    emit_block_header(fp_pwr, step);
    for (int j = 0; j < nused_out; j += stride_out) {
      double nu = j * dnu;
      if (do_molecule) {
        double dt = dos_trans.empty() ? 0.0 : dos_trans[j];
        double dr = dos_rot.empty()   ? 0.0 : dos_rot[j];
        double dv = dos_vib.empty()   ? 0.0 : dos_vib[j];
        double gt = gas_trans.empty() ? 0.0 : gas_trans[j];
        double gr = gas_rot.empty()   ? 0.0 : gas_rot[j];
        double ct = ((int)cage_trans.size() > j) ? cage_trans[j] : 0.0;
        double cr = ((int)cage_rot.size()   > j) ? cage_rot[j]   : 0.0;
        fprintf(fp_pwr, " %14.4f %14.4f %14.4f %14.4f %14.4f %14.4f %14.4f %14.4f %14.4f %14.4f %14.4f\n",
                nu, dt, gt, dt - gt, dr, gr, dr - gr, dv, dos[j], ct, cr);
      } else {
        double gm = gas_mono.empty() ? 0.0 : gas_mono[j];
        double cm = ((int)cage_mono.size() > j) ? cage_mono[j] : 0.0;
        fprintf(fp_pwr, " %14.4f %14.4f %14.4f %14.4f %14.4f\n",
                nu, dos[j], gm, dos[j] - gm, cm);
      }
    }
    fprintf(fp_pwr, "\n");
    fflush(fp_pwr);
  }

  // ── VAC file ─────────────────────────────────────────────────────────
  if (fp_vac && !vac.empty()) {
    emit_block_header(fp_vac, step);
    for (int k = 0; k < ns_out; k += stride_out) {
      double t = k * dt_vac;
      if (do_molecule) {
        double vt = vac_trans.empty() ? 0.0 : vac_trans[k];
        double vr = vac_rot.empty()   ? 0.0 : vac_rot[k];
        double vv = vac_vib.empty()   ? 0.0 : vac_vib[k];
        fprintf(fp_vac, " %14.4f %14.4f %14.4f %14.4f %14.4f\n",
                t, vt, vr, vv, vac[k]);
      } else {
        fprintf(fp_vac, " %14.4f %14.4f\n", t, vac[k]);
      }
    }
    fprintf(fp_vac, "\n");
    fflush(fp_vac);
  }
}

