/* ----------------------------------------------------------------------
   fix_xpt_const.h — shared internal definitions for the fix_xpt
   translation units (physical constants + debug log/timer macros).

   Included by fix_xpt.cpp and its sibling TUs so one definition is shared
   across the split FixXPT implementation.
   Include AFTER the LAMMPS headers (the macros expand to utils::logmesg /
   MPI_Wtime at their use sites inside FixXPT methods).

   This is NOT a public LAMMPS header — internal to the XPT package.
   The constants are `static constexpr` (internal linkage).

   Contributing author: Tod A Pascal (UCSD)
------------------------------------------------------------------------- */

#ifndef LMP_FIX_XPT_CONST_H
#define LMP_FIX_XPT_CONST_H

// Enables the FIXXPT_LOG diagnostic stream + the per-window timing report.
#define FIX_XPT_DEBUG 1

#include <mpi.h>

// ── Physical constants (SI) ────────────────────────────────────────────
static constexpr double PI      = 3.14159265358979323846;
static constexpr double KB      = 1.380649e-23;     // J/K
static constexpr double NA      = 6.02214076e23;    // 1/mol
static constexpr double R       = 8.314462618;      // J/(mol·K)
static constexpr double H_SI    = 6.62607015e-34;   // J·s
static constexpr double VLIGHT  = 2.99792458e8;     // m/s  (dt[ps]*VLIGHT[m/s]*1e-10 = cm)
static constexpr double PLANCK  = 1.4387769;        // cm·K  (hc/kB)

// ── Unit conversion ────────────────────────────────────────────────────
// 1 (g/mol)*(Å/ps)² = 10 J/mol.  VAC at lag 0 in (g/mol)(Å/ps)² and R in
// J/(mol·K):  T = vac0 * 10 / (R * DOF)   [DOF = 3*natom for monoatomic].
static constexpr double VAC_TO_JMOL = 10.0;        // (g/mol)(Å/ps)² → J/mol

// ── 3PT cage truncation safeguard ──────────────────────────────────────
// Used by cage_memory_entropy: the main-lobe cutoff and the auto-cutoff
// friction-inflation fallback trigger.
static constexpr double CAGE_MAINLOBE_ALPHA = 0.02;  // main-lobe cutoff: first |K|<α|K0|
static constexpr double CAGE_TAIL_TOL       = 1.0;   // γ-inflation fallback trigger

// ── Diagnostic log + RAII timer macros (gated by FIX_XPT_DEBUG) ─────────
// All non-error diagnostic log lines route through FIXXPT_LOG so they
// compile to nothing in production builds.  Errors (error->all/one) bypass
// this gate.  FIXXPT_TIME accumulates wall time into a double on scope
// exit:  { FIXXPT_TIME(t_bucket); ... }
#ifdef FIX_XPT_DEBUG
#  define FIXXPT_LOG(...) utils::logmesg(lmp, __VA_ARGS__)
namespace {
struct FixXPTTimerScope {
  double t0; double *target;
  FixXPTTimerScope(double &t) : t0(MPI_Wtime()), target(&t) {}
  ~FixXPTTimerScope() { *target += MPI_Wtime() - t0; }
};
}
#  define FIXXPT_TIME(bucket) FixXPTTimerScope _fixxpt_t__(bucket)
#else
#  define FIXXPT_LOG(...) static_cast<void>(0)
#  define FIXXPT_TIME(bucket) static_cast<void>(0)
#endif

#endif
