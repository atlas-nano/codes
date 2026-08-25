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
#  include <fftw3.h>           // fftw_forget_wisdom cleanup
#endif
#include "math_eigen.h"       // molecular inertia-tensor diagonalization
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
// fix_xpt_accumulate.cpp — per-step sampling / buffering of `fix xpt`:
// the velocity buffer (grow/sync/push), the multi-tau correlator, the
// molecular-frame decomposition (COM / angular / vibrational) and inertia,
// and the multitau restart restore.
// ============================================================================
/* ======================================================================
   multitau_apply_restart_stash — land the stashed restart payload.

   Called once the multi-tau buffers exist (post_run's `run 0` path).  Mirrors
   the wire format written by write_restart; a shape change vs. the run that
   wrote the restart (different nframes/mt_MP/natom) is reported and discarded.
   NOTE: write_restart serialises only rank 0's partial c_sum, so a multi-rank
   restart restores the rank-0 partial (single-domain `run 0` reanalysis is
   exact).
====================================================================== */
void FixXPT::multitau_apply_restart_stash()
{
  if (!mt_restart_pending || mt_restart_stash.empty()) return;

  const double *list = mt_restart_stash.data();
  long m = 0;
  const int raw0             = (int)list[0];
  const bool inertia_in      = (raw0 >= 100);   // Option-1 mol_inertia block
  const int mt_n_levels_in   = (int)list[1];
  const int mt_MP_in         = (int)list[2];
  const int mt_natom_ring_in = (int)list[3];
  const int matrix_in        = (int)list[4];
  const int mol_in           = (int)list[5];
  const int n_trans_in       = (int)list[6];
  const int n_rot_in         = (int)list[7];
  const int n_vib_in         = (int)list[8];
  m = 9;

  if (mt_n_levels_in > 0
      && (mt_n_levels_in != mt_n_levels
          || mt_MP_in       != mt_MP
          || mt_natom_ring_in != mt_natom_ring)) {
    if (comm->me == 0)
      utils::logmesg(lmp, "FixXPT::{}-{}: restart multi-tau shape mismatch "
                     "(saved {}/{}/{}, current {}/{}/{}); discarding state.\n",
                     id, group->names[igroup],
                     mt_n_levels_in, mt_MP_in, mt_natom_ring_in,
                     mt_n_levels,    mt_MP,    mt_natom_ring);
    mt_restart_pending = false;
    mt_restart_stash.clear();
    return;
  }

  if (mt_n_levels_in > 0) {
    const int L  = mt_n_levels;
    const int MP = mt_MP;
    if ((int)mt_c_sum.size() < L) mt_c_sum.resize(L);
    for (int l = 0; l < L; l++) {
      if ((int)mt_c_sum[l].size() < MP) mt_c_sum[l].resize(MP, 0.0);
      for (int k = 0; k < MP; k++) mt_c_sum[l][k] = list[m++];
    }
    if ((int)mt_c_cnt.size() < L) mt_c_cnt.resize(L);
    for (int l = 0; l < L; l++) {
      if ((int)mt_c_cnt[l].size() < MP) mt_c_cnt[l].resize(MP, 0);
      for (int k = 0; k < MP; k++) mt_c_cnt[l][k] = (long)list[m++];
    }
    if ((int)mt_count_seen.size() < L) mt_count_seen.resize(L, 0);
    for (int l = 0; l < L; l++) mt_count_seen[l] = (long)list[m++];
    if ((int)mt_head.size() < L) mt_head.resize(L, 0);
    for (int l = 0; l < L; l++) mt_head[l] = (int)list[m++];
    if ((int)mt_down_n.size() < L) mt_down_n.resize(L, 0);
    for (int l = 0; l < L; l++) mt_down_n[l] = (int)list[m++];
    // A restart saved by an older binary with matrix_in=1 still parses —
    // skip its 6 (now-unused) matrix-stream components.
    if (matrix_in) m += 6L * (long)L * MP;
  }

  auto read_stream = [&](MultiTauStream &s, int n_units_in) {
    if (n_units_in <= 0) return;
    const int L  = mt_n_levels_in;   // saved level count (== mt_n_levels here)
    const int MP = mt_MP_in;
    if (s.n_units != n_units_in) {   // stream not allocated to this shape — skip
      m += (long)L * MP + 3L * L;
      return;
    }
    for (int l = 0; l < L; l++)
      for (int k = 0; k < MP; k++) s.c_sum[l][k] = list[m++];
    for (int l = 0; l < L; l++) s.count_seen[l] = (long)list[m++];
    for (int l = 0; l < L; l++) s.head[l]       = (int)list[m++];
    for (int l = 0; l < L; l++) s.down_n[l]     = (int)list[m++];
  };
  if (mol_in) {
    read_stream(mt_trans, n_trans_in);
    read_stream(mt_rot,   n_rot_in);
    read_stream(mt_vib,   n_vib_in);
  }

  // Option 1: restore the window-summed mol_inertia + count (nmol == trans
  // units).  read_stream advances m by the full per-stream size even when it
  // skips a shape-mismatched stream, so the cursor lands exactly here.
  if (inertia_in && mol_in && n_trans_in > 0) {
    const long saved_count = (long)list[m++];
    if (!mol_inertia)
      memory->create(mol_inertia, n_trans_in, 3, "fix_xpt:mol_inertia");
    for (int mm = 0; mm < n_trans_in; mm++)
      for (int d = 0; d < 3; d++) mol_inertia[mm][d] = list[m++];
    mol_inertia_count = (int)saved_count;
  }

  mt_restart_pending = false;
  mt_restart_stash.clear();
}

/* ======================================================================
   compute_mol_inertia_oneframe

   Reconstruct mol_inertia (per-molecule principal moments) from the CURRENT
   restart geometry for a `run 0` reanalysis of a restart that lacks persisted
   inertia.  mol_inertia is normally window-averaged in accumulate_mol_frame,
   but `run 0` accumulates no frames (mol_inertia_count == 0 → molecular block
   skipped, S_q = ±inf).  One snapshot approximates the window-averaged inertia
   for near-rigid molecules; it feeds only the rotational-gas entropy term
   (I_avg → hs_entropy_rot), not the VAC/DoS.  Host-only: atoms do not move
   during `run 0`, so host x equals the restart geometry on CPU and KOKKOS
   alike.  Mirrors accumulate_mol_frame's Pass 1/2 minus velocity terms.
====================================================================== */
void FixXPT::compute_mol_inertia_oneframe()
{
  const int nm = nmol_group;
  if (nm <= 0 || !mol_inertia) return;

  int      *mask   = atom->mask;
  int      *tag    = atom->tag;
  int      *type   = atom->type;
  double  **x      = atom->x;
  double   *amass  = atom->mass;
  imageint *image  = atom->image;
  int       nlocal = atom->nlocal;

  // Pass 1: mass-weighted COM position (unwrapped) + per-mol mass.
  std::vector<double> lcom_r(nm*3, 0.0), gcom_r(nm*3, 0.0);
  std::vector<double> lmass(nm, 0.0),    gmass(nm, 0.0);
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    int s = tag[i] - 1;
    if (s >= (int)slot_to_mol.size() || s >= natom_buf) continue;
    int m = slot_to_mol[s];
    if (m < 0) continue;
    double mi = amass[type[i]];
    double xu[3];
    domain->unmap(x[i], image[i], xu);
    for (int d = 0; d < 3; d++) lcom_r[m*3+d] += mi * xu[d];
    lmass[m] += mi;
  }
  reduce_or_copy(lcom_r.data(), gcom_r.data(), nm*3);
  reduce_or_copy(lmass.data(),  gmass.data(),  nm);
  std::vector<double> rcom(nm*3, 0.0);
  for (int m = 0; m < nm; m++) {
    double M = (gmass[m] > 0.0) ? gmass[m] : 1.0;
    for (int d = 0; d < 3; d++) rcom[m*3+d] = gcom_r[m*3+d] / M;
  }

  // Pass 2: inertia tensor  I_αβ = Σ_i m_i (|r'|² δ_αβ − r'_α r'_β).
  std::vector<double> lI(nm*9, 0.0), gI(nm*9, 0.0);
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    int s = tag[i] - 1;
    if (s >= (int)slot_to_mol.size() || s >= natom_buf) continue;
    int m = slot_to_mol[s];
    if (m < 0) continue;
    double mi = amass[type[i]];
    double xu[3];
    domain->unmap(x[i], image[i], xu);
    double rp[3];
    for (int d = 0; d < 3; d++) rp[d] = xu[d] - rcom[m*3+d];
    double r2 = rp[0]*rp[0] + rp[1]*rp[1] + rp[2]*rp[2];
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++) {
        double delta = (a == b) ? 1.0 : 0.0;
        lI[m*9 + a*3+b] += mi * (r2 * delta - rp[a] * rp[b]);
      }
  }
  reduce_or_copy(lI.data(), gI.data(), nm*9);

  // Diagonalise → principal moments (eig storage matches accumulate_mol_frame).
  for (int m = 0; m < nm; m++) {
    double Imat[3][3];
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++) Imat[a][b] = gI[m*9 + a*3+b];
    double eig[3], V[3][3];
    mat3_sym_eigen(Imat, eig, V);
    if (mol_is_linear[m]) {
      mol_inertia[m][0] = eig[1];
      mol_inertia[m][1] = eig[2];
      mol_inertia[m][2] = 0.0;
    } else {
      for (int d = 0; d < 3; d++) mol_inertia[m][d] = eig[d];
    }
  }
  mol_inertia_count = 1;
}

/* ======================================================================
   grow_buf — (re)allocate velocity buffer
====================================================================== */

void FixXPT::grow_buf(int n)
{
  // Buffer-sharing: consumers don't own the buffer.  Sync
  // pointers from owner — owner's grow_buf already ran this step (owner
  // is at lower index in modify->fix[], so its end_of_step runs first
  // each cycle).
  if (!owns_buffer) {
    sync_buffer_pointers_from_owner();
    return;
  }

  // Skip only if the buffer is BOTH big enough AND actually allocated. On a restart RESUME,
  // natom_buf is restored while vel_buf/vel_buf_f stays null (the buffers aren't in the restart),
  // so `n <= natom_buf` alone would wrongly early-return and leave the buffer NULL -> later null
  // deref in push_velocity_frame. Re-allocate whenever the precision-appropriate buffer is null.
  const bool buf_allocated = (buffer_precision == BUFFER_FP32) ? (vel_buf_f != nullptr)
                                                               : (vel_buf != nullptr);
  if (n <= natom_buf && buf_allocated) return;
  n = std::max(n, natom_buf + 32);

  memory->destroy(vel_buf);
  memory->destroy(mass_buf);

  // With multitau the ring buffer (mt_v_ring) carries the velocity history,
  // so vel_buf only needs the current frame (KE accum + push into the ring).
  mt_vel_buf_single_frame = (correlator == CORR_MULTITAU);
  const int vel_buf_frames = mt_vel_buf_single_frame ? 1 : nframes;
  // Allocate FP64 or FP32 mutually exclusively.
  if (buffer_precision == BUFFER_FP32) {
    memory->create(vel_buf_f, vel_buf_frames, n, 3, "fix_xpt:vel_buf_f");
  } else {
    memory->create(vel_buf,   vel_buf_frames, n, 3, "fix_xpt:vel_buf");
  }
  memory->create(mass_buf, n, "fix_xpt:mass_buf");

  // Also resize vib_vel_buf if molecular mode is active.  Single-frame
  // for multi-tau (history lives in mt_vib's ring buffer).
  if (do_molecule) {
    memory->destroy(vib_vel_buf);
    memory->destroy(vib_vel_buf_f);
    const int vib_frames =
        (correlator == CORR_MULTITAU) ? 1 : nframes;
    if (buffer_precision == BUFFER_FP32) {
      memory->create(vib_vel_buf_f, vib_frames, n, 3, "fix_xpt:vib_vel_buf_f");
    } else {
      memory->create(vib_vel_buf,   vib_frames, n, 3, "fix_xpt:vib_vel_buf");
    }
  }

  natom_buf = n;

  // After reallocation, propagate the new pointers to every consumer
  // sharing this buffer (their stale pointers would otherwise reference
  // freed memory at the next end_of_step KE-accumulation read).
  for (FixXPT *c : buffer_consumers) c->sync_buffer_pointers_from_owner();
}

/* ======================================================================
   sync_buffer_pointers_from_owner — consumer-side helper.
   Copies the owner's buffer base pointers (and natom_buf) into this
   consumer's slots.  Called whenever the owner reallocates: from the
   consumer's own grow_buf (which short-circuits to this), and from the
   owner's grow_buf when it iterates buffer_consumers.
====================================================================== */
void FixXPT::sync_buffer_pointers_from_owner()
{
  if (!buffer_owner || buffer_owner == this) return;  // defensive
  vel_buf     = buffer_owner->vel_buf;
  mass_buf    = buffer_owner->mass_buf;
  natom_buf   = buffer_owner->natom_buf;
  if (do_molecule) {
    com_vel_buf = buffer_owner->com_vel_buf;
    omega_buf   = buffer_owner->omega_buf;
    angmom_buf  = buffer_owner->angmom_buf;
    vib_vel_buf = buffer_owner->vib_vel_buf;
  }
}

/* ======================================================================
   accumulate_frame — copy local atom velocities into buffer slot iframe

   Atoms are stored at slot = (atom_tag - 1), NOT at a local counter.
   This ensures vel_buf[t][slot] refers to the same physical atom at
   all timesteps, even if LAMMPS re-sorts atoms in local memory (which
   it does by default every atom_modify::sort_freq = 1000 steps).
   Without tag-based indexing, the per-atom velocity autocorrelation
   would be corrupted whenever a sort event occurs within a window.
====================================================================== */

void FixXPT::accumulate_frame()
{
  int *mask  = atom->mask;
  int *tag   = atom->tag;
  int nlocal = atom->nlocal;

  window_idle = false;

  if (iframe == 0) {
    // First frame of a new window: rebuild slot list and buffer.
    // For dynamic groups, rebuild molecular topology every window to refresh
    // slot_to_mol, nmol_group, and molmass (group membership may have evolved).
    if (do_molecule) {
      int old_nmol = nmol_group;
      build_mol_topology();
      if (nmol_group != old_nmol && nmol_group > 0) {
        // Shared per-frame buffers: only owner reallocates;
        // consumers sync from owner via sync_buffer_pointers_from_owner
        // (called at the end of this iframe==0 block via grow_buf).
        memory->destroy(mol_inertia);
        memory->create(mol_inertia, nmol_group, 3, "fix_xpt:mol_inertia");
        if (owns_buffer) {
          memory->destroy(com_vel_buf);
          memory->destroy(omega_buf);
          memory->destroy(angmom_buf);
          const int mol_frames =
              (correlator == CORR_MULTITAU) ? 1 : nframes;
          memory->create(com_vel_buf, mol_frames, nmol_group, 3, "fix_xpt:com_vel_buf");
          memory->create(omega_buf,   mol_frames, nmol_group, 3, "fix_xpt:omega_buf");
          memory->create(angmom_buf,  mol_frames, nmol_group, 3, "fix_xpt:angmom_buf");
          // Propagate to consumers (vib_vel_buf is updated in grow_buf below).
          for (FixXPT *c : buffer_consumers) c->sync_buffer_pointers_from_owner();
        }
      }
      if (mol_inertia && nmol_group > 0) {
        for (int m = 0; m < nmol_group; m++) mol_inertia[m][0]=mol_inertia[m][1]=mol_inertia[m][2]=0.0;
        mol_inertia_count = 0;
      }
    }
    // Slots are tag-based (atom_tag - 1) so they are stable across sorts.
    group_slots.clear();
    int maxTag_local = 0;
    for (int i = 0; i < nlocal; i++) {
      if (!(mask[i] & groupbit)) continue;
      group_slots.push_back(tag[i] - 1);
      if (tag[i] > maxTag_local) maxTag_local = tag[i];
    }
    // Global max tag sets the buffer size (same on all procs via Allreduce)
    int maxTag_global = 0;
    MPI_Allreduce(&maxTag_local, &maxTag_global, 1, MPI_INT, MPI_MAX, world);
    if (maxTag_global <= 0) {
      // The group is GLOBALLY empty at this window's first frame — e.g. a
      // dynamic shell that is momentarily unpopulated (or stale right after a
      // restart resume).  The window must NOT start: leave the buffers
      // unallocated and the topology unbuilt, and signal end_of_step to hold
      // iframe at 0.  Advancing the window here is the restart-resume SEGV —
      // iframe would reach 1 with vel_buf/com_vel_buf still null, so the next
      // frame (which skips this iframe==0 prologue) derefs a null buffer in
      // push_velocity_frame / accumulate_mol_frame.  maxTag_global is the same
      // on every rank, so window_idle is globally consistent (no collective skew).
      window_idle = true;
      return;
    }
    grow_buf(maxTag_global);

    // Save fix_dof at window start so run_analysis uses a value consistent
    // with group_slots (window-start membership), not the current mask which
    // may have been updated by a dynamic group at this same step.
    //
    // `ifix->dof(igroup)` returns the GLOBAL constraint count (SHAKE reports
    // the same total on every rank), so do NOT MPI_Allreduce here — that
    // would N×-overcount and drive DOF negative.
    fix_dof_window = 0;
    for (const auto &ifix : modify->get_fix_list())
      if (ifix->dof_flag) fix_dof_window += ifix->dof(igroup);
  }

  // Accumulate local atom velocities into a temporary flat buffer (zero for
  // non-local slots).  MPI_Allreduce (SUM) then assembles the complete global
  // frame: each atom is local on exactly one rank, so the SUM has no double-
  // counting.  This handles atom migration between processor domains correctly
  // — without it, migrated atoms leave stale velocities in vel_buf and corrupt
  // the power spectrum in parallel runs.
  //
  // Buffer-sharing: only the owner does the Allreduce + writes to
  // vel_buf / mass_buf; consumers read the same memory at end_of_window and
  // skip the Allreduce (avoids redundant collective traffic).  The gather +
  // MPI_Allreduce live in the virtual push_velocity_frame() hook so the KOKKOS
  // derived class can gather on the GPU (host path defined below this function).
  // Restart-safe buffer allocation lives in FixXPT::setup() so the owner's
  // buffers are guaranteed allocated even on a mid-window resume.
  if (owns_buffer) {
    const int ibuf = mt_vel_buf_single_frame ? 0 : iframe;
    push_velocity_frame(ibuf);
  }

  // Molecular decomposition: compute per-molecule COM velocity and angular
  // velocity.  Owner-only: result lives in shared com_vel_buf / omega_buf /
  // angmom_buf / vib_vel_buf that consumers read via the alias.
  if (do_molecule && nmol_group > 0 && owns_buffer) accumulate_mol_frame();

  // Multi-tau per-step ring-buffer accumulation.  Owner-only: ring buffers
  // are per-fix, and the equivalence key includes correlator so a multi-tau
  // consumer only ever matches a multi-tau owner and inherits its ring here.
  if (correlator == CORR_MULTITAU && owns_buffer) {
    if (mt_natom_ring != natom_buf || mt_n_levels == 0) multitau_alloc(natom_buf);
    if (iframe == 0) multitau_reset_window();
    multitau_push_frame();
  }
}

/* ======================================================================
   multitau_alloc — lazy allocation of L ring buffers when correlator=multitau.
   Resolves mt_n_levels from mt_L (0 = auto) and nframes.
====================================================================== */
void FixXPT::multitau_alloc(int natom)
{
  // Resolve effective L.  Auto rule: smallest L such that the longest lag
  // (M+P−1)·S^(L−1) covers ≥ nframes/2 samples, capped at 32.
  mt_MP = mt_M + mt_P;
  int Lreq = mt_L;
  if (Lreq <= 0) {
    Lreq = 1;
    long stride = mt_S;
    while ((long)mt_MP * stride < (long)nframes && Lreq < 32) {
      Lreq++;
      stride *= mt_S;
    }
  }
  mt_n_levels   = Lreq;
  mt_natom_ring = natom;
  mt_v_ring.assign(Lreq, std::vector<double>((size_t)mt_MP * natom * 3, 0.0));
  mt_c_sum.assign(Lreq, std::vector<double>(mt_MP, 0.0));
  mt_c_cnt.assign(Lreq, std::vector<long>(mt_MP, 0L));
  mt_count_seen.assign(Lreq, 0L);
  mt_head.assign(Lreq, 0);
  // Downsample state needs (L-1) entries; allocate L for symmetric indexing,
  // the last entry is unused.
  mt_down_acc.assign(Lreq, std::vector<double>((size_t)natom * 3, 0.0));
  mt_down_n.assign(Lreq, 0);
  // Molecular per-channel streams (allocated lazily in accumulate_mol_frame
  // when nmol_group is first known; alloc only marks them inactive here).
  mt_molecular_active = false;
  mt_trans.n_units = mt_rot.n_units = mt_vib.n_units = 0;
}

void FixXPT::multitau_reset_window()
{
  for (int L = 0; L < mt_n_levels; L++) {
    std::fill(mt_v_ring[L].begin(), mt_v_ring[L].end(), 0.0);
    std::fill(mt_c_sum[L].begin(), mt_c_sum[L].end(), 0.0);
    std::fill(mt_c_cnt[L].begin(), mt_c_cnt[L].end(), 0L);
    mt_count_seen[L] = 0;
    mt_head[L] = 0;
    std::fill(mt_down_acc[L].begin(), mt_down_acc[L].end(), 0.0);
    mt_down_n[L] = 0;
  }
}

void FixXPT::multitau_push_frame()
{
  // Push level-0 velocity (from the just-completed accumulate_frame) into
  // level 0's ring buffer; recurse into deeper levels via downsample.
  // Velocities are global (post-MPI_Allreduce) on every rank — but
  // group_slots is per-rank, so each rank accumulates only its local
  // atom-slot lag products into c_sum.  c_sum is MPI-reduced once per
  // window in compute_2pt_window (lazily at output time).
  // Source for the ring push: row 0 when single-frame, iframe otherwise.
  const int ibuf = mt_vel_buf_single_frame ? 0 : iframe;
  multitau_push_level(0, &vel_buf[ibuf][0][0]);
}

void FixXPT::multitau_push_level(int Lev, const double* v_curr)
{
  if (Lev >= mt_n_levels) return;
  const int MP = mt_MP;
  const int N  = mt_natom_ring;
  double* ring = mt_v_ring[Lev].data();
  int head     = mt_head[Lev];
  std::copy(v_curr, v_curr + (size_t)N * 3, ring + (size_t)head * N * 3);
  mt_count_seen[Lev]++;
  // Lag accumulation: lag k corresponds to ring position (head − k) mod (M+P).
  // Only valid up to min(count_seen, M+P) lags.
  const int kmax = (int)std::min((long)MP, mt_count_seen[Lev]);
  for (int k = 0; k < kmax; k++) {
    const int other = (head - k + MP) % MP;
    const double* v_then = ring + (size_t)other * N * 3;
    double acc = 0.0;
    for (int slot : group_slots) {
      const double m   = mass_buf[slot];
      const double nx  = v_curr[slot*3+0], ny = v_curr[slot*3+1], nz = v_curr[slot*3+2];
      const double tx  = v_then[slot*3+0], ty = v_then[slot*3+1], tz = v_then[slot*3+2];
      acc += m * (nx*tx + ny*ty + nz*tz);
    }
    mt_c_sum[Lev][k] += acc;
    mt_c_cnt[Lev][k] += 1;
  }
  mt_head[Lev] = (head + 1) % MP;

  // Downsample by mt_S into mt_down_acc[Lev]; emit averaged sample to
  // level Lev+1 every S input samples.
  if (Lev + 1 < mt_n_levels) {
    double* acc = mt_down_acc[Lev].data();
    for (int i = 0; i < N * 3; i++) acc[i] += v_curr[i];
    mt_down_n[Lev]++;
    if (mt_down_n[Lev] >= mt_S) {
      const double inv_S = 1.0 / (double)mt_S;
      std::vector<double> v_avg((size_t)N * 3);
      for (int i = 0; i < N * 3; i++) v_avg[i] = acc[i] * inv_S;
      std::fill(mt_down_acc[Lev].begin(), mt_down_acc[Lev].end(), 0.0);
      mt_down_n[Lev] = 0;
      multitau_push_level(Lev + 1, v_avg.data());
    }
  }
}

/* ======================================================================
   Molecular per-channel multi-tau streams.  One
   MultiTauStream per VACF channel:
     mt_trans → COM velocity   (n_units = nmol_group, weights = molmass,
                                 global on every rank → not distributed)
     mt_rot   → angular vel    (n_units = nmol_group, weights = 1.0,
                                 global on every rank → not distributed)
     mt_vib   → residual vel   (n_units = natom_buf, weights = mass_buf,
                                 per-rank partial → distributed, MPI-reduced
                                 at finalisation)
   Same algorithm as the scalar multitau_push_level: ring buffer push,
   biased lag-product accumulation, recursive average-downsample into
   the next level.
====================================================================== */
void FixXPT::multitau_stream_alloc(MultiTauStream& s, int n_units,
                                     bool distributed)
{
  s.n_units        = n_units;
  s.is_distributed = distributed;
  const int Lreq = mt_n_levels;
  s.v_ring.assign(Lreq, std::vector<double>((size_t)mt_MP * n_units * 3, 0.0));
  s.c_sum.assign(Lreq, std::vector<double>(mt_MP, 0.0));
  s.count_seen.assign(Lreq, 0L);
  s.head.assign(Lreq, 0);
  s.down_acc.assign(Lreq, std::vector<double>((size_t)n_units * 3, 0.0));
  s.down_n.assign(Lreq, 0);
}

void FixXPT::multitau_stream_reset(MultiTauStream& s)
{
  for (int L = 0; L < (int)s.c_sum.size(); L++) {
    std::fill(s.v_ring[L].begin(), s.v_ring[L].end(), 0.0);
    std::fill(s.c_sum[L].begin(), s.c_sum[L].end(), 0.0);
    s.count_seen[L] = 0;
    s.head[L] = 0;
    std::fill(s.down_acc[L].begin(), s.down_acc[L].end(), 0.0);
    s.down_n[L] = 0;
  }
}

void FixXPT::multitau_stream_push(MultiTauStream& s, int Lev,
                                    const double* v_curr,
                                    const double* weights,
                                    const std::vector<int>* slots)
{
  if (Lev >= (int)s.c_sum.size()) return;
  const int MP = mt_MP;
  const int N  = s.n_units;
  if (N <= 0) return;
  double* ring = s.v_ring[Lev].data();
  int head     = s.head[Lev];
  std::copy(v_curr, v_curr + (size_t)N * 3, ring + (size_t)head * N * 3);
  s.count_seen[Lev]++;
  const int kmax = (int)std::min((long)MP, s.count_seen[Lev]);
  for (int k = 0; k < kmax; k++) {
    const int other = (head - k + MP) % MP;
    const double* v_then = ring + (size_t)other * N * 3;
    double acc = 0.0;
    if (slots) {
      for (int slot : *slots) {
        const double w = weights[slot];
        acc += w * (v_curr[slot*3+0] * v_then[slot*3+0]
                  + v_curr[slot*3+1] * v_then[slot*3+1]
                  + v_curr[slot*3+2] * v_then[slot*3+2]);
      }
    } else {
      for (int i = 0; i < N; i++) {
        const double w = weights[i];
        acc += w * (v_curr[i*3+0] * v_then[i*3+0]
                  + v_curr[i*3+1] * v_then[i*3+1]
                  + v_curr[i*3+2] * v_then[i*3+2]);
      }
    }
    s.c_sum[Lev][k] += acc;
  }
  s.head[Lev] = (head + 1) % MP;
  // Downsample into next level
  const int Lreq = (int)s.c_sum.size();
  if (Lev + 1 < Lreq) {
    double* acc = s.down_acc[Lev].data();
    for (int i = 0; i < N * 3; i++) acc[i] += v_curr[i];
    s.down_n[Lev]++;
    if (s.down_n[Lev] >= mt_S) {
      const double inv_S = 1.0 / (double)mt_S;
      std::vector<double> v_avg((size_t)N * 3);
      for (int i = 0; i < N * 3; i++) v_avg[i] = acc[i] * inv_S;
      std::fill(s.down_acc[Lev].begin(), s.down_acc[Lev].end(), 0.0);
      s.down_n[Lev] = 0;
      multitau_stream_push(s, Lev + 1, v_avg.data(), weights, slots);
    }
  }
}

/* ======================================================================
   build_mol_topology — collect molecule IDs, masses, atom counts
   Called from init() when do_molecule=1.
====================================================================== */

void FixXPT::build_mol_topology()
{
  int *mask     = atom->mask;
  int *tag      = atom->tag;
  tagint *molid = atom->molecule;
  double *amass = atom->mass;
  int *type     = atom->type;
  int nlocal    = atom->nlocal;

  // ── Step 1: local unique mol IDs ─────────────────────────────────────
  std::set<tagint> local_set;
  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) local_set.insert(molid[i]);

  int nlocal_mols = (int)local_set.size();
  std::vector<tagint> local_vec(local_set.begin(), local_set.end());

  // ── Step 2: Allgatherv to collect mol IDs from all ranks ─────────────
  int total_raw = 0;
  MPI_Allreduce(&nlocal_mols, &total_raw, 1, MPI_INT, MPI_SUM, world);

  std::vector<int> counts(nprocs), displs(nprocs);
  MPI_Allgather(&nlocal_mols, 1, MPI_INT, counts.data(), 1, MPI_INT, world);
  displs[0] = 0;
  for (int p = 1; p < nprocs; p++) displs[p] = displs[p-1] + counts[p-1];

  std::vector<tagint> all_ids(total_raw);
  MPI_Allgatherv(local_vec.data(), nlocal_mols, MPI_LMP_TAGINT,
                 all_ids.data(), counts.data(), displs.data(), MPI_LMP_TAGINT, world);

  // ── Step 3: sort + unique → global mol_id_list ───────────────────────
  std::sort(all_ids.begin(), all_ids.end());
  all_ids.erase(std::unique(all_ids.begin(), all_ids.end()), all_ids.end());
  mol_id_list = all_ids;
  nmol_group  = (int)mol_id_list.size();
  if (nmol_group == 0) return;

  // ── Step 4: mol_id → index map ───────────────────────────────────────
  std::map<tagint,int> id_to_idx;
  for (int m = 0; m < nmol_group; m++) id_to_idx[mol_id_list[m]] = m;

  // ── Step 5: per-molecule mass and natom (reduce across ranks) ─────────
  std::vector<double> lmass(nmol_group, 0.0), gmass(nmol_group, 0.0);
  std::vector<double> lnat(nmol_group, 0.0),  gnat(nmol_group, 0.0);
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    int m = id_to_idx[molid[i]];
    lmass[m] += amass[type[i]];
    lnat[m]  += 1.0;
  }
  MPI_Allreduce(lmass.data(), gmass.data(), nmol_group, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(lnat.data(),  gnat.data(),  nmol_group, MPI_DOUBLE, MPI_SUM, world);

  molmass.resize(nmol_group);
  mol_natom.resize(nmol_group);
  for (int m = 0; m < nmol_group; m++) {
    molmass[m]   = gmass[m];
    mol_natom[m] = (int)gnat[m];
  }

  // ── Step 5b: isomorphism check ──────────────────────────
  // The trans/rot/vib decomposition assumes every molecule has the same
  // chemistry (same atom count, same atom-type multiset, same mass).
  // Mixed systems (water + DMSO + ions in `molecule` mode) silently
  // give nonsense S_rot / S_vib because mol_inertia + omega/angmom math
  // are averaged across non-equivalent species.  Catch this here.
  if (!allow_mixed_molecules && nmol_group > 1) {
    // Gather (mol_idx, atom_type) pairs from every rank → rank-0 builds
    // per-mol sorted type vectors, computes a signature hash per mol,
    // checks all signatures match.
    std::vector<int> local_pairs;   // flat: [mol_idx, type, mol_idx, type, ...]
    local_pairs.reserve(2 * nlocal);
    for (int i = 0; i < nlocal; i++) {
      if (!(mask[i] & groupbit)) continue;
      auto it = id_to_idx.find(molid[i]);
      if (it == id_to_idx.end()) continue;
      local_pairs.push_back(it->second);
      local_pairs.push_back(type[i]);
    }
    const int loc_n = (int)local_pairs.size();
    std::vector<int> all_counts(nprocs), all_disp(nprocs);
    MPI_Allgather(&loc_n, 1, MPI_INT, all_counts.data(), 1, MPI_INT, world);
    int tot = 0;
    for (int p = 0; p < nprocs; p++) { all_disp[p] = tot; tot += all_counts[p]; }
    std::vector<int> all_pairs(tot);
    MPI_Allgatherv(local_pairs.data(), loc_n, MPI_INT,
                   all_pairs.data(), all_counts.data(), all_disp.data(),
                   MPI_INT, world);
    // Build per-mol sorted type vector
    std::vector<std::vector<int>> mol_types(nmol_group);
    for (int k = 0; k < tot; k += 2) {
      mol_types[all_pairs[k]].push_back(all_pairs[k + 1]);
    }
    for (auto &v : mol_types) std::sort(v.begin(), v.end());
    // Compute a simple hash per signature: H = Σ_k type_k · 31^k
    auto sig_hash = [](const std::vector<int>& v) {
      long h = 0;
      for (int t : v) h = h * 31 + t;
      return h;
    };
    const long ref_sig = sig_hash(mol_types[0]);
    const int  ref_n   = (int)mol_types[0].size();
    int n_mismatch = 0;
    int first_mismatch_idx = -1;
    for (int m = 1; m < nmol_group; m++) {
      const int sn = (int)mol_types[m].size();
      if (sn != ref_n || sig_hash(mol_types[m]) != ref_sig) {
        n_mismatch++;
        if (first_mismatch_idx < 0) first_mismatch_idx = m;
      }
    }
    if (n_mismatch > 0) {
      // Build a short summary of the mismatch (rank 0 only; error->all
      // broadcasts to every rank).  Reference mol = molecule index 0.
      std::string ref_desc = "[";
      for (size_t k = 0; k < mol_types[0].size(); k++) {
        ref_desc += std::to_string(mol_types[0][k]);
        if (k + 1 < mol_types[0].size()) ref_desc += ",";
      }
      ref_desc += "]";
      std::string mis_desc = "[";
      for (size_t k = 0; k < mol_types[first_mismatch_idx].size(); k++) {
        mis_desc += std::to_string(mol_types[first_mismatch_idx][k]);
        if (k + 1 < mol_types[first_mismatch_idx].size()) mis_desc += ",";
      }
      mis_desc += "]";
      error->all(FLERR,
        "FixXPT::{}-{}: molecule isomorphism check FAILED — {} of {} "
        "molecules don't match the reference (mol 0).\n"
        "  reference  (mol 0)         : natom={} types={}\n"
        "  first mismatch (mol {})    : natom={} types={}\n"
        "The trans/rot/vib decomposition assumes a single chemistry; "
        "mixed-species systems give wrong S_rot/S_vib.  Either group the "
        "molecules by species (separate fix xpt per chemistry), or set "
        "`allow_mixed_molecules 1` to bypass this check if you know what "
        "you're doing.",
        id, group->names[igroup],
        n_mismatch, nmol_group,
        ref_n, ref_desc.c_str(),
        first_mismatch_idx,
        (int)mol_types[first_mismatch_idx].size(), mis_desc.c_str());
    }
  }

  // ── Step 6: per-molecule linear flag ─────────────────────────────────
  mol_is_linear.resize(nmol_group);
  for (int m = 0; m < nmol_group; m++) {
    // Auto-detect: diatomic (natom==2) is always linear
    mol_is_linear[m] = (is_linear || mol_natom[m] == 2);
  }

  // ── Step 7: GLOBAL slot_to_mol map (tag-1 → mol_idx, ALL atoms) ──────
  //
  // Must be global: after atoms migrate across processor domains, a rank
  // holding an atom it did not initially own would have slot_to_mol[tag-1]
  // = -1 and silently skip it in accumulate_mol_frame's `if (m < 0)` guard,
  // dropping that molecule's contribution.  Assemble via MPI_Allreduce(MAX):
  // since -1 < 0 <= any valid mol_idx, MAX picks the populated value across
  // ranks while leaving truly-unowned tags at -1.
  int maxTag_local = 0, maxTag_global = 0;
  for (int i = 0; i < nlocal; i++) if (tag[i] > maxTag_local) maxTag_local = tag[i];
  MPI_Allreduce(&maxTag_local, &maxTag_global, 1, MPI_INT, MPI_MAX, world);

  std::vector<int> slot_to_mol_local(maxTag_global, -1);
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    slot_to_mol_local[tag[i]-1] = id_to_idx[molid[i]];
  }
  slot_to_mol.assign(maxTag_global, -1);
  MPI_Allreduce(slot_to_mol_local.data(), slot_to_mol.data(), maxTag_global,
                MPI_INT, MPI_MAX, world);


  if (me == 0) {
    const char *mode_lbl =
        (req_mode == REQ_1PT)            ? "1PT"
      : (req_mode == REQ_3PT)            ? "3PT"
      : (refinement == REF_LIN2003)      ? "2PT/lin2003"
      : (refinement == REF_DESJARLAIS)   ? "2PT/desjarlais"
      : (refinement == REF_R2PT)         ? "2PT/r2pt"
      :                                    "2PT/rigorous";
    FIXXPT_LOG( "FixXPT::{}-{}: {} molecules in group (mode={}, linear={}, rotsym={})\n",
                   id, group->names[igroup], nmol_group, mode_lbl, is_linear ? "yes" : "auto", rotsym);
  }


  // Inform any device-side cache (FixXPTKokkos) that
  // slot_to_mol / molmass / mol_natom / mol_is_linear were just (re)built.
  mol_topology_generation++;
}


/* ======================================================================
   push_velocity_frame — host-side velocity gather + tag-indexed write
   + MPI_Allreduce(SUM) into vel_buf[ibuf].  Default impl used by the
   CPU build of fix_xpt.  Overridden in FixXPTKokkos to do the gather
   on the GPU.

   Owner-only: caller in accumulate_frame() already gated on owns_buffer.
====================================================================== */
void FixXPT::push_velocity_frame(int ibuf)
{
  int    *mask  = atom->mask;
  int    *tag   = atom->tag;
  int    *type  = atom->type;
  double **v    = atom->v;
  double *mass  = atom->mass;
  int     nlocal = atom->nlocal;

  if (buffer_precision == BUFFER_FP32) {
    // FP32 storage path.
    std::vector<float> vel_local_f(natom_buf * 3, 0.0f);
    for (int i = 0; i < nlocal; i++) {
      if (!(mask[i] & groupbit)) continue;
      int slot = tag[i] - 1;
      if (slot < 0 || slot >= natom_buf) continue;   // guard BOTH bounds: tag<=0 -> slot<0 -> OOB write
      vel_local_f[slot*3 + 0] = (float)v[i][0];
      vel_local_f[slot*3 + 1] = (float)v[i][1];
      vel_local_f[slot*3 + 2] = (float)v[i][2];
      mass_buf[slot]           = mass[type[i]];
    }
    reduce_or_copy_f(vel_local_f.data(), vel_buf_f[ibuf][0], natom_buf * 3);
  } else {
    std::vector<double> vel_local(natom_buf * 3, 0.0);
    for (int i = 0; i < nlocal; i++) {
      if (!(mask[i] & groupbit)) continue;
      int slot = tag[i] - 1;
      if (slot < 0 || slot >= natom_buf) continue;   // guard BOTH bounds: tag<=0 -> slot<0 -> OOB write
      vel_local[slot*3 + 0] = v[i][0];
      vel_local[slot*3 + 1] = v[i][1];
      vel_local[slot*3 + 2] = v[i][2];
      mass_buf[slot]         = mass[type[i]];
    }
    reduce_or_copy(vel_local.data(), vel_buf[ibuf][0], natom_buf * 3);
  }
}

/* ======================================================================
   reduce_or_copy — MPI_Allreduce(SUM) of n doubles from src → dst on
   multi-rank runs; plain memcpy on single rank (skips collective traffic).
====================================================================== */
void FixXPT::reduce_or_copy(const double *src, double *dst, int n)
{
  if (comm->nprocs == 1) {
    std::memcpy(dst, src, size_t(n) * sizeof(double));
  } else {
    MPI_Allreduce(src, dst, n, MPI_DOUBLE, MPI_SUM, world);
  }
}

// FP32 companion.
void FixXPT::reduce_or_copy_f(const float *src, float *dst, int n)
{
  if (comm->nprocs == 1) {
    std::memcpy(dst, src, size_t(n) * sizeof(float));
  } else {
    MPI_Allreduce(src, dst, n, MPI_FLOAT, MPI_SUM, world);
  }
}

/* ======================================================================
   accumulate_mol_frame — per-frame molecular decomposition
   Computes COM velocity and angular velocity for each molecule.
   Called from accumulate_frame() when do_molecule=1.

   Units: v in native LAMMPS (Å/fs real, Å/ps metal, σ/τ lj)
          r in Å (real/metal) or σ (lj)
          mass in g/mol
          L = r × (m·v) in  g/mol·Å·(Å/fs) = g/mol·Å²/fs  (real)
          I = m·r²       in  g/mol·Å²
          ω = I⁻¹·L      in  1/fs (real), 1/ps (metal), 1/τ (lj)
====================================================================== */

void FixXPT::accumulate_mol_frame()
{
  // ibuf_mol: 0 in multitau mode (com/omega/angmom/vib buffers are
  // single-frame, history lives in the ring buffers), live iframe
  // otherwise.  Mirrors mt_vel_buf_single_frame's semantics for the
  // monatomic vel_buf.
  const int ibuf_mol = (correlator == CORR_MULTITAU) ? 0 : iframe;
  // ibuf_vel: where this frame's per-atom velocities live in vel_buf.  Reading
  // vel_buf (written by push_velocity_frame at top of accumulate_frame) instead
  // of atom->v means this routine doesn't depend on host atom->v being fresh —
  // important under KOKKOS where the device integrator may leave it stale.
  const int ibuf_vel = mt_vel_buf_single_frame ? 0 : iframe;
  // The per-atom loops of Passes 1/2/3 are extracted into virtual hooks
  // (compute_mol_pass{1,2,3}_atoms) so the KOKKOS class can run them on-device.

  // Local per-molecule accumulators:
  //   com_p[m][3]    = sum(m_i * v_i)  (COM momentum)
  //   com_r[m][3]    = sum(m_i * r_i)  (mass-weighted position for COM)
  //   L_lab[m][3]    = angular momentum in lab frame (computed after COM)
  //   I_ten[m][3][3] = inertia tensor (computed after COM)

  int nm = nmol_group;
  std::vector<double> lcom_p(nm*3, 0.0), gcom_p(nm*3, 0.0);
  std::vector<double> lcom_r(nm*3, 0.0), gcom_r(nm*3, 0.0);
  std::vector<double> lL(nm*3, 0.0),     gL(nm*3, 0.0);
  std::vector<double> lI(nm*9, 0.0),     gI(nm*9, 0.0);

  // Pass 1: accumulate COM momentum and mass-weighted position (unwrapped).
  compute_mol_pass1_atoms(ibuf_vel, lcom_p.data(), lcom_r.data());
  reduce_or_copy(lcom_p.data(), gcom_p.data(), nm*3);
  reduce_or_copy(lcom_r.data(), gcom_r.data(), nm*3);

  // COM velocity = momentum / total mass
  // COM position = mass-weighted sum / total mass
  std::vector<double> vcom(nm*3), rcom(nm*3);
  for (int m = 0; m < nm; m++) {
    double M = molmass[m];
    for (int d = 0; d < 3; d++) {
      vcom[m*3+d] = gcom_p[m*3+d] / M;
      rcom[m*3+d] = gcom_r[m*3+d] / M;
    }
  }

  // Pass 2: accumulate angular momentum L = sum_i m_i * r'_i × v'_i
  //         and inertia tensor  I_αβ = sum_i m_i (|r'|² δ_αβ - r'_α r'_β)
  compute_mol_pass2_atoms(ibuf_vel, vcom.data(), rcom.data(),
                          lL.data(), lI.data());
  reduce_or_copy(lL.data(), gL.data(), nm*3);
  reduce_or_copy(lI.data(), gI.data(), nm*9);


  // Compute ω = I⁻¹ · L for each molecule
  for (int m = 0; m < nm; m++) {
    for (int d = 0; d < 3; d++)
      com_vel_buf[ibuf_mol][m][d] = vcom[m*3+d];

    double Imat[3][3];
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++)
        Imat[a][b] = gI[m*9 + a*3+b];

    // Diagonalize I → eigenvalues eig, eigenvectors V[:,a]
    double eig[3], V[3][3];
    mat3_sym_eigen(Imat, eig, V);
    // L in principal frame: Lv[a] = V[:,a] · L_lab
    double Lv[3] = {0,0,0};
    for (int a = 0; a < 3; a++)
      for (int d = 0; d < 3; d++) Lv[a] += V[d][a] * gL[m*3+d];

    // anguv_body[a] = Lv[a] / sqrt(I_a)  (zero for linear axis a=0)
    // omega_body[a] = Lv[a] / I_a        (zero for linear axis a=0)
    int lin_start = mol_is_linear[m] ? 1 : 0;
    double anguv_body[3] = {0,0,0};
    double omega_body[3] = {0,0,0};
    for (int a = lin_start; a < 3; a++) {
      if (eig[a] > 1e-30) {
        anguv_body[a] = Lv[a] / sqrt(eig[a]);
        omega_body[a] = Lv[a] / eig[a];
      }
    }

    // Rotate anguv and omega back to lab frame
    double anguv_lab[3] = {0,0,0}, omega_lab[3] = {0,0,0};
    for (int d = 0; d < 3; d++)
      for (int a = 0; a < 3; a++) {
        anguv_lab[d] += V[d][a] * anguv_body[a];
        omega_lab[d] += V[d][a] * omega_body[a];
      }
    // omega_buf  stores anguv = L/√I (angular unit velocity) for rotational DoS
    // angmom_buf stores ω = L/I   (angular velocity)         for vib. subtraction (Pass 3)
    for (int d = 0; d < 3; d++) omega_buf[ibuf_mol][m][d]  = anguv_lab[d];
    for (int d = 0; d < 3; d++) angmom_buf[ibuf_mol][m][d] = omega_lab[d];


    // Accumulate I eigenvalues for window average
    if (mol_is_linear[m]) {
      mol_inertia[m][0] += eig[1];
      mol_inertia[m][1] += eig[2];
      mol_inertia[m][2] += 0.0;
    } else {
      for (int d = 0; d < 3; d++) mol_inertia[m][d] += eig[d];
    }
  }
  mol_inertia_count++;


  // Pass 3: per-atom vibrational velocity v_vib,i = v_i - v_COM - ω×r'_i
  // (omega_lab per molecule lives in angmom_buf[ibuf_mol][m][:]).  Accumulate
  // into a zero-initialized local buffer then MPI_Allreduce(SUM) to assemble
  // the global frame — same atom-migration fix as vel_buf: without it, atoms
  // that migrate between domains leave stale vib velocities on the origin rank.
  {
    std::vector<double> omega_lab_flat(nm * 3);
    for (int m = 0; m < nm; m++)
      for (int d = 0; d < 3; d++)
        omega_lab_flat[m*3+d] = angmom_buf[ibuf_mol][m][d];
    std::vector<double> vib_local(natom_buf * 3, 0.0);
    compute_mol_pass3_atoms(ibuf_vel, vcom.data(), rcom.data(),
                            omega_lab_flat.data(), vib_local.data());
    if (buffer_precision == BUFFER_FP32) {
      // Convert + reduce-or-copy into FP32 destination.
      std::vector<float> vib_local_f(natom_buf * 3);
      for (int j = 0; j < natom_buf * 3; j++)
        vib_local_f[j] = (float)vib_local[j];
      reduce_or_copy_f(vib_local_f.data(),
                       vib_vel_buf_f[ibuf_mol][0], natom_buf * 3);
    } else {
      reduce_or_copy(vib_local.data(),
                     vib_vel_buf[ibuf_mol][0], natom_buf * 3);
    }
  }


  // Multi-tau per-channel push.  Lazy alloc on the
  // first frame of the first window (when nmol_group is known).  Then
  // every subsequent step pushes the three channels into their
  // respective ring buffers.
  if (correlator == CORR_MULTITAU && mt_n_levels > 0) {
    if (mt_trans.n_units != nmol_group) {
      multitau_stream_alloc(mt_trans, nmol_group, /*distributed=*/false);
    }
    if (mt_rot.n_units != nmol_group) {
      multitau_stream_alloc(mt_rot,   nmol_group, /*distributed=*/false);
    }
    if (mt_vib.n_units != natom_buf) {
      multitau_stream_alloc(mt_vib,   natom_buf,  /*distributed=*/true);
    }
    if (iframe == 0) {
      multitau_stream_reset(mt_trans);
      multitau_stream_reset(mt_rot);
      multitau_stream_reset(mt_vib);
    }
    mt_molecular_active = true;

    // Unit-weight vector for the rotational channel (vac_rot uses omega_buf
    // with weight = 1.0 in the FFT path).
    std::vector<double> ones(nmol_group, 1.0);
    {
      FIXXPT_TIME(t_multitau_push);
      multitau_stream_push(mt_trans, 0, &com_vel_buf[ibuf_mol][0][0],
                              molmass.data(), nullptr);
      multitau_stream_push(mt_rot,   0, &omega_buf[ibuf_mol][0][0],
                              ones.data(),  nullptr);
      multitau_stream_push(mt_vib,   0, &vib_vel_buf[ibuf_mol][0][0],
                              mass_buf,    &group_slots);
    }
  }
}

/* ======================================================================
   compute_mol_pass1_atoms — default host impl of the Pass 1 per-atom loop.

   Writes LOCAL accumulators lcom_p/lcom_r (COM momentum + mass-weighted
   position, per molecule).  MUST NOT MPI-reduce: the caller
   (accumulate_mol_frame) zero-inits the buffers and runs the Allreduce(SUM)
   host-side, independent of where this atom loop ran.
====================================================================== */

void FixXPT::compute_mol_pass1_atoms(int ibuf_vel,
                                     double *lcom_p,
                                     double *lcom_r)
{
  int       *mask   = atom->mask;
  int       *tag    = atom->tag;
  double   **x      = atom->x;
  imageint  *image  = atom->image;
  double    *amass  = atom->mass;
  int       *type   = atom->type;
  int        nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    int s = tag[i] - 1;
    if (s >= (int)slot_to_mol.size() || s >= natom_buf) continue;
    int m = slot_to_mol[s];
    if (m < 0) continue;
    double mi = amass[type[i]];
    double xu[3];
    domain->unmap(x[i], image[i], xu);
    for (int d = 0; d < 3; d++) {
      lcom_p[m*3+d] += mi * vread(ibuf_vel, s, d);
      lcom_r[m*3+d] += mi * xu[d];
    }
  }
}

/* ======================================================================
   compute_mol_pass2_atoms — default host impl of Pass 2 (Phase 5c).
   Accumulates per-mol L = Σ m·r'×v' and I = Σ m·(|r'|²δ - r'⊗r') locally.
====================================================================== */

void FixXPT::compute_mol_pass2_atoms(int           ibuf_vel,
                                     const double *vcom,
                                     const double *rcom,
                                     double       *lL,
                                     double       *lI)
{
  int       *mask   = atom->mask;
  int       *tag    = atom->tag;
  double   **x      = atom->x;
  imageint  *image  = atom->image;
  double    *amass  = atom->mass;
  int       *type   = atom->type;
  int        nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    int s = tag[i] - 1;
    if (s >= (int)slot_to_mol.size() || s >= natom_buf) continue;
    int m = slot_to_mol[s];
    if (m < 0) continue;
    double mi = amass[type[i]];
    double xu[3];
    domain->unmap(x[i], image[i], xu);
    double rp[3], vp[3];
    for (int d = 0; d < 3; d++) {
      rp[d] = xu[d] - rcom[m*3+d];
      vp[d] = vread(ibuf_vel, s, d) - vcom[m*3+d];
    }
    // L += m * r' × v'
    double Lcontrib[3];
    cross3(rp, vp, Lcontrib);
    for (int d = 0; d < 3; d++) lL[m*3+d] += mi * Lcontrib[d];
    // I += m * (|r'|²·I₃ - r'⊗r')
    double r2 = rp[0]*rp[0] + rp[1]*rp[1] + rp[2]*rp[2];
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++) {
        double delta = (a == b) ? 1.0 : 0.0;
        lI[m*9 + a*3+b] += mi * (r2 * delta - rp[a] * rp[b]);
      }
  }
}

/* ======================================================================
   compute_mol_pass3_atoms — default host impl of Pass 3 (Phase 5c).
   Per-atom vib velocity v_vib,i = v_i - v_COM - ω × r'_i written into
   vib_local[s*3..s*3+2]; entries for atoms not in the group remain zero
   (caller zero-initialised), preserving the atom-migration MPI_Allreduce
   correctness invariant.
====================================================================== */

void FixXPT::compute_mol_pass3_atoms(int           ibuf_vel,
                                     const double *vcom,
                                     const double *rcom,
                                     const double *omega_lab,
                                     double       *vib_local)
{
  int       *mask   = atom->mask;
  int       *tag    = atom->tag;
  double   **x      = atom->x;
  imageint  *image  = atom->image;
  int        nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    int s = tag[i] - 1;
    if (s >= natom_buf || s >= (int)slot_to_mol.size()) continue;
    int m = slot_to_mol[s];
    if (m < 0) continue;
    double xu[3];
    domain->unmap(x[i], image[i], xu);
    double rp[3];
    for (int d = 0; d < 3; d++) rp[d] = xu[d] - rcom[m*3+d];
    const double omx = omega_lab[m*3+0];
    const double omy = omega_lab[m*3+1];
    const double omz = omega_lab[m*3+2];
    double vrot[3];
    vrot[0] = omy*rp[2] - omz*rp[1];
    vrot[1] = omz*rp[0] - omx*rp[2];
    vrot[2] = omx*rp[1] - omy*rp[0];
    for (int d = 0; d < 3; d++)
      vib_local[s*3+d] = vread(ibuf_vel, s, d) - vcom[m*3+d] - vrot[d];
  }
}

/* ======================================================================
   cross3 — c = a × b
====================================================================== */

void FixXPT::cross3(const double a[3], const double b[3], double c[3])
{
  c[0] = a[1]*b[2] - a[2]*b[1];
  c[1] = a[2]*b[0] - a[0]*b[2];
  c[2] = a[0]*b[1] - a[1]*b[0];
}


/* ======================================================================
   mat3_sym_eigen — eigenvalues+eigenvectors of 3×3 symmetric A via
   Jacobi sweeps.  eig[3] returned ascending; cols of V are eigenvectors.
====================================================================== */

void FixXPT::mat3_sym_eigen(const double A[3][3], double eig[3], double V[3][3])
{
  double a[3][3];
  for (int i=0;i<3;i++) { eig[i]=0; for(int j=0;j<3;j++) { a[i][j]=A[i][j]; V[i][j]=(i==j)?1:0; } }

  for (int sweep = 0; sweep < 50; sweep++) {
    double offdiag = fabs(a[0][1]) + fabs(a[0][2]) + fabs(a[1][2]);
    if (offdiag < 1e-15) break;
    for (int p=0; p<2; p++) for (int q=p+1; q<3; q++) {
      if (fabs(a[p][q]) < 1e-15) continue;
      double theta = 0.5*(a[q][q]-a[p][p]) / a[p][q];
      double t = (theta>=0) ? 1.0/(theta+sqrt(1+theta*theta)) : 1.0/(theta-sqrt(1+theta*theta));
      double c = 1.0/sqrt(1+t*t), s = t*c;
      double tau = s/(1+c);
      // Update a
      double apq = a[p][q]; a[p][q] = a[q][p] = 0;
      a[p][p] -= t*apq; a[q][q] += t*apq;
      for (int r=0; r<3; r++) if (r!=p && r!=q) {
        double apr=a[p][r], aqr=a[q][r];
        a[p][r]=a[r][p]=apr-s*(aqr+tau*apr);
        a[q][r]=a[r][q]=aqr+s*(apr-tau*aqr);
      }
      // Update V
      for (int r=0; r<3; r++) {
        double vpr=V[r][p], vqr=V[r][q];
        V[r][p]=vpr-s*(vqr+tau*vpr);
        V[r][q]=vqr+s*(vpr-tau*vqr);
      }
    }
  }
  eig[0]=a[0][0]; eig[1]=a[1][1]; eig[2]=a[2][2];
  // Sort ascending
  for (int i=0; i<2; i++) for (int j=i+1; j<3; j++) if (eig[i]>eig[j]) {
    std::swap(eig[i],eig[j]);
    for (int r=0;r<3;r++) std::swap(V[r][i],V[r][j]);
  }
}


