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
   pppm/samqeq : a stock-PPPM subclass that exposes the per-atom reciprocal-
   space Coulomb potential (compute_vector) needed by fix qeq/sam's
   long-range (lr_ewald) charge solve, WITHOUT depending on the ELECTRODE
   package. The compute_vector / start_compute / make_rho_in_brick /
   project_psi machinery is the per-atom-potential subset of PPPMElectrode
   (Ahrens-Iwers, Tee, Meissner), reimplemented as a minimal subclass of the
   current PPPM: it reuses stock PPPM's grid setup (new Grid3d API), and only
   adds the electrolyte (source-group) density bricks + the potential
   projection. The Yeh-Berkowitz EW3DC slab correction (kspace_modify slab
   <volfactor>, slabflag==1) IS supported: forces/energy via the inherited
   PPPM::slabcorr(), and the SAME dipole term added analytically to the
   per-atom potential in compute_vector so fix qeq/sam's charge solve stays
   consistent with the forces. Wire corrections, slab nozforce (slabflag==2),
   the ELECTRODE package's matrix methods, and group/group interactions are
   intentionally NOT ported.
-------------------------------------------------------------------------*/

#ifdef KSPACE_CLASS
// clang-format off
KSpaceStyle(pppm/samqeq, PPPMSamqeq);
// clang-format on
#else

#ifndef LMP_PPPM_SAMQEQ_H
#define LMP_PPPM_SAMQEQ_H

#include "pppm.h"
#include "samqeq_kspace.h"
#include <vector>

namespace LAMMPS_NS {

class PPPMSamqeq : public PPPM, public SamqeqKspace {
 public:
  PPPMSamqeq(class LAMMPS *);
  ~PPPMSamqeq() override;

  // Reject the geometries this subclass does not implement in compute_vector: triclinic boxes and
  // slab nozforce (slabflag==2). The EW3DC slab correction (slabflag==1) is supported: PPPM::compute()
  // (which our compute() chains to) applies the inherited slabcorr() to the FORCES/energy, and
  // compute_vector adds the matching dipole term to the per-atom potential (fix qeq/sam's charge-solve
  // RHS/matvec), so solve and force stay consistent (same E functional -> no nonconservation).
  void init() override;

  // Per-atom reciprocal-space potential of the source group's charges, sampled
  // at the sensor group's atoms. ACCUMULATES into vec (caller pre-zeros);
  // returns the RAW reciprocal potential (no qqrd2e prefactor) so fix qeq/sam
  // can scale it consistently with its real-space erfc block. invert_source
  // flips group membership. Mirrors PPPMElectrode::compute_vector.
  // Under kspace_modify slab (slabflag==1) the EW3DC dipole term
  // dE_slab/dq_i = (2pi/V_slab)(2 z_i M_z - S2 - qsum z_i^2 - qsum L^2/6),
  // with M_z/S2/qsum summed over the SOURCE group, is added for sensor atoms —
  // the exact charge-gradient of PPPM::slabcorr()'s energy, so the charge solve
  // minimizes the same functional whose forces slabcorr() applies.
  // implements SamqeqKspace; pppm/samqeq/kk provides a device FFT version (Phase 3). The fix
  // dynamic_casts force->kspace to SamqeqKspace*, so the right backend's compute_vector runs.
  void compute_vector(double *vec, int sensor_grpbit, int source_grpbit, bool invert_source) override;

  // Refresh qsqsum/qsum from the CURRENT charges before forming the reciprocal energy. Stock PPPM caches
  // them at setup() and only re-measures on atom-COUNT change (correct for FIXED point charges), so for
  // fix qeq/sam's fluctuating charges the reported E_long carries a stale self-energy (-g_ewald*qsqsum/sqrt(pi))
  // + net-charge term frozen at the setup-time charges -> E_long wrong by qscale*g_ewald*Δqsqsum/sqrt(pi).
  // Position-independent => ZERO force impact (dynamics unaffected); this corrects the reported energy only.
  void compute(int eflag, int vflag) override;

  // R4: exact per-atom grid self-coefficient (see samqeq_kspace.h). Kernel table rebuilt when the grid
  // signature {nx,ny,nz,g_ewald,box} changes (one backward FFT); per atom ~ (2P-1)^3 flops.
  bool compute_self_peratom(double *out, int grpbit) override;
  double self_image_term() const override { return self_xi; }

 protected:
  FFT_SCALAR ***electrolyte_density_brick;    // source-group density on the 3d brick grid
  FFT_SCALAR *electrolyte_density_fft;         // ... in FFT decomposition

  void allocate() override;            // PPPM::allocate + electrolyte bricks + always own u_brick
  void deallocate() override;
  void allocate_peratom() override;    // like PPPM but u_brick is owned by allocate(), not here
  void deallocate_peratom() override;


 private:
  int compute_step;    // last timestep particle_map was refreshed (-1 = needs setup)
  void start_compute();
  void make_rho_in_brick(int source_grpbit, FFT_SCALAR ***scratch_brick, bool invert_source);
  void project_psi(double *vec, int sensor_grpbit);
  // R4: real-space grid kernel Kr(D) for |D| <= order-1 in each direction ((2P-1)^3 values, replicated)
  std::vector<double> self_kern;
  int self_kern_nx = -1, self_kern_ny = -1, self_kern_nz = -1;
  double self_kern_gewald = 0.0, self_kern_prd[3] = {0.0, 0.0, 0.0};
  double self_xi = 0.0;   // analytic Ewald image self-potential of the (grid-)periodic box, see build_self_kernel
  void build_self_kernel();
};

}    // namespace LAMMPS_NS

#endif
#endif
