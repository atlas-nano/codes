/* -*- c++ -*- ----------------------------------------------------------
   SamqeqKspace — tiny abstract interface for the per-atom reciprocal-space
   Coulomb potential that fix qeq/sam needs. Both pppm/samqeq (CPU) and
   pppm/samqeq/kk (device) implement it, so the fix can dynamic_cast to ONE
   type regardless of backend. (PPPMSamqeqKokkos derives from PPPMKokkos, NOT
   PPPMSamqeq, so they have no common concrete base beyond PPPM — this interface
   is the shared compute_vector contract.)
------------------------------------------------------------------------*/

#ifndef LMP_SAMQEQ_KSPACE_H
#define LMP_SAMQEQ_KSPACE_H

namespace LAMMPS_NS {

class SamqeqKspace {
 public:
  // accumulate the RAW reciprocal potential (no qqrd2e) of the source group's charges
  // at the sensor group's atoms into vec[0,nlocal). invert_source flips source membership.
  virtual void compute_vector(double *vec, int sensor_grpbit, int source_grpbit, bool invert_source) = 0;
  // The EXACT per-atom PPPM grid self-coefficient c_i (raw, no
  // qqrd2e) = the reciprocal potential atom i's own unit charge produces at its own position through the grid:
  // c_i = (delvolinv/N_grid) sum_{g,g'} W_i(g) W_i(g') Kr(g-g'), Kr = backward FFT of greensfn. Fills
  // out[i] for local atoms in grpbit. Returns false when the backend cannot (Kokkos) -> caller must error.
  virtual bool compute_self_peratom(double * /*out*/, int /*grpbit*/) { return false; }
  virtual double self_image_term() const { return 0.0; }   // xi = psi_{k!=0}(0) - 2a/sqrt(pi) (+ real images), raw
 protected:
  // protected + non-virtual: objects are only ever deleted through KSpace*, never through this
  // mix-in interface, so no virtual dtor is needed (and a virtual one would clash with PPPM's
  // noexcept(false) destructor exception spec).
  ~SamqeqKspace() = default;
};

}    // namespace LAMMPS_NS

#endif
