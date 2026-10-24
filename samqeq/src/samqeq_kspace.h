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

#include "atom.h"
#include "math_const.h"
#include <mpi.h>

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
  // Stock PPPM::init() sizes the mesh from the charges present at `run` start (error estimate proportional to Sum q^2),
  // so a run that starts at q~0 would keep a coarse grid -- and the recip_self calibrated on it -- for its whole length.
  // If the current mesh misses the kspace accuracy for the CURRENT atom->q at the HELD g_ewald (the real-space pair
  // cached it at init_style), re-size by the stock search at that g_ewald and adopt the result if it is finer, via a
  // full init(). Returns true iff the mesh changed; old/new dims in od/nd. No-op when the mesh is pinned.
  virtual bool resize_mesh_from_charges(int * /*od*/, int * /*nd*/) { return false; }
  // The OPERATOR self term. fix qeq/sam's solve removes the grid self-coefficient c (calibrated or pinned scalar
  // recip_self, or per-atom c_i under `recip_self peratom`) from the reciprocal diagonal; stock PPPM's E_long removes
  // the analytic 2g/sqrt(pi). The fix pushes the c it uses (add_reciprocal, host and device); compute() then adds
  // qscale*sum_{i in grp} (g/sqrt(pi) - c_i/2) q_i^2, so pe is the energy the solve minimises (mu = dpe/dq). No push
  // (fixed-charge decks, no reciprocal in the solve) => 0 => stock energy. Position-independent => no force, no virial.
  // c_atom (if non-null) is indexed by LOCAL atom and valid for i < c_len; beyond that the scalar c is used.
  void set_operator_self(double c, const double *c_atom, int c_len, int grpbit) { op_self_c = c; op_self_atom = c_atom;
    op_self_len = c_atom ? c_len : 0; op_self_grpbit = grpbit; }
  void clear_operator_self() { op_self_grpbit = 0; op_self_atom = nullptr; op_self_len = 0; op_self_c = 0.0; }
 protected:
  double op_self_c = 0.0;
  const double *op_self_atom = nullptr;
  int op_self_len = 0;
  int op_self_grpbit = 0;
  int op_self_logged = 0;   // one log line per run (reset by the PPPM init)
  // returns the GLOBAL energy delta (collective: call on every rank); adds the per-atom share to eatom if non-null
  double operator_self_delta(Atom *atom, MPI_Comm world, double g_ewald, double qscale, double *eatom) const {
    if (!op_self_grpbit) return 0.0;
    const double *q = atom->q; const int *mask = atom->mask; const int nlocal = atom->nlocal;
    const double ga = g_ewald / MathConst::MY_PIS;
    double loc = 0.0;
    for (int i = 0; i < nlocal; i++) { if (!(mask[i] & op_self_grpbit)) continue;
      const double c = (op_self_atom && i < op_self_len) ? op_self_atom[i] : op_self_c;
      const double e = qscale * (ga - 0.5*c) * q[i]*q[i];
      loc += e; if (eatom) eatom[i] += e; }
    double tot = 0.0; MPI_Allreduce(&loc, &tot, 1, MPI_DOUBLE, MPI_SUM, world);
    return tot;
  }
 protected:
  // protected + non-virtual: objects are only ever deleted through KSpace*, never through this
  // mix-in interface, so no virtual dtor is needed (and a virtual one would clash with PPPM's
  // noexcept(false) destructor exception spec).
  ~SamqeqKspace() = default;
};

}    // namespace LAMMPS_NS

#endif
