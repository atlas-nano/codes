/* -*- c++ -*- ----------------------------------------------------------
   compute dipole/samqeq — system + per-molecule dipole for samQEq/samQEq
   fluctuating-charge models (point charges; q from fix qeq/sam).

   Modeled on compute_dipole_pqeq (total dipole) but cleaned for point
   charges (no core/shell rsx, no hardcoded box) and EXTENDED with the
   per-molecule mean dipole — the headline FQ observable (e.g. SPC-FQ
   liquid 2.69 D) — so it can be read straight from a thermo column.
-------------------------------------------------------------------------*/

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(dipole/samqeq,ComputeDipoleSamQEq);
// clang-format on
#else

#ifndef LMP_COMPUTE_DIPOLE_SAMQEQ_H
#define LMP_COMPUTE_DIPOLE_SAMQEQ_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeDipoleSamQEq : public Compute {
 public:
  ComputeDipoleSamQEq(class LAMMPS *, int, char **);
  ~ComputeDipoleSamQEq() override;
  void init() override {}
  double compute_scalar() override;   // = mean per-molecule |dipole| in Debye (the FQ headline number)
  void compute_vector() override;     // [0..2]=total dipole (e·Å), [3]=|total| (e·Å), [4]=<|µ_mol|> (Debye)

 private:
  double conv;            // e·Å -> Debye (4.803204...)
  int unit_debye;         // 1 = report the vector in Debye too (keyword "debye"); default 0 = e·Å
  // per-molecule accumulators (indexed by molecule-id; spans procs via Allreduce)
  bigint nmol_alloc;
  double *moldip;         // 3*nmol local Σ q_i r_i (unwrapped) per molecule
  double *moldip_all;     // reduced
  double *molcnt, *molcnt_all;   // atom count per molecule (to count populated molecules)
  void grow_mol(bigint);
};

}    // namespace LAMMPS_NS
#endif
#endif
