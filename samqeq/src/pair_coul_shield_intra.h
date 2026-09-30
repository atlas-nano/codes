/* -*- c++ -*- ----------------------------------------------------------
   coul/shield/intra — INTRA-molecular Coulomb shielding correction for samQEq.

   samQEq's long-range solve uses a ReaxFF-shielded J(r)=1/∛(r³+1/γ³) for INTRA-
   molecular (same-molecule) pairs and bare 1/r for inter-molecular, while the
   pair forces come from lj/cut/coul/long + pppm (bare 1/r everywhere). This
   pair_style overlays the missing piece so the FORCES/ENERGY match the solve:
   for intra pairs it adds qqrd2e·q_i·q_j·(J_shield(r) − 1/r) (a short-range,
   →0-at-large-r correction). Net intra = J_shield; inter untouched (=bare 1/r).
   Use: pair_style hybrid/overlay lj/cut <rc> coul/long <rc> coul/shield/intra <rc_s>
         pair_coeff * * coul/shield/intra <gamma> (gamma = the samQEq shielding)
   γ_ij defaults to √(γ_ii·γ_jj) for unset pairs (matches FixQEqSam compute_H).
-------------------------------------------------------------------------*/

#ifdef PAIR_CLASS
// clang-format off
PairStyle(coul/shield/intra,PairCoulShieldIntra);
// clang-format on
#else

#ifndef LMP_PAIR_COUL_SHIELD_INTRA_H
#define LMP_PAIR_COUL_SHIELD_INTRA_H

#include "pair.h"
#include <cmath>
#include "slater_jtable.h"   // Rick Slater-overlap J(r) kernel + per-type-pair table (settings keyword `slater`)
#include "samqeq_ttdamp.h"   // Tang-Toennies f_n(br) for the damped interionic kernel (settings keyword `iondamp`)

#include <vector>

namespace LAMMPS_NS {

class PairCoulShieldIntra : public Pair {
 public:
  PairCoulShieldIntra(class LAMMPS *);
  ~PairCoulShieldIntra() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  void init_list(int, class NeighList *) override;   // special_bonds + exclusion guards
  double init_one(int, int) override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_restart_settings(FILE *) override;
  void read_restart_settings(FILE *) override;
  double single(int, int, int, int, double, double, double, double &) override;

  // iondamp read-only accessors for the FIX-side consistency cross-check: FixQEqSam::init() compares its
  // fix_modify iondamp table against this pair's settings and WARNS on any mismatch (a silent fix/pair split
  // means the solve and the forces disagree on the ion-ion kernel).
  int    iondamp_active() const { return iondamp_on; }
  double iondamp_bget(int i, int j) const { return iondamp_b ? iondamp_b[i][j] : 0.0; }
  int    iondamp_nget(int i, int j) const { return iondamp_n ? iondamp_n[i][j] : 4; }
  // SHIELDING-KERNEL CONSISTENCY: read-only accessors for the four pair<->fix divergence channels,
  // asserted in FixQEqSam::check_shield_consistency(). The pair style and the fix each select the
  // shielding kernel INDEPENDENTLY; a split means the forces integrate one kernel while the charge
  // solve equilibrates against another, which breaks energy conservation.
  double shield_cut_get() const    { return cut_global; }     // channel 1: range (vs the fix's swb)
  int    shield_mode_get() const   { return shield_gauss; }   // channel 2: cbrt|gaussian|slater
  double shield_lambda_get() const { return shield_lambda; }  // channel 3: PQEq lambda
  int    shield_is2s_get(int t) const { return is2s ? is2s[t] : 0; }   // channel 4: Slater 2s/1s per type
  int    shield_intra_only() const { return intra_only; }     // scope (intra|all), for the error message
  // Per-type-pair PQEq Gaussian radius override, pushed by fix qeq/sam (single source of truth)
  void   shield_rpair_set(int i, int j, double r);
  double shield_rpair_get(int i, int j) const { return shield_rpair ? shield_rpair[i][j] : 0.0; }
  inline double pqeq_aij(int itype, int jtype) const {
    if (shield_rpair && shield_rpair[itype][jtype] > 0.0) return sqrt(shield_lambda)/(2.0*shield_rpair[itype][jtype]);
    const double ai = shield_lambda*0.5/(gamma[itype][itype]*gamma[itype][itype]);
    const double aj = shield_lambda*0.5/(gamma[jtype][jtype]*gamma[jtype][jtype]);
    return sqrt(ai*aj/(ai+aj));
  }

 protected:
  double cut_global;
  int intra_only;        // 1 = correct intra-molecular pairs only (lr_ewald=1); 0 = all pairs (lr_ewald=2)
  // shield_gauss: TRI-STATE kernel mode (see fix_qeq_sam.h for the fuller rationale -- this pair mirrors the fix's shielded_coulomb() so forces stay consistent with the solve).
  static constexpr int SHIELD_CBRT = 0, SHIELD_GAUSSIAN = 1, SHIELD_SLATER = 2;
  int shield_gauss;      // 0 = cbrt J_shield (default); 1 = PQEq Gaussian erf(α_ij r)/r; 2 = Slater J(r) (Rick)
  double shield_lambda;  // PQEq λ (Gaussian-overlap → shielding); α_i = λ/(2 Rc_i²), Rc = per-type diagonal gamma[i][i]
  double **cut;
  double **gamma;        // per-type-pair shielding γ_ij (cbrt: J_shield=1/∛(r³+1/γ³)); when gauss/slater, diagonal
  double **shield_rpair = nullptr;   // per-type-pair Gaussian radius override (0 = combination rule)
                         // gamma[i][i] = Rc_i (pqeq) or zeta_i, the Slater exponent 1/Å (slater) -- per-type only.
  int *is2s;             // SLATER per-type 2s(O/M)-vs-1s(H) flag (ntypes+1, default 0=1s); settings `2s <t>...`
  std::vector<SlaterJTable> slater_tabs;   // SLATER per-(ti<=tj)-type-pair table (built in init_style())
  void build_slater_tables();
  // IONDAMP: Tang-Toennies-damped interionic kernel for
  // DESIGNATED type pairs -- E = K·(f_n(b r)·J(r) − factor_coul/r), mirroring the fix's damped
  // shielded_coulomb() so pair forces/energy stay consistent with the solve (as for slater; the TT
  // function is the shared samqeq_ttdamp.h -- one transcription). Settings keyword (repeatable):
  // `iondamp <typeI> <typeJ> <b> [<n>]` (wildcards ok; b 1/Angstrom; n = TT order 1..8, default 4).
  // Allocated at settings() time like is2s (independent of allocate()); NOT persisted in restarts
  // (write_restart_settings' byte format is fixed) -- re-issue the pair_style line after read_restart.
  // iondamp_on==0 (default) skips every iondamp code path.
  int      iondamp_on;   // 1 iff any type pair carries b>0
  double **iondamp_b;    // per-type-pair TT exponent b_ij (1/Å); 0 => pair undamped ((ntypes+1)^2, symmetric)
  int    **iondamp_n;    // per-type-pair TT order n (default 4)

  virtual void allocate();
};

}    // namespace LAMMPS_NS
#endif
#endif
