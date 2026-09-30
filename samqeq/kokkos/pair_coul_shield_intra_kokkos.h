/* -*- c++ -*- ----------------------------------------------------------
   coul/shield/intra/kk — Kokkos (device) port of coul/shield/intra.

   Mirrors pair_coul_cut_kokkos: reuses the generic pair_compute machinery
   (PairComputeFunctor / pair_compute_neighlist) and only supplies
   compute_fcoul / compute_ecoul. The intra-molecular shielding gate is
   applied INSIDE those device functions by reading the per-atom molecule
   view at i and j (both passed live by the functor):

     intra_only==1 (lr_ewald=1): correct same-molecule pairs only.
     intra_only==0 (lr_ewald=2): correct all in-cutoff pairs.

   Net intra Coulomb = J_shield(r) = 1/∛(r³+1/γ³); inter untouched. This
   keeps the kk pair forces/energy byte-consistent with the CPU pair and
   with FixQEqSam's solve, unblocking production -sf kk samQEq decks (the
   CPU-only pair otherwise crashes the kk Serial neighbor build).
-------------------------------------------------------------------------*/

#ifdef PAIR_CLASS
// clang-format off
PairStyle(coul/shield/intra/kk,PairCoulShieldIntraKokkos<LMPDeviceType>);
PairStyle(coul/shield/intra/kk/device,PairCoulShieldIntraKokkos<LMPDeviceType>);
PairStyle(coul/shield/intra/kk/host,PairCoulShieldIntraKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_COUL_SHIELD_INTRA_KOKKOS_H
#define LMP_PAIR_COUL_SHIELD_INTRA_KOKKOS_H

#include "pair_kokkos.h"
#include "pair_coul_shield_intra.h"
#include "neigh_list_kokkos.h"

namespace LAMMPS_NS {

template<class DeviceType>
class PairCoulShieldIntraKokkos : public PairCoulShieldIntra {
 public:
  enum {EnabledNeighFlags=FULL|HALFTHREAD|HALF};
  enum {COUL_FLAG=1};
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  PairCoulShieldIntraKokkos(class LAMMPS *);
  ~PairCoulShieldIntraKokkos() override;

  void compute(int, int) override;

  void init_style() override;
  double init_one(int, int) override;

  struct params_coulshield {
// NOLINTNEXTLINE
    KOKKOS_INLINE_FUNCTION
    params_coulshield() {cutsq=0;gamma=0;aij=0;};
// NOLINTNEXTLINE
    KOKKOS_INLINE_FUNCTION
    params_coulshield(int /*i*/) {cutsq=0;gamma=0;aij=0;};
    // aij: PQEq per-type-pair shielding exponent alpha_ij = sqrt(ai*aj/(ai+aj)), ai = shield_lambda*0.5/Rc_i^2
    // (Rc_i = the per-type diagonal gamma[i][i]); precomputed on the HOST in init_one() (mirrors the CPU pair's
    // per-call computation in compute()/single()). Only meaningful when shield_gauss==SHIELD_GAUSSIAN; left 0
    // (unused) in cbrt mode -- does not affect the untouched cbrt math.
    KK_FLOAT cutsq, gamma, aij;
  };

  // Slater (Rick J(r), shield_gauss==SHIELD_SLATER) device table (, backlog item b): dense
  // [itype][jtype][node] view, built in init_style() from the CPU base class' `slater_tabs`
  // (PairCoulShieldIntra::init_style() -> build_slater_tables() runs FIRST, see the .cpp) via a
  // host mirror + one deep_copy. Every type-pair table shares the SAME (rmin,dr,npts) grid (see
  // slater_jtable.h: SlaterJTable::build always spans [rmin, cut_global] with a fixed npoints), so
  // only one scalar triple is stored, not per-pair. Left default-constructed (empty, zero device
  // memory) for cbrt/pqeq decks -- only allocated when shield_gauss==SHIELD_SLATER.
  typedef Kokkos::View<KK_FLOAT***, Kokkos::LayoutRight, DeviceType> t_slater_tab;
  t_slater_tab d_slater_J, d_slater_dJ;
  KK_FLOAT slater_rmin, slater_dr;
  int slater_npts;

  // Device mirror of SlaterJTable::eval() (slater_jtable.h) -- same Hermite-cubic basis, reading
  // from the device table instead of a std::vector. Kept byte-for-byte structurally identical to
  // the host version so kk and CPU interpolate identically given the identical table.
  KOKKOS_INLINE_FUNCTION
  void slater_eval(int itype, int jtype, const KK_FLOAT& r, KK_FLOAT& J, KK_FLOAT& dJdr) const
  {
    if (slater_npts < 2 || r <= slater_rmin) {
      J = d_slater_J(itype,jtype,0);
      dJdr = d_slater_dJ(itype,jtype,0);
      return;
    }
    KK_FLOAT x = (r - slater_rmin)/slater_dr;
    int n0 = (int) x;
    if (n0 >= slater_npts - 1) n0 = slater_npts - 2;
    const KK_FLOAT t = x - n0;
    const KK_FLOAT h = slater_dr;
    const KK_FLOAT J0 = d_slater_J(itype,jtype,n0),   J1 = d_slater_J(itype,jtype,n0+1);
    const KK_FLOAT D0 = d_slater_dJ(itype,jtype,n0),  D1 = d_slater_dJ(itype,jtype,n0+1);
    const KK_FLOAT t2 = t*t, t3 = t2*t;
    const KK_FLOAT h00 = 2.0*t3 - 3.0*t2 + 1.0, h10 = t3 - 2.0*t2 + t;
    const KK_FLOAT h01 = -2.0*t3 + 3.0*t2,       h11 = t3 - t2;
    const KK_FLOAT h00d = 6.0*t2 - 6.0*t,        h10d = 3.0*t2 - 4.0*t + 1.0;
    const KK_FLOAT h01d = -6.0*t2 + 6.0*t,       h11d = 3.0*t2 - 2.0*t;
    dJdr = (h00d*J0 + h10d*h*D0 + h01d*J1 + h11d*h*D1) / h;
    J = h00*J0 + h10*h*D0 + h01*J1 + h11*h*D1;
  }

 protected:
  template<bool STACKPARAMS, class Specialisation>
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  KK_FLOAT compute_fpair(const KK_FLOAT& /*rsq*/, const int& /*i*/, const int& /*j*/,
                        const int& /*itype*/, const int& /*jtype*/) const { return 0.0; }

  template<bool STACKPARAMS, class Specialisation>
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  KK_FLOAT compute_fcoul(const KK_FLOAT& rsq, const int& i, const int&j,
                        const int& itype, const int& jtype, const KK_FLOAT& factor_coul, const KK_FLOAT& qtmp) const;

  template<bool STACKPARAMS, class Specialisation>
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  KK_FLOAT compute_evdwl(const KK_FLOAT& /*rsq*/, const int& /*i*/, const int& /*j*/,
                        const int& /*itype*/, const int& /*jtype*/) const { return 0; }

  template<bool STACKPARAMS, class Specialisation>
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  KK_FLOAT compute_ecoul(const KK_FLOAT& rsq, const int& i, const int&j,
                        const int& itype, const int& jtype, const KK_FLOAT& factor_coul, const KK_FLOAT& qtmp) const;

  // true when this in-cutoff pair must receive the shield−bare correction
  // (intra_only==0 -> all pairs; intra_only==1 -> same nonzero molecule).
  KOKKOS_INLINE_FUNCTION
  bool shielded_pair(const int& i, const int& j) const {
    if (!intra_only) return true;
    const tagint mi = d_molecule(i);
    return (mi != 0 && mi == d_molecule(j));
  }

  Kokkos::DualView<params_coulshield**,Kokkos::LayoutRight,DeviceType> k_params;
  typename Kokkos::DualView<params_coulshield**,
    Kokkos::LayoutRight,DeviceType>::t_dev_const_um params;
  // hardwired to space for 12 atom types
  params_coulshield m_params[MAX_TYPES_STACKPARAMS+1][MAX_TYPES_STACKPARAMS+1];

  KK_FLOAT m_cutsq[MAX_TYPES_STACKPARAMS+1][MAX_TYPES_STACKPARAMS+1];
  KK_FLOAT m_cut_ljsq[MAX_TYPES_STACKPARAMS+1][MAX_TYPES_STACKPARAMS+1];
  KK_FLOAT m_cut_coulsq[MAX_TYPES_STACKPARAMS+1][MAX_TYPES_STACKPARAMS+1];
  typename AT::t_kkfloat_1d_3_lr_randomread x;
  typename AT::t_kkfloat_1d_3_lr c_x;
  typename AT::t_kkacc_1d_3 f;
  typename AT::t_kkfloat_1d_randomread q;
  typename AT::t_int_1d_randomread type;
  typename AT::t_tagint_1d_randomread d_molecule;

  DAT::ttransform_kkacc_1d k_eatom;
  DAT::ttransform_kkacc_1d_6 k_vatom;
  typename AT::t_kkacc_1d d_eatom;
  typename AT::t_kkacc_1d_6 d_vatom;

  int newton_pair;

  DAT::ttransform_kkfloat_2d k_cutsq;
  typename AT::t_kkfloat_2d d_cutsq;
  DAT::tdual_kkfloat_2d k_cut_ljsq;
  typename AT::t_kkfloat_2d d_cut_ljsq;
  DAT::tdual_kkfloat_2d k_cut_coulsq;
  typename AT::t_kkfloat_2d d_cut_coulsq;

  int neighflag;
  int nlocal,nall,eflag,vflag;

  KK_FLOAT special_coul[4];
  KK_FLOAT special_lj[4];
  KK_FLOAT qqrd2e;

  void allocate() override;
  friend struct PairComputeFunctor<PairCoulShieldIntraKokkos,FULL,true,0>;
  friend struct PairComputeFunctor<PairCoulShieldIntraKokkos,FULL,true,1>;
  friend struct PairComputeFunctor<PairCoulShieldIntraKokkos,HALF,true>;
  friend struct PairComputeFunctor<PairCoulShieldIntraKokkos,HALFTHREAD,true>;
  friend struct PairComputeFunctor<PairCoulShieldIntraKokkos,FULL,false,0>;
  friend struct PairComputeFunctor<PairCoulShieldIntraKokkos,FULL,false,1>;
  friend struct PairComputeFunctor<PairCoulShieldIntraKokkos,HALF,false>;
  friend struct PairComputeFunctor<PairCoulShieldIntraKokkos,HALFTHREAD,false>;
  friend EV_FLOAT pair_compute_neighlist<PairCoulShieldIntraKokkos,FULL,0>(PairCoulShieldIntraKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairCoulShieldIntraKokkos,FULL,1>(PairCoulShieldIntraKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairCoulShieldIntraKokkos,HALF>(PairCoulShieldIntraKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairCoulShieldIntraKokkos,HALFTHREAD>(PairCoulShieldIntraKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute<PairCoulShieldIntraKokkos,void>(PairCoulShieldIntraKokkos*,
                                                       NeighListKokkos<DeviceType>*);
  friend void pair_virial_fdotr_compute<PairCoulShieldIntraKokkos>(PairCoulShieldIntraKokkos*);

};

}

#endif
#endif
