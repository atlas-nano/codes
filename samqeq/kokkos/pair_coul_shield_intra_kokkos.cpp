// clang-format off
/* ----------------------------------------------------------------------
   coul/shield/intra/kk — Kokkos (device) port of coul/shield/intra.
   See pair_coul_shield_intra_kokkos.h. Device intra-molecular Coulomb
   shielding correction: adds qqrd2e q_i q_j (J_shield(r) − 1/r) on the GPU
   so -sf kk samQEq decks get the same forces/energy as the CPU pair.
-------------------------------------------------------------------------*/

#include "pair_coul_shield_intra_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "math_const.h"
#include "memory_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"

#include <cmath>

using namespace LAMMPS_NS;

/* ----------------------------------------------------------------------*/

template<class DeviceType>
PairCoulShieldIntraKokkos<DeviceType>::PairCoulShieldIntraKokkos(LAMMPS *lmp) : PairCoulShieldIntra(lmp)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TYPE_MASK | Q_MASK | MOLECULE_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
  slater_npts = 0;
  slater_rmin = 0.0;
  slater_dr = 0.0;
}

/* ----------------------------------------------------------------------*/

template<class DeviceType>
PairCoulShieldIntraKokkos<DeviceType>::~PairCoulShieldIntraKokkos()
{
  if (copymode) return;

  if (allocated) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->destroy_kokkos(k_vatom,vatom);
    memoryKK->destroy_kokkos(k_cutsq, cutsq);
  }
}

/* ----------------------------------------------------------------------*/

template<class DeviceType>
void PairCoulShieldIntraKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  eflag = eflag_in;
  vflag = vflag_in;

  if (neighflag == FULL) no_virial_fdotr_compute = 1;

  ev_init(eflag,vflag,0);

  // reallocate per-atom arrays if necessary

  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
    d_eatom = k_eatom.view<DeviceType>();
  }
  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom,vatom);
    memoryKK->create_kokkos(k_vatom,vatom,maxvatom,"pair:vatom");
    d_vatom = k_vatom.view<DeviceType>();
  }

  atomKK->sync(execution_space,datamask_read);
  k_cutsq.template sync<DeviceType>();
  k_cut_ljsq.template sync<DeviceType>();
  k_cut_coulsq.template sync<DeviceType>();
  k_params.template sync<DeviceType>();
  if (eflag || vflag) atomKK->modified(execution_space,datamask_modify);
  else atomKK->modified(execution_space,F_MASK);

  x = atomKK->k_x.view<DeviceType>();
  c_x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  q = atomKK->k_q.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  d_molecule = atomKK->k_molecule.view<DeviceType>();
  nlocal = atom->nlocal;
  nall = atom->nlocal + atom->nghost;
  newton_pair = force->newton_pair;
  special_lj[0] = force->special_lj[0];
  special_lj[1] = force->special_lj[1];
  special_lj[2] = force->special_lj[2];
  special_lj[3] = force->special_lj[3];
  special_coul[0] = force->special_coul[0];
  special_coul[1] = force->special_coul[1];
  special_coul[2] = force->special_coul[2];
  special_coul[3] = force->special_coul[3];
  qqrd2e = force->qqrd2e;

  // loop over neighbors of my atoms

  EV_FLOAT ev = pair_compute<PairCoulShieldIntraKokkos<DeviceType>,void >
    (this,(NeighListKokkos<DeviceType>*)list);

  if (eflag) eng_coul += ev.ecoul;
  if (vflag_global) {
    virial[0] += ev.v[0];
    virial[1] += ev.v[1];
    virial[2] += ev.v[2];
    virial[3] += ev.v[3];
    virial[4] += ev.v[4];
    virial[5] += ev.v[5];
  }

  if (eflag_atom) {
    k_eatom.template modify<DeviceType>();
    k_eatom.sync_host();
  }

  if (vflag_atom) {
    k_vatom.template modify<DeviceType>();
    k_vatom.sync_host();
  }

  if (vflag_fdotr) pair_virial_fdotr_compute(this);
}

/* ----------------------------------------------------------------------
   fpair for the shield−bare correction, gated on the intra-molecular mask.
   cbrt (shield_gauss==SHIELD_CBRT, default): K (r·sh⁴ − factor_coul/r³) -- matches the CPU pair
   exactly (factor_coul fix, backlog item a; byte-identical to pre-fix at factor_coul==1, the
   only case in the CPU-only 24-case suite). pqeq (shield_gauss==SHIELD_GAUSSIAN):
   K[(erf(α_ij r) − factor_coul)/r³ − (2α_ij/√π)exp(−α_ij²r²)/r²], the exact
   device mirror of the CPU pair's PQEq branch in compute()/single() (α_ij
   precomputed per-type-pair on the HOST in init_one(), stored in
   params_coulshield::aij -- see the header). slater (shield_gauss==SHIELD_SLATER, backlog item
   b): −K(dJ/dr + factor_coul/r²)/r using the tabulated Rick J(r)/dJ/dr (device table built in
   init_style(), evaluated via slater_eval() -- device mirror of SlaterJTable::eval()), the exact
   device counterpart of the CPU pair's SLATER branch in compute()/single().
-------------------------------------------------------------------------*/

template<class DeviceType>
template<bool STACKPARAMS, class Specialisation>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
KK_FLOAT PairCoulShieldIntraKokkos<DeviceType>::
compute_fcoul(const KK_FLOAT& rsq, const int& i, const int&j, const int& itype,
              const int& jtype, const KK_FLOAT& factor_coul, const KK_FLOAT& qtmp) const {
  if (!shielded_pair(i,j)) return 0.0;

  const KK_FLOAT r2inv = 1.0/rsq;
  const KK_FLOAT rinv = sqrt(r2inv);
  const KK_FLOAT r = 1.0/rinv;

  if (shield_gauss == SHIELD_SLATER) {
    KK_FLOAT J, dJdr;
    slater_eval(itype, jtype, r, J, dJdr);
    const KK_FLOAT K = qqrd2e * qtmp * q(j);
    // fpair = −(1/r)dE/dr, E = K(J − factor_coul/r) -> matches CPU pair_coul_shield_intra.cpp:117.
    return -K * (dJdr + factor_coul*r2inv) * rinv;
  }

  if (shield_gauss == SHIELD_GAUSSIAN) {
    const KK_FLOAT aij = (STACKPARAMS?m_params[itype][jtype].aij:params(itype,jtype).aij);
    const KK_FLOAT erfar = erf(aij*r);
    const KK_FLOAT expt = exp(-aij*aij*rsq);
    const KK_FLOAT r3inv = rinv*r2inv;
    const KK_FLOAT K = qqrd2e * qtmp * q(j);
    // fpair = −(1/r)dE/dr = K[(erf−factor_coul)/r³ − (2α_ij/√π)exp(−α²r²)/r²]
    return K * ((erfar - factor_coul)*r3inv -
                (static_cast<KK_FLOAT>(2.0)*aij/static_cast<KK_FLOAT>(MathConst::MY_PIS))*expt*r2inv);
  }

  const KK_FLOAT g = (STACKPARAMS?m_params[itype][jtype].gamma:params(itype,jtype).gamma);
  const KK_FLOAT sh = 1.0/cbrt(rsq*r + 1.0/(g*g*g));     // J_shield = 1/∛(r³+1/γ³)
  const KK_FLOAT sh4 = sh*sh*sh*sh;
  const KK_FLOAT K = qqrd2e * qtmp * q(j);
  // fpair = −(1/r) dE/dr = K (r·sh⁴ − factor_coul/r³) -- ONLY the −1/r compensation term tracks
  // factor_coul (matches the CPU pair's compute()/single(), pair_coul_shield_intra.cpp:135/410).
  // FIX (backlog item a): was `factor_coul * K * (r*sh4 - rinv*r2inv)` -- scaled the WHOLE
  // bracket, which only agrees with the CPU formula at factor_coul==1 (flagged UNSURE in
  // NOTE_B10_kk_pqeq.md). Byte-identical at factor_coul==1 (the only case exercised by the CPU-only
  // 24-case suite, which has no kk cases); changes device forces only for fractional special_bonds
  // coul weights on bonded intramolecular topology under -sf kk.
  return K * (r*sh4 - factor_coul*rinv*r2inv);
}

/* ----------------------------------------------------------------------
   ecoul for the shield−bare correction, gated on the mask.
   cbrt: K (sh − factor_coul/r) -- factor_coul fix (item a), matches CPU exactly. pqeq:
   K(erf(α_ij r) − factor_coul)/r, the device mirror of the CPU pair's PQEq branch. slater (backlog item b): K(J − factor_coul/r), the device mirror of the CPU pair's SLATER branch.
-------------------------------------------------------------------------*/

template<class DeviceType>
template<bool STACKPARAMS, class Specialisation>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
KK_FLOAT PairCoulShieldIntraKokkos<DeviceType>::
compute_ecoul(const KK_FLOAT& rsq, const int& i, const int&j, const int& itype,
              const int& jtype, const KK_FLOAT& factor_coul, const KK_FLOAT& qtmp) const {
  if (!shielded_pair(i,j)) return 0.0;

  const KK_FLOAT r2inv = 1.0/rsq;
  const KK_FLOAT rinv = sqrt(r2inv);
  const KK_FLOAT r = 1.0/rinv;

  if (shield_gauss == SHIELD_SLATER) {
    KK_FLOAT J, dJdr;
    slater_eval(itype, jtype, r, J, dJdr);
    const KK_FLOAT K = qqrd2e * qtmp * q(j);
    return K * (J - factor_coul*rinv);
  }

  if (shield_gauss == SHIELD_GAUSSIAN) {
    const KK_FLOAT aij = (STACKPARAMS?m_params[itype][jtype].aij:params(itype,jtype).aij);
    const KK_FLOAT erfar = erf(aij*r);
    const KK_FLOAT K = qqrd2e * qtmp * q(j);
    return K * (erfar - factor_coul) * rinv;
  }

  const KK_FLOAT g = (STACKPARAMS?m_params[itype][jtype].gamma:params(itype,jtype).gamma);
  const KK_FLOAT sh = 1.0/cbrt(rsq*r + 1.0/(g*g*g));
  const KK_FLOAT K = qqrd2e * qtmp * q(j);
  // FIX (backlog item a): matches CPU E = K(sh - factor_coul/r) exactly; was
  // `factor_coul * K * (sh - rinv)` (whole-bracket scaling, see compute_fcoul comment above).
  return K * (sh - factor_coul*rinv);
}

/* ----------------------------------------------------------------------
   allocate all arrays
-------------------------------------------------------------------------*/

template<class DeviceType>
void PairCoulShieldIntraKokkos<DeviceType>::allocate()
{
  PairCoulShieldIntra::allocate();

  int n = atom->ntypes;
  memory->destroy(cutsq);
  memoryKK->create_kokkos(k_cutsq,cutsq,n+1,n+1,"pair:cutsq");
  d_cutsq = k_cutsq.template view<DeviceType>();

  k_cut_ljsq = DAT::tdual_kkfloat_2d("pair:cut_ljsq",n+1,n+1);
  d_cut_ljsq = k_cut_ljsq.template view<DeviceType>();
  k_cut_coulsq = DAT::tdual_kkfloat_2d("pair:cut_coulsq",n+1,n+1);
  d_cut_coulsq = k_cut_coulsq.template view<DeviceType>();

  k_params = Kokkos::DualView<params_coulshield**,Kokkos::LayoutRight,DeviceType>("PairCoulShieldIntra::params",n+1,n+1);
  params = k_params.template view<DeviceType>();
}

/* ----------------------------------------------------------------------
   init specific to this pair style
-------------------------------------------------------------------------*/

template<class DeviceType>
void PairCoulShieldIntraKokkos<DeviceType>::init_style()
{
  // PairCoulShieldIntra::init_style() (re)builds slater_tabs (the CPU per-type-pair Slater J(r)
  // tables, see build_slater_tables()) when shield_gauss==SHIELD_SLATER -- MUST run before the
  // device-table upload below, which copies FROM slater_tabs.
  PairCoulShieldIntra::init_style();

  // compute_fcoul/compute_ecoul now implement all three shielding kernels: cbrt (default,
  // untouched), PQEq Gaussian (shield_gauss==SHIELD_GAUSSIAN; B10 port, params_coulshield::aij), and
  // Slater (shield_gauss==SHIELD_SLATER; backlog item b, ported below -- dense device table +
  // slater_eval(), mirroring how PQEq was ported in B10).
  if (shield_gauss == SHIELD_SLATER) {
    const int nt = atom->ntypes;
    const int npts = slater_tabs.empty() ? 0 : slater_tabs[0].npts;
    slater_npts = npts;
    slater_rmin = slater_tabs.empty() ? 0.0 : static_cast<KK_FLOAT>(slater_tabs[0].rmin);
    slater_dr   = slater_tabs.empty() ? 0.0 : static_cast<KK_FLOAT>(slater_tabs[0].dr);

    // Every type-pair table shares the SAME (rmin,dr,npts) grid (slater_jtable.h: SlaterJTable::
    // build always spans [rmin, cut_global] with the same npoints), so a single dense
    // [itype][jtype][node] view (built once, here, from the CPU tables) suffices -- no per-pair
    // grid bookkeeping needed on device.
    d_slater_J  = t_slater_tab("PairCoulShieldIntraKokkos::slater_J",  nt+1, nt+1, npts>0?npts:1);
    d_slater_dJ = t_slater_tab("PairCoulShieldIntraKokkos::slater_dJ", nt+1, nt+1, npts>0?npts:1);
    auto h_J  = Kokkos::create_mirror_view(d_slater_J);
    auto h_dJ = Kokkos::create_mirror_view(d_slater_dJ);
    for (int ti = 1; ti <= nt; ti++)
      for (int tj = 1; tj <= nt; tj++) {
        const int idx = slater_tri_index(ti,tj,nt);
        const SlaterJTable &tab = slater_tabs[idx];
        for (int n = 0; n < npts; n++) {
          h_J(ti,tj,n)  = static_cast<KK_FLOAT>(tab.Jv[n]);
          h_dJ(ti,tj,n) = static_cast<KK_FLOAT>(tab.dJv[n]);
        }
      }
    Kokkos::deep_copy(d_slater_J, h_J);
    Kokkos::deep_copy(d_slater_dJ, h_dJ);
  }

  // adjust neighbor list request for KOKKOS

  neighflag = lmp->kokkos->neighflag;
  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same_v<DeviceType,LMPHostType> &&
                           !std::is_same_v<DeviceType,LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType,LMPDeviceType>);
  if (neighflag == FULL) request->enable_full();
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
-------------------------------------------------------------------------*/

template<class DeviceType>
double PairCoulShieldIntraKokkos<DeviceType>::init_one(int i, int j)
{
  double cutone = PairCoulShieldIntra::init_one(i,j);

  // PQEq alpha_ij: built from the per-type DIAGONAL gamma[i][i]/gamma[j][j] (reinterpreted as Rc_i/Rc_j when
  // shield_gauss==SHIELD_GAUSSIAN -- same convention as the CPU pair's compute()/single()), NOT the (possibly
  // mixed) off-diagonal gamma[i][j] above -- mirrors the CPU formula exactly. Left 0 in cbrt mode (unused).
  double aij_val = 0.0;
  if (shield_gauss == SHIELD_GAUSSIAN) {
    double ai = shield_lambda*0.5/(gamma[i][i]*gamma[i][i]);
    double aj = shield_lambda*0.5/(gamma[j][j]*gamma[j][j]);
    aij_val = sqrt(ai*aj/(ai+aj));
  }

  k_params.view_host()(i,j).gamma = gamma[i][j];
  k_params.view_host()(i,j).aij = aij_val;
  k_params.view_host()(i,j).cutsq = cutone*cutone;
  k_params.view_host()(j,i) = k_params.view_host()(i,j);

  if (i<MAX_TYPES_STACKPARAMS+1 && j<MAX_TYPES_STACKPARAMS+1) {
    m_params[i][j] = m_params[j][i] = k_params.view_host()(i,j);
    m_cutsq[j][i] = m_cutsq[i][j] = cutone*cutone;
    m_cut_ljsq[j][i] = m_cut_ljsq[i][j] = cutone*cutone;
    m_cut_coulsq[j][i] = m_cut_coulsq[i][j] = cutone*cutone;
  }
  k_cutsq.view_host()(i,j) = cutone*cutone;
  k_cutsq.modify_host();
  k_cut_ljsq.view_host()(i,j) = cutone*cutone;
  k_cut_ljsq.modify_host();
  k_cut_coulsq.view_host()(i,j) = cutone*cutone;
  k_cut_coulsq.modify_host();
  k_params.modify_host();

  return cutone;
}

namespace LAMMPS_NS {
template class PairCoulShieldIntraKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairCoulShieldIntraKokkos<LMPHostType>;
#endif
}
