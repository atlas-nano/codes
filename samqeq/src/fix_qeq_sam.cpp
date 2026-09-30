// clang-format off
/* ----------------------------------------------------------------------
   samQEq — shielded-Coulomb charge equilibration.
   Inherits FixACKS2Sam (self-contained ACKS2 base in SAMQEQ; no REAXFF pkg):
   reuses the augmented (Δ,u) BiCGStab solver, sparse_matvec_acks2, all MPI comm,
   shielded-Coulomb compute_H, storage and vector ops UNCHANGED. Only the
   response-matrix weights (calc_w), the reference-charge field, and q=q0+Δ differ.

   This translation unit holds the CORE methods: ctor/dtor, init, allocate,
   pertype_parameters, the response kernel (compute_H/compute_X/calc_w),
   init_storage/init_matvec/calculate_Q, and the comm/exchange callbacks.
   The rest of FixQEqSam is split across sibling TUs (same class):
     fix_qeq_sam_lr.cpp long-range Ewald projected-CG (pre_force, qeq_*)
     fix_qeq_sam_xl.cpp extended-Lagrangian dynamics (setmask, integrate)
     fix_qeq_sam_quartic.cpp the quartic on-site diagonal (apply_quartic_eta)
     fix_qeq_sam_modify.cpp fix_modify keywords + on-site self-energy (compute_scalar)
-------------------------------------------------------------------------*/

#include "fix_qeq_sam.h"

#include "atom.h"
#include "molecule.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "memory.h"
#include "modify.h"   // check_drude_consistency() scans modify->fix for a `fix drude`
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "text_file_reader.h"
#include "tokenizer.h"
#include "update.h"
#include "kspace.h"
#include "pair_coul_shield_intra.h"   // iondamp fix<->pair consistency cross-check (init())
#include "pppm_samqeq.h"   // self-contained per-atom kspace potential (compute_vector); no ELECTRODE pkg
#include "random_mars.h"   // xl_random (the Langevin charge thermostat)
#include "math_const.h"

#include <cmath>
#include <cstring>
#include <exception>
#include <array>
#include <map>
#include <vector>

using namespace LAMMPS_NS;
using namespace FixConst;      // INITIAL_INTEGRATE / FINAL_INTEGRATE / PRE_FORCE masks
using MathConst::MY_PIS;       // sqrt(pi), for the DSF self/shift terms

/* ----------------------------------------------------------------------*/

FixQEqSam::FixQEqSam(LAMMPS *lmp, int narg, char **arg) : FixACKS2Sam(lmp, narg, arg)
{
  // Command mirrors acks2/reaxff: fix ID grp samqeq Nevery cutlo cuthi tol paramfile
  // (γ_align and κ_bond are read from the parameter-file header, not the command line,
  //  so the fix command is interchangeable with acks2/qeq.)
  q0 = ehomo = elumo = nullptr;
  eta_diag = nullptr;
  eta0 = nullptr; lr_alpha = 0.0; e_shift = f_shift = 0.0;   // long-range off by default (shielded)
  virial_global_flag = 1;   // the fix-owned position forces tally an fdotr virial; inert unless thermo_virial is set
  lr_ewald = 0; eksp = nullptr; qsave = nullptr; prec = nullptr;   // Route B: plain-QEq long-range state
  lr_pppm = nullptr; lr_nx = lr_ny = lr_nz = -1; lr_gewald = -1.0;  // recip_self grid-signature cache (grid_changed)
  lr_prd[0] = lr_prd[1] = lr_prd[2] = -1.0;
  lr_tagA = lr_tagB = 0;
  qs = qb = q_r = q_d = q_p = q_q = nullptr; comm_v = nullptr;
  nmol_ = 0; molinv = molsum = nullptr; lr_calibrated = 0; lr_self_meas = 0.0;
  maxmol_ = 0; mol2c = nullptr; c2mol = nullptr; cmol = nullptr;   // compact per-molecule projector reindex
  molinv_valid = 0; molinv_natoms = -1;                            // molinv cache starts invalid
  molinv_active = nullptr; molinv_active_cap = 0;                  // persistent pass-1 scratch
  fixq_field = nullptr;   // fixed non-group charge field (ionfield)
  ionfield_flag = 1;   // ionfield: fixed non-group ION charges -> QEq RHS. ON BY DEFAULT: the
                       // field of a fixed charge belongs in b, not nowhere.
                       // `fix_modify <id> ionfield off` selects the ion-blind solve and says so.
                       // add_fixed_charge_field returns early when no non-group charged atom exists, so
                       // decks without fixed charges pay nothing for it.
  q0field = nullptr; has_q0ref = 0;     // q0 reference-charge field (energy conservation; off iff all q0=0)
  lr_xl = 0; q_mass = 5.0e-5; q_tdamp = 0.1; qdot = qddot = pchi = nullptr; xl_started = 0;
  xl_qspike = 2.0; xl_kspike = 10.0*ev_scale;   // XL spike-guard defaults (off only if xl_qspike<=0)
                                        // qspike is charge (e, invariant); kspike is eV/e^2-anchored ->
                                        // deck units (fix_modify spikeguard overwrites with native values)
  xl_Tq = 0.0; xl_random = nullptr;      // Langevin charge thermostat OFF (pure friction) by default
  xl_masswt = 0;                         // per-type mass-weighted charge mass OFF (scalar q_mass) by default
  gamma_align = 0.0; kappa_bond = 0.0; r_ov = 0.0; r_loc = 0.0; lr_ridge = 0.0; ridge_cur = 0.0;
  ridge_qonset = 4.0; ridge_gain = 0.0;   // smooth-ridge: off by default (discrete escalation)
  ridge_mode = 0; lam_floor = 1.0*ev_scale; nlanczos = 24;   // ridge mode 0=discrete (default); 1=smooth|q|; 2=Lanczos λ_min; 3=local
                                          // lam_floor default is eV-anchored (target λ_min) -> deck units;
                                          // `fix_modify ridge eig` overwrites with the user's native value (no double-scale)
  ridge_every = 1; lmin_cache = 0.0; lmin_step = -1;   // efficiency: Lanczos λ_min cache (recompute cadence)
  ridge_atom = nullptr; ridge_local = false;        // local per-atom ridge (mode 3); off unless enabled
  m_t = nullptr; m_w2 = nullptr; use_minres = false; minres_qcap = 0.0;   // MINRES solver; off (=CG) unless enabled
  mv_out_t = nullptr; mv_out_nth = 0; mv_out_cap = 0;         // OMP per-thread matvec accumulator (lazy)
  ps_molsum_t = nullptr; ps_molsum_nth = 0; ps_molsum_cap = 0;   // OMP project_neutral thread scratch (lazy)
  r_bond = 2.6;   // header fields 8-11 (r_bond, graph_wedge, pd_lam_min, graph_hop_max): read and unused
  graph_wedge = 0.8*ev_scale; pd_lam_min = 0.6*ev_scale; graph_hop_max = 3;   // defaults for header
                  // fields that are accepted and unused
  xreg = 1.0e-6*ev_scale; // u-block null-space regularization (small vs physical X scale ~O(1) eV-family;
                          // summed with the native-unit lr_ridge on X_diag -> shares its energy scale)
  precond_mode = 0;       // diagonal preconditioner by default (ILU opt-in via fix_modify precond)
  ilu_rp = ilu_ci = ilu_dptr = nullptr; ilu_lu = nullptr;
  ilu_Dinv = nullptr;
  ilu_Y0 = ilu_Y1 = ilu_scr = ilu_w = nullptr; ilu_n = ilu_nnz = 0; ilu_droptol = 1.0e-3;
  ilu_shift = 0.0;        // eV/e^2 Manteuffel shift (no fix_modify setter; inert at 0.0 -- a nonzero
                          // internal default MUST be written *ev_scale). droptol = relative fraction.
  // quartic_etafloor (header in-class default 2.0) is eV/e^2-anchored and has NO fix_modify override
  // (the `quartic` parser stops at qref) -> convert the internal default to deck units here, once.
  quartic_etafloor = 2.0*ev_scale;
  list_full = nullptr;    // FULL neighbor list (ionfield sweep, the device solve)
  reaxflag = 0;            // file-mode params (no reaxff pair coupling)

  // --- Expose the on-site samQEq self-energy as the fix's global scalar (accessible as f_ID),
  //   and let it count toward the system PE/etotal when the user sets `fix_modify ID energy yes`.
  //   E_self = Σ_i(χ_i q_i + ½ η_i q_i²): the diagonal of the QEq functional. The OFF-diagonal Coulomb
  //   (½ Σ_{i≠j} J_ij q_i q_j) is already in pe via the coul/shield/intra + coul/long pairs + kspace, so
  //   pe + E_self = the complete samQEq energy. E_self depends only on charges (not positions) ⇒ zero force,
  //   zero virial ⇒ inclusion CANNOT change dynamics, density, or pressure.
  //   ★ DEFAULT ON: thermo pe/etotal INCLUDE E_self unless the deck says `fix_modify ID energy no`. Without
  //   it, every fitted target (E_QM - pe) and every pe-derived ΔH/ΔU omits E_self (several kcal/mol toward
  //   an ion contact; the force is -d(pe + E_self)/dx, not -d(pe)/dx). With `energy no`, pe differs by
  //   EXACTLY f_ID (the fix scalar). init() prints a banner either way, so the accounting is never silent.
  scalar_flag = 1;
  extscalar = 1;            // extensive (Σ over atoms)
  energy_global_flag = 1;   // may contribute to the global potential energy (gated by fix_modify energy yes|no)
  thermo_energy = 1;        // ★ E_self counts toward pe/etotal BY DEFAULT (see the block above); `fix_modify ID energy no` opts out
  global_freq = 1;
  // We skip init_bondcut (bcut is set up lazily for bondsoft), so bcut_acks2 is never allocated.
  // The base ctor null-inits bcut but NOT bcut_acks2; its dtor frees bcut_acks2 when reaxflag=0,
  // so null it here or shutdown segfaults on free(garbage).
  bcut = nullptr; bcut_acks2 = nullptr;
  // qdot/qddot are sized by grow_arrays, which FixACKS2Sam::post_constructor() already calls
  // (virtual dispatch -> our override) after construction — so no explicit ctor call is needed here.

  // --- restart persistence ---
  // GLOBAL: the recip_self calibration (write_restart/restart). PER-ATOM: qdot/qddot
  // (pack_restart/unpack_restart below), riding the SAME GROW-callback machinery that migrates them
  // (grow_arrays/copy_arrays/pack_exchange/unpack_exchange) so they migrate AND restart-round-trip.
  restart_global = 1;
  restart_peratom = 1;
  atom->add_callback(Atom::RESTART);
  // exchange payload = the FixACKS2Sam block (2*nprev+6) + qdot/qddot — see pack_exchange.
  maxexchange = 2*nprev + 8;
}

/* ----------------------------------------------------------------------*/

FixQEqSam::~FixQEqSam()
{
  if (copymode) return;
  atom->delete_callback(id, Atom::RESTART);   // unregister the per-atom qdot/qddot restart callback (added in the ctor)
  memory->destroy(q0); memory->destroy(ehomo); memory->destroy(elumo); memory->destroy(eta0);
  memory->destroy(shield_rpair);
  memory->destroy(lr_self_atom);
  memory->destroy(c3_type); memory->destroy(c4_type);   // per-type staircase coeffs
  // gself_type is per-type CONFIG from `fix_modify ... gself types <...>`, not per-run storage, so it is
  // freed here and NOT in deallocate_storage(), which runs BETWEEN RUNS: freeing it there while gself_flag
  // survives would turn a type-scoped fold silently global on the next run.
  memory->destroy(gself_type);
  memory->destroy(is2s);   // shield slater per-type 2s/1s flag
  memory->destroy(iondamp_b); memory->destroy(iondamp_n);   // damped interionic kernel (fix_modify iondamp)
  memory->destroy(qsave); memory->destroy(prec);
  memory->destroy(qs); memory->destroy(qb);
  memory->destroy(q_r); memory->destroy(q_d); memory->destroy(q_p); memory->destroy(q_q);
  memory->destroy(ridge_atom);    // local ridge
  memory->destroy(m_t); memory->destroy(m_w2);   // MINRES work
  memory->destroy(mv_out_t);        // OMP per-thread matvec accumulator
  memory->destroy(ps_molsum_t);     // OMP project_neutral thread scratch
  memory->destroy(molinv); memory->destroy(molsum);
  memory->destroy(mol2c); memory->destroy(c2mol); memory->destroy(cmol);   // compact projector reindex
  memory->destroy(molinv_active);                                          // persistent pass-1 scratch
  delete xl_random;               // Langevin charge-thermostat RNG
  memory->destroy(fixq_field);   // fixed non-group charge field
  memory->destroy(q0field);        // q0 reference-charge field
  memory->destroy(qdot); memory->destroy(qddot); memory->destroy(pchi);
  ilu_free();             // ILU CSR + Schur scratch
  // deallocate_storage() is virtual and NOT reached from here via base-class dtors (by the time
  // ~FixQEqBaseSam/~FixACKS2Sam run, the vtable no longer points at FixQEqSam::deallocate_storage) ->
  // free our allocate_storage() arrays explicitly. chi/eta/gamma freed by base; bcut/bcut_acks2 never
  // allocated (we skip init_bondcut) -> base destroy() on nullptr is safe.
  memory->destroy(eta_diag); memory->destroy(quartic_gate);
  // freed (and nulled) by deallocate_storage() too; freed here so nothing leaks at exit.
  memory->destroy(deta_anh);
}

/* ----------------------------------------------------------------------*/
// allocate eta_diag alongside the inherited ACKS2 (2N+2) storage.
void FixQEqSam::allocate_storage()
{
  FixACKS2Sam::allocate_storage();
  memory->create(eta_diag, atom->nmax, "samqeq:eta_diag");
  memory->create(deta_anh,   atom->nmax, "samqeq:deta_anh");    // saddle on-site anharmonic diagonal
  for (int _i = 0; _i < atom->nmax; _i++) deta_anh[_i] = 0.0;   // matvec-safe before the first fill
  onsite_extra = nullptr;   // (re)point after realloc (compute_saddle_onsite re-points again on the saddle path)
  memory->create(qsave, atom->nmax, "samqeq:qsave");
  memory->create(prec,  atom->nmax, "samqeq:prec");
  memory->create(fixq_field, atom->nmax, "samqeq:fixq_field");   // fixed non-group charge field
  for (int i = 0; i < atom->nmax; i++) fixq_field[i] = 0.0;
  memory->create(q0field, atom->nmax, "samqeq:q0field");           // q0 reference-charge Coulomb field on cores
  for (int i = 0; i < atom->nmax; i++) q0field[i] = 0.0;           // stays 0 unless has_q0ref
  memory->create(quartic_gate, atom->nmax, "samqeq:quartic_gate"); // per-atom field gate for the quartic
  for (int i = 0; i < atom->nmax; i++) quartic_gate[i] = 1.0;      // 1 ⇒ ungated (set <1 only when fld0>0)
  memory->create(cmol, atom->nmax, "samqeq:cmol");     // per-atom compact molecule slot (filled every
  for (int i = 0; i < atom->nmax; i++) cmol[i] = 0;    // build_molinv; harmless default before the first solve)
  int n = atom->nmax;                                 // plain-QEq vectors are N-dim (no augmented X/last-rows)
  memory->create(qs, n, "samqeq:qs");   memory->create(qb, n, "samqeq:qb");
  memory->create(q_r, n, "samqeq:q_r"); memory->create(q_d, n, "samqeq:q_d");
  memory->create(q_p, n, "samqeq:q_p"); memory->create(q_q, n, "samqeq:q_q");
  memory->create(pchi, n, "samqeq:pchi");
  memory->create(ridge_atom, n, "samqeq:ridge_atom");                   // local ridge
  for (int i = 0; i < n; i++) ridge_atom[i] = 0.0;
  memory->create(m_t, n, "samqeq:m_t");   memory->create(m_w2, n, "samqeq:m_w2");   // MINRES work
  // DEFENCE IN DEPTH. Every writer of these vectors is group-masked, so the slots of atoms outside the
  // fix group are never assigned and would carry whatever the allocator returns. Nothing reads them
  // (compute_H tests the column group), and zeroing here makes any unfiltered read deterministic
  // instead of heap-dependent, so it fails visibly and reproducibly rather than at random.
  for (int i = 0; i < n; i++) {
    qs[i] = qb[i] = q_r[i] = q_d[i] = q_p[i] = q_q[i] = 0.0;
    pchi[i] = m_t[i] = m_w2[i] = 0.0;
  }
  // NOTE: qdot/qddot are NOT created here — they are persistent per-atom XL state managed by the GROW
  // callback (grow_arrays) so they MIGRATE with atoms across procs. allocate_storage holds only
  // scratch/CG vectors that are recomputed every solve.
}
void FixQEqSam::deallocate_storage()
{
  memory->destroy(eta_diag); eta_diag = nullptr;
  memory->destroy(quartic_gate); quartic_gate = nullptr;
  onsite_extra = nullptr;
  memory->destroy(deta_anh); deta_anh = nullptr;
  memory->destroy(qsave); qsave = nullptr;
  memory->destroy(prec); prec = nullptr;
  memory->destroy(fixq_field); fixq_field = nullptr;     // fixed non-group charge field
  memory->destroy(q0field); q0field = nullptr;             // q0 reference-charge field
  memory->destroy(cmol); cmol = nullptr;                   // per-atom compact molecule slot
  memory->destroy(qs); memory->destroy(qb);
  memory->destroy(q_r); memory->destroy(q_d); memory->destroy(q_p); memory->destroy(q_q);
  memory->destroy(molinv); memory->destroy(molsum);
  memory->destroy(mol2c); memory->destroy(c2mol);          // compact projector reindex maps
  memory->destroy(pchi);
  memory->destroy(ridge_atom); ridge_atom = nullptr;   // local ridge
  memory->destroy(m_t); m_t = nullptr; memory->destroy(m_w2); m_w2 = nullptr;   // MINRES work
  // qdot/qddot are NOT freed here — they are GROW-callback per-atom arrays (freed in the destructor),
  // and must SURVIVE reallocate_storage() (= deallocate+allocate) so XL charge velocities persist.
  qs=qb=q_r=q_d=q_p=q_q=nullptr; molinv=molsum=nullptr; pchi=nullptr;
  mol2c=nullptr; c2mol=nullptr;
  nmol_ = 0; maxmol_ = 0;   // molinv/molsum just freed -> force build_molinv to re-create them (else
                 // a second run's setup_pre_force sees nmol==nmol_, skips create, derefs the null molinv);
                 // maxmol_=0 forces mol2c to rebuild too (same reasoning, for the raw-ID map)
  molinv_valid = 0;   // cached projector state just freed -> force a full rebuild.
                      // ★ this clear is NOT always rank-symmetric: reallocate_storage() reaches here from
                      // `atom->nmax > nmax` — a PER-RANK trigger: one rank regrows at an exchange event and
                      // drops the cache alone. build_molinv's gate is therefore a collective Allreduce(MIN)
                      // that absorbs a one-rank clear as one forced, byte-identical rebuild on all ranks.
  FixACKS2Sam::deallocate_storage();
}

/* ----------------------------------------------------------------------
   init: do the FixQEqBaseSam setup (taper, shielding, neighbor request) but
   SKIP FixACKS2Sam::init_bondcut (bcut is set up lazily for bondsoft).
   FixQEqBaseSam::init() runs without a reaxff pair (file mode, as ACKS2 does).
-------------------------------------------------------------------------*/
/* Warm-start the run-BOUNDARY re-solve. The base setup_pre_force re-inits storage and re-solves the charges at
   every `run`. On a fresh first run (ntimestep==0, incl. reset_timestep 0) that is a cold solve from q0.
   On a 2nd+ run / restart / NVT->NVE / heating switch (ntimestep>0) the
   charges in atom->q are already the previous run's CONVERGED values (init_storage zeros only s/t, NOT
   atom->q); cold-solving from q0 there can drop into the over-polarization catastrophe -> "Out of range atoms -
   cannot compute PPPM" blow-up. So at a boundary we SKIP this re-solve and reuse atom->q (pre_force returns early).
   This is independent of the recip_self grid recalibration at a boundary.*/
void FixQEqSam::setup_pre_force(int vflag)
{
  setup_resolve = (update->ntimestep > 0) ? 1 : 0;     // boundary/restart -> skip re-solve; fresh first setup -> solve
  // ---- Disclose the ACTIVE self-term at every run start (one line, so a carried-over outlier is visible
  // without parsing hundreds of calibration lines), and warn ONCE per run when the single-pair route is
  // about to be re-drawn under a changing box. ------------------
  lr_self_atom_logged = 0;                              // per-atom stats line once per run
  if (lr_ewald >= 2 && !lr_nrecip && lr_self_peratom && comm->me == 0)
    utils::logmesg(lmp, "samqeq: recip_self ACTIVE at run start (step {}): EXACT PER-ATOM grid self-coefficient"
                        "(stats follow at the first solve)\n", update->ntimestep);
  if (lr_ewald >= 2 && !lr_nrecip && !lr_self_peratom && comm->me == 0) {
    const double f = force->qqrd2e/ev_scale;
    const std::string route = lr_recip_probes ? fmt::format("K={} deterministic-probe", lr_recip_probes)
                                              : std::string("single-pair");
    if (lr_self_pinned)
      utils::logmesg(lmp, "samqeq: recip_self ACTIVE at run start (step {}): {:.5f} raw = {:.4f} eV/e, PINNED"
                          "(fix_modify recip_self)\n", update->ntimestep, lr_self_meas, lr_self_meas*f);
    else if (!lr_calibrated)
      utils::logmesg(lmp, "samqeq: recip_self ACTIVE at run start (step {}): not yet calibrated -- measured at"
                          "the first solve by the {} route\n", update->ntimestep, route);
    else if (lr_ncalib == 0)     // calibrated but never measured in THIS run series: restored from a restart file
      utils::logmesg(lmp, "samqeq: recip_self ACTIVE at run start (step {}): {:.5f} raw = {:.4f} eV/e, RESTORED from"
                          "the restart file (the writing run's route is not recorded); kept until the grid changes\n",
                     update->ntimestep, lr_self_meas, lr_self_meas*f);
    else
      utils::logmesg(lmp, "samqeq: recip_self ACTIVE at run start (step {}): {:.5f} raw = {:.4f} eV/e, {} route,"
                          "calibration #{} at step {} on the {}x{}x{} mesh; first at this mesh {:.5f} (step {}),"
                          "drift {:+.3f}%\n", update->ntimestep, lr_self_meas, lr_self_meas*f, route, lr_ncalib,
                     lr_self_step, lr_nx, lr_ny, lr_nz, lr_self_first, lr_first_step,
                     lr_self_first != 0.0 ? 100.0*(lr_self_meas - lr_self_first)/lr_self_first : 0.0);
    if (!lr_self_pinned && lr_recip_probes == 0 && domain->box_change_size)
      error->warning(FLERR, "samqeq: this run changes the box (NPT/deform) with the single-pair recip_self"
                            "calibration: every >0.1% box-length change re-draws the self-term from ONE probe pair,"
                            "which is not deterministic across draws (under NPT a draw can land several percent"
                            "low). Use `fix_modify {} recip_probes"
                            "16` (deterministic multi-probe), or pin the fixed-box value with `fix_modify {}"
                            "recip_self <raw>` for a reproduction control. The drift guard aborts this run at"
                            "2% from the first calibration.", id, id);
  }
  FixQEqBaseSam::setup_pre_force(vflag);                // -> init_storage + pre_force (skips the re-solve if setup_resolve)
  setup_resolve = 0;
}

void FixQEqSam::init()
{
  // ★ E_self accounting banner — printed on EVERY init so the energy that thermo reports is
  //   never ambiguous. E_self = Σ_i(χ_i q_i + ½ η_i qs_i²) is the on-site (diagonal) term of the samQEq
  //   functional; pe + E_self is the energy the forces integrate (zero force / zero virial ⇒ dynamics identical).
  if (comm->me == 0) {
    if (thermo_energy)
      utils::logmesg(lmp, "samqeq: E_self = sum(chi q + 1/2 eta qs^2) INCLUDED in pe/etotal (fix_modify {} energy yes;"
                          "eselfref {}) -- pe + E_self is the energy the forces integrate\n",
                     id, eself_qs ? "qs = q - q0" : "q (physical-q basis)");
    else
      error->warning(FLERR, "samqeq: E_self EXCLUDED from pe/etotal (fix_modify {} energy no): thermo pe is NOT the energy"
                            "the forces integrate; wall targets, dH and dU derived from pe are incomplete by exactly f_{}",
                     id, id);
    if (thermo_energy && !eself_qs)
      error->warning(FLERR, "samqeq: eselfref q (physical-q basis): reported etotal carries sum(eta*q0*q(t)), which is"
                            "not constant when q0 != 0 -- use `fix_modify {} eselfref qs` for the conserved functional", id);
  }
  // gself: (re)fold the finite-width charge self-energy into eta from the eta0 backup.
  // Idempotent across run boundaries by construction -- eta is ALWAYS rebuilt from eta0 here, so toggling
  // gself off restores the bare param exactly and a second `run` never double-adds. Done before anything
  // below reads eta, so the solve diagonal (solve_diag_of) and the XL force see it, and compute_scalar
  // reports it through eself_diag_of (fix_qeq_sam_lr.cpp). Off => eta == eta0 == the param.
  // Per-run warning budgets and ASPC accuracy counters restart with every run
  nwarn_ridge = nwarn_overpol = nwarn_picard = 0; cg_nonconv = 0;
  aspc_nabove = aspc_nabove_run = 0; aspc_above_warned = 0;
  if (eta0) {
    const int nt = atom->ntypes;
    for (int t = 1; t <= nt; t++) {
      eta[t] = eta0[t];
      // skip types excluded by `gself ... types <...>` (see the header note: the self-energy is
      // a lattice regulariser and a double count on a molecular site).
      if (gself_flag && gself_type && t < gself_ntype && !gself_type[t]) continue;
      if (gself_flag) {
        // width w (default 0.5 A): E_self = K_e*sqrt(2 alpha/pi) with alpha = 1/(2 w^2), in eV/e^2, converted to
        // native deck energy with ev_scale (eta is native-unit). The default is not the type's own Gaussian: at
        // that width the Coulomb block of a metal is only semidefinite and a slab's layer charges stagger.
        const double w = gself_width;
        const double alpha = 0.5/(w*w);
        eta[t] += 14.399645*sqrt(2.0*alpha/MathConst::MY_PI)*ev_scale;
      }
    }
    if (gself_flag && comm->me == 0) {   // logmesg writes screen AND log; `&& screen` hid it under -screen none
      std::string line = "samqeq: gself on-site self-energy per type (eV/e^2, added to eta; 0.000 = excluded):";
      for (int t = 1; t <= nt; t++) line += fmt::format(" {}:{:.3f}", t, (eta[t]-eta0[t])/ev_scale);
      utils::logmesg(lmp, line + "\n");
    }
  }
  FixQEqBaseSam::init();



  // Invalidate the build_molinv cache at every run start (init() runs on all ranks at each `run`) —
  // covers deck-level changes between runs that no in-code hook can see (`set ... mol`, group edits,
  // delete_atoms/create_atoms). Cost: one full rebuild on the first solve of each run.
  molinv_valid = 0;

  // recip_self (lr_self_meas) is purely a function of the PPPM grid {nx,ny,nz}_pppm + g_ewald. It is
  // (re)calibrated lazily in pre_force ONLY when that signature changes (grid_changed()) -- NOT blanket-reset
  // every run. A blanket per-run reset would force a recip_self recalibration inside setup_pre_force,
  // whose compute_vector/particle_map runs BEFORE kspace->setup() (verlet.cpp ordering) on a not-yet-finalized
  // grid + stale end-of-run atom positions -> "Out of range atoms - cannot compute PPPM" on an in-memory 2nd
  // run, and would measure the OLD grid on an NPT resize. Signature-gating avoids both. See grid_changed().

  // A FULL neighbor list (id 1) for the helpers that need each atom's complete neighborhood (the ionfield
  // sweep, the device solve); the half list (id 0) still drives compute_H/compute_X (the matvec
  // symmetrizes, so half is correct there).
  neighbor->add_request(this, NeighConst::REQ_FULL)->set_id(1);

  // Molecule-optional solve: without molecule IDs (atom_style charge) build_molinv/project_neutral
  // treat every in-group atom as fragment 0 -> ONE global fragment -> the projector enforces GLOBAL
  // charge neutrality (the standard QEq constraint for a bare solid).
  if (lr_ewald && !atom->molecule_flag) {
    if (comm->me == 0)
      utils::logmesg(lmp, "samqeq: no molecule IDs -> one global fragment"
                          "(global charge-neutrality constraint)\n");
  }

  // shield slater: (re)build the per-type-pair J(r) tables. Unconditional on every init() while active (see
  // build_slater_tables()'s header comment for why that IS the "rebuild if swb changes" policy). Placed before
  // the lr_ewald/DSF branch below only so the guard there can assume the tables (if needed) already exist.
  if (shield_gauss == SHIELD_SLATER) build_slater_tables();

  // iondamp path guards (mirror the slater DSF guard: error on silent-no-op combinations).
  // lr_ewald=1 treats INTER pairs (which is what ion-ion pairs are) as bare 1/r without ever calling
  // shielded_coulomb() for them; plain DSF (lr_ewald=0, lr_alpha>0) computes erfc/r-shift directly and never
  // calls shielded_coulomb()/calc_Hval() at all -- in both, iondamp would be silently inert. Supported paths:
  // lr_ewald>=2 (full Ewald + `fix_modify cutoff on` taper) and the gas path (lr_ewald=0,lr_alpha=0).
  if (lr_iondamp) {
    if (lr_ewald == 1)
      error->all(FLERR, "samqeq: fix_modify iondamp is not supported with lr_ewald=1 (inter-molecular -- i.e."
                        "ion-ion -- pairs are BARE 1/r there, never shielded); use lr_ewald>=2");
    if (!lr_ewald && lr_alpha > 0.0)
      error->all(FLERR, "samqeq: fix_modify iondamp is not supported with plain DSF (lr_alpha>0, lr_ewald=0) --"
                        "that branch never calls the shielded kernel; supported paths are lr_ewald>=2 and the"
                        "gas path (lr_ewald=0, lr_alpha=0)");
    // A fix/pair iondamp SPLIT is silent otherwise -- the
    // solve equilibrates on the damped kernel while the forces/energy ride the undamped one (or vice versa).
    // Cross-check this fix's table against the coul/shield/intra pair's settings and WARN on any mismatch
    // (warning, not error: gas / taper decks may legitimately carry force consistency elsewhere).
    if (comm->me == 0) {
      // exact=0, NOT 1. `-sf kk` stores the style name WITH its suffix ("coul/shield/intra/kk"),
      // which an exact match misses -> pair_match would return nullptr and the check would silently
      // no-op on every device deck.
      auto *csp = dynamic_cast<PairCoulShieldIntra *>(force->pair_match("coul/shield/intra", 0));
      if (!csp)
        error->warning(FLERR, "iondamp: no pair coul/shield/intra found -- the pair FORCES/ENERGY are not"
                              "damped (solve<->force inconsistency) unless another route provides them");
      else {
        int nbad = 0;
        for (int i = 1; i <= atom->ntypes; i++)
          for (int j = i; j <= atom->ntypes; j++)
            if (iondamp_b[i][j] != csp->iondamp_bget(i, j) ||
                (iondamp_b[i][j] > 0.0 && iondamp_n[i][j] != csp->iondamp_nget(i, j))) nbad++;
        if (!csp->iondamp_active() || nbad)
          error->warning(FLERR, "iondamp: fix_modify and pair coul/shield/intra iondamp settings DISAGREE for"
                                "{} type pair(s) (pair iondamp {}) -- solve and forces see different ion-ion"
                                "kernels; make both carry the same `iondamp <I> <J> <b> [<n>]`",
                                nbad, csp->iondamp_active() ? "on" : "OFF");
      }
    }
  }

  // Assert the pair<->fix shielding-kernel channels (range/mode/lambda/2s/scope).
  // Placed AFTER build_slater_tables() above — the range residual evaluates shielded_coulomb(), which
  // reads slater_tabs when the Slater kernel is active.
  push_shield_pairs();          // per-pair shielding radii -> pair style (before the consistency check)
  check_shield_consistency();

  // Refuse `fix drude` (see the header note). Placed beside the shielding check because it is the same
  // class of pair<->fix inconsistency — there the two sides integrate different kernels, here the solve
  // and the forces would see different CHARGE SETS.
  check_drude_consistency();

  // The same guard, generalized to plain fixed-charge ions outside the solve group
  // (no `fix drude` involved). See the function's header note for the escape hatches.
  check_ionfield_consistency();

  // --- long-range (DSF) setup: off-diagonal shift constants (f_shift, e_shift) matching pair coul/dsf ---
  // NOTE: the Wolf SELF-term (qqrd2e*(e_shift+2α/√π), ~3.3 eV) is intentionally NOT folded into eta:
  //   it over-softens water's tight intramolecular hardness -> the charge-transfer mode goes negative
  //   (indefinite) -> solve diverges. It is position-independent (zero force, zero virial) so it does
  //   NOT affect density/pressure. We keep the BARE eta diagonal (stiff, PD) + DSF off-diagonals, exactly
  //   as the working coul/shield path (which also has no self-term). Pair forces: coul/dsf (e_self there
  //   is a per-atom energy offset with no force; harmless for dynamics/pressure). [Energy conservation of
  //   the absolute NVE energy may need NVT or a self-term-free DSF pair.]
  if (lr_ewald) {
    // Route B: full Ewald, reciprocal IN the matvec. Real-space = plain erfc(g_ewald r)/r in compute_H;
    // reciprocal applied each matvec via PPPMSamqeq::compute_vector, with the MEASURED grid self
    // (calibrate_recip_self) subtracted so the diagonal stays = eta (NOT the analytic 2g_ewald/√π, which is
    // wrong on the finite PPPM grid). See add_reciprocal. The bare-eta diagonal (Ewald/Wolf self deliberately
    // omitted, see the DSF block above) is what keeps the operator PD — necessary but NOT
    // sufficient: the near-critical soft mode is the OFF-diagonal collective k=0 reciprocal coupling, which only
    // a real-space cutoff operator (lr_nrecip, the bond-softness path) removes.
    if (!force->kspace)
      error->all(FLERR, "samqeq lr_ewald requires kspace_style pppm/samqeq + an lj/cut/coul/long pair");
    eksp = dynamic_cast<SamqeqKspace *>(force->kspace);   // CPU pppm/samqeq OR device pppm/samqeq/kk
    if (!eksp)
      error->all(FLERR, "samqeq lr_ewald requires kspace_style pppm/samqeq[/kk] (provides compute_vector)");
    lr_pppm = dynamic_cast<PPPM *>(force->kspace);   // SAME object; PPPM base exposes the grid dims for grid_changed()
    lr_alpha = force->kspace->g_ewald;      // erfc damping == the kspace Ewald split
    e_shift = f_shift = 0.0;                 // no DSF shift in Ewald mode (reciprocal completes 1/r)
    if (comm->me == 0 && screen)
      utils::logmesg(lmp, "samqeq: long-range EWALD (reciprocal-in-matvec) on (alpha=g_ewald={:.4f},"
                          "swb={:.2f}); pair MUST be lj/cut/coul/long + kspace pppm/samqeq\n", lr_alpha, swb);
  } else if (lr_alpha > 0.0) {
    // shield slater is untested combined with plain DSF: compute_H's DSF branch (below) computes
    // erfc(αr)/r-e_shift-r*f_shift directly and NEVER calls shielded_coulomb() at all, so `shield slater` would
    // silently have ZERO effect here (and the combination is not validated) -- error instead.
    if (shield_gauss == SHIELD_SLATER)
      error->all(FLERR, "samqeq: fix_modify shield slater is not supported with plain DSF (lr_alpha>0,"
                        "lr_ewald=0) -- supported paths are lr_ewald>=2 (full Ewald), `fix_modify cutoff"
                        "on` (taper-cutoff), and the gas path (lr_ewald=0, lr_alpha=0)");
    double a = lr_alpha, rc = swb;
    double erfcc = erfc(a*rc), erfcd = exp(-a*a*rc*rc);
    f_shift = -(erfcc/(rc*rc) + 2.0*a/MY_PIS*erfcd/rc);
    e_shift = erfcc/rc - f_shift*rc;
    if (comm->me == 0 && screen)
      utils::logmesg(lmp, "samqeq: long-range DSF on (alpha={:.3f}, swb={:.2f}); bare eta diagonal"
                          "(no self-term); pair MUST be coul/dsf {:.3f} {:.2f}\n", a, rc, a, rc);
  }

}

/* ----------------------------------------------------------------------
   check_shield_consistency(). Assert that the pair style (which owns forces/energy) and
   this fix (which owns the charge solve) agree on the shielding kernel. The two select it INDEPENDENTLY;
   unchecked, a deck could integrate forces from erf(α_ij r)/r while equilibrating charges against
   1/∛(r³+1/γ³), or truncate the correction at 5 Å on one side and 8 Å on the other — silently, with
   plausible-looking output (the latter heats an NVE melt by thousands of kelvin within picoseconds).
   See the declaration banner in fix_qeq_sam.h for the channels.

   ★ WHY THE RANGE CHANNEL IS A MAGNITUDE TEST, NOT `cut == swb`. A cutoff difference only matters to the
   extent the correction (J − 1/r) is still nonzero where one side stops applying it. That depends entirely
   on the kernel and the col-4 parameters, and it spans FIVE orders of magnitude:

     model kernel, col4 cut/swb residual at r_trunc
     TIP4P-FQ (Rick zetas) slater z 3.08 5 / 10 0.00001 eV/e² physically nil
     water, narrow Gaussian pqeq Rc 0.5 5 / 8 0.0001 eV/e² nil
     SPC-FQ water cbrt γ 1.11 5 / 10 0.0056 eV/e² nil
     cbrt γ=1.0 cbrt γ 1.0 5 / 10 0.0076 eV/e² nil
     ---------------------------------------------------------- tol 0.02 -------
     cbrt γ=0.25 cbrt γ 0.25 8 / 12 0.0693 eV/e²
     cbrt γ=0.2171 cbrt γ 0.2171 8 / 12 0.1018 eV/e²
     wide Gaussian pqeq Rc 1.856 5 / 8 0.5616 eV/e² melt runaway

   Exact equality would reject nearly every deck with a cutoff difference, including the shipped examples,
   to catch the family that is genuinely broken. The residual test instead leaves a 7.4x gap between
   0.0076 and 0.0693, with the default tolerance roughly geometrically between them, and 74x to the wide
   Gaussian. cut == swb passes unconditionally (both sides truncate
   identically => they cannot disagree; a large SHARED discontinuity there is an energy-conservation
   matter for `taper`, not a force<->solve inconsistency).
-------------------------------------------------------------------------*/

void FixQEqSam::check_shield_consistency()
{
  if (!shieldchk_on) return;
  // plain DSF (lr_alpha>0, lr_ewald=0) computes erfc(αr)/r − shift inline in compute_H and never calls
  // shielded_coulomb() at all, so this fix carries no shielding kernel to disagree about. (Reached BEFORE
  // the lr_ewald block below overwrites lr_alpha with g_ewald, so this reads the fix_modify-set DSF alpha.)
  if (!lr_ewald && lr_alpha > 0.0) return;

  // exact=0: under `-sf kk` the style name carries a /kk suffix ("coul/shield/intra/kk"), which an exact
  // match misses, and the check would silently no-op on device decks.
  auto *csp = dynamic_cast<PairCoulShieldIntra *>(force->pair_match("coul/shield/intra", 0));
  if (!csp) return;   // file-mode / gas decks carry no pair-side correction: nothing to compare

  const char *kname[3] = {"cbrt", "gaussian", "slater"};
  const int pmode = csp->shield_mode_get();
  std::string bad;

  // --- channel 2: kernel MODE ---
  if (pmode != shield_gauss)
    bad += fmt::format("\n MODE pair `{}` vs fix `{}`", kname[pmode], kname[shield_gauss]);

  // --- channel 3: PQEq lambda. A deck that retunes it on the fix must set the same value on the pair
  // (`pair_style ... lambda <val>`).
  if (pmode == shield_gauss && shield_gauss == SHIELD_GAUSSIAN &&
      fabs(csp->shield_lambda_get() - shield_lambda) > 1.0e-12)
    bad += fmt::format("\n LAMBDA pair {:.6f} vs fix {:.6f}", csp->shield_lambda_get(), shield_lambda);
  if (pmode == shield_gauss && shield_gauss == SHIELD_GAUSSIAN) {   // per-pair radius table
    std::string t;
    for (int i = 1; i <= atom->ntypes; i++)
      for (int j = i; j <= atom->ntypes; j++) {
        const double rf = shield_rpair ? shield_rpair[i][j] : 0.0, rp = csp->shield_rpair_get(i, j);
        if (fabs(rf - rp) > 1.0e-12) t += fmt::format(" ({},{}) pair {:.4f}/fix {:.4f}", i, j, rp, rf);
      }
    if (!t.empty()) bad += "\n RPAIR" + t;
  }

  // --- channel 4: Slater per-type 2s/1s form-factor flags ---
  if (pmode == shield_gauss && shield_gauss == SHIELD_SLATER && is2s) {
    std::string t;
    for (int i = 1; i <= atom->ntypes; i++)
      if ((csp->shield_is2s_get(i) != 0) != (is2s[i] != 0))
        t += fmt::format(" {}(pair {}/fix {})", i, csp->shield_is2s_get(i) ? "2s" : "1s",
                         is2s[i] ? "2s" : "1s");
    if (!t.empty()) bad += "\n 2S type" + t;
  }

  // --- channel 1: RANGE, by residual magnitude (see the banner above) ---
  const double pcut = csp->shield_cut_get();
  if (pmode == shield_gauss && fabs(pcut - swb) > 1.0e-9) {
    const double rt = MIN(pcut, swb);       // truncation radius: beyond it ONE side keeps correcting
    double worst = 0.0;
    int wi = 0, wj = 0;
    for (int i = 1; i <= atom->ntypes; i++) {
      if (gamma[i] <= 0.0) continue;        // types absent from the param file (col 4 is a radius/exponent:
      for (int j = i; j <= atom->ntypes; j++) {          // 0 is "not parameterised", and would divide by 0)
        if (gamma[j] <= 0.0) continue;
        const double d = fabs(shielded_coulomb(i, j, rt) - 1.0/rt);
        if (d > worst) { worst = d; wi = i; wj = j; }
      }
    }
    // -> eV per unit charge pair: qqrd2e is in native deck units, ev_scale converts real->eV (metal: 1.0),
    // so the tolerance means the same physical thing in both unit styles.
    worst *= force->qqrd2e / ev_scale;
    if (worst > shieldchk_tol)
      bad += fmt::format("\n RANGE pair cutoff {:.4g} vs fix swb {:.4g} -- the shielding correction is"
                         "still {:.4g} eV/e^2 at r = {:.4g} (tol {:.4g}; worst type pair {}-{}, col4"
                         "{:.4g}/{:.4g})", pcut, swb, worst, rt, shieldchk_tol, wi, wj, gamma[wi], gamma[wj]);
  }

  // --- channel 5: SCOPE, i.e. WHICH pairs each side corrects. The pair style corrects the
  //   set selected by its `all` keyword (intra_only=0 -> every pair in cutoff; 1 -> same-molecule
  //   only). The solve's set is fixed by lr_ewald, which comes from the param-file header and NOT
  //   from the deck: lr_ewald>=2 shields every pair in the H-block, lr_ewald==1 shields only
  //   same-molecule pairs and leaves inter bare (compute_H, the lr_ewald==2||3 branch vs the
  //   lr_ewald==1 branch). The two are configured in different files.
  //
  //   ★ WHY THIS ONE IS AN EQUALITY TEST WHERE `RANGE` IS A MAGNITUDE TEST. Range asks how much
  //   correction is left where one side stops applying it, and that is a genuine matter of degree
  //   (five orders of magnitude). Scope is not: with lr_ewald=2 and no `all`,
  //   EVERY intermolecular pair carries the correction in the charge matrix and none of it in the
  //   forces, at all separations including contact. There is no small-residual regime to tolerate.
  //
  //   On examples/tip4pfq_liquid/in.tip4pfq_xl_slater (lr_ewald=2, no `all`) the mismatch leaves the
  //   solved charges bit-identical -- the pair scope never enters the solve -- while pe moves and the
  //   LIQUID dipole moves 2.4631 +/- 0.0028 -> 2.3396 +/- 0.0014 D, a 5% shift in a reported
  //   observable with no other diagnostic: a converged residual on a neutral charge set proves nothing.
  //
  //   Guarded to the Ewald paths (lr_ewald>=1). The gas path (lr_ewald=0, lr_alpha=0)
  //   assembles its H-block through the base class and is not audited here; DSF already returned
  //   above. Override with `fix_modify <id> shieldcheck off`, as for every other channel.
  if (lr_ewald >= 1) {
    const int pair_all  = !csp->shield_intra_only();
    const int solve_all = (lr_ewald >= 2);
    if (pair_all != solve_all)
      bad += fmt::format("\n SCOPE pair corrects {} pairs, solve (lr_ewald={}) shields {} -- {}",
                         pair_all ? "ALL" : "INTRA-molecular", lr_ewald,
                         solve_all ? "ALL pairs" : "INTRA-molecular pairs only",
                         solve_all ? "add the `all` keyword to pair_style coul/shield/intra, or set"
                                     "the param-file header's 6th field (lr_ewald) to 1"
                                   : "drop the `all` keyword from pair_style coul/shield/intra, or"
                                     "set the param-file header's 6th field (lr_ewald) to 2");
  }

  // --- channel 6: COVERAGE, i.e. whether the pair style corrects every solved type pair at all. Under
  //   pair_style hybrid/overlay LAMMPS mixes a sub-style's coefficients only for type pairs that no other
  //   sub-style already covers, so per-type lines (`pair_coeff 1 1 coul/shield/intra ...`, `2 2 ...`)
  //   beside `pair_coeff * * coul/long` leave the cross pairs with no shielding correction in the forces
  //   while the solve applies it to them. The sub-style's setflag records exactly the pairs it was given.
  //   Pairs checked: every pair of solved types under the all-pairs scope, the pairs that share a
  //   molecule under the intramolecular scope. A lone (non-hybrid) coul/shield/intra mixes as usual.
  if (force->pair != csp && csp->setflag) {
    const int nt = atom->ntypes;
    std::vector<int> need((nt + 1) * (nt + 1), 0), all((nt + 1) * (nt + 1), 0);
    const int *type = atom->type, *mask = atom->mask;
    const int nlocal = atom->nlocal;
    if (lr_ewald >= 2 || !atom->molecule_flag) {
      std::vector<int> has(nt + 1, 0);
      for (int i = 0; i < nlocal; i++) if (mask[i] & groupbit) has[type[i]] = 1;
      for (int a = 1; a <= nt; a++)
        for (int b = a; b <= nt; b++) need[a * (nt + 1) + b] = has[a] && has[b] && (lr_ewald >= 2);
    } else {
      std::map<tagint, std::vector<int>> mol;
      for (int i = 0; i < nlocal; i++)
        if (mask[i] & groupbit) mol[atom->molecule[i]].push_back(type[i]);
      for (auto &m : mol)
        for (size_t u = 0; u < m.second.size(); u++)
          for (size_t v = u + 1; v < m.second.size(); v++) {
            const int a = MIN(m.second[u], m.second[v]), b = MAX(m.second[u], m.second[v]);
            need[a * (nt + 1) + b] = 1;
          }
    }
    MPI_Allreduce(need.data(), all.data(), (nt + 1) * (nt + 1), MPI_INT, MPI_MAX, world);
    std::string t;
    for (int a = 1; a <= nt; a++)
      for (int b = a; b <= nt; b++)
        if (all[a * (nt + 1) + b] && !csp->setflag[a][b]) t += fmt::format(" ({},{})", a, b);
    if (!t.empty())
      bad += "\n COVERAGE no coul/shield/intra coefficients for solved type pair(s)" + t +
             " -- under pair_style hybrid/overlay LAMMPS does not mix a sub-style across type pairs another"
             "sub-style already covers; give `pair_coeff * * coul/shield/intra ...` or one line per pair";
  }

  if (bad.empty()) return;

  error->all(FLERR, "samqeq: shielding-kernel MISMATCH between pair coul/shield/intra and fix {} -- the"
                    "pair FORCES/ENERGY and the charge SOLVE would use different kernels:{}\n"
                    "  fix {}: swb {:.4g}, kernel {}\n"
                    "  pair coul/shield/intra: cutoff {:.4g}, scope {}, kernel {}\n"
                    "  Fix: make both sides agree -- match the pair cutoff to swb, and select the same"
                    "SET of pairs on both sides (pair `all` <-> param-file lr_ewald>=2)."
                    "See the channel notes in fix_qeq_sam.h."
                    "Override with `fix_modify {} shieldcheck <tol>` or `... shieldcheck off`.",
             id, bad, id, swb, kname[shield_gauss],
             pcut, csp->shield_intra_only() ? "intra" : "all", kname[pmode], id);
}

/* ----------------------------------------------------------------------
   check_drude_consistency(): refuse a deck with Drude shells. fix qeq/sam does not couple `fix drude` shells into
   the charge solve, so their charges would be present in the forces and absent from the solve.
-------------------------------------------------------------------------*/

void FixQEqSam::check_drude_consistency()
{
  for (int ifix = 0; ifix < modify->nfix; ifix++)
    if (strcmp(modify->fix[ifix]->style, "drude") == 0)
      error->all(FLERR, "fix qeq/sam does not support Drude shells (`fix drude`): the shell charges would be in the"
                        "forces and absent from the charge solve");
}

/* ----------------------------------------------------------------------
   check_ionfield_consistency(). The check_drude_consistency() guard, generalized to
   plain FIXED-CHARGE atoms (ions) outside the solve group.

   compute_H correctly skips non-group columns, so the Coulomb field of a fixed non-group charge
   reaches the solved charges ONLY through the RHS channel (add_fixed_charge_field), which is gated
   behind drude_flag / ionfield_flag. A deck whose solve group excludes a charged species therefore
   runs an inconsistent Hamiltonian: the ions are in the FORCES and the dynamics and ABSENT from the
   charge solve: moving a Cl⁻ from 10.0 to 2.6 Å leaves q_H unchanged, and nothing in the output
   shows it.

   Escapes: `fix_modify <id> ionfield on` (couple the field) or `... ionfield off` (acknowledged
   ion-blind control).
-------------------------------------------------------------------------*/

void FixQEqSam::check_ionfield_consistency()
{
  if (ionfield_explicit) return;                // the deck stated a choice and the fix_modify handler logged it

  bigint nout_local = 0;                        // fixed charged atoms OUTSIDE the solve group — exactly the
  int *mask = atom->mask;                       //   sources add_fixed_charge_field would use
  double *q = atom->q;
  for (int i = 0; i < atom->nlocal; i++)
    if (!(mask[i] & groupbit) && q[i] != 0.0) nout_local++;
  bigint nout = nout_local;
  MPI_Allreduce(&nout_local, &nout, 1, MPI_LMP_BIGINT, MPI_SUM, world);
  if (nout == 0) return;                        // no fixed charges outside the solve group -> nothing to say

  // ionfield is ON by default. Couple the field -- but never silently: the choice changes the solved
  // charges, so it announces itself. One line, rank 0, setup only.
  if (ionfield_flag) {
    if (comm->me == 0)
      error->warning(FLERR,
        "samqeq: {} fixed charged atom(s) lie outside the charge-solve group; their shielded field is coupled"
        "into the QEq right-hand side (ionfield ON by default). This supplies the ions' electrostatic field"
        "only -- necessary but NOT sufficient (~56% of the QM dipole response at contact)."
        "`fix_modify {} ionfield off` selects the ion-blind solve.", nout, id);
    return;
  }

  // Unreachable through fix_modify (`off` sets ionfield_explicit); a backstop so that a cleared flag
  // without a stated choice is an error rather than a silent ion-blind solve.
  error->all(FLERR,
    "fix qeq/sam: {} fixed charged atom(s) lie OUTSIDE the charge-solve group, and their Coulomb field is\n"
    "  NOT in the QEq right-hand side. They are present in the FORCES and the dynamics and ABSENT from the\n"
    "  charge solve -- an inconsistent Hamiltonian: the solved charges cannot respond to them at all\n"
    "  (moving a Cl- from 10.0 to 2.6 A leaves q_H unchanged).\n"
    "  State the choice explicitly with one of:\n"
    "  FIX: fix_modify {} ionfield on (fixed non-group charges' shielded field -> QEq RHS)\n"
    "  CONTROL: fix_modify {} ionfield off (run ion-blind ON PURPOSE; silences this error, logs the choice)\n"
    "  VERIFY either way: move or recharge a non-group ion and confirm the solved charges move.",
    nout, id, id);
}

/* ----------------------------------------------------------------------
   route the two neighbor lists: id 0 = the base HALF list (charge solve, member `list`);
   id 1 = the FULL list requested in init() (ionfield sweep and device solve, member `list_full`).
-------------------------------------------------------------------------*/
void FixQEqSam::init_list(int id, NeighList *ptr)
{
  if (id == 0) list = ptr;
  else         list_full = ptr;
}

/* ----------------------------------------------------------------------
   parameter file:
     line 1 (header): gamma_align kappa_bond
     then one line per atom type: type chi eta gamma q0 eHOMO eLUMO
   (chi,eta,gamma are the usual qeq params: gamma = Coulomb shielding.
    eHOMO/eLUMO (eV) are read and ignored; supply placeholders, e.g. -7 3.)
-------------------------------------------------------------------------*/
void FixQEqSam::pertype_parameters(char *arg)
{
  reaxflag = 0;
  const int nt = atom->ntypes;
  memory->create(chi,   nt+1, "samqeq:chi");
  memory->create(eta,   nt+1, "samqeq:eta");
  memory->create(gamma, nt+1, "samqeq:gamma");
  memory->create(q0,    nt+1, "samqeq:q0");
  memory->create(c3_type, nt+1, "samqeq:c3_type");   // per-type IP-staircase cubic on-site coeff
  memory->create(c4_type, nt+1, "samqeq:c4_type");   // per-type IP-staircase quartic on-site coeff
  memory->create(ehomo, nt+1, "samqeq:ehomo");
  memory->create(elumo, nt+1, "samqeq:elumo");
  memory->create(is2s,  nt+1, "samqeq:is2s");   // shield slater per-type 2s/1s flag; default 0 (1s) below,
                                                // overridden by `fix_modify shield slater 2s <type>...`
  for (int i = 0; i <= nt; i++) is2s[i] = 0;

  if (comm->me == 0) {
    int *set = new int[nt+1];
    for (int i = 0; i <= nt; i++) { set[i]=0; chi[i]=eta[i]=gamma[i]=q0[i]=ehomo[i]=elumo[i]=0.0; c3_type[i]=c4_type[i]=0.0; }
    try {
      TextFileReader reader(arg, "samqeq parameters");
      // header line: gamma_align kappa_bond r_ov [r_loc] ... (r_loc is read and unused)
      char *hdr = reader.next_line();
      if (!hdr) throw TokenizerException("samqeq: missing header (gamma_align kappa_bond r_ov [r_loc])", "");
      ValueTokenizer hv(hdr);
      if (hv.count() < 3) throw TokenizerException("samqeq: header needs >=3 values", hdr);
      gamma_align = hv.next_double();
      kappa_bond  = hv.next_double();
      r_ov        = hv.next_double();
      r_loc       = hv.has_next() ? hv.next_double() : 2.5;   // read, unused
      lr_alpha    = hv.has_next() ? hv.next_double() : 0.0;   // DSF damping (0 = shielded, long-range off)
      lr_ewald    = hv.has_next() ? (int)(hv.next_double()+0.5) : 0;  // 1 = full Ewald (erfc+reciprocal SCF)
      if (lr_ewald < 0 || lr_ewald > 2)
        throw TokenizerException("samqeq: lr_ewald must be 0, 1 or 2", hdr);
      lr_ridge    = hv.has_next() ? hv.next_double() : 0.0;   // Tikhonov diagonal hardness (eV/e²): lifts the
                                                             // near-catastrophe soft mode so the solve stays
                                                             // bounded under wide (NVE) sampling (0 = off)
      // Header values 8-11 are accepted and read; no solve path uses them.
      r_bond      = hv.has_next() ? hv.next_double() : 2.6;
      graph_wedge   = hv.has_next() ? hv.next_double() : 0.8*ev_scale;
      pd_lam_min    = hv.has_next() ? hv.next_double() : 0.6*ev_scale;
      graph_hop_max = hv.has_next() ? (int)(hv.next_double()+0.5) : 3;
      // per-type lines
      char *line;
      while ((line = reader.next_line(7))) {
        ValueTokenizer v(line);
        int it = v.next_int();
        if (it < 1 || it > nt) throw TokenizerException("samqeq: bad type", std::to_string(it));
        chi[it]=v.next_double(); eta[it]=v.next_double(); gamma[it]=v.next_double();
        q0[it]=v.next_double();
        ehomo[it]=v.next_double(); elumo[it]=v.next_double();   // read and ignored
        // optional columns 8/9 = IP-staircase cubic/quartic on-site coeffs (default 0 ⇒ plain
        // parabola). next_line(7) admits 7-column lines.
        c3_type[it] = v.has_next() ? v.next_double() : 0.0;
        c4_type[it] = v.has_next() ? v.next_double() : 0.0;
        set[it]=1;
      }
    } catch (EOFException &) {
    } catch (std::exception &e) { error->one(FLERR, e.what()); }
    for (int i = 1; i <= nt; i++) if (!set[i]) error->one(FLERR,"samqeq: missing params type {}",i);
    delete[] set;
  }
  MPI_Bcast(&gamma_align,1,MPI_DOUBLE,0,world); MPI_Bcast(&kappa_bond,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&r_ov,1,MPI_DOUBLE,0,world);   MPI_Bcast(&r_loc,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&lr_alpha,1,MPI_DOUBLE,0,world); MPI_Bcast(&lr_ewald,1,MPI_INT,0,world);
  MPI_Bcast(&lr_ridge,1,MPI_DOUBLE,0,world); MPI_Bcast(&r_bond,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&graph_wedge,1,MPI_DOUBLE,0,world); MPI_Bcast(&pd_lam_min,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&graph_hop_max,1,MPI_INT,0,world);
  MPI_Bcast(chi,nt+1,MPI_DOUBLE,0,world);  MPI_Bcast(eta,nt+1,MPI_DOUBLE,0,world);
  MPI_Bcast(gamma,nt+1,MPI_DOUBLE,0,world);MPI_Bcast(q0,nt+1,MPI_DOUBLE,0,world);
  MPI_Bcast(ehomo,nt+1,MPI_DOUBLE,0,world);MPI_Bcast(elumo,nt+1,MPI_DOUBLE,0,world);
  MPI_Bcast(c3_type,nt+1,MPI_DOUBLE,0,world); MPI_Bcast(c4_type,nt+1,MPI_DOUBLE,0,world);

  if (gamma_align <= 0.0) error->all(FLERR,"samqeq: gamma_align must be > 0");
  if (r_ov <= 0.0)        error->all(FLERR,"samqeq: r_ov must be > 0");
  if (r_loc <= 0.0)       error->all(FLERR,"samqeq: r_loc must be > 0");
  if (lr_alpha < 0.0)     error->all(FLERR,"samqeq: lr_alpha must be >= 0");
  for (int itype = 1; itype <= nt; itype++)
    if (eta[itype] <= 0.0) error->all(FLERR,"samqeq: eta must be > 0 for type {}", itype);

  // backup the param eta so the DSF self-term can be (re)applied idempotently in init()
  memory->create(eta0, nt+1, "samqeq:eta0");
  for (int i = 0; i <= nt; i++) eta0[i] = eta[i];

  // q0 reference-charge field: needed only when some type carries a nonzero q0 (e.g. the PQEq/Drude core
  // q0=1.0). For q0=0 models (SPC-FQ, TIP4P-FQ, metals) the omitted field is identically zero -> skip it
  // entirely so those runs pay no extra matvec. (See the lr RHS q0field term.)
  has_q0ref = 0;
  for (int i = 1; i <= nt; i++) if (q0[i] != 0.0) has_q0ref = 1;

  // Enable the charge-dependent (Picard) diagonal whenever any type carries IP-staircase coeffs,
  // so the solve routes through quartic_scf / ASPC-quartic even with no `fix_modify quartic` group. With
  // c3/c4 = 0 lr_quartic is untouched. (A later fix_modify quartic only ever turns lr_quartic ON.)
  for (int i = 1; i <= nt; i++) if (c3_type[i] != 0.0 || c4_type[i] != 0.0) {
    lr_quartic = 1;
    if (comm->me == 0)
      utils::logmesg(lmp, "samqeq: IP-STAIRCASE on-site anharmonicity active — per-type c3/c4 give a"
                          "charge-dependent hardness eta_eff(q)=eta+1/2 c3 q+1/6 c4 q^2 via the Picard/ASPC-quartic"
                          "solve; oxidation state floats with q0+field from ONE element param set\n");
    break;
  }
}

/* ----------------------------------------------------------------------
   shielded_coulomb(ti,tj,r): the NET short-range Coulomb kernel for the lr_ewald=2 solve (and the fixed-
   charge field). Three forms, selected by shield_gauss (SHIELD_CBRT/PQEQ/SLATER, fix_qeq_sam.h):
     0 cbrt (default): J_shield = 1/∛(r³ + 1/γ³), γ = sqrt(gamma[ti] gamma[tj]) (param 4th col).
     1 PQEq: Gaussian-Gaussian overlap erf(α_ij r)/r, α_i = λ/(2 Rc_i²) (Rc = param 4th col
                         reinterpreted as the Gaussian core radius), α_ij = sqrt(α_i α_j/(α_i+α_j)).
     2 Slater: Rick's exact Slater-orbital Coulomb-overlap integral J(r) (JCP 101, 6141), read from
                         the per-(ti,tj) table built by build_slater_tables() (param 4th col reinterpreted as
                         the Slater exponent zeta, 1/Å; is2s selects the 1s/2s form factor per type).
   All three → 1/r at large r and are finite at r→0; the caller subtracts erf(g_ewald r)/r (the reciprocal adds
   it back), so the NET off-diagonal = this kernel.
   iondamp: DESIGNATED type pairs (fix_modify iondamp; iondamp_b[ti][tj]>0) get the Tang-Toennies-damped
   kernel J_damp = f_n(b r)·J on top of whichever branch is active — applied HERE so every fix-side consumer
   (compute_H Ewald + taper, add_fixed_charge_field, field_diag, and through the H matrix: q0field/quartic-gate/XL/
   ASPC) sees the identical damped operator. lr_iondamp=0 (default) skips it; undamped pairs evaluate
   the three kernel branches unchanged.
-------------------------------------------------------------------------*/
/* ---- Where the HOST-side per-step time goes when the solve runs on device.
   Enabled by SAMQEQ_HOST_TIME=1; the kokkos fix's own post_run chains to this. ----*/
void FixQEqSam::post_run()
{
  if (th_on != 1 || comm->me != 0 || th_steps == 0) return;
  const double other = th_solve - th_molinv - th_q0field - th_proj - th_cg;
  utils::logmesg(lmp, "samqeq HOST TIMING over {} solves\n"
                      "   qeq_solve total {:8.3f} s\n"
                      "     build_molinv {:8.3f} s {:5.1f}%\n"
                      "     q0 reference field {:8.3f} s {:5.1f}%\n"
                      "     project_neutral ({} calls) {:8.3f} s {:5.1f}%\n"
                      "     CG (device when engaged) {:8.3f} s {:5.1f}%\n"
                      "     everything else (RHS fill, calculate_Q, ASPC bookkeeping) {:8.3f} s {:5.1f}%\n",
                 th_steps, th_solve,
                 th_molinv, 100*th_molinv/th_solve, th_q0field, 100*th_q0field/th_solve,
                 th_projcalls, th_proj, 100*th_proj/th_solve, th_cg, 100*th_cg/th_solve,
                 other, 100*other/th_solve);
}

double FixQEqSam::shielded_coulomb(int ti, int tj, double r)
{
  double J;
  if (shield_gauss == SHIELD_SLATER) {
    double dJdr;   // unused here (energy-only caller); compute_H/add_fixed_charge_field don't need the force form
                   // (the on-site quadratic solve has no r-dependent force term from this kernel by itself --
                   // the pair force consistency is carried entirely by pair_coul_shield_intra's OWN table eval)
    int idx = slater_tri_index(ti, tj, atom->ntypes);
    J = slater_tabs[idx].eval(r, dJdr);
  } else if (shield_gauss == SHIELD_GAUSSIAN) {
    double aij;
    if (shield_rpair && shield_rpair[ti][tj] > 0.0) {        // per-pair radius override
      aij = sqrt(shield_lambda)/(2.0*shield_rpair[ti][tj]);
    } else {
      double ai = shield_lambda*0.5/(gamma[ti]*gamma[ti]);   // gamma reinterpreted as Rc when Gaussian
      double aj = shield_lambda*0.5/(gamma[tj]*gamma[tj]);
      aij = sqrt(ai*aj/(ai+aj));
    }
    J = erf(aij*r)/r;
  } else {
    double g = sqrt(gamma[ti]*gamma[tj]);
    J = 1.0/cbrt(r*r*r + 1.0/(g*g*g));
  }
  if (lr_iondamp) {
    const double b = iondamp_b[ti][tj];
    if (b > 0.0) { double dfdx; J *= samqeq_tt_damp(b*r, iondamp_n[ti][tj], dfdx); }
  }
  return J;
}

/* ----------------------------------------------------------------------
   build_slater_tables(): (re)build the per-(ti<=tj)-type-pair Slater J(r) tables from is2s/gamma
   (reinterpreted as zeta when shield_gauss==SHIELD_SLATER) and swb (the fix's cutoff). Called from init()
   -- LAZILY in the sense that it only runs when shield_gauss==SHIELD_SLATER, but UNCONDITIONALLY on every
   init() call while active (init() runs once per run-setup, not per step/matvec) -- the simplest correct
   policy for "rebuild if swb changes": swb is const after the fix's ctor, and is2s/gamma can only change via
   fix_modify between runs, so a full rebuild at every init() is always up to date and cheap (see
   slater_jtable.h for the O(nint) cost -- nint=1000 quadrature points * npts=2000 table nodes, ONCE per
   unique type pair, never per force/solve call).
-------------------------------------------------------------------------*/
void FixQEqSam::build_slater_tables()
{
  const int nt = atom->ntypes;
  const int npairs = nt * (nt + 1) / 2;
  slater_tabs.assign(npairs, SlaterJTable());
  for (int ti = 1; ti <= nt; ti++)
    for (int tj = ti; tj <= nt; tj++) {
      int idx = slater_tri_index(ti, tj, nt);
      // gamma is REINTERPRETED as the Slater exponent zeta (1/Å) when shield_gauss==SHIELD_SLATER (same
      // "reuse the 4th param column" convention as the pqeq Rc reinterpretation above).
      slater_tabs[idx].build(is2s[ti] != 0, is2s[tj] != 0, gamma[ti], gamma[tj], swb);
    }
  if (comm->me == 0)
    utils::logmesg(lmp, "samqeq: Slater-overlap J(r) tables built ({} unique type pairs, {} pts each,"
                        "r in [{:.3g},{:.3g}])\n", npairs, slater_tabs.empty() ? 0 : slater_tabs[0].npts,
                        slater_tabs.empty() ? 0.0 : slater_tabs[0].rmin, swb);
}

/* ----------------------------------------------------------------------
   calc_Hval(r,ti,tj): gas-path (lr_ewald=0, lr_alpha=0) H.val hook, called from the INHERITED
   FixQEqBaseSam::compute_H (that function inlines cbrt via shld/calculate_H and is NOT reachable
   from shielded_coulomb() at all -- the two live in different classes/paths). Slater mode
   routes through the SAME per-type-pair table shielded_coulomb() uses; cbrt/pqeq delegate to the base
   (an explicit branch calling a small helper, not a rewrite of the base's Taper/shld machinery).
-------------------------------------------------------------------------*/
double FixQEqSam::calc_Hval(double r, int ti, int tj)
{
  double v;
  if (shield_gauss != SHIELD_SLATER) v = FixQEqBaseSam::calc_Hval(r, ti, tj);   // cbrt/pqeq: the base kernel
  else {
    double dJdr;
    int idx = slater_tri_index(ti, tj, atom->ntypes);
    double J = slater_tabs[idx].eval(r, dJdr);
    // Same 7th-order Taper (Tap, via taper_poly()) as calculate_H's cbrt, so slater's cutoff-at-swb behavior
    // matches the cbrt/pqeq gas path exactly (a smooth taper to 0, not a hard truncation of the table).
    v = taper_poly(r) * force->qqrd2e * J;
  }
  // iondamp on the gas path: the TT factor rides ON TOP of the base value (Taper·qqrd2e·J is linear
  // in J, so multiplying the result by f_n(b r) IS the damped kernel) — keeps gas-phase cluster scans
  // usable with iondamp. lr_iondamp=0 / b=0 ⇒ undamped.
  if (lr_iondamp) {
    const double b = iondamp_b[ti][tj];
    if (b > 0.0) { double dfdx; v *= samqeq_tt_damp(b*r, iondamp_n[ti][tj], dfdx); }
  }
  return v;
}

/* ----------------------------------------------------------------------
   compute_H: off-diagonal Coulomb hardness for the charge solve.
     lr_alpha = 0 -> shielded short-range (inherited FixQEqBaseSam::compute_H).
     lr_alpha > 0 -> damped-shifted-force (DSF): H_ij = qqrd2e*(erfc(α r)/r − e_shift − r·f_shift),
                     IDENTICAL to pair coul/dsf <α> <swb>. The Wolf self-term is folded into the
                     diagonal eta in init(). This captures the long-range Coulomb (the truncated
                     shielded form under-binds polar liquids -> wrong density/pressure).
   Same CSR assembly + half-dedup as the inherited compute_H; only the per-pair value differs.
-------------------------------------------------------------------------*/
/* 7th-order Taper polynomial (Tap from init_taper): 1 at swa, smoothly -> 0 at swb. Used by the
   Gaussian-shielded taper-cutoff solve (lr_nrecip) to truncate the full shielded Coulomb at swb.*/
double FixQEqSam::taper_poly(double r)
{
  double T = Tap[7];
  for (int n = 6; n >= 0; --n) T = T * r + Tap[n];
  return T;
}

void FixQEqSam::compute_H()
{
  if (lr_alpha <= 0.0) { FixQEqBaseSam::compute_H(); return; }
  ensure_matrix_capacity();

  int *mask = atom->mask, *type = atom->type;
  tagint *tag = atom->tag; double **x = atom->x;
  constexpr double EPS = 1e-4;
  double a = lr_alpha;

  m_fill = 0;
  for (int ii = 0; ii < nn; ii++) {
    int i = ilist[ii];
    if (!(mask[i] & groupbit)) continue;
    int *jlist = firstneigh[i]; int jnum = numneigh[i];
    H.firstnbr[i] = m_fill;
    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj] & NEIGHMASK;
      // COLUMN group test. H multiplies the SOLVE VECTOR, so a column for an atom
      // that is not a solve variable is never meaningful: the matvec reads x[j], which for a
      // non-group atom is either never-written workspace (uninitialized -> silent garbage) or
      // zero (the atom's real fixed charge silently absent). Neither is the physics. The Coulomb
      // field of fixed non-group charges belongs in the RHS, where add_fixed_charge_field() already puts
      // it using their ACTUAL charges q[j] -- that is the correct channel.
      // Without this test, a group-restricted solve would assemble core->non-group entries, and CG
      // would converge exactly onto the WRONG system -- "converged" charges that no residual test can
      // flag, non-deterministic across runs (heap-history dependent).
      // Whole-system solves (every atom in the group) are unaffected.
      if (!(mask[j] & groupbit)) continue;
      double dx=x[j][0]-x[i][0], dy=x[j][1]-x[i][1], dz=x[j][2]-x[i][2];
      double rsq = dx*dx+dy*dy+dz*dz;
      if (rsq > swb*swb) continue;
      int flag = 0;                              // full-list -> half dedup (qeq/reaxff convention)
      if (j < atom->nlocal) flag = 1;
      else if (tag[i] < tag[j]) flag = 1;
      else if (tag[i] == tag[j]) {
        if (dz > EPS) flag = 1;
        else if (fabs(dz) < EPS) { if (dy > EPS) flag = 1;
          else if (fabs(dy) < EPS && dx > EPS) flag = 1; }
      }
      if (!flag) continue;
      double r = sqrt(rsq);
      if (m_fill >= H.m)
        error->one(FLERR, "samqeq: H matrix overflow at atom {} (m_fill {} >= {})", i, m_fill, H.m);
      H.jlist[m_fill] = j;
      if (lr_ewald == 2) {
        // lr_ewald=2 H-block real-space:
        // FULLY-SHIELDED Ewald (Rick electrostatics). ALL pairs use J_shield(r)=1/∛(r³+1/γ³),
        // →1/r at long range. Real-space = J_shield(r) − erf(αr)/r; the reciprocal (compute_vector) adds
        // erf(αr)/r back ⇒ net off-diagonal = J_shield for every pair. Shielding the INTER pairs too keeps
        // the model OFF the bare-1/r polarization catastrophe and is consistent with the shielded form the
        // SPC-FQ params were fit to → stable, experiment-tunable via γ (param-file knob).
        double shielded = shielded_coulomb(type[i], type[j], r);             // cbrt J_shield or PQEq Gaussian
        if (lr_nrecip)
          // GAUSSIAN-SHIELDED TAPER CUTOFF (PQEq-faithful): full shielded Coulomb tapered to 0 at swb, NO Ewald
          // split, NO reciprocal -> strictly short-ranged operator (no k=0 collective soft mode). Forces unchanged.
          H.val[m_fill] = force->qqrd2e * shielded * taper_poly(r);
        else
          H.val[m_fill] = force->qqrd2e * (shielded - (1.0 - erfc(a*r))/r);   // net = shielded (all pairs, via Ewald)
      } else if (lr_ewald) {
        // lr_ewald=1: INTRA-molecular shielding only; INTER = bare 1/r (force-consistent with stock coul/long).
        // Intra (rigid) net J_shield tames the ~1Å O–H over-softening; inter net bare 1/r (3.01 D, near-critical).
        // ★ intra-shield + BARE-inter is EXACTLY Rick's TIP4P-FQ structure (mdtpn.f applies
        // the Slater J only to the fixed intramolecular pairs; ALL inter pairs are bare erfc+recip) — lr_ewald=2's
        // all-pair shielding under-polarizes the liquid ~7%. This branch honors `fix_modify shield pqeq|slater`
        // for the INTRA kernel; the default cbrt is inlined.
        tagint *mol = atom->molecule;
        if (mol && mol[i] != 0 && mol[i] == mol[j]) {
          double shielded;
          if (shield_gauss != SHIELD_CBRT) shielded = shielded_coulomb(type[i], type[j], r);
          else { double g = sqrt(gamma[type[i]]*gamma[type[j]]); shielded = 1.0/cbrt(r*r*r + 1.0/(g*g*g)); }
          H.val[m_fill] = force->qqrd2e * (shielded - (1.0 - erfc(a*r))/r);   // intra: net J_shield
        } else {
          H.val[m_fill] = force->qqrd2e * (erfc(a*r)/r);                       // inter: net bare 1/r
        }
      } else {
        H.val[m_fill] = force->qqrd2e * (erfc(a*r)/r - e_shift - r*f_shift);  // DSF (== pair coul/dsf)
      }
      m_fill++;
    }
    H.numnbrs[i] = m_fill - H.firstnbr[i];
  }
  if (m_fill >= H.m) error->all(FLERR,"samqeq: H overflow (lr) m_fill={} H.m={}", m_fill, H.m);
}

/* ====================== RESPONSE KERNEL =============*/



double FixQEqSam::calc_w(int i, int j, double r)
{
  // Overlap prefactor on the BOND/contact scale r_ov (distinct from the long Coulomb cutoff swb).
  // Gaussian, peak 1 at contact. (ACKS2's d^3(1-d)^6 needs a
  // short per-type bcut + large amplitude; here r_ov sets the scale and kappa sets the amplitude.)
  tagint *mol = atom->molecule;
  int intra = (mol && mol[i] != 0 && mol[i] == mol[j]);
  if (intra) return kappa_bond * exp(-(r*r) / (r_ov*r_ov));               // INTRA: Gaussian bond softness
  // INTER: no inter-fragment response. A zero weight makes compute_X skip every inter-fragment pair.
  return 0.0;
}

/* response X = -L(W). Same CSR assembly / dedup as ACKS2 compute_X,
   only the per-pair weight calc_w(i,j,r) differs. (Shadows base compute_X;
   invoked from our init_matvec below.)*/
/* fill bcut[i][j] = bcut_global for the bond-softness ACKS2 mode (samQEq normally SKIPS init_bondcut). Lazy:
   allocate on first use; bond_softness (the kappa scale) is set by fix_modify bondsoft.*/
void FixQEqSam::setup_bondsoft_bcut()
{
  int nt = atom->ntypes;
  if (bcut == nullptr) memory->create(bcut, nt+1, nt+1, "samqeq:bondsoft_bcut");
  for (int i = 1; i <= nt; ++i)
    for (int j = 1; j <= nt; ++j) bcut[i][j] = bcut_global;
}

void FixQEqSam::compute_X()
{
  if (lr_bondsoft) { FixACKS2Sam::compute_X(); return; }   // standard Verstraelen bcut bond-softness (not the gate)
  tagint *tag = atom->tag; double **x = atom->x; int *mask = atom->mask;
  constexpr double SMALL = 1e-4;
  memset(X_diag, 0, atom->nmax*sizeof(double));
  m_fill = 0;
  for (int ii = 0; ii < nn; ii++) {
    int i = ilist[ii];
    if (!(mask[i] & groupbit)) continue;
    int *jl = firstneigh[i]; int jnum = numneigh[i];
    X.firstnbr[i] = m_fill;
    for (int jj = 0; jj < jnum; jj++) {
      int j = jl[jj] & NEIGHMASK;
      if (!(mask[j] & groupbit)) continue;       // skip non-group atoms (e.g. uncharged TIP4P O):
                                                 // they carry no charge variable, so they must not
                                                 // enter the response graph. Including them welds the
                                                 // 0.15 A M-O pair (kappa_bond).
      double dx=x[j][0]-x[i][0], dy=x[j][1]-x[i][1], dz=x[j][2]-x[i][2];
      double rsq = dx*dx+dy*dy+dz*dz;
      if (rsq > swb*swb) continue;
      int flag = 0;                              // full-list -> half dedup (ACKS2 convention)
      if (j < atom->nlocal) flag = 1;
      else if (tag[i] < tag[j]) flag = 1;
      else if (tag[i] == tag[j]) {
        if (dz > SMALL) flag = 1;
        else if (fabs(dz) < SMALL) { if (dy > SMALL) flag = 1;
          else if (fabs(dy) < SMALL && dx > SMALL) flag = 1; }
      }
      if (!flag) continue;
      double w = calc_w(i, j, sqrt(rsq));
      if (w == 0.0) continue;
      if (m_fill >= X.m)
        error->one(FLERR, "samqeq: X matrix overflow at atom {} (m_fill {} >= {})", i, m_fill, X.m);
      X.jlist[m_fill] = j; X.val[m_fill] = w;
      X_diag[i] -= w; X_diag[j] -= w;            // X_ii = -Σ w => X = -graph Laplacian (NSD)
      m_fill++;
    }
    X.numnbrs[i] = m_fill - X.firstnbr[i];
  }
  if (m_fill >= X.m) error->all(FLERR,"samqeq: X overflow m_fill={} X.m={}",m_fill,X.m);
}

/* ----------------------------------------------------------------------
   init_storage: ghost-safe override of FixACKS2Sam::init_storage.
   The base loops ii < NN (= nlocal+nghost) and dereferences ilist[ii]; that is
   only valid when the neighbor list spans ghost centers (reaxff mode). In file
   mode our list is QEq's REQ_NEWTON_OFF half list with ilist of length nn = inum
   = nlocal, so the base loop would read past ilist for any periodic system with ghosts
   (a gas with nghost=0 is the only safe case). Loop over nn only;
   ghost b_s/s entries are filled by forward_comm, and init_matvec rebuilds b_s/s
   every step regardless.
-------------------------------------------------------------------------*/
void FixQEqSam::init_storage()
{
  if (efield) get_chi_field();
  for (int ii = 0; ii < nn; ii++) {
    int i = ilist[ii];
    if (atom->mask[i] & groupbit) {
      b_s[i] = -chi[atom->type[i]];
      if (efield) b_s[i] -= chi_field[i];
      b_s[NN + i] = 0.0;
      s[i] = 0.0;
      s[NN + i] = 0.0;
    }
  }
  for (int i = 0; i < 2; i++) { b_s[2*NN + i] = 0.0; s[2*NN + i] = 0.0; }   // last rows (match base: all procs)
}

/* ----------------------------------------------------------------------
   init_matvec: the ACKS2 (H, X) set-up of the augmented system, which
     (1) calls OUR compute_X (intra-fragment weights), and
     (2) folds the reference-charge Coulomb field into b_s:
         b_s[i] = -(chi[type] + Σ_j J_ij q0[type_j]) [+ chi_field if efield].
   The q0 field is obtained convention-safely by applying the H block of the
   augmented matvec to the vector Δ=q0 (so it matches compute_H exactly):
   field_i = (M·[q0,0,..])_i - eta_i*q0_i. Scratch reuses d (overwritten in BiCGStab).
-------------------------------------------------------------------------*/
void FixQEqSam::init_matvec()
{
  h_ridge = lr_ridge;                // Tikhonov ridge on the H-block diagonal (metal-limit regularization);
                                     // applied consistently to the matvec, preconditioner, ref-field below.
  compute_H();                       // inherited shielded-Coulomb
  compute_X();                       // X-block response (calc_w)
  compute_saddle_onsite();            // staircase c3/c4 + quartic wall -> onsite_extra
  pack_flag = 4; comm->reverse_comm(this);     // collect X_diag from ghosts (ACKS2)

  if (efield) get_chi_field();

  int *type = atom->type, *mask = atom->mask;

  // --- reference-charge field via matvec on Δ=q0 (into p; result d) ---
  for (int i = 0; i < NN; i++) { p[i] = q0[type[i]]; p[NN+i] = 0.0; }   // q0 known on ghosts (per-type)
  p[2*NN] = 0.0; p[2*NN+1] = 0.0;   // ON EVERY RANK -- sparse_matvec_acks2 adds x[2NN] / x[2NN+1] into every
                                    // owned row and this probe is not broadcast first; guarding with
                                    // last_rows_flag would leave them uninitialised on every other rank (a
                                    // per-rank chi shift at np >= 2)
  sparse_matvec_acks2(&H, &X, p, d);
  pack_flag = 1; comm->reverse_comm(this); more_reverse_comm(d);        // gather ghost contributions to d
  // field_i = d[i] - eta_i*q0_i (subtract the on-site eta diagonal the matvec added)

  for (int ii = 0; ii < nn; ii++) {
    int i = ilist[ii];
    if (!(mask[i] & groupbit)) continue;
    Hdia_inv[i] = 1.0 / (eta[type[i]] + (onsite_extra ? onsite_extra[i] : 0.0) + lr_ridge);   // precond matches the ridged matvec
    X_diag[i] -= (xreg + lr_ridge);     // X→X−(xreg+ridge)·I : lift per-fragment null modes + metal-limit ridge
    Xdia_inv[i] = (X_diag[i] != 0.0) ? 1.0/X_diag[i] : 1.0;
    double reffield = d[i] - (eta[type[i]] + (onsite_extra ? onsite_extra[i] : 0.0) + lr_ridge) * q0[type[i]];   // subtract the ridged on-site diagonal
    b_s[i]    = -(chi_b(i) + reffield);
    if (ionfield_flag) b_s[i] -= fixq_field[i];   // ionfield: fixed-charge field -> RHS only (sources not
                                                   //   in the operator: add_reciprocal sources group charges, compute_H
                                                   //   is core-core) => counted exactly once. Same sign/guard as
                                                   //   Route B's qb[i] = -(chi_b + fixq_field + q0field), with
                                                   //   reffield playing q0field's role. Charge rows only (a potential
                                                   //   on cores), so the augmented X/last rows need nothing.
    if (efield) b_s[i] -= chi_field[i];            // (Ewald reciprocal is in the matvec operator, not b_s)
    b_s[NN+i] = 0.0;
    s[i]    = 4*(s_hist[i][0]+s_hist[i][2]) - (6*s_hist[i][1]+s_hist[i][3]);
    s[NN+i] = 4*(s_hist_X[i][0]+s_hist_X[i][2]) - (6*s_hist_X[i][1]+s_hist_X[i][3]);
  }
  if (last_rows_flag) for (int i=0;i<2;i++){ b_s[2*NN+i]=0.0;
    s[2*NN+i]=4*(s_hist_last[i][0]+s_hist_last[i][2])-(6*s_hist_last[i][1]+s_hist_last[i][3]); }

  pack_flag = 2; comm->forward_comm(this); more_forward_comm(s);

  if (precond_mode == 1) {              // (re)factor the ILU saddle preconditioner — but only on a reneighbor
    // step (the saddle structure changes most then); reuse the cached factor otherwise. A preconditioner only
    // sets the iteration count, not the solution, so a stale ILU between rebuilds is safe (BiCGStab still
    // converges to the same answer). Single-point/run-0 ⇒ one build.
    if (!ilu_valid || neighbor->lastcall == update->ntimestep) { ilu_build(); ilu_valid = 1; }
  }
}

/* q = q0 + Δ (SQE+Q0 references). Otherwise identical to ACKS2 calculate_Q.*/
void FixQEqSam::calculate_Q()
{
  double *qa = atom->q; int *type = atom->type, *mask = atom->mask;
  pack_flag = 2; comm->forward_comm(this);

  // robustness: detect a diverged solve (non-finite or runaway charge) and stop cleanly here,
  // BEFORE the blown-up charges reach the force/neighbor code (which otherwise segfaults as atoms
  // fly apart and the matrix overflows). A diverged solve means the system is under-constrained —
  // e.g. truncated real-space Coulomb under-stiffening a dense polar phase.
  constexpr double QMAX = 50.0;     // far beyond any physical equilibration charge (e)
  int bad = 0;
  for (int i = 0; i < atom->nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    double qi = s[i];                 // divergence metric = DEVIATION from q0 (formal-charge ions legal)
    if (!std::isfinite(qi) || fabs(qi) > QMAX) { bad = 1; break; }
  }
  int badall = 0; MPI_Allreduce(&bad, &badall, 1, MPI_INT, MPI_MAX, world);
  if (badall)
    error->all(FLERR, "samqeq: charge solve diverged (non-finite or |q-q0|>{} e) at step {} — system"
                      "under-constrained (e.g. truncated Coulomb in a dense polar phase); check"
                      "params / cutoff / kspace", QMAX, update->ntimestep);

  for (int i = 0; i < NN; i++) {
    if (!(mask[i] & groupbit)) continue;
    qa[i] = q0[type[i]] + s[i];                       // <-- + q0 (vs ACKS2: qa[i]=s[i])
    if (i < atom->nlocal) {
      for (int k = nprev-1; k > 0; k--) { s_hist[i][k]=s_hist[i][k-1]; s_hist_X[i][k]=s_hist_X[i][k-1]; }
      s_hist[i][0]=s[i]; s_hist_X[i][0]=s[NN+i];
    }
  }
  if (last_rows_flag) for (int i=0;i<2;i++){
    for (int k=nprev-1;k>0;k--) s_hist_last[i][k]=s_hist_last[i][k-1];
    s_hist_last[i][0]=s[2*NN+i]; }
}

/* ----------------------------------------------------------------------*/
// pack_flag==6: comm_v, the current N-dim plain-QEq vector
// (forward = distribute x to ghosts; reverse = sum ghost matvec contributions to owners). Else -> base.
int FixQEqSam::pack_forward_comm(int n, int *lst, double *buf, int pbc_flag, int *pbc)
{
  if (pack_flag == 6) { for (int i=0;i<n;i++) buf[i]=comm_v[lst[i]]; return n; }
  return FixACKS2Sam::pack_forward_comm(n, lst, buf, pbc_flag, pbc);
}
void FixQEqSam::unpack_forward_comm(int n, int first, double *buf)
{
  if (pack_flag == 6) { int m=0; for (int i=first;i<first+n;i++) comm_v[i]=buf[m++]; return; }
  FixACKS2Sam::unpack_forward_comm(n, first, buf);
}
int FixQEqSam::pack_reverse_comm(int n, int first, double *buf)
{
  if (pack_flag == 6) { int m=0; for (int i=first;i<first+n;i++) buf[m++]=comm_v[i]; return m; }
  return FixACKS2Sam::pack_reverse_comm(n, first, buf);
}
void FixQEqSam::unpack_reverse_comm(int n, int *lst, double *buf)
{
  if (pack_flag == 6) { int m=0; for (int i=0;i<n;i++) comm_v[lst[i]] += buf[m++]; return; }
  FixACKS2Sam::unpack_reverse_comm(n, lst, buf);
}

/* ----------------------------------------------------------------------
   Per-atom XL state migration. Extend the base GROW-callback arrays (s_hist/t_hist) so the
   persistent charge velocity/acceleration travel WITH the atoms on reneighbor/exchange. Without this,
   qdot/qddot stay at stale local indices after an atom swaps procs -> wrong per-atom association ->
   per-molecule charge-sum drifts off zero -> net charge -> long-range blow-up (the XL over-polarization).
-------------------------------------------------------------------------*/
// NOTE: chain to the IMMEDIATE base FixACKS2Sam (which manages s_hist + s_hist_X, NOT t_hist) — its
// post_constructor() calls grow_arrays() and then zeroes s_hist_X, so skipping it (e.g. via FixQEqBaseSam)
// leaves s_hist_X unallocated -> segfault.
void FixQEqSam::grow_arrays(int nmax)
{
  FixACKS2Sam::grow_arrays(nmax);               // s_hist, s_hist_X (ACKS2 augmented history)
  memory->grow(qdot,  nmax, "samqeq:qdot");
  memory->grow(qddot, nmax, "samqeq:qddot");
  // Zero the NEW slots. Non-XL decks never write qdot/qddot, yet pack_restart writes them, so restart files
  // would otherwise carry heap garbage (harmless: XL zeroes them when it starts).
  for (int i = qdot_nalloc; i < nmax; i++) { qdot[i] = 0.0; qddot[i] = 0.0; }
  if (nmax > qdot_nalloc) qdot_nalloc = nmax;
}
void FixQEqSam::copy_arrays(int i, int j, int delflag)
{
  FixACKS2Sam::copy_arrays(i, j, delflag);
  qdot[j] = qdot[i]; qddot[j] = qddot[i];
}
int FixQEqSam::pack_exchange(int i, double *buf)
{
  int n = FixACKS2Sam::pack_exchange(i, buf);
  buf[n++] = qdot[i]; buf[n++] = qddot[i];
  return n;
}
int FixQEqSam::unpack_exchange(int nlocal, double *buf)
{
  int n = FixACKS2Sam::unpack_exchange(nlocal, buf);
  qdot[nlocal] = buf[n++]; qddot[nlocal] = buf[n++];
  return n;
}

/* ----------------------------------------------------------------------
   write_restart / restart: the recip_self calibration state (the sticky probe-pair tags, the measured value and
   the grid signature it was measured on), so a restarted run keeps the calibration its parameters were tuned
   against. The grid-signature check is deferred to the first long-range pre_force (the new run's PPPM grid does
   not exist at fix-definition time): a matching signature keeps the restored value, a mismatch recalibrates.
   Layout: [format tag -5] + 11 fields. A restart file in any other layout carries fields this build does not
   read, so it is ignored with a warning and the fix recalibrates. write_restart runs on every rank; rank 0
   writes (the members are rank-uniform by construction).
-------------------------------------------------------------------------*/
void FixQEqSam::write_restart(FILE *fp)
{
  double list[12];
  int m = 0;
  list[m++] = ubuf((bigint) -5).d;   // format tag
  list[m++] = ubuf((bigint) lr_tagA).d;
  list[m++] = ubuf((bigint) lr_tagB).d;
  list[m++] = lr_self_meas;
  list[m++] = ubuf((bigint) lr_calibrated).d;
  list[m++] = ubuf((bigint) lr_nx).d;
  list[m++] = ubuf((bigint) lr_ny).d;
  list[m++] = ubuf((bigint) lr_nz).d;
  list[m++] = lr_gewald;
  list[m++] = lr_prd[0];
  list[m++] = lr_prd[1];
  list[m++] = lr_prd[2];
  if (comm->me == 0) {
    int size = m * sizeof(double);
    fwrite(&size, sizeof(int), 1, fp);
    fwrite(list, sizeof(double), m, fp);
  }
}

void FixQEqSam::restart(char *buf)
{
  auto *list = (double *) buf;
  int m = 0;
  if ((bigint) ubuf(list[m++]).i != -5) {
    if (comm->me == 0)
      error->warning(FLERR, "fix {}: restart data in an unrecognised layout; ignored (recip_self is recalibrated)", id);
    return;
  }
  lr_tagA       = (tagint) ubuf(list[m++]).i;
  lr_tagB       = (tagint) ubuf(list[m++]).i;
  lr_self_meas  = list[m++];
  lr_calibrated = (int) ubuf(list[m++]).i;
  lr_nx         = (int) ubuf(list[m++]).i;
  lr_ny         = (int) ubuf(list[m++]).i;
  lr_nz         = (int) ubuf(list[m++]).i;
  lr_gewald     = list[m++];
  lr_prd[0]     = list[m++];
  lr_prd[1]     = list[m++];
  lr_prd[2]     = list[m++];
}

/* ----------------------------------------------------------------------
   Per-atom RESTART persistence for the XL charge-dynamics state qdot/qddot -- the same two values
   migrated across procs (grow_arrays/copy_arrays/pack_exchange/unpack_exchange above), also
   round-tripped through a restart file, so resuming `fix_modify xl` after a restart continues the charge
   velocities instead of applying a discontinuous kick. q_hist (ASPC) is deliberately NOT persisted here: it is a short predictor history that
   re-warms benignly over a few steps (the code falls back to the exact within-step Picard/CG solve while it
   refills -- quartic_scf / qeq_solve), unlike qdot/qddot, which are true integrated dynamical variables whose
   loss is a hard discontinuity, not a benign re-warm.
   Mirrors the standard LAMMPS per-atom-restart pattern (e.g. fix_move.cpp pack_restart/unpack_restart):
   buf[0] is reserved for THIS block's own size (so a later fix's restart blob on the same atom can skip past
   it); maxsize_restart()==size_restart()==3 (1 header + qdot + qddot, constant per atom).
   CONTINUITY NOTE: unpack_restart is only invoked (by Modify::add_fix, once per local atom) when the restart
   file actually carries a matching fix-ID+style block, i.e. only on a genuine restart round-trip -- so it is
   the right (and only) place to also mark xl_started=1: `initial_integrate` would otherwise see the FRESH
   fix's default xl_started==0 on the very first post-restart step and re-zero the just-restored qdot/qddot
   before they are ever used (xl_started itself is NOT restart_global state -- see the header note in
   fix_qeq_sam.h). Harmless no-op for decks that never enable `fix_modify xl` (lr_xl gates initial_integrate
   first) and for restart files without this block (unpack_restart is then simply never called).
-------------------------------------------------------------------------*/
int FixQEqSam::pack_restart(int i, double *buf)
{
  int n = 1;
  buf[n++] = qdot[i];
  buf[n++] = qddot[i];
  buf[0] = n;
  return n;
}

void FixQEqSam::unpack_restart(int nlocal, int nth)
{
  double **extra = atom->extra;
  int m = 0;
  for (int i = 0; i < nth; i++) m += static_cast<int>(extra[nlocal][m]);
  m++;
  qdot[nlocal]  = extra[nlocal][m++];
  qddot[nlocal] = extra[nlocal][m++];
  xl_started = 1;   // the restored qdot/qddot ARE a valid continuation state; do not let the next
                    // initial_integrate() re-zero them (see the CONTINUITY NOTE above).
}

int FixQEqSam::maxsize_restart()
{
  return 3;
}

int FixQEqSam::size_restart(int /*nlocal*/)
{
  return 3;
}

/* ----------------------------------------------------------------------
   On-site anharmonic diagonal for the ACKS2 saddle. Mirrors apply_quartic_eta()
   (fix_qeq_sam_quartic.cpp) term-for-term -- same anh_secant_add(), same lagged atom->q,
   same quartic_etafloor clamp -- but as an ADDEND to eta[type] (through onsite_extra) rather
   than a replacement of eta_diag.
   The floor is applied to the TOTAL (eta + addend), as the standard path applies it to eta_diag.
-------------------------------------------------------------------------*/
void FixQEqSam::compute_saddle_onsite()
{
  int *type = atom->type; int *mask = atom->mask; double *qa = atom->q;
  const int nlocal = atom->nlocal;
  if (!lr_quartic) { onsite_extra = nullptr; return; }   // nothing anharmonic
  compute_quartic_gate();            // per-atom field gate for the wall (no-op gate=1 when fld0<=0); the
                                     // standard path runs this in fix_qeq_sam_lr.cpp, which the saddle
                                     // path never reaches
  for (int i = 0; i < nlocal; i++) {
    deta_anh[i] = 0.0;
    if (!(mask[i] & groupbit)) continue;
    const int t = type[i];
    double e = 0.0;                                    // accumulate ONLY the anharmonic secant
    const bool anh = anh_secant_add(e, qa[i], t, mask[i] & quartic_groupbit, quartic_gate[i]);
    if (!anh) continue;
    const double base = eta[t];
    if (base + e < quartic_etafloor) e = quartic_etafloor - base;   // floor on the TOTAL diagonal
    deta_anh[i] = e;
  }
  onsite_extra = deta_anh;
}

/* ----------------------------------------------------------------------
   post_force(): apply every FIX-OWNED position force here,
   AFTER pair->compute, which is LAMMPS's contract for a force-adding fix (cf. fix_efield). Applied in pre_force,
   `Pair::virial_fdotr_compute` -- which sums x (x) f over the WHOLE accumulated atom->f under `newton on`, the
   host default -- would pick them up a second time as the wrong-image local sum Sum_local x.F, corrupting the
   pressure while pe, forces and charges stay exact.
   v_init(vflag) runs here, in the hook that tallies. Cheap early-out first: nothing is pending on any deck without
   these features.
-------------------------------------------------------------------------*/
void FixQEqSam::post_force(int vflag)
{
  // v_init FIRST, before the early-out: compute pressure sums virial on every step thermo_virial is raised,
  // pending or not, and Fix::Fix never initialises virial[6].
  v_init(vflag);                       // zeroes virial and sets vflag_global iff thermo_virial was raised
}

void FixQEqSam::setup(int vflag) { post_force(vflag); }          // the run-boundary force evaluation
void FixQEqSam::min_setup(int vflag) { post_force(vflag); }
void FixQEqSam::min_post_force(int vflag) { post_force(vflag); }
void FixQEqSam::post_force_respa(int vflag, int ilevel, int /*iloop*/)
{
  if (ilevel == nlevels_respa-1) post_force(vflag);              // mirrors FixQEqBaseSam::pre_force_respa
}

/* ----------------------------------------------------------------------
   Per-type-pair PQEq Gaussian radius overrides (`fix_modify shieldpair`). The fix owns the
   table; init() pushes it into pair coul/shield/intra so forces and the solve use ONE kernel, and
   check_shield_consistency() (channel 5) verifies the copy. Kokkos device kernels (pair /kk and the fix's
   device solve) carry only the per-type Rc, so an override there is refused rather than silently ignored.
-------------------------------------------------------------------------*/
void FixQEqSam::shield_rpair_ensure()
{
  if (shield_rpair) return;
  const int np1 = atom->ntypes + 1;
  memory->create(shield_rpair, np1, np1, "samqeq:shield_rpair");
  for (int i = 0; i < np1; i++) for (int j = 0; j < np1; j++) shield_rpair[i][j] = 0.0;
}

void FixQEqSam::push_shield_pairs()
{
  if (!shield_rpair) return;
  int nset = 0;
  for (int i = 1; i <= atom->ntypes; i++) for (int j = i; j <= atom->ntypes; j++) if (shield_rpair[i][j] > 0.0) nset++;
  if (nset == 0) return;
  if (shield_gauss != SHIELD_GAUSSIAN)
    error->all(FLERR, "samqeq: fix_modify shieldpair is defined for the PQEq Gaussian kernel only (`fix_modify {} shield gaussian`)", id);
  if (strstr(style, "/kk") || force->pair_match("coul/shield/intra/kk", 1))
    error->all(FLERR, "samqeq: fix_modify shieldpair is not supported on the Kokkos kernels (fix {} / pair coul/shield/intra/kk carry per-type Rc only)", style);
  auto *csp = dynamic_cast<PairCoulShieldIntra *>(force->pair_match("coul/shield/intra", 0));
  std::string msg;
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      if (shield_rpair[i][j] > 0.0) {
        if (csp) csp->shield_rpair_set(i, j, shield_rpair[i][j]);
        msg += fmt::format(" ({},{}) R={:.4f} [rule {:.4f}]", i, j, shield_rpair[i][j],
                           sqrt(0.5*(gamma[i]*gamma[i] + gamma[j]*gamma[j])));
      }
  if (comm->me == 0)
    utils::logmesg(lmp, "samqeq: shieldpair overrides{} -> {}\n", msg,
                   csp ? "pushed to pair coul/shield/intra (forces + solve share the kernel)"
                       : "fix-side only (no pair coul/shield/intra in this deck)");
}
