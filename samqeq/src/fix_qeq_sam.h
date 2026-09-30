/* -*- c++ -*- ----------------------------------------------------------
   samQEq — a FAMILY of QEq functionals for condensed-phase charge response.

   ARCHITECTURE: samQEq is built on ACKS2,
   so it INHERITS FixACKS2Sam and overrides only what differs. The entire
   augmented (Δ,u) saddle-point machinery — allocate_storage (2N+2), BiCGStab,
   sparse_matvec_acks2, all MPI comm, the shielded-Coulomb compute_H, vector
   ops — is reused unchanged. This makes samQEq a drop-in ACKS2/QEq-family fix
   (switchable with `fix qeq/*` / `fix acks2/*`; usable by any code that
   already drives ACKS2). NOTE: lives in / depends on the REAXFF package
   (where the ACKS2 base lives).

   OVERRIDES (all virtual in FixQEqBaseSam):
     pertype_parameters — read samQEq file (adds q0, eHOMO, eLUMO; header γ_align, κ_bond)
     init — skip ACKS2 init_bondcut (bcut is set up lazily for bondsoft)
     init_matvec — build X + reference-charge field into b_s
     calculate_Q — q = q0 + Δ (SQE+Q0 references)
   NEW (response kernel): compute_X, calc_w
   Inherited verbatim: everything else.

   UNITS: samQEq supports units metal (energy eV) and
   units real (energy kcal/mol; 1 eV = 23.060549 kcal/mol; charge = e in BOTH).
     R1 — ALL user-facing values are NATIVE DECK UNITS: param-file columns (chi/eta/ehomo/elumo/c3/c4)
          and header values (gamma_align, kappa_bond, lr_ridge, pd_lam_min, ...), plus every fix_modify
          numeric arg (ridge onset/gain/eig, quartic c4/c/fld0, xl q_mass/tdamp, spikeguard k, ...).
          A real-units deck supplies kcal/mol
          numbers (e.g. nod_11.4_real.param = metal params x23.0605).
     R2 — internal hardcoded constants are written eV-ANCHORED and scaled by the inherited ev_scale
          member at point of use (or once in the ctor for stored defaults).
     R3 — solver tolerances are corrected per-criterion so convergence means the same PHYSICAL
          tightness in both unit systems (e.g. qeq_cg's e/sqrt(E) criterion uses tolerance/sqrt(ev_scale)).
   ev_scale == 1.0 in metal, so every unit scaling is an exact x1.0 / /1.0 there.
-------------------------------------------------------------------------*/

#ifdef FIX_CLASS
// clang-format off
FixStyle(qeq/sam,FixQEqSam);
// clang-format on
#else

#ifndef LMP_FIX_QEQ_SAM_H
#define LMP_FIX_QEQ_SAM_H

#include "fix_acks2_sam.h"     // base: augmented ACKS2 solver (self-contained in SAMQEQ; no REAXFF pkg)
#include "slater_jtable.h"     // Rick Slater-overlap J(r) kernel + per-type-pair table (shield slater)
#include "samqeq_ttdamp.h"     // Tang-Toennies f_n(br) for the damped interionic kernel (fix_modify iondamp)

#include <cmath>               // fabs in the inline on-site functional below (strict GCC: no transitive include)
#include <map>
#include <vector>

namespace LAMMPS_NS {

class FixQEqSam : public FixACKS2Sam {
 public:
  FixQEqSam(class LAMMPS *, int, char **);
  ~FixQEqSam() override;
  void init() override;
  double calc_Hval(double r, int ti, int tj) override;   // gas-path H.val hook: routes through the
                                                         // Slater J(r) table when shield_gauss==SHIELD_SLATER,
                                                         // else delegates to the base (cbrt).
  double compute_scalar() override;   // on-site self-energy E_self = Σ_i(χ_i q_i + ½ η_i q_i²) [deck E units];
                                      // completes pe (off-diagonal Coulomb is already in pairs+kspace).
                                      // Enable in thermo with: fix_modify <ID> energy yes

  // ---- THE on-site anharmonic model, single source of truth ----
  // E_anh(q) = (1/6)c3_t q³ + (1/24)c4_t q⁴ [per-type IP staircase, ungated]
  //          + (1/24)c4·gate·d⁴ + (1/6)c·q³, d = max(|q|−qref, 0) [global field-gated quartic wall]
  // Three derivative orders, consumed by compute_scalar (energy), xl_chargeforce
  // (−dE/dq), apply_quartic_eta (secant (dE_anh/dq)/q). They ACCUMULATE into the caller's variable in a
  // fixed term order. Any change to the model happens HERE, in all derivative orders at once.
  inline double anh_q(double qi, int /*t*/, bool /*in_qgrp*/) const { return qi; }   // the anharmonic wall acts on q

  inline void anh_energy_add(double &e, double qi, int t, bool in_qgrp, double gate) const {
    if (lr_quartic && (c3_type[t] != 0.0 || c4_type[t] != 0.0))
      e += (1.0/24.0)*c4_type[t]*qi*qi*qi*qi + (1.0/6.0)*c3_type[t]*qi*qi*qi;
    if (lr_quartic && in_qgrp) {
      const double qw = anh_q(qi, t, in_qgrp);         // wall argument
      double c4e;                                      // offset one-sided wall (qref=0 ⇒ centered)
      if (quartic_qref > 0.0) { const double qd = fabs(qw) - quartic_qref; c4e = (qd > 0.0) ? qd*qd*qd*qd : 0.0; }
      else c4e = qw*qw*qw*qw;
      e += (1.0/24.0)*quartic_c4*gate*c4e + (1.0/6.0)*quartic_c*qw*qw*qw;
    }
  }
  inline void anh_force_add(double &f, double qi, int t, bool in_qgrp, double gate) const {  // f += −dE_anh/dq
    if (c3_type[t] != 0.0 || c4_type[t] != 0.0)        // staircase types imply lr_quartic (auto-set at load)
      f -= 0.5*c3_type[t]*qi*qi + (1.0/6.0)*c4_type[t]*qi*qi*qi;
    if (lr_quartic && in_qgrp) {
      const double qw = anh_q(qi, t, in_qgrp);         // wall argument
      double c4f;                                      // E'_c4 = (1/6)c4·gate·d³·sign(q)
      if (quartic_qref > 0.0) { const double aq = fabs(qw), qd = aq - quartic_qref;
        c4f = (qd > 0.0) ? qd*qd*qd*(qw > 0.0 ? 1.0 : -1.0) : 0.0; }
      else c4f = qw*qw*qw;
      f -= (1.0/6.0)*quartic_c4*gate*c4f + 0.5*quartic_c*qw*qw;
    }
  }
  inline bool anh_secant_add(double &e, double qi, int t, bool in_qgrp, double gate) const {
    bool anh = false;                                  // e += (dE_anh/dq)/q; callers only run under lr_quartic
    if (c3_type[t] != 0.0 || c4_type[t] != 0.0) {
      e += 0.5*c3_type[t]*qi + (1.0/6.0)*c4_type[t]*qi*qi;
      anh = true;
    }
    if (in_qgrp) {
      // The secant the solve uses (apply_quartic_eta, fix_qeq_sam_quartic.cpp, builds eta_diag from it).
      // The divisor is the wall argument qw itself, never a different charge: dividing by a quantity of
      // opposite sign would soften the diagonal instead of stiffening it and put a pole at zero.
      // (dE/dq)/qw = (1/6)c4*qw^3/qw = (1/6)c4*qw^2 >= 0 -- ALWAYS stiffens, no pole.
      const double qw = anh_q(qi, t, in_qgrp);
      double c4diag;                                   // (1/6)c4·gate·d³/|q|; qref=0 ⇒ q²
      if (quartic_qref > 0.0) {
        const double aq = fabs(qw), qd = aq - quartic_qref;
        c4diag = (qd > 0.0) ? qd*qd*qd/aq : 0.0;
      } else c4diag = qw*qw;
      e += (1.0/6.0)*quartic_c4*gate*c4diag + 0.5*quartic_c*qw;
      anh = true;
    }
    return anh;
  }


  // --- restart persistence ---
  // GLOBAL: the recip_self calibration (probe pair, measured value, grid signature), so a restarted run
  // keeps its calibration continuity check. PER-ATOM: the XL qdot/qddot
  // charge-velocity state (fix_qeq_sam.cpp, alongside the GROW/exchange migration). q_hist (ASPC) is
  // deliberately NOT persisted -- it re-warms benignly over a few steps.
  void   write_restart(FILE *) override;
  void   restart(char *) override;
  int    pack_restart(int, double *) override;
  void   unpack_restart(int, int) override;
  int    maxsize_restart() override;
  int    size_restart(int) override;

  // Does the HOST need the CSR H matrix this step? Its only consumers are the host matvec
  // (csr_matvec_add, i.e. the host solves) and the ILU preconditioner, so when the whole solve runs on
  // device the host build is wasted serial erf/erfc work.
  // Overridden in the kokkos fix; the base always says yes.
  virtual bool host_H_needed() { return true; }
  /* ---- HOST-side phase timing (env SAMQEQ_HOST_TIME=1), printed by post_run ----
     Breaks down the host work that remains when the solve runs on device. Free when the env var is
     unset.*/
  int    th_on = -1;
  double th_pre = 0.0, th_molinv = 0.0, th_q0field = 0.0, th_proj = 0.0, th_cg = 0.0, th_solve = 0.0;
  long   th_steps = 0, th_projcalls = 0;
  void   post_run() override;

  long host_mv_calls = 0;   // probe: how many times the HOST csr matvec ran (it is the
                            // only consumer of the CSR H, so a nonzero count on a device-solve step
                            // means skipping the host build is unsafe)

 protected:
  // samQEq-specific per-type params (inherited: chi, eta, gamma[shielding], H, X, s, b_s, ...)
  double *q0;                 // reference (formal/oxidation-state) charge per type
  // FLOATING DIAGONAL: per-type on-site anharmonicity from the IP STAIRCASE. The on-site energy
  // is E(q)=chi q + 1/2 eta q^2 + 1/6 c3 q^3 + 1/24 c4 q^4, so the solve diagonal is the running curvature
  // eta_eff(q)=E''(q)=eta + c3 q + 1/2 c4 q^2 (the charge-dependent hardness). c3_type/c4_type are param
  // columns 8/9 (default 0 => the plain parabola). Fed into apply_quartic_eta IN ADDITION to
  // the global fix_modify-quartic scalars (orthogonal: the staircase is the ion's true hardness, ungated; the
  // global quartic stays the water-O over-pol bounding knob). lr_quartic auto-set if any type carries c3/c4.
  double *c3_type = nullptr;  // per-type cubic on-site coeff (eV/e^3); E'''(0). staircase: spans oxidation states
  double *c4_type = nullptr;  // per-type quartic on-site coeff (eV/e^4); E''''. trivalent 3-charge curvature fit
  double *ehomo, *elumo;      // per-type frontier levels (eV): read from the param file and ignored
  double gamma_align;         // param-file header field 1: read, must be > 0, otherwise unused
  double kappa_bond;          // fixed intra-fragment bond softness
  double r_ov;                // overlap length scale for Ω(r)=exp(-(r/r_ov)^2) (X-matrix; distinct from
                              // the long Coulomb cutoff swb used by compute_H)
  double xreg;                // tiny u-block regularization X→X−xreg·I: lifts the per-fragment null
                              // modes of the X-block ⇒ unique, well-conditioned, reproducible solve

  // Per-atom solve diagonal: eta[type] plus the quartic secant (apply_quartic_eta), read through solve_diag_of.
  double *eta_diag;
  // Parameter-file header fields that are read and not used by any solve path.
  double r_loc, r_bond, graph_wedge, pd_lam_min;
  int    graph_hop_max;
  // Per-atom on-site ANHARMONIC diagonal for the ACKS2 saddle. The IP-staircase c3/c4
  // and the `fix_modify quartic` wall enter the STANDARD solve through apply_quartic_eta() -> eta_diag,
  // but init_matvec's H diagonal is the bare eta[type]. deta_anh[i] = (dE_anh/dq)/q, the SAME secant
  // anh_secant_add() builds for the standard path at the same lagged atom->q, routed through the existing
  // onsite_extra hook. Zero whenever no type carries
  // c3/c4 and no quartic group is set. eta_diag is NOT touched.
  double *deta_anh = nullptr;
  void   compute_saddle_onsite();  // fill deta_anh, re-point onsite_extra; saddle path only

  // --- QUARTIC near-criticality bound on the FQ on-site energy ---
  // E_i(q) = chi q + 1/2 eta q^2 + 1/24 c4 q^4 [+ 1/6 c q^3] <=> eta_eff(q) = eta + 1/6 c4 q^2 [+ 1/2 c q].
  // The quartic (c4>0) Landau-bounds the soft collective FQ mode -> a FINITE polarization through k<=0, replacing
  // the artificial adaptive ridge with a physical bound. The term is q-ONLY (no field/position dependence) =>
  // variational at the SCF charge minimum => NO new force term (positions feel only the existing pairwise Coulomb).
  // The diagonal becomes charge-dependent => damped-Picard SCF (quartic_niter, quartic_mix) wraps qeq_solve.
  // Default off (lr_quartic=0) => zero diagonal/energy change. Applied to quartic_groupbit atoms
  // (e.g. water O). `fix_modify <id> quartic <group> <c4> [<c>] [<niter>] [<mix>]`.
  int    lr_quartic = 0;           // 0 = off; 1 = quartic/cubic on-site term active
  int    quartic_groupbit = 0;     // atoms the term applies to (e.g. water O cores)
  double quartic_c4 = 0.0;         // quartic coefficient c4 (the 1/24 c4 q^4 term)
  double quartic_c  = 0.0;         // optional cubic coefficient c (1/6 c q^3); 0 = pure quartic (the near-crit cure)
  int    quartic_niter = 20;       // max damped-Picard SCF iterations (the nonlinear diagonal)
  double quartic_mix = 0.5;        // SCF charge mixing alpha: q <- (1-a) q_old + a q_new
  double quartic_etafloor = 2.0;   // clamp eta_eff >= floor (guard a negative solve diagonal). eV/e^2-
                                   // anchored internal default, NO fix_modify override exists -> re-set to
                                   // 2.0*ev_scale in the FixQEqSam ctor body (this in-class value is a
                                   // placeholder; ev_scale isn't safely usable in an in-class initializer).
  double quartic_fld0 = 0.0;       // FIELD GATE scale: c4_eff(i)=c4·φ0²/(φ0²+fld0²), φ0=|J·q0|_i (reference/ion field).
                                   // The quartic hardens ONLY where the field is strong (shell), not the bulk
                                   // equilibrium charge. 0 (default) ⇒ gate≡1 everywhere.
  double quartic_qref = 0.0;       // CHARGE-OFFSET (one-sided wall): the c4 term acts on d=max(|q|-qref,0), i.e.
                                   // E_c4 = (1/24)c4·gate·d^4 — FLAT (no curvature) for |q|<qref, a stiffening wall
                                   // beyond. Keeps the physical range (|q|<qref) at the bare soft η (full
                                   // polarizability / gas→liquid response) and caps only the unphysical near-critical
                                   // runaway tail |q|>qref. 0 (default) ⇒ d=|q| ⇒ centered quartic.
  double *quartic_gate = nullptr;  // per-atom field gate G(i)∈[0,1] (geometry-only, computed once per step)
  void   compute_quartic_gate();   // fill quartic_gate from the unprojected reference-charge (q0/ion) Coulomb field
  void   apply_quartic_eta();      // fill eta_diag[i] = eta + 1/6 c4·gate·q^2 [+ 1/2 c q] for the quartic group (lagged q)
  void   quartic_scf();            // damped-Picard outer loop around qeq_solve for the charge-dependent diagonal

  // --- shield_gauss: the shielding-kernel MODE for the lr_ewald=2 solve (vs the default cbrt J_shield) ---
  // TRI-STATE: every call site is an explicit `shield_gauss == SHIELD_*` check, not a bare truthiness test,
  // EXCEPT the two device (kokkos) guards that intentionally reject ANY nonzero mode -- see those sites:
  //   SHIELD_CBRT (0, default): J_shield = 1/∛(r³+1/γ³), γ = sqrt(gamma[ti] gamma[tj]) (param 4th col).
  //   SHIELD_GAUSSIAN (1): PQEq Gaussian-Gaussian overlap erf(α_ij r)/r, α_i = λ/(2 Rc_i²) with Rc = the
  //     per-type Gaussian core radius (param 4th col, reinterpreted), α_ij = sqrt(α_i α_j/(α_i+α_j)). Conditions
  //     the FQ operator the way PQEq does (stable at η_O=13.36 WITHOUT a depolarizing ridge), unlike cbrt.
  //   SHIELD_SLATER (2): Rick's EXACT Slater-orbital Coulomb-overlap integral J(r) (Rick, Stuart & Berne, JCP
  //     101, 6141 (1994)) -- the true kernel TIP4P-FQ's fluctuating charges (1s H, 2s O/M) actually interact
  //     through, valid at ALL r (not just the two rigid intramolecular distances a cbrt/gauss refit is pinned
  //     to). param 4th col REINTERPRETED as the Slater exponent zeta (1/Å, RAW -- zeta_bohr/0.529177, NO
  //     ev_scale: zeta is a length⁻¹, not an energy). Per-type 2s/1s selector: is2s (default 1s; `fix_modify
  //     <id> shield slater [2s <type> <type> ...]`, list runs to the end of that fix_modify invocation). Tables
  //     (slater_tabs, one per unique ti<=tj type pair) built once per init() from slater_jtable.h -- see
  //     build_slater_tables(). Reference: examples/tip4pfq_liquid/jrab.py (jrab_raw) + check_slater_table.py
  //     (parity oracle).
  // All three forms → 1/r at large r and are finite at r→0; the caller subtracts erf(g_ewald r)/r (the
  // reciprocal adds it back), so the NET off-diagonal = this kernel.
  static constexpr int SHIELD_CBRT = 0, SHIELD_GAUSSIAN = 1, SHIELD_SLATER = 2;
  int    shield_gauss = 0;          // 0 = cbrt J_shield (default); 1 = PQEq Gaussian; 2 = Slater J(r) (see above)
  double shield_lambda = 0.462770;  // PQEq λ (Gaussian-overlap → effective-shielding conversion)
  // Per-TYPE-PAIR Gaussian radius override
  // for the PQEq kernel. Unset (0) reproduces the combination rule a_ij = sqrt(lambda/(2(Rc_i^2+Rc_j^2))), i.e.
  // R_ij,default = sqrt((Rc_i^2+Rc_j^2)/2); set, a_ij = sqrt(lambda)/(2 R_ij). `fix_modify <id> shieldpair ti tj R`.
  // The fix owns the table and PUSHES it into pair coul/shield/intra at init() (single source of truth; the
  // consistency check verifies). Not supported on the Kokkos device kernels (init() errors if set there).
  double **shield_rpair = nullptr;  // [ntypes+1][ntypes+1], symmetric, 0 = unset
  void   shield_rpair_ensure();     // lazy allocation
  void   push_shield_pairs();       // init(): copy the table into the pair style + log
  // --- gself: finite-width charge SELF-ENERGY on the on-site diagonal (opt-in) ---
  // The Gaussian (PQEq) charge of width w carries a self-energy E_self = K_e·sqrt(2α/π), α = 1/(2w²),
  // which is the i==j limit of the erf(α_ij r)/r pair kernel. H stores only i≠j, and the lr path subtracts
  // the grid self-term so the diagonal is EXACTLY η (fix_qeq_sam_lr.cpp:361) -- correct for a molecule,
  // whose fitted η already IS the full on-site curvature of a finite-width atom; WRONG for a metal lattice,
  // whose η is a bulk Thomas–Fermi quantity (η ≈ e²l_TF²d/ε0, Scalfi JCP 153,174704) with no width in it.
  // Without it a point-charge metal slab under Ewald carries a staggered Nyquist mode (layer charges
  // alternate at 1.8–2.5× the Gauss limit at physical η_Au, diverging as η→0); with it the slab screens
  // like a conductor and the η→0 limit is well-behaved -- Scalfi's electrode runs with exactly this term.
  // `fix_modify ID gself on [width <w_Angstrom>]` adds E_self(w) to η for every type, folded idempotently
  // from eta0 in init() so the solve diagonal (solve_diag_of) and the XL force see it, and compute_scalar
  // reports it through eself_diag_of (fix_qeq_sam_lr.cpp). Default width 0.5 Å: the type's own Gaussian
  // (the PQEq Rc_Au = 1.62 Å gives a 2.4 Å cloud on a 2.4 Å lattice) is too diffuse to regularise. `width` is the Gaussian's standard deviation w (kernel erf(r/2w)/r,
  // E_self = K_e/(w sqrt(pi))); Scalfi's width eta^-1 is w*sqrt(2), so w = 0.5 Å is eta^-1 = 0.71 Å
  // (Scalfi et al. used eta^-1 = 0.56 Å, i.e. w = 0.40 Å).
  // On a 240-atom Au(111) slab at E = 0.5 V/Å (face charge / Gauss limit):
  //   no gself 2.2× (staggered) · gself, Rc width 1.5× · gself, 0.5 Å 1.09× (monotone, interior 0.1)
  // Off by default.
  int    gself_flag  = 0;
  // OPTIONAL type restriction. The self-energy is the electrostatic cost of a charge having a
  // WIDTH, and it is the right regulariser for a metal LATTICE, where eta is a bulk Thomas-Fermi
  // quantity with no width in it. On a MOLECULAR site it is a DOUBLE COUNT: eta there was fit to the
  // IP/EA of a finite-width atom and already contains that width (on a Cl-/H2O dimer with gself on
  // every type, the water's charges collapse q_O -0.89 -> -0.29, q_H +0.45 -> +0.15). A mixed
  // metal+adsorbate cell therefore needs the term on the electrode types only. Empty list (the
  // default) = every type.
  int   *gself_type  = nullptr;     // per-type 0/1; nullptr => all types
  int    gself_ntype = 0;           // allocated length (ntypes+1) when the list is set
  double gself_width = 0.5;         // Å; `fix_modify gself ... width <w>` (must be > 0)
  int   *is2s = nullptr;            // SLATER per-type 2s(O/M)-vs-1s(H) Slater form-factor flag (nt+1, default 0=1s)
  std::vector<SlaterJTable> slater_tabs;   // SLATER per-(ti<=tj)-type-pair table (see build_slater_tables())
  void   build_slater_tables();     // (re)build slater_tabs from is2s/gamma(=zeta)/swb; called from init() -- see
                                    // there for why an unconditional rebuild-every-init is the chosen "rebuild
                                    // if swb changes" policy (init() runs once per run-setup; swb is otherwise
                                    // const post-ctor, so this is cheap and trivially always-correct).
  double shielded_coulomb(int ti, int tj, double r);   // net shielded Coulomb kernel (cbrt, Gaussian, or Slater)

  // --- SHIELDING-KERNEL FORCE<->SOLVE CONSISTENCY ---
  // The pair style (forces/energy) and this fix (the solve) each select the shielding kernel INDEPENDENTLY.
  // Without a check, a deck could integrate forces from erf(αr)/r while equilibrating charges against
  // 1/∛(r³+1/γ³), silently, with plausible-looking output (a 5 Å pair cutoff against an 8 Å swb heats an
  // NVE melt by thousands of kelvin within picoseconds). check_shield_consistency() asserts these channels
  // at init():
  //   1 RANGE min(pair cutoff, swb) is where ONE side stops applying the correction while the other
  //             continues. Checked by MAGNITUDE, not equality: the residual qqrd2e·|J(r_trunc) − 1/r_trunc|
  //             (per unit charge pair, in eV) is what the two sides actually disagree by. Equality would be
  //             the wrong predicate — it flags every narrow-kernel deck (whose residual is ~1e-4 eV,
  //             physically nil) to catch the wide-Gaussian ones (~0.5 eV, orders of magnitude larger).
  //             cut == swb passes unconditionally: both sides truncate identically, so they cannot disagree.
  //   2 MODE cbrt/pqeq/slater must match exactly.
  //   3 LAMBDA PQEq λ must match (set on the pair with its `lambda` keyword).
  //   4 2S Slater per-type 2s/1s form-factor flags must match.
  //   5 SCOPE WHICH pairs each side corrects: the pair's `all` keyword (intra_only) against the solve's
  //             lr_ewald, which comes from the param-file header, not the deck. Equality, not magnitude:
  //             with lr_ewald=2 and no `all` every intermolecular pair is shielded in the charge matrix
  //             and bare in the forces at all separations, so there is no small-residual regime. On
  //             in.tip4pfq_xl_slater the mismatch leaves the charges unchanged (the pair scope never enters
  //             the solve) but moves the liquid dipole 2.4631(28) -> 2.3396(14) D. Guarded to lr_ewald>=1.
  // No-ops when no coul/shield/intra pair exists (file-mode / gas decks) and on the plain-DSF path
  // (which never calls shielded_coulomb()). Adds only an error path.
  void   check_shield_consistency();
  int    shieldchk_on = 1;          // `fix_modify <id> shieldcheck off` disables (escape hatch, logs a warning)

  // check_drude_consistency() refuses a deck with `fix drude`: shells would be in the forces and absent from the solve.
  void   check_drude_consistency();
  // ionfield: the SAME guard class for plain FIXED-CHARGE atoms (ions) outside the solve
  // group. check_ionfield_consistency() hard-errors when fixed charged atoms sit outside the solve group
  // and ionfield_flag is not set, UNLESS the
  // deck acknowledged the choice with `fix_modify <id> ionfield off`. See the .cpp header note.
  void   check_ionfield_consistency();
  int    ionfield_ack = 0;          // set by `fix_modify <id> ionfield off` — "ion-blind solve is intended"
  int    ionfield_explicit = 0;     // the deck stated a choice either way; the fix_modify handler logged it,
                                    // so check_ionfield_consistency() stays quiet
  double shieldchk_tol = 0.02;      // channel-1 residual tolerance, eV per unit charge pair (see above).
                                    // Separation at this value: cbrt γ=1.0 gives 0.0076 eV (passes),
                                    // cbrt γ=0.25 gives 0.0693, pqeq Rc=1.856 gives 0.562. Table in the .cpp.

  // --- iondamp: Tang-Toennies-DAMPED interionic off-diagonal ---
  // For DESIGNATED type pairs (fix_modify <id> iondamp <typeI> <typeJ> <b> [<n>]) the net kernel becomes
  // J_damp(r) = f_n(b_ij r)·J_shield(r), f_n = the TT function (samqeq_ttdamp.h). f→1 at large r (the Ewald
  // erf/reciprocal complement is untouched inside swb up to the 1−f(b·swb) residual, warned when >1e-3);
  // f→0 at contact, which quenches the mutual-field/charge-transfer coupling that drives the Zn–Cl contact
  // FQ sloshing (qZn→+4/qCl→−2) — the CT-mode curvature η_i+η_j−2 f·J stays positive.
  // Applied INSIDE shielded_coulomb() (covers compute_H lr_ewald=2 + the lr_nrecip taper + add_fixed_charge_field
  // + field_diag; q0field/quartic-gate/XL/ASPC inherit it through the H matrix) and in calc_Hval()
  // (gas path). Pair-side consistency: pair coul/shield/intra's matching `iondamp` settings keyword
  // damps its energy/force identically (force↔solve, as for slater). SYMMETRIC per unordered type
  // pair; off-diagonal-only ⇒ charge conservation/projector untouched. Guards: lr_ewald=1
  // (ion pairs are bare-1/r inter there) and plain DSF error out; kokkos fix rejects at parse (host-only).
  // Default OFF (lr_iondamp=0); b=0 pairs evaluate the undamped expressions. b is 1/Å (length⁻¹, NO
  // ev_scale, same as the Slater zeta).
  int      lr_iondamp = 0;          // 0 = off; 1 = at least one type pair carries b>0
  double **iondamp_b = nullptr;     // per-type-pair TT exponent b_ij (1/Å); 0 ⇒ pair undamped ((nt+1)², symmetric)
  int    **iondamp_n = nullptr;     // per-type-pair TT order n (1..8; default 4 = the CL&Pol convention)

  double chi_b(int i) const;  // the electronegativity the solve uses at site i (chi[type])

  // --- LONG-RANGE Coulomb: damped-shifted-force (DSF/Wolf) in the charge solve ---
  // lr_alpha = 0 -> shielded short-range compute_H (default)
  // lr_alpha > 0 -> H_ij = qqrd2e*(erfc(α r)/r − e_shift − r·f_shift) (== pair coul/dsf), + Wolf self-term
  //                  folded into the diagonal eta. Pair forces MUST use coul/dsf <α> <swb> for consistency.
  double lr_alpha;            // damping α (1/Å); 0 disables long-range. DSF: from param.
                              // Ewald: set = force->kspace->g_ewald in init().
  double e_shift, f_shift;    // DSF energy/force shift constants (set in init from α, swb)
  double *eta0;               // backup of param eta (idempotent self-term application across init() calls)

  // --- Route B: full Ewald long-range, RECIPROCAL IN THE MATVEC (single PD solve, no SCF/lag) ---
  // lr_ewald=1: compute_H gives plain erfc(g_ewald r)/r (real-space Ewald split); the reciprocal + Ewald
  // point-self are applied IMPLICITLY inside the matvec via the self-contained per-atom kspace
  // potential (PPPMSamqeq::compute_vector). Total operator (eta + erfc + recip + self) = full 1/r + eta
  // is positive-definite ⇒ one stable BiCGStab solve, exact (no outer-SCF, no lag, no DIIS, no reentrant
  // stock-PPPM compute). Pair MUST be lj/cut/coul/long + kspace_style pppm/samqeq. (6th param value.)
  int lr_ewald;
  // GAUSSIAN-SHIELDED TAPER-CUTOFF solve (PQEq-faithful). Default 0 = full Ewald. When 1, the
  // lr_ewald=2 Gaussian path builds H from the FULL shielded Coulomb TAPERED to 0 at swb (no erf Ewald split)
  // and SKIPS the reciprocal in qeq_matvec/add_fixed_charge_field -> a strictly short-ranged QEq operator (≤swb), like
  // native coul/pqeqgauss. Removes the long-wavelength (k=0) collective polarization mode that makes the
  // full-Ewald FQ operator near-critical. Forces are UNCHANGED (still Ewald) -> isolates the solve operator.
  // Set by `fix_modify <id> bondsoft`, whose saddle runs on this operator.
  int lr_nrecip = 0;
  double taper_poly(double r);   // 7th-order Taper polynomial (Tap) -> 1 at swa, 0 at swb (for the cutoff kernel)
  // BOND-SOFTNESS ACKS2 (standard Verstraelen X-block) for water — gaps the FQ close-pair soft mode so the
  // intrinsically near-critical operator (η≈J_shield(0)) stays solvable at LOW η (target polarization), where
  // plain QEq is indefinite (λ_min≈−128). Routes pre_force to the FixACKS2Sam saddle with the BASE bcut X-block
  // + Gaussian-shielded cutoff H (lr_nrecip, noise-free). Default off.
  // `fix_modify <id> bondsoft <kappa> <bcut>` (kappa=bond_softness scale, bcut=bond-softness cutoff Å).
  int    lr_bondsoft = 0;
  double bcut_global = 0.0;       // single global ACKS2 bond cutoff (Å) used to fill bcut[i][j] when bondsoft
  void   setup_bondsoft_bcut();   // lazily allocate+fill bcut = bcut_global (replaces the skipped init_bondcut)
  class SamqeqKspace *eksp;   // force->kspace cross-cast to the SamqeqKspace per-atom-potential interface
                             // (pppm/samqeq CPU or pppm/samqeq/kk device — both implement it)
  double *qsave;              // scratch: save/restore atom->q while feeding trial charges to compute_vector
  double *prec;              // scratch: reciprocal per-atom potential of the trial vector (compute_vector out)
  void   add_reciprocal(double *x, double *b);  // b[i] += qqrd2e*(φ_recip,i(x) − 2g_ewald/√π·x[i])
  void   field_diag();                          // DEBUG (env SAMQEQ_FIELDDIAG): dump matvec field on tags 4/5

  // --- PLAIN-QEq long-range path (lr_ewald): SPD η+full-Coulomb solve, not the ACKS2 saddle ---
  // The ACKS2 saddle is symmetric-INDEFINITE and stagnates with dense long-range Coulomb.
  // Plain QEq (Rick FQ): solve the SPD H = η + full 1/r Coulomb (erfc in compute_H + reciprocal in the
  // matvec) by CG (well-conditioned, diagonally dominant), then enforce PER-MOLECULE neutrality via the
  // two-solve trick: H·s = −χ, H·t = −1, q_i = s_i − (Σ_mol s / Σ_mol t)·t_i.
  // The UNCONSTRAINED H (η + full Coulomb over all atoms) is INDEFINITE — it has a charge-delocalization
  // mode (inter-molecular CT is Coulomb-favorable). So solve in the PER-MOLECULE-NEUTRAL subspace, where H
  // is PD: projected CG with à = P·H (P removes each molecule's mean), RHS = P(−χ), preconditioner P·D⁻¹.
  double *qs;                // CG solution (per-molecule neutral)
  double *qb;                // RHS (−χ, projected)
  double *q_r, *q_d, *q_p, *q_q;   // CG work vectors (N-dim)
  double *m_t, *m_w2;        // MINRES extra work vectors (matvec temp + w2 recurrence)

  // --- OpenMP threading --- The CSR matvec (qeq_matvec/coulomb_field share the
  // pattern) does out[i]+=H.val*x[j]; out[j]+=H.val*x[i] -- a race on out[j] when the ii-loop is partitioned
  // across threads (j is not co-partitioned with i). So each thread accumulates into its OWN private
  // nall-length slice of mv_out_t, then a deterministic thread-ORDERED sweep (t=0..nthreads-1, fixed order --
  // NOT dependent on which physical core finishes first) folds all slices into the caller's out. Lazily
  // allocated on the first THREADED matvec call (comm->nthreads==1 never touches/allocates this -> zero extra
  // memory, serial loop unchanged). Regrown whenever
  // nthreads changes or atom->nmax grows (ensure_mv_out, fix_qeq_sam_lr.cpp), mirroring how q_r/qb/etc. regrow
  // via reallocate_storage. Freed in the destructor (fix_qeq_sam.cpp).
  double *mv_out_t;          // nthreads * mv_out_cap scratch; thread t's slice = mv_out_t + t*mv_out_cap
  int     mv_out_nth;        // nthreads this buffer is currently sized for (0 = not yet allocated)
  int     mv_out_cap;        // per-thread slice length (>= atom->nmax) this buffer is currently sized for
  void    ensure_mv_out(int nth);   // lazily (re)allocate mv_out_t for nth*atom->nmax doubles

  // project_neutral's per-molecule accumulation (molsum[cmol[i]] += v[i]) is a SCATTER-reduction -- two
  // atoms of the SAME molecule landing in different threads' ii-chunks race on the same molsum[m] slot. Each
  // thread gets its own nmol_-length partial array (ps_molsum_t), then a deterministic thread-ordered sweep
  // folds the partials into molsum BEFORE the MPI_Allreduce. Lazily allocated/regrown on nmol_
  // or nthreads change (build_molinv rebuilds nmol_ every solve, so this is cheap: nthreads*nmol_ doubles,
  // typically nthreads*O(few hundred) = negligible vs mv_out_t).
  double *ps_molsum_t;       // nthreads * ps_molsum_cap scratch (thread t's slice = ps_molsum_t + t*ps_molsum_cap)
  int     ps_molsum_nth;     // nthreads this buffer is currently sized for (0 = not yet allocated)
  int     ps_molsum_cap;     // per-thread slice length (>= nmol_) this buffer is currently sized for
  void    ensure_ps_molsum(int nth);   // lazily (re)allocate ps_molsum_t for nth*nmol_ doubles

  // GRAIN THRESHOLD: on a SMALL deck the threading is a net LOSS (a 639-atom deck at np2 runs ~36% SLOWER
  // with 4 threads than with 1): dozens of fork/join parallel regions per CG iteration x ~56 matvecs/step on
  // ~320-atom-per-rank loops means the per-region fork/join overhead exceeds the loop work. Below OMP_GRAIN
  // iterations the threaded branch is skipped and the serial loop runs. 4096 = atoms-per-rank scale at
  // which the loop-body work comfortably dominates the fork/join cost; the win-case is large EChem cells —
  // small cells (<~4k atoms/rank) intentionally never thread.
  static constexpr int OMP_GRAIN = 4096;
  // uniform gate for EVERY threaded branch: thread only when >1 thread AND the loop is big enough.
  // n = the loop's actual trip count (nn / nall / atom->nmax / ninner / nlocal as appropriate per site).
  static bool omp_go(int nth, int n) { return nth > 1 && n >= OMP_GRAIN; }
  bool    use_minres;        // true ⇒ qeq_solve uses projected MINRES (symmetric, indefinite-safe); false
                             // (default) ⇒ CG. `fix_modify solver minres`. See the note at
                             // FixQEqSam::qeq_minres on when MINRES is and is not needed.
  double  minres_qcap;       // optional MINRES truncation: stop the Krylov expansion once max|q|>this
                             // (iterative regularization ⇒ BOUNDED solution on near-singular frames). 0 = off.
  // AUTO-FALLBACK: if the (default) CG path diverges (|q|>50 even after the adaptive ridge), retry the SAME step
  // with the indefinite-safe MINRES instead of crashing. On by default (only fires on configs
  // that would otherwise error — CG keeps the well-conditioned cases). `fix_modify <id> solver cg nofallback`.
  int     lr_autofb = 1;
  // FREEZE-AND-RIDE-THROUGH: if the solve still over-polarizes (max|q| > lr_qfreeze) after the adaptive ridge AND
  // the MINRES auto-fallback, committing the distorted/runaway charge would INJECT ENERGY into the dynamics (the
  // dynamics then heat up chaotically). Instead keep the PREVIOUS step's charges in atom->q, do NOT
  // advance the ASPC history, and ride through the transient near-singular config (frozen-charge dynamics is
  // stable). Default 0 = OFF; enable with `fix_modify <id> rnd <qfreeze> [<maxconsec>]`
  // (rnd = "ride not die": ride through the near-critical config instead of committing a divergent solve).
  double  lr_qfreeze = 0.0;      // max|q| above which to freeze instead of commit (0 = off)
  int     freeze_max = 500;      // error out if frozen this many CONSECUTIVE steps (config genuinely stuck)
  bigint  freeze_nconsec = 0;    // running count of consecutive frozen steps (reset on any good commit)
  bool    freeze_overpolarized(double maxq);   // returns true (and freezes) if maxq > lr_qfreeze
  // CHARGE PREDICTOR (warm-start): seed the solve's deviation qs from the PREVIOUS converged charges
  // (qs_init = q − q0 − Δ_p, already migrated with the atoms in atom->q) instead of cold-starting at 0. Cuts the
  // iteration count in BO dynamics (consecutive configs differ little). Default off ⇒ first/single-point solves
  // cold-start; opt-in via `fix_modify <id> solver warmstart on`. The seeded guess is
  // per-fragment-neutral (a previous projected solution), so it stays in the constraint subspace.
  int     warmstart = 0;

  bigint  q0field_step = -1;     // timestep q0field was last computed (bit-identical reuse stamp)
  int     setup_resolve = 0;  // 1 only during a run-BOUNDARY setup (ntimestep>0: a 2nd+ run, restart, or
                              // NVT->NVE/heating/staged switch). Makes pre_force SKIP that re-solve and reuse the
                              // previous run's converged atom->q (preserved by init_storage) -> avoids both the
                              // "Out of range atoms" PPPM-before-kspace-setup error AND the cold-resolve
                              // over-polarization blow-up. Fresh first setup (ntimestep==0, incl. reset_timestep 0)
                              // -> setup_resolve=0 -> normal solve. Set in FixQEqSam::setup_pre_force.
  int     setup_skip_logged = 0;  // the skip's log line (lr.cpp pre_force): every `run 0`, once per fix otherwise
  // The fix must not spend LAMMPS' global 100-warning budget on per-step warnings (that
  // silences every later warning of the run). warn_budget emits the first `first`, then every `every`-th occurrence.
  bool   warn_budget(bigint &n, int first = 5, int every = 100) { ++n; return n <= first || (every > 0 && n % every == 0); }
  bigint nwarn_ridge = 0, nwarn_overpol = 0, nwarn_picard = 0;
  bool   aspc_corr_call = false;  // qeq_cg is running as the ASPC capped corrector (imax = n_corr by design)
  bigint cg_nonconv = 0;          // genuine CG non-convergences (residual above tol at imax), warned 10x per run
  double cg_bnorm = 1.0;          // |b| of the last qeq_cg call (the resid/b denominator), for the ASPC committed residual
  bigint aspc_nabove = 0, aspc_nabove_run = 0;   // accepted ASPC steps whose corrector resid/b > tolerance (total, run)
  int    aspc_above_warned = 0;
  double *ridge_atom;        // LOCAL ridge (mode 3): per-atom Tikhonov hardness on the diagonal, nonzero
                             // only on runaway atoms (|q_i| above the onset). N-dim (local atoms).
  bool    ridge_local;       // true ⇒ matvec/Hdia use ridge_atom[i] (mode 3); false ⇒ scalar ridge_cur (modes
                             // 0/1/2 + Lanczos). Set only during the mode-3 solve.
  double *comm_v;            // array currently being forward/reverse-communicated (pack_flag 6, N-dim)
  double  lr_ridge;          // BASE Tikhonov diagonal hardness (eV/e², 7th hdr value, default 0). Floor for
                             // the adaptive ridge below; usually 0 so the dipole is unaffected.
  double  ridge_cur;         // ACTIVE ridge for the current solve (= lr_ridge, escalated only at near-critical
                             // configs to bound a runaway charge). Position-independent ⇒ force-safe.
  double  ridge_qonset;      // SMOOTH ridge onset (e): ridge stays = lr_ridge below this max|q|. Default 4.0.
  double  ridge_gain;        // SMOOTH ridge stiffness (eV/e⁴): if >0, ridge = lr_ridge + ridge_gain·(max|q|−
                             // ridge_qonset)² (C1 ramp, damped fixed point) ⇒ ridge & charges CONTINUOUS in
                             // config (avoids the discrete-escalation bimodality). Default 0 = discrete
                             // ×4 escalation. Set via `fix_modify <id> ridge <onset> <gain>`.
  int     ridge_mode;        // 0 = discrete ×4 escalation (default); 1 = smooth max|q| ramp (a);
                             // 3 = LOCAL per-atom smooth ridge: ridge_atom[i]=lr_ridge+ridge_gain·(|q_i|−
                             // ridge_qonset)² applied ONLY to runaway atoms ⇒ bounds runaway charges
                             // without globally over-hardening (preserves bulk dipole). `fix_modify ridge local <onset> <gain>`.
                             // 2 = Lanczos λ_min-driven (b): ridge = max(lr_ridge, lam_floor − λ_min) so the
                             // projected operator's smallest eigenvalue ≥ lam_floor. λ_min varies SMOOTHLY
                             // through criticality (unlike max|q|, which diverges) ⇒ ridge & charges continuous,
                             // and the matrix is PD ⇒ the projected CG converges even at indefinite configs.
  double  lam_floor;         // (b) target smallest eigenvalue of the projected operator (eV). Default 1.0.
  double  ridge_delta = 0.0;  // (b) Ritz-estimate error budget: target = lam_floor + ridge_delta, so lambda_min(A+ridge)
                              // >= lam_floor whenever the m-step estimate is within delta of the true lambda_min (an
                              // upper bound can only miss a negative mode). Default 0.
  int     nlanczos;          // (b) Lanczos steps for the λ_min estimate (extreme eig converges fast). Default 24.
  int     lmin_logged = 0;   // estimate_lambda_min: first estimate + one unconverged warning are logged;
                             //   1 = first estimate logged, 2 = unconverged warning issued too
  // efficiency: cache the (slowly-varying) Lanczos λ_min across steps (the per-step estimate is nlanczos PPPM matvecs)
  int     ridge_every;       // (b) recompute λ_min every N steps (default 1 = every step)
  double  lmin_cache;        // (b) cached λ_min estimate
  bigint  lmin_step;         // (b) timestep of the last λ_min recompute (-1 = none)
  // COMPACT per-molecule projector reindexing. Raw molecule IDs can be arbitrarily sparse (e.g.
  // an ion tagged `mol 9001` on a 216-water box ⇒ raw max(mol)+1 = 9003 for ~219 actual fragments); a
  // raw-sized per-ITERATION allreduce/array (molsum) would scale with the MAX raw ID, not the fragment COUNT,
  // on the hottest path in the code (project_neutral, called every CG/MINRES iteration). So build_molinv compacts the ACTIVE
  // raw IDs (any in-group atom, on any rank) into a dense [0,nactive) index space, ONCE per solve (not per
  // iteration), deterministically (ascending raw-ID order ⇒ identical on every rank/np).
  int     nmol_;             // #ACTIVE per-molecule projector slots = nactive (the compact fragment
                             // COUNT, not raw max(mol)+1). Every reader below indexes by the compact slot,
                             // never by the raw atom->molecule[i] directly.
  // molsum is over-allocated by PN_TAIL_MAX extra slots so small serialized reductions
  // can ride the SAME MPI_Allreduce message as the per-molecule sums (project_neutral): [nmol_] an optional
  // fused dot partial. The reduced values in [0,nmol_) are untouched.
  static constexpr int PN_TAIL_MAX = 1;
  double *molinv, *molsum;   // per-molecule 1/count (cached) and sum scratch (for the projector P); COMPACT
                             // (length nmol_ = nactive) — index by the per-atom compact slot cmol[i] below.
  int     maxmol_;           // raw mol-ID space size (= max raw ID + 1); the length of mol2c.
  int    *mol2c;             // raw mol id -> compact index (length maxmol_), rebuilt every build_molinv;
                             // -1 = that raw ID currently has no in-group atom anywhere (compacted away).
  int    *c2mol;             // compact index -> raw mol id (length nmol_ = nactive), the inverse of
                             // mol2c — needed wherever a compact-space result must be reported by its RAW
                             // mol id (log messages: all user-facing state stays RAW, never compact).
  int    *cmol;              // per-LOCAL-atom compact index (nmax array; filled in build_molinv for
                             // ilist atoms only — project_neutral/qeq_solve never touch a
                             // ghost's cmol, so ghosts are neither filled nor read). The per-iteration
                             // replacement for atom->molecule[i] on every projector read.
                             // mol==nullptr ⇒ nmol_=1, cmol[i]=0 for every in-group atom (one global
                             // fragment).
  // --- build_molinv cache. The compact projector state (mol2c/c2mol/nmol_/molinv/
  // molsum counts) is invariant under atom MIGRATION (it depends only on the global multiset of
  // raw mol ids of in-group atoms), so the O(max-raw-mol-ID) pass-1 Allreduce + alloc
  // (~3.6 MB/solve under a `set group metal mol 900000` convention) is skipped between
  // invalidation events; only the LOCAL cmol fill is refreshed per call.
  // ★ molinv_valid is NOT rank-uniform. Most clear sites ARE collective (ctor, init(), fix_modify,
  // restart()) — but deallocate_storage() also clears it, and
  // reallocate_storage() reaches deallocate_storage() from `atom->nmax > nmax`, a PER-RANK trigger:
  // whichever rank's atom arrays regrow at an exchange event drops the cache (and nulls molinv/mol2c)
  // alone. A gate on the local flag would split — the regrown subset entering the collective rebuild while
  // the rest served the cache — a mismatched collective and a silent hang. build_molinv therefore
  // Allreduce(MIN)s the per-rank verdict so the branch is unanimous BY CONSTRUCTION, and trips
  // molinv_desync_note (one warning) when verdicts diverged. The forced rebuild on
  // a one-rank drop recomputes byte-identical values (counts are exact integer-valued FP; pass 1 is a
  // {0,1} MAX), so recovery changes nothing numeric.
  int     molinv_valid;      // 1 = cached compact projector state is current on THIS rank; 0 = rebuild.
                             // Rank-LOCAL — only the Allreduce(MIN)'d verdict in build_molinv may gate
                             // a collective; never branch a collective on this flag directly.
  bigint  molinv_natoms;     // global atom count at build time (atom insertion/deletion detector)
  int    *molinv_active;     // persistent pass-1 scratch (length molinv_active_cap), grown monotonically
  int     molinv_active_cap; // current capacity of molinv_active (>= maxmol_ after a build)
  int     molinv_desync_note = 0;  // one-shot tripwire: set (on all ranks) the first time the cache-
                                   // validity Allreduce sees rank disagreement; gates the warning only

  double solve_diag_of(int i);      // the lr solve-diagonal base at site i: eta[type], plus the quartic secant under lr_quartic
  double eself_diag_of(int i, bool in_group);   // the HARMONIC part of that diagonal (no ridge, no quartic secant):
                                                 // the ONLY curvature compute_scalar may report
  int     lr_calibrated;     // measured the grid reciprocal self-term yet?
  int     lr_recip_probes = 16; // K deterministic probe pairs for the self-term calibration (default 16).
                                // K>=1 selects the K lowest GLOBAL tags and averages, which is order-
                                // independent by construction. 0 = a single pair by local ilist order
                                // (decomposition- and atom-order-dependent), which wanders under NPT by up to
                                // several percent where K=16 drifts ~0.004%. Clamped to the group size, floor
                                // >=2 atoms; falls back to the single pair on a G2-window miss under the default K.
  int     lr_recip_probes_user = 0;   // 1 once `fix_modify recip_probes` was issued (then <2 atoms / a G2 miss is an error)
  // `fix_modify recip_self peratom` -- the EXACT per-atom grid self-coefficient from pppm/samqeq
  // (compute_self_peratom) replaces the calibrated scalar in add_reciprocal: no probes, no calibration
  // events, no drift; the sub-grid position dependence is carried per atom. Opt-in.
  int     lr_self_peratom = 0;
  double *lr_self_atom = nullptr;     // per-atom c_i (raw), refreshed at every solve (positions move)
  int     lr_self_atom_nmax = 0;
  int     lr_self_atom_logged = 0;    // stats line once per run (reset in setup_pre_force)
  void    compute_self_peratom_now();  // fill lr_self_atom for this solve (+ stats line once per run)
  double  lr_self_sigma = 0.0;  // spread over the K probes (0 if K<2); disclosed in the calibration log
  // The FIRST calibration at the current mesh COUNTS plus a
  // running census, so a random walk of sub-2% re-calibration steps is caught against its ORIGIN, not its
  // predecessor (G1 compares against the previous value only and cannot see a drift). Reset when the counts change.
  double  lr_self_first = 0.0;                    // first calibration at this {nx,ny,nz} (raw)
  int     lr_first_nx = -1, lr_first_ny = -1, lr_first_nz = -1;
  bigint  lr_first_step = -1;                     // step of that first calibration
  double  lr_first_vol = 0.0;                     // box volume at that first calibration (drift message only)
  bigint  lr_self_step = -1;                      // step of the LAST calibration
  int     lr_ncalib = 0;                          // calibrations at this mesh
  int     lr_drift_band = 0;                      // last 0.5%-band of |rs-first|/first warned about (throttle)
  int     lr_self_pinned = 0;   // `fix_modify <id> recip_self <raw>` -- user-pinned self-term.
                                // Skips measurement entirely, so the operator is reproducible across
                                // rank counts, atom orderings and code versions. This is the path for
                                // reproducing a published run whose calibration is known.
  double  lr_pin_gchk = 0.0;   // g_ewald at which a pinned recip_self was last checked against 2 g_ewald/sqrt(pi)
  double  lr_self_meas;      // PPPM on-site reciprocal self-potential (raw, measured via a lone unit charge);
                             // subtracted from prec so the i=i grid self cancels -> QEq diagonal = eta (PD).
  class PPPM *lr_pppm;       // force->kspace cross-cast to PPPM (same object as eksp) — exposes the grid dims
  int     lr_nx, lr_ny, lr_nz;  // PPPM grid {nx,ny,nz}_pppm at the last calibration (-1 = never)
  double  lr_gewald;            // g_ewald at the last calibration
  double  lr_prd[3];            // box lengths at the last calibration: recip_self scales with the grid
                                // SPACING (box/counts), not just the counts -> NPT changes it at fixed counts
  tagint  lr_tagA, lr_tagB;     // STICKY probe pair: cached at the first calibration, re-measured (min-image)
                                // at every re-calibration so rs stays on one consistent measurement
  bool   grid_changed();        // true if the PPPM grid signature differs from the last calibration -> recalibrate
  // Optional FUSED dot -- when dot_out != nullptr, the local partial of the in-group dot
  // <dot_a,dot_b> (computed by qeq_dot_local, byte-identical accumulation to qeq_dot) is appended as a tail
  // slot of the molsum Allreduce and the reduced value returned in *dot_out, saving one 1-double Allreduce
  // latency. ONLY legal when dot_a/dot_b are NOT modified by this projection (the local partial is formed
  // BEFORE the reduce) -- e.g. qeq_cg's ||b|| alongside project_neutral(q_d). The per-CG-iteration dots
  // (q_d.q_q, q_r.q_p) read the PROJECTED vector and therefore CANNOT be fused byte-identically (they need
  // the reduced molsum first); they keep their own qeq_dot calls.
  void   project_neutral(double *v, double *dot_a = nullptr, double *dot_b = nullptr,
                         double *dot_out = nullptr);   // v -> v − per-molecule mean (enforce Σ_mol v = 0)
  void   calibrate_recip_self();                // grid reciprocal self-term measurement (re-run iff grid_changed)
  bool   calibrate_recip_self_probes();         // deterministic K-probe route (lr_recip_probes>0); false =
                                                //   a G2 miss or < 2 atoms under the DEFAULT K -> caller runs the single pair
  bool   measure_recip_probe(tagint tA, tagint &tB, double hmax, double &sk);
                                                // ONE co-residency-free pair measurement shared by both
                                                //   routes (A by global tag; B pinned or nearest-in-group;
                                                //   hmax>0 = G2 window, <=0 = none). false = no partner.
  void   finish_recip_self(double rs, int nprobe);  // shared guardrails + commit for both routes
  void   build_molinv();                        // (re)build per-molecule 1/count for the projector

  // --- fixed (non-group) charges as an external field in the QEq RHS ---
  // ionfield: fixed charges on atoms OUTSIDE the qeq/sam group enter the QEq right-hand side as a shielded field
  // (real-space J_shield − erf split from the FULL list, plus the reciprocal potential at the solved atoms).
  int     ionfield_flag;       // ionfield: 1 = on (default): fixed non-group ION
                               //   charges -> QEq RHS via add_fixed_charge_field().
                               //   supplies the ion's electrostatic field only (~56% of the QM dipole response
                               //   at contact in an ion-water dimer benchmark). 0 = off (ion-blind solve).
  double *fixq_field;         // per-atom field from the fixed non-group charges (real+recip), eV/e; sized nmax
  void   add_fixed_charge_field();   // (re)compute the field from the fixed non-group charges (once per solve);
                               //   collective; returns after zeroing when no non-group charged atom exists.

  // ---- on-site self-energy REPORTING basis ------------------------------------------------------
  // The lr/base solves are variational in the SHIFTED variable qs = q - q0 (q0 = the reference-charge
  // parabola center): the model's on-site term is chi*q + 1/2 eta*qs^2. Reporting chi*q + 1/2 eta*q^2
  // (physical q) instead lets the difference Sum_i eta_i*q0_i*q_i(t) (+const) ride on the reported
  // etotal for any q0 != 0 model, where it fluctuates by tens of kcal/mol and looks like an NVE
  // non-conservation. `fix_modify <id> eselfref qs` (the default) reports 1/2 eta qs^2 (gradient
  // chi + eta*qs = the solve's stationarity condition), so the monitored etotal IS the conserved
  // functional; `eselfref q` restores the physical-q report. Reporting-only: no force, no virial,
  // trajectories identical either way. Host+device agnostic (pure compute_scalar).
  int    eself_qs = 1;            // 0 = chi q + 1/2 eta q^2 ; 1 = chi q + 1/2 eta (q-q0)^2 (default)

  // --- q0 REFERENCE-CHARGE FIELD (energy conservation). The lr_ewald solve varies qs (qa = qs+q0); the
  // physical pair/PPPM forces use qa, so the energy the solve stationarizes must include the Coulomb field
  // from the FIXED q0 reference charges (cores carry q0=1.0 in the Drude/PQEq partition). Omitting it makes
  // ∂E_pe/∂qs ≠ const at the solve -> the Hellmann-Feynman force drops the [J·q0]·dq/dr term -> BO heating
  // (flexible+Drude amplify dq/dr; q0=0 models are unaffected). The base ACKS2 path already adds this as
  // `reffield` (fix_qeq_sam.cpp); here we add the matching term to the lr RHS via q0field. has_q0ref gates it
  // (any q0≠0) so q0=0 models skip the extra matvec entirely.
  double *q0field;             // per-core projected Coulomb field from the q0 reference charges, eV/e; sized nmax
  int     has_q0ref;          // 1 iff any type has q0≠0 (computed at param load); else q0field stays 0
  virtual void coulomb_field(double *x, double *out, bool project = true);   // out = [P](full off-diagonal
                                // 1/r Coulomb · x); project=false ⇒ raw field (for the quartic gate).
                                // VIRTUAL so the kokkos fix can serve it from the device
                                // operator — it is the per-step q0-reference-field consumer of the CSR H.

  // --- optional block-ILUT saddle preconditioner (CENTRALIZED/replicated parallel) for the metal ---
  // Enable with `fix_modify <id> precond ilu` (default 0 = diagonal Jacobi). The metal saddle is
  // well-posed but ill-conditioned -> diagonal needs ~thousands of BiCGStab iters; ILUT -> ~tens. In
  // parallel every rank factors the FULL global saddle (see fix_qeq_sam_ilu.cpp); proc-stable to np=1.
  int     precond_mode;        // 0 = diagonal (default), 1 = block-ILUT saddle, 2 = molecular block-Jacobi
  /* ---- MOLECULAR BLOCK-JACOBI preconditioner (`fix_modify <id> precond mol`) ----
     A preconditioner cannot change the answer, only the iteration count, which makes this the one
     lever here that is numerically free. In a fluctuating-charge model the stiffest couplings by far
     are INTRAmolecular (bonded neighbours at ~1 Å, J ~ 10 eV against eta ~ 12-20), and diagonal
     Jacobi ignores every one of them. Each molecule's dense block (eta_i delta_ij + J_ij over its own
     atoms) is small — 3 atoms for water, tens for the anions — so a Cholesky factor per molecule is
     cheap to build (n_m^2 kernel evaluations, no neighbour list needed) and cheap to apply.
     With ASPC supplying a good predictor the gain is small; the target is the ASPC-off / near-critical
     / cold-start regime, where the solve is slowest. Off by default.*/
  int     precond_mol_ok = 0;  // 1 once the blocks are built for this solve
  std::vector<int>    mb_off, mb_n, mb_idx;   // per-molecule offset / size / member atom indices
  std::vector<double> mb_L;                   // packed lower-triangular Cholesky factors (n(n+1)/2 each)
  std::vector<char>   mb_ok;                  // 0 => that block fell back to Jacobi (not PD)
  void   build_mol_blocks();                  // factorize once per solve (positions fixed within it)
  void   apply_precond(double *r, double *z); // z = M^-1 r (Jacobi, or the molecular blocks)
  // GLOBAL BLOCK-CSR over atoms (2x2 blocks), replicated on every rank: row-ptr[n+1], block-col[nnz],
  // per-row diag-block slot[n], block values[4*nnz] (row-major a00,a01,a10,a11). Block-ILUT with 2x2
  // pivots is stable on the indefinite ACKS2 saddle (diagonal block [[eta,1],[1,X_diag]] is nonsingular).
  int    *ilu_rp, *ilu_ci, *ilu_dptr;
  double *ilu_lu;              // block-ILU factors (L blocks + D_i + U blocks)
  double *ilu_Dinv;            // inverted diagonal 2x2 pivots [4*n]
  int     ilu_valid = 0;       // cache: ILU factorization reused until the next reneighbor (a preconditioner
                               // only affects iteration count, not the solution, so a stale ILU between neighbor
                               // rebuilds is safe). Reset on parse/realloc/reneighbor. Single-point ⇒ 1 build.
  int     ilu_n, ilu_nnz;      // ilu_n = N = global atom count (block rows); ilu_nnz = #blocks
  double  ilu_shift;           // optional Manteuffel diagonal shift (usually 0 with block pivots)
  double  ilu_droptol;         // ILUT relative drop tolerance for fill (default 1e-3; smaller = more fill)
  double *ilu_Y0, *ilu_Y1, *ilu_scr, *ilu_w;   // A^{-1} C columns + rhs/solve scratch (global block vec, 2*N)
  double  ilu_Sinv[4];         // 2x2 constraint Schur inverse
  // CENTRALIZED / REPLICATED parallel: every rank assembles + factors the FULL global
  // saddle (gather H/X couplings by global tag) and applies it to the globally-gathered residual.
  // Block-Jacobi Schwarz (with or without a per-proc-constant coarse correction) breaks down on the
  // near-singular metal; the replicated solve is decomposition-independent -> proc-stable to np=1.
  void precond_apply(double *in, double *out) override;   // dispatch diagonal vs centralized ILU (fix_qeq_sam_ilu.cpp)
  void ilu_build();            // assemble+factor the GLOBAL block-ILUT (replicated on every rank)
  void ilu_solveM2(const double *rhs, double *sol);       // serial block-ILUT solve (global, 2*ilu_n)
  int  ilu_find(int i, int col);
  void ilu_free();

  // --- Extended-Lagrangian charge dynamics (true NVE: charges are dynamic DOF, never re-minimized) ---
  // fix_modify ID xl <q_mass> <q_tdamp> enables it. Velocity-Verlet propagate q with the electronegativity
  // force μ_i=χ_i+Σ_j J_ij q_j (per-molecule-projected so Σ_mol q is conserved); a weak friction (q_tdamp)
  // keeps the charge DOF cold/adiabatic. Avoids the per-step BO minimization that crashes near criticality.
  int     lr_xl;             // 1 = extended-Lagrangian charge dynamics
  double  q_mass, q_tdamp;   // fictitious charge mass (eV·ps²/e²) and charge-friction damping time (ps)
  double *qdot, *qddot;      // per-atom charge velocity / acceleration
  int     qdot_nalloc = 0;   // slots of qdot/qddot already zeroed by grow_arrays
  double *pchi;              // cached P(χ) (per-molecule-projected electronegativity offset)
  int     xl_started;        // qdot/qddot initialized?
  // XL SPIKE-GUARD: a soft one-sided restoring "force" on charges whose |q| exceeds xl_qspike, added to
  // qddot (and PROJECTED per-molecule so it can't leak charge between molecules). Bounds the rare
  // near-critical XL charge spikes that otherwise run away -> "Out of range atoms - cannot compute PPPM"
  // after tens of thousands of steps. With xl_qspike above the physical |q| range it NEVER fires in normal
  // dynamics, so it does not over-harden the dipole/density the way a global base ridge does.
  double  xl_qspike;         // |q| threshold (e) above which the guard engages; <=0 disables. Default 2.0.
  double  xl_kspike;         // guard stiffness (eV/e²): restoring force = xl_kspike·(|q|−xl_qspike). Default 10.
  // FD-BALANCED Langevin charge thermostat: the pure friction
  // qdot*=exp(-dt/q_tdamp) is a one-sided drag with NO fluctuation-dissipation balance -> drains energy
  // (the NVE freeze) and can't reach the BO charge distribution (a few-percent over-polarization). Adding the FDT
  // noise qdot = c1*qdot + c2*gauss (c2=sqrt((1-c1^2) kB T_q / q_mass), projected per-molecule) restores
  // FD balance -> zero net drain + charges sampled at T_q. xl_Tq=0 -> c2=0 -> pure friction.
  double  xl_Tq;             // Langevin charge-thermostat target temperature (K); 0 = pure friction (default)
  class RanMars *xl_random;  // FDT noise RNG for the Langevin charge thermostat (allocated iff xl_Tq>0)
  // MASS-WEIGHTED (per-type) charge mass — the Car-Parrinello remedy for a WIDE eta spectrum. A single scalar q_mass gives each type a fictitious charge
  // frequency ω_i=√(eta_i/q_mass); when the spread of eta across types is large (alkane: C eta 3.8 vs H 29,
  // ratio ~8), the SOFT types (C) drop into resonance with the nuclear band (C-H stretch) → non-adiabatic
  // pumping → localized over-polarization on the ion's first shell → corrupted transport D. Setting the mass
  // ∝ stiffness, m_i = m0·eta[type_i], makes EVERY mode oscillate at one frequency ω0=√(1/m0), chosen above
  // the nuclear band. When xl_masswt: the fix_modify `xl <q_mass>` value is reinterpreted as m0 and the
  // effective per-atom mass is q_mass·eta[type]. Default OFF → scalar q_mass.
  int     xl_masswt;         // 1 = per-type mass-weighted charge mass (m_i = q_mass·eta[type_i]); 0 = scalar (default)
  void   xl_chargeforce();   // qddot = −(P(χ) + qeq_matvec(q))/q_mass (per-molecule-neutral) + spike-guard
  int    setmask() override;
  void   initial_integrate(int) override;
  void   final_integrate() override;
  void   qeq_matvec(double *x, double *out);    // out = P(η·x + erfc-Coulomb·x + reciprocal·x)
  void   csr_matvec_add(double *x, double *out);   // out[i]+=H.val*x[j]; out[j]+=H.val*x[i]
                                                    // over ii in nn -- shared by qeq_matvec/coulomb_field
  double qeq_dot(double *a, double *b);         // MPI-reduced in-group dot over local atoms
  double qeq_dot_local(double *a, double *b);   // the LOCAL partial of qeq_dot (no MPI) -- shared by
                                                // qeq_dot and project_neutral's fused-dot tail slot so the
                                                // accumulation is byte-identical on both paths
  virtual int qeq_cg(double *b, double *x);     // projected Jacobi-preconditioned CG (PD on the subspace);
                                                // virtual so qeq/sam/kk routes the lr_ewald=2 corrector to the device CG
  int    qeq_minres(double *b, double *x, double bref = -1.0);   // projected MINRES (symmetric INDEFINITE-
                                                // safe). bref>0 = external reference norm for the relative
                                                // stopping criterion (the FULL-BO RHS norm when solving the
                                                // seeded DELTA system, so warm and cold solves stop at the SAME
                                                // absolute residual); bref<=0 = ||b|| (default)
  void   qeq_solve_minres();                    // the MINRES solve path (primary when use_minres, or CG fallback)
  double estimate_lambda_min(int m);            // (b) Lanczos estimate of the projected operator's λ_min
  double tridiag_lambda_min(const double *d, const double *e, int n);  // smallest eig of a symmetric tridiagonal
  void   qeq_solve();                           // build P(−χ), projected CG -> atom->q (per-molecule neutral)
  void pre_force(int) override;   // Ewald: plain-QEq SPD CG solve (reciprocal in matvec); else base
  // The fix applies no position forces in this release; post_force only zeroes virial so a
  // `compute pressure` that sums fix virials reads an initialised zero.
  void post_force(int) override;
  void setup(int) override;
  void min_setup(int) override;
  void min_post_force(int) override;
  void post_force_respa(int, int, int) override;
  void setup_pre_force(int) override;   // warm-start the run-boundary re-solve (ntimestep>0) -> no blow-up

  void pertype_parameters(char *) override;
  int  modify_param(int, char **) override;
  // A FULL neighbor list (id 1) for the ionfield sweep and the device solve. init_list routes both;
  // the charge solve uses the half list.
  class NeighList *list_full;
  void init_list(int, class NeighList *) override;
  void allocate_storage() override;       // + eta_diag
  void deallocate_storage() override;
  void init_storage() override;       // loop ilist over nn (local), not NN — file-mode ghost-safe
  void init_matvec() override;
  void calculate_Q() override;
  int  saddle_fallback(double *b, double *x) override;   // one ILU retry of a grossly unconverged saddle (fix_qeq_sam_ilu.cpp)

  // response kernel (shadows ACKS2 compute_X; called from our init_matvec)
  void compute_H() override;   // off-diagonals: shielded (lr_alpha=0) or DSF (lr_alpha>0)
  void compute_X();
  double calc_w(int i, int j, double r);   // X-block weight: kappa_bond·exp(-(r/r_ov)^2), intra pairs only

  // comm overrides: pack_flag==6 carries the N-dim comm_v (plain-QEq CG);
  // all other flags delegate to FixACKS2Sam.
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;

  // qdot/qddot are PERSISTENT per-atom XL state -> they must MIGRATE with atoms on reneighbor.
  // Extend the base per-atom GROW-callback machinery (s_hist/t_hist) to carry qdot,qddot too; without
  // this they stay at stale local indices after atom exchange -> per-molecule neutrality leaks -> blow-up.
  void grow_arrays(int) override;
  void copy_arrays(int, int, int) override;
  int  pack_exchange(int, double *) override;
  int  unpack_exchange(int, double *) override;
};

}    // namespace LAMMPS_NS
#endif
#endif
