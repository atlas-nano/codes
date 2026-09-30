// clang-format off
/* ----------------------------------------------------------------------
   samQEq (fix qeq/sam): extended-Lagrangian charge dynamics.
   Split out of fix_qeq_sam.cpp; all routines are members of class FixQEqSam
   (declared in fix_qeq_sam.h) -- a separate translation unit, not a new class.
-------------------------------------------------------------------------*/

#include "fix_qeq_sam.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "memory.h"
#include "neigh_list.h"
#include "update.h"
#include "kspace.h"
#include "pppm_samqeq.h"
#include "random_mars.h"   // variant-1 Langevin charge-thermostat FDT noise
#include "math_const.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;
using MathConst::MY_PIS;

/* ====================== EXTENDED-LAGRANGIAN CHARGE DYNAMICS =========*/

int FixQEqSam::setmask()
{
  // ALWAYS claim both the BO (PRE_FORCE) and XL (INITIAL/FINAL_INTEGRATE) hooks — the fix mask is cached at
  // creation (modify.cpp) and can't be changed by fix_modify, so we dispatch at RUNTIME on lr_xl instead:
  // pre_force no-ops in XL mode; initial/final_integrate no-op in BO mode.
  // + the POST_FORCE-class hooks, which only zero the fix virial (the fix applies no position forces).
  return FixQEqBaseSam::setmask() | INITIAL_INTEGRATE | FINAL_INTEGRATE |
         POST_FORCE | POST_FORCE_RESPA | MIN_POST_FORCE;
}

/* charge "force" (negative electronegativity), per-molecule-projected:
   qddot_i = −(P(χ)_i + [P(H·q)]_i) / q_mass, where H = η + full 1/r Coulomb (qeq_matvec).
   The projection makes Σ_mol qddot = 0 ⇒ per-molecule charge is conserved by the dynamics.*/
void FixQEqSam::xl_chargeforce()
{
  int *mask = atom->mask, *type = atom->type;
  ridge_cur = lr_ridge;                          // XL: base ridge only (smooth dynamics, no adaptive guard)
  // matvec on the VARIABLE charge qs = qa - q0 (NOT qa), to match the BO solve operator. Combined with
  // pchi = P(chi + fixq_field + q0field) above, qddot = -(pchi + P(H·qs)) is the EXACT negative BO gradient ->
  // zero at the BO minimum (charges stay put) instead of the spurious ~fixq_field kick.
  double *qa = atom->q;
  for (int i=0;i<atom->nmax;i++) qs[i]=0.0;
  for (int ii=0;ii<nn;ii++){ int i=ilist[ii]; if(mask[i]&groupbit) qs[i]=qa[i]-q0[type[i]]; }
  qeq_matvec(qs, q_q);                            // q_q = P(H·qs) (η qs + full Coulomb qs, projected)
  // 1/(fictitious charge mass). Scalar (default): inv = 1/q_mass -> byte-identical. Mass-weighted (xl_masswt):
  // per-atom m_i = q_mass·eta_eff_i, so every mode has ONE frequency ω0=√(eta_eff/(q_mass·eta_eff))=√(1/q_mass)
  // -- constant for all q, chosen above the nuclear band (the Car-Parrinello cure for a wide/anharmonic eta
  // spectrum; see fix_qeq_sam.h xl_masswt). ★ eta_eff (NOT bare eta): under a c4 quartic guard the real on-site
  // stiffness is eta_eff(q)=eta+½c4q² (charge-dependent, Duffing). Weighting by BARE eta lets ω(q)=√(eta_eff/
  // (q_mass·eta)) DIVERGE with |q| -> crosses the Verlet ceiling -> dt-insensitive blow-up. Weighting by
  // eta_eff cancels that exactly (mass grows in lockstep with stiffness). eta_eff = eta_diag[i] (filled by
  // apply_quartic_eta in final_integrate BEFORE this call, whenever lr_quartic) -- the diagonal the BO solve
  // preconditions with. !lr_quartic -> bare eta (unchanged). Floored via
  // quartic_etafloor inside apply_quartic_eta, so no near-zero-mass runaway.
  const double inv = 1.0/q_mass;                  // scalar path
  auto inv_mass = [&](int i) -> double {
    if (!xl_masswt) return inv;
    double eeff = lr_quartic ? eta_diag[i] : eta[type[i]];
    return 1.0/(q_mass*eeff); };

  // EXTRA per-atom charge forces folded into qddot via a projected scratch vector (q_p, free during XL):
  //   (1) QUARTIC near-crit cure — the XL analog of the BOMD `fix_modify quartic`. The BOMD quartic puts
  //       eta_eff(q)=eta+1/6 c4 q^2 on the solve diagonal; the equivalent on-site energy is
  //       (1/24)c4 q^4 + (1/6)c q^3, so the charge force gains -dE/dq = -(1/6 c4 q^3 + 1/2 c q^2). This
  //       cubic restoring STIFFENS the indefinite near-critical soft mode (lambda_min<0) that otherwise
  //       makes the XL charge dynamics blow up at liquid density. q-only, per-molecule projected.
  //   (2) #22 spike-guard — soft one-sided restoring above |q|>xl_qspike (last-resort clamp).
  //   (3) #13 Phase B IP-STAIRCASE — per-type c3_type/c4_type on-site anharmonicity (UNGATED, independent of
  //       quartic_groupbit membership; mirrors compute_scalar's self-energy at electrode.cpp:67-68 and the
  //       Picard secant diagonal at levels.cpp:377-380): E(q)=...+1/6 c3_type q^3+1/24 c4_type q^4 ⇒
  //       -dE/dq = -(1/2 c3_type q^2 + 1/6 c4_type q^3). Without this term XL propagates staircase-carrying
  //       ions on the wrong PES (BO-audit finding #5): the diagonal stiffens (lr_quartic auto-set whenever
  //       any type carries c3/c4, at fix_qeq_sam.cpp:433) but the restoring force never gets applied.
  // PROJECTED per-molecule (project_neutral) so neither can leak charge between molecules. With the quartic
  // off, no staircase type, AND xl_qspike above the physical |q| range, q_p is all-zero -> byte-identical to
  // the unguarded path.
  bool any_staircase = false;                        // computed ONCE (not per atom): any type carries c3/c4
  for (int t = 1; t <= atom->ntypes; t++) if (c3_type[t] != 0.0 || c4_type[t] != 0.0) { any_staircase = true; break; }
  bool have_extra = (lr_quartic || xl_qspike > 0.0 || any_staircase);
  if (have_extra) {
    for (int ii=0; ii<nn; ii++){ int i=ilist[ii];
      if (!(mask[i]&groupbit)) { q_p[i] = 0.0; continue; }
      double qi = qa[i], f = 0.0;
      // staircase + global gated quartic −dE/dq: the shared on-site anharmonic model (fix_qeq_sam.h, Tier-A #1)
      anh_force_add(f, qi, type[i], mask[i] & quartic_groupbit, quartic_gate[i]);
      if (xl_qspike > 0.0) { double aq = fabs(qi);
        if (aq > xl_qspike) f -= xl_kspike*(aq - xl_qspike)*(qi > 0.0 ? 1.0 : -1.0); }
      q_p[i] = f;
    }
    project_neutral(q_p);                        // keep Σ_mol guard = 0 (per-molecule charge conserved)
    for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(!(mask[i]&groupbit)) continue;
      qddot[i] = (-(pchi[i] + q_q[i]) + q_p[i]) * inv_mass(i); }
  } else {
    for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(!(mask[i]&groupbit)) continue;
      qddot[i] = -(pchi[i] + q_q[i]) * inv_mass(i); }    // already per-molecule-neutral (pchi & q_q both projected)
  }
}

/* velocity-Verlet first half-step on the charges (uses qddot from the previous step).*/
void FixQEqSam::initial_integrate(int /*vflag*/)
{
  if (!lr_xl) return;
  double dt = update->dt; double *qa = atom->q; int *mask = atom->mask;
  if (!xl_started) { for (int i=0;i<atom->nmax;i++){ qdot[i]=0.0; qddot[i]=0.0; } xl_started=1; }
  for (int i=0; i<atom->nlocal; i++) if (mask[i]&groupbit) {
    qa[i]   += dt*qdot[i] + 0.5*dt*dt*qddot[i];
    qdot[i] += 0.5*dt*qddot[i];
  }
  comm_v = qa; pack_flag = 6; comm->forward_comm(this);   // distribute new q to ghosts for the force compute
}

/* velocity-Verlet second half-step: recompute qddot at the new q/positions, advance qdot, cool charges.*/
void FixQEqSam::final_integrate()
{
  if (!lr_xl) return;
  double dt = update->dt; int *mask = atom->mask;
  NN = atom->nlocal + atom->nghost;
  nn = list->inum; ilist = list->ilist; numneigh = list->numneigh; firstneigh = list->firstneigh;
  lr_alpha = force->kspace->g_ewald;
  if (atom->nmax > nmax) reallocate_storage();
  if (atom->nlocal > n_cap*0.90 || m_fill > m_cap*0.90) reallocate_matrix();   // XL under NPT: density rises ->
                                              // neighbor count grows -> H CSR must grow too, else compute_H
                                              // overflows m_cap -> qeq_matvec reads bad indices -> segfault.
  if (!lr_calibrated || grid_changed()) calibrate_recip_self();   // (re)measure recip_self iff the PPPM grid changed
  build_molinv();
  compute_H();                                    // erfc real-space at the current positions (q0field needs H built)
  // ★ XL FORCE CONSISTENCY (Drude/q0): the charge force must be the NEGATIVE BO GRADIENT, so pchi must carry the
  // SAME RHS terms the BO solve uses — the fixed-charge field (add_fixed_charge_field) AND the q0 reference-charge
  // (q0field). Without these the XL force is wrong by ~fixq_field (~-20 eV/e) every step -> the Drude-XL blow-up
  // (XL was only ever validated for non-Drude SPC-FQ, where drude_flag=0 & has_q0ref=0 -> both 0 -> byte-id).
  if (ionfield_flag) add_fixed_charge_field();   // ionfield (no find_drude(); see fix_qeq_sam_lr.cpp)
  if (has_q0ref) {
    int *type = atom->type;
    for (int i=0;i<atom->nmax;i++) m_t[i]=0.0;
    for (int ii=0;ii<nn;ii++){ int i=ilist[ii]; if(mask[i]&groupbit) m_t[i]=q0[type[i]]; }
    coulomb_field(m_t, q0field);                   // q0field = P(J_offdiag·q0) (same as the BO path)
  } else for (int i=0;i<atom->nmax;i++) q0field[i]=0.0;
  // B3.6: chi_field (fix efield coupling) -- the BO RHS folds it in the SAME bracket as chi_b/reffield before
  // negating (fix_qeq_sam.cpp:697,745: b_s[i] = -(chi_b(i)+reffield); b_s[i] -= chi_field[i]), so it enters pchi
  // with the SAME (+) sign as chi_b/q0field here (qddot = -(pchi+H·qs)/q_mass is the negative BO gradient; a
  // field-coupled deck without this term silently got zero charge response under XL, mirroring BO finding #10).
  // No efield fix present -> get_chi_field() is never called and chi_field is untouched -> byte-identical.
  if (efield) get_chi_field();
  for (int ii=0; ii<nn; ii++){ int i=ilist[ii]; if(mask[i]&groupbit)
    pchi[i]=chi_b(i) + (ionfield_flag?fixq_field[i]:0.0) + q0field[i] + (efield?chi_field[i]:0.0); }
  project_neutral(pchi);                          // cache P(chi + fixq_field + q0field [+ chi_field]) for the charge force
  if (lr_quartic) apply_quartic_eta();   // fill eta_diag[i] = eta0[type] + anharmonic secant (lagged q)
  // (chemical seed since 2026-09-16; solve_diag_of re-adds the gself delta for the FORCE matvec. The mass weight
  // and shadow preconditioner reads of raw eta_diag below therefore omit E_self under XL+quartic+gself -- a
  // conditioning choice only, left as is; XL+gself is not a validated combination.)
  // B3.3: the field gate quartic_gate is otherwise only recomputed in BO pre_force -> frozen at its cold-start
  // allocation value (1.0) for the life of an XL run (audit finding #3/gate). compute_quartic_gate() is a no-op
  // (gate≡1, byte-identical) when fld0<=0 or no q0 reference, so only pay its one-FFT PPPM compute_vector cost
  // (via coulomb_field) when the gate is actually active.
  if (lr_quartic && quartic_fld0 > 0.0) compute_quartic_gate();
  xl_chargeforce();                               // qddot = -g/q_mass (g = pchi + H·qs + quartic) at new q/positions
  for (int i=0; i<atom->nlocal; i++) if (mask[i]&groupbit) qdot[i] += 0.5*dt*qddot[i];
  // CHARGE THERMOSTAT on the velocities. Pure friction (xl_Tq=0) is a one-sided drag -> drains energy (the NVE
  // freeze) and floors the over-pol (no FD balance). With xl_Tq>0 add the FDT noise (variant 1): the
  // Ornstein-Uhlenbeck update qdot = c1*qdot + c2*gauss, c2=sqrt((1-c1^2) kB T_q / q_mass), drives the charge
  // DOF to temperature T_q with ZERO net drain (noise balances friction). xl_Tq=0 -> c2=0 -> exactly qdot*=c1.
  double fric = exp(-dt/q_tdamp);                 // = c1
  if (xl_Tq > 0.0 && xl_random) {
    // FDT amplitude c2 = sqrt((1-c1²)·kB·T_q / mass). force->boltz is unit-style-aware (eV/K metal, kcal/mol/K
    // real) and q_mass is a native-unit fix_modify value (R1) -> self-consistent, no ev_scale (A7). Mass-weighted:
    // the mass is per-atom (q_mass·eta[type]) so c2 becomes per-atom -> factor out the mass-independent part.
    int *type = atom->type;
    const double c2_scalar = sqrt((1.0 - fric*fric) * force->boltz * xl_Tq / q_mass);   // EXACT original expr (byte-id)
    const double c2base = xl_masswt ? sqrt((1.0 - fric*fric) * force->boltz * xl_Tq) : 0.0;   // = c2·sqrt(mass) (masswt only)
    for (int i=0;i<atom->nmax;i++) q_p[i]=0.0;    // reuse q_p (free after xl_chargeforce) as the noise scratch
    for (int i=0;i<atom->nlocal;i++) if (mask[i]&groupbit) {
      // mass-weighted FDT amplitude uses the SAME per-atom mass as qddot: m_i = q_mass·eta_eff (eta_diag under
      // lr_quartic, mirroring xl_chargeforce's inv_mass) so noise and drag balance at T_q on the true mass.
      double eeff = lr_quartic ? eta_diag[i] : eta[type[i]];
      double c2 = xl_masswt ? c2base/sqrt(q_mass*eeff) : c2_scalar;
      q_p[i] = c2 * xl_random->gaussian(); }
    project_neutral(q_p);                         // per-molecule-neutral noise -> conserves Σ_mol q (like the drag)
    for (int i=0;i<atom->nlocal;i++) if (mask[i]&groupbit) qdot[i] = fric*qdot[i] + q_p[i];
  } else {
    for (int i=0; i<atom->nlocal; i++) if (mask[i]&groupbit) qdot[i] *= fric;   // pure friction (byte-id default)
  }
}

