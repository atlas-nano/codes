// clang-format off
/* ----------------------------------------------------------------------
   samQEq (fix qeq/sam): fix_modify keywords + on-site self-energy (compute_scalar).
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
#include "random_mars.h"   // xl_random (the Langevin charge thermostat)

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ----------------------------------------------------------------------
   compute_scalar: the on-site samQEq self-energy (the diagonal of the QEq
   energy functional), summed over the fix group:

       E_self = Σ_i (χ_b(i)·q_i + ½ η_i q_i²) [eV]

   χ_b(i) = χ[type]. η_i is the solve's harmonic
   diagonal, eself_diag_of (gself-folded; the adaptive ridge is a crash-guard, not physical, so it is
   excluded). The OFF-diagonal Coulomb ½Σ_{i≠j} J_ij q_i q_j
   is already accounted for in pe by the coul/shield/intra + coul/long pairs
   and kspace, so pe + E_self = the complete samQEq potential energy.
   Position-independent ⇒ zero force / zero virial.
   CONSISTENCY NOTES (forceless, so dynamics/density/pressure are unaffected):
   (1) In the Ewald path q* is variational w.r.t. pe + E_self (the off-diagonal
       solve operator matches the i≠j Coulomb in pe; the reciprocal grid self is
       removed in add_reciprocal so the diagonal == eself_diag_of). The DSF path
       (lr_ewald=0) instead carries a per-atom e_self ∝ q² inside pe (coul/dsf)
       that the bare-eta solve omits ⇒ a forceless absolute-energy offset there.
   (2) Under an ACTIVE adaptive ridge the SOLVE uses (eself_diag_of+ridge) ⇒ q* is the
       ridged/depolarized charge, but E_self here evaluates ½ eself_diag_of q*² ⇒ when the
       ridge fires the reported energy is NOT the exact functional minimum either.
-------------------------------------------------------------------------*/
double FixQEqSam::compute_scalar()
{
  const double *q = atom->q;
  const int *type = atom->type;
  const int *mask = atom->mask;
  const int nlocal = atom->nlocal;
  // The curvature reported here is the solve's own harmonic diagonal, eself_diag_of (fix_qeq_sam_lr.cpp): the
  // gself-folded eta. Pre-quartic; anh_energy_add below supplies the quartic. The forces integrate pe + E_self only
  // when this equals the solve diagonal.
  double e = 0.0;
  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      const double qi = q[i];
      // T3 : optional solve-basis quadratic (`fix_modify eselfref qs`): the lr/base solves are
      // variational in qs = q - q0, so reporting 1/2 eta qs^2 (gradient chi + eta*qs =
      // the solve's stationarity) makes etotal the conserved functional for q0!=0 models (the Drude
      // melt's apparent NVE band was Sum eta*q0*q riding on the legacy physical-q report — see the
      // eself_qs block in fix_qeq_sam.h). Default eself_qs=1 since.
      const double qref = eself_qs ? q0[type[i]] : 0.0;
      const double qsi = qi - qref;
      e += chi_b(i) * qi + 0.5 * eself_diag_of(i, true) * qsi * qsi;   // loop is already in-group
      // staircase + global gated quartic: the shared on-site anharmonic model (fix_qeq_sam.h, Tier-A #1)
      // (NB stays on the PHYSICAL q in either basis — the quartic bounds physical over-polarization.)
      anh_energy_add(e, qi, type[i], mask[i] & quartic_groupbit, quartic_gate[i]);
    }
  double eall = 0.0;
  MPI_Allreduce(&e, &eall, 1, MPI_DOUBLE, MPI_SUM, world);
  return eall;
}

/* ----------------------------------------------------------------------
   fix_modify keywords of fix qeq/sam.
-------------------------------------------------------------------------*/
int FixQEqSam::modify_param(int narg, char **arg)
{

  if (strcmp(arg[0], "xl") == 0) {     // fix_modify ID xl <q_mass> <q_tdamp> [<T_q> [<seed>]] [masswt] -> extended-Lagrangian charges
    if (narg < 3) error->all(FLERR, "Illegal fix_modify xl: need <q_mass> <q_tdamp> [<T_q> [<seed>]] [masswt]");
    if (!lr_ewald) error->all(FLERR, "fix_modify xl requires lr_ewald=1 (long-range Ewald mode)");
    lr_xl = 1;
    xl_masswt = 0;                              // reset (a re-issue without the keyword clears mass-weighting)
    q_mass  = utils::numeric(FLERR, arg[1], false, lmp);
    q_tdamp = utils::numeric(FLERR, arg[2], false, lmp);
    // optional 4th arg: FD-balanced Langevin charge-thermostat target temperature T_q (K). 0 = pure friction (byte-id).
    // optional 5th arg: RNG seed for the per-proc Langevin noise stream (default 54321, matching the prior hardcode).
    xl_Tq = 0.0; int nret = 3; int seed = 54321;
    if (narg >= 4 && strcmp(arg[3], "masswt") != 0) { xl_Tq = utils::numeric(FLERR, arg[3], false, lmp); nret = 4; }
    if (narg >= 5 && strcmp(arg[4], "masswt") != 0) { seed  = utils::inumeric(FLERR, arg[4], false, lmp); nret = 5; }
    // optional trailing `masswt` keyword: reinterpret <q_mass> as m0 and set per-atom mass m_i = m0·eta[type]
    // (uniform fictitious frequency ω0=√(1/m0) across all types; the Car-Parrinello cure for a wide eta spectrum).
    if (narg > nret && strcmp(arg[nret], "masswt") == 0) { xl_masswt = 1; nret++; }
    delete xl_random; xl_random = nullptr;
    if (xl_Tq > 0.0) xl_random = new RanMars(lmp, seed + comm->me);   // per-proc FDT noise RNG (noise is local per-atom)
    if (comm->me == 0) {
      const char *mw = xl_masswt ? " [MASS-WEIGHTED: per-atom mass m0*eta[type], m0=q_mass, uniform charge freq]" : "";
      if (xl_Tq > 0.0)
        utils::logmesg(lmp, "samqeq: extended-Lagrangian charge dynamics ON (q_mass={:.4g} eV ps^2/e^2,"
                            "q_tdamp={:.4g} ps); FD-balanced Langevin charge thermostat T_q={:.4g} K (noise projected per-molecule){}\n",
                            q_mass, q_tdamp, xl_Tq, mw);
      else
        utils::logmesg(lmp, "samqeq: extended-Lagrangian charge dynamics ON (q_mass={:.4g} eV ps^2/e^2,"
                            "q_tdamp={:.4g} ps); pure friction (T_q=0); charges propagate by velocity-Verlet (no per-step solve){}\n",
                            q_mass, q_tdamp, mw);
    }
    return nret;
  }
  if (strcmp(arg[0], "aspc") == 0) {   // fix_modify ID aspc <n_corr> [<order k>] | off | rtol <val> | saddle allow|refuse (#16)
    if (narg < 2) error->all(FLERR, "Illegal fix_modify aspc: need <n_corr> [<order>] | off | rtol <val>");
    if (strcmp(arg[1], "off") == 0) { aspc_on = 0; if (comm->me==0) utils::logmesg(lmp,"samqeq: ASPC off (Born-Oppenheimer)\n"); return 2; }
    if (strcmp(arg[1], "saddle") == 0) {   // : opt in/out of ASPC on the ACKS2 saddle (default: refused)
      if (narg < 3) error->all(FLERR, "fix_modify aspc saddle: need allow|refuse");
      if (strcmp(arg[2], "allow") == 0)       aspc_saddle_allow = 1;
      else if (strcmp(arg[2], "refuse") == 0) aspc_saddle_allow = 0;
      else error->all(FLERR, "fix_modify aspc saddle: use allow|refuse");
      if (comm->me == 0)
        utils::logmesg(lmp, "samqeq: ASPC on the ACKS2 saddle = {}{}\n",
                       aspc_saddle_allow ? "ALLOWED" : "REFUSED (default)",
                       aspc_saddle_allow ? " -- behind the accept/reject gate. MEASURED COST on the gated"
                                           "saddle: 581x worse charge conservation and ~30% corruption of the"
                                           "CT observable for 9.6% wall-clock. You almost certainly want `aspc off`."
                                         : "; a deck that asks for ASPC there runs exact Born-Oppenheimer instead");
      return 3;
    }
    if (strcmp(arg[1], "rtol") == 0) {   // accept the capped corrector only if its rel-residual < val (else BO). SCOPE 2c.
      if (narg < 3) error->all(FLERR, "fix_modify aspc rtol: need <value>");
      aspc_rtol = utils::numeric(FLERR, arg[2], false, lmp);
      if (aspc_rtol <= 0.0) error->all(FLERR, "fix_modify aspc rtol: value must be > 0");
      if (comm->me == 0) utils::logmesg(lmp, "samqeq: ASPC accept rel-residual tol = {:.3g} (corrector falls to BO above this)\n", aspc_rtol);
      return 3;
    }
    // ASPC works on the per-molecule-neutral subspace.
    aspc_on = 1;
    aspc_ncorr = utils::inumeric(FLERR, arg[1], false, lmp);
    if (narg > 2) aspc_korder = utils::inumeric(FLERR, arg[2], false, lmp);
    if (aspc_ncorr < 1) error->all(FLERR, "fix_modify aspc: n_corr must be >= 1");
    if (aspc_korder < 1 || aspc_korder > 4) error->all(FLERR, "fix_modify aspc: order must be 1..4");
    aspc_setup();
    if (comm->me == 0)
      utils::logmesg(lmp, "samqeq: ASPC q-direct predictor-corrector ON (order k={}, n_corr={}, omega={:.4g},"
                          "rtol={:.3g}); fixed-iteration corrector off a time-reversible predictor, accept only if"
                          "rel-resid<rtol else BO (BO<->XL middle ground, #16)\n",
                          aspc_korder, aspc_ncorr, aspc_omega, aspc_rtol);
    return (narg > 2) ? 3 : 2;
  }
  if (strcmp(arg[0], "eselfref") == 0) {   // T3 : fix_modify ID eselfref qs|q — self-energy reporting basis
    if (narg < 2) error->all(FLERR, "Illegal fix_modify eselfref: need qs|q");
    if (strcmp(arg[1], "qs") == 0) eself_qs = 1;
    else if (strcmp(arg[1], "q") == 0) eself_qs = 0;
    else error->all(FLERR, "fix_modify eselfref: unknown basis {} (use qs|q)", arg[1]);
    if (comm->me == 0)
      utils::logmesg(lmp, "samqeq: on-site self-energy reporting basis = {} (reporting-only, no forces;"
                          "qs basis makes the monitored etotal the solve's conserved functional for"
                          "q0!=0 models — T3)\n", eself_qs ? "qs = q - q0" : "q (legacy)");
    return 2;
  }
  if (strcmp(arg[0], "cutoff") == 0) {    // fix_modify ID cutoff on|off (taper-cutoff QEq solve, no reciprocal)
    if (narg < 2) error->all(FLERR, "Illegal fix_modify cutoff: use `cutoff on|off`");
    if (strcmp(arg[1], "on") == 0)       lr_nrecip = 1;
    else if (strcmp(arg[1], "off") == 0) lr_nrecip = 0;
    else error->all(FLERR, "fix_modify cutoff: use on|off");
    if (lr_nrecip) lr_calibrated = 0;     // drop any stale reciprocal calibration; the cutoff solve never uses it
    if (comm->me == 0)
      utils::logmesg(lmp, "samqeq: QEq solve long-range = {} -- the SOLVE operator is {} (forces unchanged)\n",
                     lr_nrecip ? "TAPER-CUTOFF at swb (no reciprocal)" : "full Ewald (reciprocal)",
                     lr_nrecip ? "strictly short-ranged" : "all-to-all");
    return 2;
  }
  if (strcmp(arg[0], "precond") == 0) {   // fix_modify ID precond ilu|diag (#20 metal-limit solver)
    if (narg < 2) error->all(FLERR, "Illegal fix_modify precond: need ilu|diag");
    if (strcmp(arg[1], "ilu") == 0) {
      precond_mode = 1; ilu_valid = 0;             // #16: (re)config ⇒ rebuild the ILU on the next solve
      if (narg > 2) { ilu_droptol = utils::numeric(FLERR, arg[2], false, lmp); }   // ILUT drop tolerance
    }
    else if (strcmp(arg[1], "diag") == 0) precond_mode = 0;
    else if (strcmp(arg[1], "mol") == 0) {   // (#30 item 5): molecular block-Jacobi (q-direct CG)
      precond_mode = 2;
      if (lr_ewald < 2)
        error->all(FLERR, "fix_modify precond mol needs the Ewald-split operator (lr_ewald>=2); the"
                          "block is built from the same J_shield - erf(ar)/r the solve uses");
    }
    else error->all(FLERR, "fix_modify precond: unknown mode {} (use ilu|diag|mol)", arg[1]);
    if (comm->me == 0)
      utils::logmesg(lmp, "samqeq: preconditioner = {} (saddle BiCGStab), drop_tol={:.2g}\n",
                          precond_mode == 1 ? "block-ILUT saddle [centralized]" :
                          (precond_mode == 2 ? "molecular block-Jacobi (per-molecule Cholesky)" : "diagonal"),
                          ilu_droptol);
    return (precond_mode == 1 && narg > 2) ? 3 : 2;
  }
  if (strcmp(arg[0], "ionfield") == 0) {   // fix_modify ID ionfield on|off
    // Fixed non-group ION charges -> QEq RHS, through the SAME channel as `drude`
    // (add_fixed_charge_field), WITHOUT the own-shell special case and WITHOUT needing a `fix drude`.
    // `off` is the explicit acknowledgement that an ion-blind solve is intended — it silences
    // check_ionfield_consistency() and changes no physics (the same contract as `drude off`).
    if (narg < 2 || (strcmp(arg[1], "on") != 0 && strcmp(arg[1], "off") != 0))
      error->all(FLERR, "fix_modify ionfield: need `ionfield on` or `ionfield off`");
    if (strcmp(arg[1], "off") == 0) {
      ionfield_flag = 0; ionfield_ack = 1; ionfield_explicit = 1;
      if (comm->me == 0)
        error->warning(FLERR, "samqeq: ionfield explicitly OFF — fixed non-group charged atoms (ions) are in"
                              "the FORCES but NOT in the QEq solve (ion-blind Hamiltonian, acknowledged by the"
                              "deck). Valid as a control or a legacy reproduction; say so when quoting physics.");
      return 2;
    }
    ionfield_flag = 1; ionfield_ack = 0; ionfield_explicit = 1;
    if (comm->me == 0)
      utils::logmesg(lmp, "samqeq: ionfield ON — fixed non-group charges enter the QEq RHS as a shielded field"
                          "(real + reciprocal, Ewald-complete; same channel as the Drude shell field, no own-shell"
                          "case). NOTE: this supplies the ion's electrostatic field only — necessary but NOT"
                          "sufficient (~56% of the QM dipole response at contact)\n");
    return 2;
  }
  if (strcmp(arg[0], "shieldcheck") == 0) {
    // Change A escape hatch: `fix_modify ID shieldcheck <tol> | off | on`. <tol> = the channel-1
    // residual tolerance in eV per unit charge pair (default 0.02; see check_shield_consistency()'s
    // banner for the repo-wide separation). `off` disables ALL four channels -- it makes a known
    // force<->solve kernel split runnable, so it warns loudly rather than passing quietly.
    if (narg < 2) error->all(FLERR, "Illegal fix_modify shieldcheck: use `shieldcheck <tol>|off|on`");
    if (strcmp(arg[1], "off") == 0) {
      shieldchk_on = 0;
      if (comm->me == 0)
        error->warning(FLERR, "samqeq: shieldcheck OFF -- the pair<->fix shielding-kernel consistency"
                              "assertion is disabled; forces and the charge solve may integrate different"
                              "kernels (the melt-NVE-runaway bug class)");
    } else if (strcmp(arg[1], "on") == 0) {
      shieldchk_on = 1;
    } else {
      shieldchk_tol = utils::numeric(FLERR, arg[1], false, lmp);
      if (shieldchk_tol <= 0.0)
        error->all(FLERR, "fix_modify shieldcheck: tol must be > 0 eV (use `off` to disable the check)");
      shieldchk_on = 1;
      if (comm->me == 0)
        utils::logmesg(lmp, "samqeq: shieldcheck range tolerance = {:.4g} eV/e^2 (default 0.02)\n",
                       shieldchk_tol);
    }
    return 2;
  }
  if (strcmp(arg[0], "recip_probes") == 0) {   // fix_modify ID recip_probes <K>
    // K=0 keeps the LEGACY single-pair calibration (decomposition- and atom-order-dependent).
    // K>=4 selects K deterministic probe pairs and averages, which is order-independent by construction.
    // The default is K=16 (fix_qeq_sam.h).
    if (narg < 2) error->all(FLERR, "Illegal fix_modify recip_probes: use `recip_probes <K>` (0 = legacy)");
    lr_recip_probes = utils::inumeric(FLERR, arg[1], false, lmp);
    lr_recip_probes_user = 1;                              // : explicit => <4 atoms is an error, not a fallback
    if (lr_recip_probes < 0 || lr_recip_probes == 1 || lr_recip_probes == 2 || lr_recip_probes == 3)
      error->all(FLERR, "fix_modify recip_probes: use 0 (legacy single pair) or >=4");
    lr_calibrated = 0;                                     // re-arm so the new route measures
    lr_self_first = 0.0;                                   // : a route switch starts a new drift origin (R1 guard)
    if (comm->me == 0)
      utils::logmesg(lmp, "samqeq: recip_self calibration = {}\n",
                     lr_recip_probes ? "deterministic multi-probe (K lowest global tags)"
                                     : "LEGACY single pair (ilist order; decomposition-dependent)");
    return 2;
  }

  if (strcmp(arg[0], "recip_self") == 0) {   // fix_modify ID recip_self <raw> | auto
    // Pin the PPPM grid reciprocal self-term instead of measuring it. The measurement picks a probe
    // pair by local ilist order, so it depends on the MPI decomposition AND on atom ordering in the
    // data file: measured 0.35516/0.35516/0.35311/0.33655 at np 1/2/4/8 on one configuration, moving
    // the liquid dipole 2.4377 -> 2.1055 D, and 0.33655 -> 0.34339 from permuting the data file alone.
    // Pinning makes the operator reproducible and is how a published run is re-run exactly.
    if (narg < 2) error->all(FLERR, "Illegal fix_modify recip_self: use `recip_self <raw value> | auto | peratom`");
    if (strcmp(arg[1], "peratom") == 0) {   // R4: exact per-atom grid self (no calibration)
      if (strstr(style, "/kk"))
        error->all(FLERR, "fix_modify recip_self peratom is not supported by fix {} (device add_reciprocal carries a scalar self-term)", style);
      lr_self_peratom = 1; lr_self_pinned = 0; lr_calibrated = 0; lr_self_first = 0.0;
      if (comm->me == 0)
        utils::logmesg(lmp, "samqeq: recip_self = EXACT PER-ATOM grid self-coefficient (pppm/samqeq compute_self_peratom):"
                            "no probe calibration, no drift guard; refreshed at every solve\n");
      return 2;
    }
    if (strcmp(arg[1], "auto") == 0) {
      lr_self_peratom = 0;
      lr_self_pinned = 0; lr_calibrated = 0; lr_self_first = 0.0;   // : un-pinning re-measures from a fresh drift origin
      if (comm->me == 0) utils::logmesg(lmp, "samqeq: recip_self un-pinned (will be re-measured)\n");
      return 2;
    }
    lr_self_meas = utils::numeric(FLERR, arg[1], false, lmp);
    lr_self_pinned = 1; lr_calibrated = 1; lr_self_peratom = 0;
    // force->qqrd2e is assigned ONLY in Force::init(), which runs at setup -- i.e. AFTER every
    // fix_modify is parsed -- so it is still 0 here and the eV/e column printed 0.0000. Use the
    // identical expression init() will use, from members the `units` command has already set
    // (qqr2e at units time, dielectric = 1.0 from the Force ctor). Same number, valid earlier.
    if (comm->me == 0)
      utils::logmesg(lmp, "samqeq: grid reciprocal self-term PINNED at recip_self={:.5f} (raw) = {:.4f} eV/e"
                          "(not measured; operator is decomposition- and atom-order-independent)\n",
                     lr_self_meas, lr_self_meas*(force->qqr2e/force->dielectric)/ev_scale);
    return 2;
  }

  if (strcmp(arg[0], "gself") == 0) {     // fix_modify ID gself on|off [width <A>] [types <t1> <t2> ...]
    if (narg < 2) error->all(FLERR, "Illegal fix_modify gself: use `gself on|off [width <Angstrom>] [types <t1> ...]`");
    if (strcmp(arg[1], "on") == 0) gself_flag = 1;
    else if (strcmp(arg[1], "off") == 0) gself_flag = 0;
    else error->all(FLERR, "fix_modify gself: expected on|off, got {}", arg[1]);
    int used = 2;
    if (narg > used && strcmp(arg[used], "width") == 0) {
      if (narg < used + 2)
        error->all(FLERR, "fix_modify gself width: need <Angstrom>");
      gself_width = utils::numeric(FLERR, arg[used+1], false, lmp);
      if (gself_width <= 0.0) error->all(FLERR, "fix_modify gself width must be > 0");
      used += 2;
    }
    // : `types <t1> <t2> ...` restricts the self-energy to those atom types. Consumes to the end
    // of the keyword list. Omitted => every type (legacy, byte-identical).
    if (narg > used && strcmp(arg[used], "types") == 0) {
      const int nt = atom->ntypes;
      if (narg < used + 2) error->all(FLERR, "fix_modify gself types: need at least one type");
      memory->destroy(gself_type);
      memory->create(gself_type, nt+1, "samqeq:gself_type");
      for (int t = 0; t <= nt; t++) gself_type[t] = 0;
      gself_ntype = nt+1;
      int n = 0;
      for (int k = used+1; k < narg; k++) {
        if (!utils::is_integer(arg[k])) break;
        int t = utils::inumeric(FLERR, arg[k], false, lmp);
        if (t < 1 || t > nt)
          error->all(FLERR, "fix_modify gself types: type {} out of range 1..{}", t, nt);
        gself_type[t] = 1; n++; used++;
      }
      if (n == 0) error->all(FLERR, "fix_modify gself types: no valid type given");
      used++;   // the `types` keyword itself
      if (comm->me == 0) {
        std::string ts;
        for (int t = 1; t <= nt; t++) if (gself_type[t]) ts += fmt::format(" {}", t);
        utils::logmesg(lmp, "samqeq: gself RESTRICTED to type(s){} -- the self-energy is a lattice"
                            "regulariser (metal sites, where eta is a bulk TF quantity with no width"
                            "in it) and a DOUBLE COUNT on molecular sites, whose eta was fit to the"
                            "IP/EA of a finite-width atom\n", ts);
      }
    } else if (narg > used) {
      error->all(FLERR, "fix_modify gself: unknown trailing keyword {} (use `width <A>` and/or"
                        "`types <t1> ...`)", arg[used]);
    }
    if (comm->me == 0)
      utils::logmesg(lmp, "samqeq: gself {} -- finite-width charge self-energy K_e*sqrt(2a/pi) {} the on-site"
                          "diagonal (width {}); folded into eta from eta0 at init()\n",
                     gself_flag ? "ON" : "OFF", gself_flag ? "ADDED to" : "removed from",
                     fmt::format("{:.3f} A", gself_width)
                       + (gself_type ? std::string(" for the listed type(s) only") : std::string(" for all types")));
    return used;
  }

  if (strcmp(arg[0], "shieldpair") == 0) {   // fix_modify ID shieldpair <ti> <tj> <R_ij>|off (option A)
    if (narg < 4) error->all(FLERR, "Illegal fix_modify shieldpair: use `shieldpair <ti> <tj> <R_ij Angstrom>|off`");
    const int ti = utils::inumeric(FLERR, arg[1], false, lmp), tj = utils::inumeric(FLERR, arg[2], false, lmp);
    if (ti < 1 || tj < 1 || ti > atom->ntypes || tj > atom->ntypes)
      error->all(FLERR, "fix_modify shieldpair: type pair ({},{}) out of range 1..{}", ti, tj, atom->ntypes);
    shield_rpair_ensure();
    double r = 0.0;
    if (strcmp(arg[3], "off") != 0) {
      r = utils::numeric(FLERR, arg[3], false, lmp);
      if (r <= 0.0) error->all(FLERR, "fix_modify shieldpair: R_ij must be > 0 (or `off`)");
    }
    shield_rpair[ti][tj] = shield_rpair[tj][ti] = r;
    if (comm->me == 0) {
      const double rdef = sqrt(0.5*(gamma[ti]*gamma[ti] + gamma[tj]*gamma[tj]));
      utils::logmesg(lmp, "samqeq: shieldpair ({},{}) pair Gaussian radius {} (combination-rule value {:.4f} A;"
                          "a_ij = sqrt(lambda)/(2 R)); pushed to pair coul/shield/intra at init\n",
                     ti, tj, r > 0.0 ? fmt::format("= {:.4f} A", r) : std::string("OFF (rule)"), rdef);
    }
    return 4;
  }
  if (strcmp(arg[0], "shield") == 0) {    // fix_modify ID shield gaussian|cbrt|slater [<lambda>] [2s <type>...]
    if (narg < 2) error->all(FLERR, "Illegal fix_modify shield: use `shield gaussian|cbrt|slater [lambda|2s ...]`");
    if (strcmp(arg[1], "gaussian") == 0)     shield_gauss = SHIELD_GAUSSIAN;
    else if (strcmp(arg[1], "pqeq") == 0)    // : renamed `gaussian`; alias removed
      error->all(FLERR, "fix_modify shield:" "the Gaussian shielding kernel keyword `pqeq` was renamed `gaussian` in samQEq (same kernel, same numbers); replace `pqeq` with `gaussian` in the deck");
    else if (strcmp(arg[1], "cbrt") == 0)    shield_gauss = SHIELD_CBRT;
    else if (strcmp(arg[1], "slater") == 0)  shield_gauss = SHIELD_SLATER;
    else error->all(FLERR, "fix_modify shield: unknown mode {} (use gaussian|cbrt|slater)", arg[1]);
    int used = 2;
    if (shield_gauss == SHIELD_GAUSSIAN) {
      if (narg > 2) { shield_lambda = utils::numeric(FLERR, arg[2], false, lmp); used = 3; }
    } else if (shield_gauss == SHIELD_SLATER) {
      // Slater per-type 2s(O/M)-vs-1s(H) form-factor selector: `shield slater 2s <type> <type> ...` -- the
      // list runs to the END of this fix_modify invocation (no closing keyword), mirroring the spec's
      // documented syntax; issue "2s" as its own separate fix_modify command if other keywords follow it.
      if (narg > 2) {
        if (strcmp(arg[2], "2s") != 0)
          error->all(FLERR, "fix_modify shield slater: unknown trailing keyword {} (use `2s <type>...`)", arg[2]);
        for (int k = 3; k < narg; k++) {
          int t = utils::inumeric(FLERR, arg[k], false, lmp);
          if (t < 1 || t > atom->ntypes)
            error->all(FLERR, "fix_modify shield slater 2s: bad type {} (ntypes={})", t, atom->ntypes);
          is2s[t] = 1;
        }
        used = narg;
      }
    }
    if (comm->me == 0) {
      const char *modestr = shield_gauss == SHIELD_SLATER ? "Slater J(r) (Rick JCP101,6141)" :
                            (shield_gauss == SHIELD_GAUSSIAN  ? "PQEq Gaussian erf(a_ij r)/r" : "cbrt J_shield");
      const char *colstr  = shield_gauss == SHIELD_SLATER ? "zeta (Slater exponent, 1/Angstrom)" :
                            (shield_gauss == SHIELD_GAUSSIAN  ? "Rc (Gaussian radius)" : "gamma");
      utils::logmesg(lmp, "samqeq: lr-solve shielding = {} (param 4th col = {}); routed through compute_H +"
                          "add_fixed_charge_field + the legacy gas path (calc_Hval)\n", modestr, colstr);
    }
    return used;
  }
  if (strcmp(arg[0], "iondamp") == 0) {   // fix_modify ID iondamp <typeI> <typeJ> <b> [<n>] | off (#5 damped interionic kernel)
    // Tang-Toennies-damp the OFF-DIAGONAL shielded Coulomb between designated (ion) type pairs:
    // J_damp(r) = f_n(b r)·J_shield(r) -- bounds the contact charge-transfer sloshing (qZn->+4/qCl->-2,
    // an ion-pair benchmark) while leaving the large-r Coulomb/Ewald complement untouched. Repeatable
    // per type pair (LAMMPS type wildcards, e.g. `iondamp 3*5 3*5 2.0`); stored SYMMETRIC in both triangles
    // (A3 reversibility). The matching pair-side keyword (pair_style coul/shield/intra ... iondamp ...) MUST
    // carry the same pairs/b/n for force<->solve consistency. b is 1/Angstrom (length^-1 family, NO ev_scale
    // -- A7, same convention as the slater zeta)..
    if (narg < 2) error->all(FLERR, "Illegal fix_modify iondamp: need <typeI> <typeJ> <b> [<n>] | off");
    if (strcmp(arg[1], "off") == 0) {
      lr_iondamp = 0;
      if (iondamp_b)
        for (int i = 0; i <= atom->ntypes; i++)
          for (int j = 0; j <= atom->ntypes; j++) { iondamp_b[i][j] = 0.0; iondamp_n[i][j] = 4; }
      if (comm->me == 0) utils::logmesg(lmp, "samqeq: interionic TT damping (iondamp) OFF -- all pairs cleared\n");
      return 2;
    }
    // HOST-ONLY (device-fallback precedent, like slater): the kokkos fix's device H-build/matvec hardcode the
    // undamped kernels -- reject at parse rather than silently solving an undamped operator on the device.
    if (kokkosable)
      error->all(FLERR, "fix_modify iondamp is host-only (the qeq/sam/kk device H-build does not apply the"
                        "TT damping); use the host fix qeq/sam");
    if (narg < 4) error->all(FLERR, "Illegal fix_modify iondamp: need <typeI> <typeJ> <b> [<n>] | off");
    const int nt = atom->ntypes;
    if (!iondamp_b) {                     // lazy (nt+1)^2 alloc; freed in the dtor
      memory->create(iondamp_b, nt+1, nt+1, "samqeq:iondamp_b");
      memory->create(iondamp_n, nt+1, nt+1, "samqeq:iondamp_n");
      for (int i = 0; i <= nt; i++)
        for (int j = 0; j <= nt; j++) { iondamp_b[i][j] = 0.0; iondamp_n[i][j] = 4; }
    }
    int ilo, ihi, jlo, jhi;
    utils::bounds(FLERR, arg[1], 1, nt, ilo, ihi, error);
    utils::bounds(FLERR, arg[2], 1, nt, jlo, jhi, error);
    double bdamp = utils::numeric(FLERR, arg[3], false, lmp);
    if (bdamp <= 0.0) error->all(FLERR, "fix_modify iondamp: b must be > 0 (use `iondamp off` to clear)");
    int used = 4, ttn = 4;                // default TT order n=4 (the CL&Pol convention)
    if (narg > 4 && utils::is_integer(arg[4])) { ttn = utils::inumeric(FLERR, arg[4], false, lmp); used = 5; }
    if (ttn < 1 || ttn > 8) error->all(FLERR, "fix_modify iondamp: TT order n must be 1..8 (got {})", ttn);
    for (int i = ilo; i <= ihi; i++)
      for (int j = jlo; j <= jhi; j++) {  // fill BOTH triangles -> symmetric kernel by construction
        iondamp_b[i][j] = iondamp_b[j][i] = bdamp;
        iondamp_n[i][j] = iondamp_n[j][i] = ttn;
      }
    lr_iondamp = 1;
    if (comm->me == 0) {
      double dfdx; double fres = samqeq_tt_damp(bdamp*swb, ttn, dfdx);
      utils::logmesg(lmp, "samqeq: interionic TT damping ON -- types {}..{} x {}..{}: J_damp = f_{}(b r)*J_shield,"
                          "b={:.4g} 1/A (f at swb={:.2f}: {:.6f}); pair coul/shield/intra must carry the SAME"
                          "iondamp pairs (force<->solve consistency)\n", ilo, ihi, jlo, jhi, ttn, bdamp, swb, fres);
      if (1.0 - fres > 1.0e-3)
        error->warning(FLERR, "iondamp: TT damping not converged at the solve cutoff (1-f(b*swb)={:.2e} at"
                              "swb={:.2f}) -- the net Ewald kernel is discontinuous there; raise b (or swb)",
                              1.0 - fres, swb);
    }
    return used;
  }
  if (strcmp(arg[0], "bondsoft") == 0) {  // fix_modify ID bondsoft {off | <kappa> <bcut>} (standard ACKS2 bond-softness)
    if (narg < 2) error->all(FLERR, "Illegal fix_modify bondsoft: use `bondsoft off` or `bondsoft <kappa> <bcut>`");
    if (strcmp(arg[1], "off") == 0) {
      lr_bondsoft = 0;
      if (comm->me == 0) utils::logmesg(lmp, "samqeq: bond-softness ACKS2 = OFF\n");
      return 2;
    }
    if (narg < 3) error->all(FLERR, "fix_modify bondsoft: need <kappa> <bcut>");
    lr_bondsoft   = 1;
    bond_softness = utils::numeric(FLERR, arg[1], false, lmp);   // ACKS2 bond-softness scale (kappa)
    bcut_global   = utils::numeric(FLERR, arg[2], false, lmp);   // bond-softness cutoff (Å)
    lr_nrecip     = 1;                                           // bondsoft uses the Gaussian taper-cutoff H (noise-free)
    acks2_use_minres = 1;                                       // the indefinite KKT saddle needs MINRES (BiCGStab breaks)
    if (bcut) { memory->destroy(bcut); bcut = nullptr; }         // force re-fill at next setup
    if (bond_softness <= 0.0 || bcut_global <= 0.0)
      error->all(FLERR, "fix_modify bondsoft: kappa and bcut must be > 0");
    if (comm->me == 0)
      utils::logmesg(lmp, "samqeq: bond-softness ACKS2 ON (Verstraelen X-block, kappa={:.4g}, bcut={:.3g} Ang) +"
                          "Gaussian taper-cutoff H — gaps the FQ close-pair soft mode so the eta~J_shield(0)"
                          "operator stays solvable at low eta (target polarization)\n", bond_softness, bcut_global);
    return 3;
  }
  if (strcmp(arg[0], "quartic") == 0) {   // fix_modify ID quartic <group> <c4> [<c>] [<niter>] [<mix>] [<fld0>] [<qref>] ((B) near-crit cure)
    if (narg < 3) error->all(FLERR, "Illegal fix_modify quartic: use `quartic <group> <c4> [c] [niter] [mix] [fld0] [qref]`");
    int g = group->find(arg[1]);
    if (g < 0) error->all(FLERR, "fix_modify quartic: group {} does not exist", arg[1]);
    quartic_groupbit = group->bitmask[g];
    quartic_c4 = utils::numeric(FLERR, arg[2], false, lmp);
    if (narg > 3) quartic_c     = utils::numeric(FLERR, arg[3], false, lmp);
    if (narg > 4) quartic_niter = utils::inumeric(FLERR, arg[4], false, lmp);
    if (narg > 5) quartic_mix   = utils::numeric(FLERR, arg[5], false, lmp);
    if (narg > 6) quartic_fld0  = utils::numeric(FLERR, arg[6], false, lmp);   // (B') FIELD-GATE scale (eV/e); 0 = un-gated
    if (narg > 7) quartic_qref  = utils::numeric(FLERR, arg[7], false, lmp);   // CHARGE-OFFSET wall: c4 acts on max(|q|-qref,0); 0 = centered
    int stair = 0;   // C7: per-type IP-staircase (c3_type/c4_type) also auto-enables lr_quartic, else it's silently inert
    for (int t = 1; t <= atom->ntypes; t++)
      if ((c3_type && c3_type[t] != 0.0) || (c4_type && c4_type[t] != 0.0)) { stair = 1; break; }
    lr_quartic = (quartic_c4 != 0.0 || quartic_c != 0.0 || stair) ? 1 : 0;
    if (comm->me == 0)
      utils::logmesg(lmp, "samqeq: QUARTIC near-crit cure {} (group {} c4={:.4g} c={:.4g} niter={} mix={:.2f}"
                          "fld0={:.4g} qref={:.4g}); c4 term {}{}; damped-Picard SCF, q-only (no force term)\n",
                          lr_quartic ? "ON" : "OFF", arg[1], quartic_c4, quartic_c, quartic_niter, quartic_mix,
                          quartic_fld0, quartic_qref,
                          quartic_qref > 0.0 ? "ONE-SIDED WALL on (|q|-qref)" : "centered on q",
                          quartic_fld0 > 0.0 ? " ·gate(field)" : "");
    return narg;
  }
  if (strcmp(arg[0], "spikeguard") == 0) {  // fix_modify ID spikeguard <q_spike> <k_spike> | off (#22 XL)
    if (narg < 2) error->all(FLERR, "Illegal fix_modify spikeguard: need <q_spike> <k_spike> | off");
    if (strcmp(arg[1], "off") == 0) {
      xl_qspike = -1.0;
      if (comm->me == 0) utils::logmesg(lmp, "samqeq: XL charge spike-guard OFF\n");
      return 2;
    }
    if (narg < 3) error->all(FLERR, "Illegal fix_modify spikeguard: need <q_spike> <k_spike>");
    xl_qspike = utils::numeric(FLERR, arg[1], false, lmp);
    xl_kspike = utils::numeric(FLERR, arg[2], false, lmp);
    if (comm->me == 0)
      utils::logmesg(lmp, "samqeq: XL charge spike-guard q_spike={:.3g} e, k_spike={:.3g} eV/e^2"
                          "(soft restoring force on |q| above q_spike, per-molecule projected)\n",
                          xl_qspike, xl_kspike);
    return 3;
  }
  if (strcmp(arg[0], "solver") == 0) {  // fix_modify ID solver {cg | minres [<qcap>] | warmstart on|off} (#M-A,#20)
    if (narg < 2) error->all(FLERR, "Illegal fix_modify solver: need cg | minres [<qcap>] | warmstart on|off");
    if (strcmp(arg[1], "warmstart") == 0) {       // #20 charge predictor: seed qs from the previous solve
      if (narg < 3) error->all(FLERR, "Illegal fix_modify solver warmstart: need on|off");
      warmstart = (strcmp(arg[2], "on") == 0);
      if (!warmstart && strcmp(arg[2], "off") != 0) error->all(FLERR, "fix_modify solver warmstart: use on|off");
      if (comm->me == 0)
        utils::logmesg(lmp, "samqeq: charge predictor (warm-start from the previous converged charges) = {}"
                            "(#20; cuts BO iteration count; first/single-point solve still cold-starts)\n",
                            warmstart ? "ON" : "off");
      return 3;
    }
    use_minres = (strcmp(arg[1], "minres") == 0);
    if (!use_minres && strcmp(arg[1], "cg") != 0) error->all(FLERR, "fix_modify solver: must be cg or minres");
    int used = 2;
    minres_qcap = 0.0;
    if (use_minres && narg > 2) { minres_qcap = utils::numeric(FLERR, arg[2], false, lmp); used = 3; }
    if (!use_minres && narg > 2 && strcmp(arg[2], "nofallback") == 0) { lr_autofb = 0; used = 3; }  // disable CG->MINRES rescue
    if (comm->me == 0)
      // NB: MINRES is indefinite-safe as an algorithm, which is why it is kept for the ACKS2
      // saddles. It does NOT cure a "close-pair" indefiniteness of the plain QEq operator — that
      // mechanism was refuted by direct diagonalization; see the correction note at FixQEqSam::qeq_minres.
      utils::logmesg(lmp, "samqeq: lr solver = {}{} (symmetric, indefinite-safe){}\n",
                     use_minres ? "MINRES" : "CG",
                     (use_minres && minres_qcap > 0.0) ? fmt::format(", |q|-truncate at {:.3g}", minres_qcap) : "",
                     (!use_minres && lr_autofb) ? "; CG->MINRES auto-fallback on divergence" : "");
    return used;
  }
  if (strcmp(arg[0], "rnd") == 0) {  // fix_modify ID rnd {off | <qfreeze> [<maxconsec>]} (RIDE-NOT-DIE: ride through near-crit)
    if (narg < 2) error->all(FLERR, "Illegal fix_modify rnd: need off | <qfreeze> [<maxconsec>]");
    if (strcmp(arg[1], "off") == 0) {
      lr_qfreeze = 0.0; freeze_nconsec = 0;
      if (comm->me == 0) utils::logmesg(lmp, "samqeq: ride-not-die (rnd) = OFF\n");
      return 2;
    }
    lr_qfreeze = utils::numeric(FLERR, arg[1], false, lmp);
    if (lr_qfreeze <= 0.0) error->all(FLERR, "fix_modify rnd: <qfreeze> must be > 0 (or 'off')");
    int used = 2;
    if (narg > 2) { freeze_max = utils::inumeric(FLERR, arg[2], false, lmp); used = 3; }   // optional consecutive cap
    freeze_nconsec = 0;
    if (comm->me == 0)
      utils::logmesg(lmp, "samqeq: ride-not-die (rnd) ON (max|q|>{:.3g} -> hold previous charges; cap"
                          "{} consecutive steps) -- bounds energy injection from over-polarized near-critical"
                          "solves without committing them\n", lr_qfreeze, freeze_max);
    return used;
  }
  if (strcmp(arg[0], "ridge") == 0) {   // fix_modify ID ridge {off | <q_onset> <gain> | eig <lam_floor> [<m>] | local <q_onset> <gain>}
    if (narg < 2) error->all(FLERR, "Illegal fix_modify ridge: need off | <q_onset> <gain> | eig <lam_floor> [<m>] | local <q_onset> <gain>");
    if (strcmp(arg[1], "off") == 0) {                 // mode 0: legacy discrete x4 escalation
      ridge_mode = 0;
      if (comm->me == 0) utils::logmesg(lmp, "samqeq: ridge mode = discrete x4 escalation (legacy)\n");
      return 2;
    }
    if (strcmp(arg[1], "local") == 0) {               // mode 3 (#M-3): per-atom LOCAL ratchet ridge
      if (narg < 4) error->all(FLERR, "Illegal fix_modify ridge local: need <q_guard> <factor>");
      ridge_mode = 3;
      ridge_qonset = utils::numeric(FLERR, arg[2], false, lmp);   // per-atom runaway threshold (e)
      ridge_gain   = utils::numeric(FLERR, arg[3], false, lmp);   // escalation factor (>=... default 4 if <=1)
      if (comm->me == 0)
        utils::logmesg(lmp, "samqeq: ridge mode = LOCAL per-atom ratchet: where |q_i|>{:.3g} e, multiply that atom's"
                            "diagonal ridge by {:.3g} and re-solve (bounds runaway charges; bulk untouched)\n",
                            ridge_qonset, (ridge_gain > 1.0 ? ridge_gain : 4.0));
      return 4;
    }
    if (strcmp(arg[1], "eig") == 0) {                 // mode 2: Lanczos lambda_min-driven (b)
      if (narg < 3) error->all(FLERR, "Illegal fix_modify ridge eig: need <lam_floor> [<m>] [<every>] [<delta>]");
      ridge_mode = 2;
      lam_floor = utils::numeric(FLERR, arg[2], false, lmp);
      int used = 3;
      if (narg > 3) { nlanczos = utils::inumeric(FLERR, arg[3], false, lmp); used = 4; }
      if (nlanczos < 4 || nlanczos > 200) {           // estimate_lambda_min clamps to [4,200]; it used to do so silently
        const int mc = nlanczos < 4 ? 4 : 200;
        if (comm->me == 0)
          error->warning(FLERR, "samqeq ridge eig: m = {} is outside [4, 200]; using m = {}", nlanczos, mc);
        nlanczos = mc;
      }
      if (narg > 4) { ridge_every = utils::inumeric(FLERR, arg[4], false, lmp); used = 5; }  // λ_min recompute cadence
      if (ridge_every < 1) ridge_every = 1;
      ridge_delta = 0.0;                              // : Ritz-estimate error budget (engage target = floor + delta);
      if (narg > 5 && utils::is_double(arg[5])) {     // a re-issue without it resets it; consumed only if numeric
        ridge_delta = utils::numeric(FLERR, arg[5], false, lmp); used = 6;
        if (ridge_delta < 0.0) error->all(FLERR, "fix_modify ridge eig: delta must be >= 0");
      }
      lmin_step = -1;                                 // force a fresh estimate on the next solve
      if (comm->me == 0)
        utils::logmesg(lmp, "samqeq: ridge mode = Lanczos lambda_min: ridge = max(lr_ridge, {:.4g} + {:.4g} - lambda_min)"
                            "eV (floor + delta), {} Lanczos steps, recomputed every {} step(s) (continuous in config;"
                            "matrix kept PD). The Ritz estimate is an UPPER bound on lambda_min: an unconverged m can"
                            "only MISS a negative mode, never invent one -- m >= 60 recommended for interface cells;"
                            "delta is the estimate-error budget, so lambda_min(A+ridge) >= floor whenever the estimate"
                            "is within delta of the true value\n",
                            lam_floor, ridge_delta, nlanczos, ridge_every);
      return used;
    }
    if (narg < 3) error->all(FLERR, "Illegal fix_modify ridge: need <q_onset> <gain>");
    ridge_mode = 1;                                   // mode 1: smooth max|q| ramp (a)
    ridge_qonset = utils::numeric(FLERR, arg[1], false, lmp);
    ridge_gain   = utils::numeric(FLERR, arg[2], false, lmp);
    if (comm->me == 0)
      utils::logmesg(lmp, "samqeq: ridge mode = smooth max|q|: ridge = lr_ridge + {:.4g}*(max|q|-{:.3g})^2 eV\n",
                          ridge_gain, ridge_qonset);
    return 3;
  }
  return 0;
}

/* the electronegativity the solve uses at site i*/
double FixQEqSam::chi_b(int i) const
{
  return chi[atom->type[i]];
}
