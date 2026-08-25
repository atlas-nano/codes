/* -*- c++ -*- ----------------------------------------------------------
   fix_xpt.h  —  on-the-fly Two-Phase Thermodynamics (2PT)

   Syntax:
     fix ID group-ID xpt Nevery Nframes prefix [keyword value ...]

   Keywords:
     classical        also write classical thermo columns
     normalize        divide extensive outputs by number of atoms (or molecules)
     epsilon E        LJ energy ε [kcal/mol] — enables ħ* scaling; auto-detected if omitted
     sigma   S        LJ length σ [Å]       — enables ħ* scaling; auto-detected if omitted
     mass    M        LJ mass   m [g/mol]   — enables ħ* scaling; auto-detected if omitted
     volume  V        override box volume; V is a positive decimal or v_<name> (LAMMPS equal-style variable)

   Molecular mode keywords (require atom_style molecular):
     molecule         enable molecular 2PT: S decomposed into trans+rot+vib
     mode M           partitioning mode (string, case-insensitive):
                        1PT = all-solid harmonic baseline (no gas; f=0)  [diagnostic]
                        2PT = gas + solid hard-sphere partition       [default]
                        3PT = 2PT-rigorous + parameter-free cage (gas|cage|solid)
     refinement R     mode-dependent sub-method.
                      mode 2PT (3PT forces rigorous-HS as its base):
                        rigorous   = thermodynamically exact HS entropy  [default]
                        lin2003    = Lin 2003 +ln(Z) convention
                        desjarlais = moment-matched memory-function gas DoS
                                     (uses the lin2003 +lnZ entropy convention)
                        r2pt       = Sun 2017 revised-2PT (δ-tuned f, F_a-inclusive)
                      mode 3PT (parameter-free cage):
                        none       = bare parameter-free memory cage  [default]
     r2pt_delta D     Sun 2017 δ for refinement=r2pt (default 1.5)
     linear           flag molecule as linear; auto-set for diatomics
     symmetry <sym>   point-group Schoenflies symbol (default C1).  Drives
                      the rotational symmetry number σ_rot for the molecular
                      partition function (looked up from an internal table:
                      C1=1, C2v=2, C3v=3, D2h=4, D3h=6, D6h=12, Td=12,
                      Oh=24, etc.).
     show_split       also write gas/solid split for S_trans and S_rot

   Hard-sphere reference choice:
     (the entropy convention is no longer a standalone keyword — it is derived
      from `refinement`: rigorous→rigorous, lin2003/desjarlais→+lnZ, r2pt→rigorous)


   Thermo vector (0-based internally, 1-based in thermo_style f_ID[n]):

   Monoatomic (10 elements):
     f_ID[1]  S_q    quantum entropy      (J/mol/K  or  kB/atom)
     f_ID[2]  A_q    Helmholtz free energy (kJ/mol  or  ε/atom)
     f_ID[3]  E_q    internal energy      (kJ/mol  or  ε/atom)
     f_ID[4]  Cv_q   heat capacity        (J/mol/K or  kB/atom)
     f_ID[5]  D      self-diffusivity     (cm²/s   or  σ²/τ)
     f_ID[6]  f      fluidicity           (dimensionless)
     f_ID[7]  T_vac  temperature from VAC (K        or  ε/kB)
     f_ID[8]  ZPE_q  zero-point energy    (kJ/mol  or  ε/atom)
     f_ID[9]  μ_q    quantum chemical potential   (kJ/mol or ε/atom)
     f_ID[10] μ_c    classical chemical potential (kJ/mol or ε/atom)

   Molecular mode (14 elements, indices 1-7 same as above):
     f_ID[8]  S_trans  translational entropy
     f_ID[9]  S_rot    rotational entropy
     f_ID[10] S_vib    vibrational entropy
     f_ID[11] f_rot    rotational fluidicity
     f_ID[12] ZPE_q    zero-point energy
     f_ID[13] μ_q      quantum chemical potential   (kJ/mol or ε/mol)
     f_ID[14] μ_c      classical chemical potential (kJ/mol or ε/mol)

   Molecular + show_split (18 elements):
     f_ID[12] S_trans_gas    translational gas entropy
     f_ID[13] S_trans_solid  translational solid entropy
     f_ID[14] S_rot_gas      rotational gas entropy
     f_ID[15] S_rot_solid    rotational solid entropy
     f_ID[16] ZPE_q          zero-point energy
     f_ID[17] μ_q            quantum chemical potential
     f_ID[18] μ_c            classical chemical potential

   Chemical-potential definition (matching Lin 2003):
     μ_q = A_q + Σ_components (HSDF · R T · Z_HS(y) / 3)
     where Z_HS(y) is the hard-sphere compressibility factor at the 2PT-derived
     packing fraction:
       Z_CS(y)    = (1 + y + y² − y³) / (1−y)³           (default)
       Z_BMCSL(y) = (1 + y + y²)     / (1−y)³           (one-component limit)
     Solid component contributes no PV term (μ_solid = A_solid).

   References:
     Lin, Blanco & Goddard, J. Chem. Phys. 119, 11792 (2003)
     Lin, Maiti & Goddard, J. Phys. Chem. B 114, 8191 (2010)
     Pascal, Lin & Goddard, PCCP 13, 169 (2011)
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(xpt, FixXPT)
// clang-format on
#else

#ifndef LMP_FIX_XPT_H
#define LMP_FIX_XPT_H

#include "fix.h"
#include "fft3d_wrap.h"
#include <cstdio>
#include <string>
#include <vector>

namespace LAMMPS_NS {

class FixXPT : public Fix {
 public:
  FixXPT(class LAMMPS *, int, char **);
  ~FixXPT() override;

  int setmask() override;
  void init() override;
  void setup(int) override;
  void end_of_step() override;
  void post_run() override;
  double compute_vector(int) override;
  double memory_usage() override;
  // Persist multi-tau correlator accumulators across restart cycles.
  // v_ring + down_acc are NOT serialized (reset to zero on restart) to keep
  // the payload small — trades up to MP frames of correlation per boundary.
  void write_restart(FILE *fp) override;
  void restart(char *buf) override;

 protected:
  // Members are protected (not private) so FixXPTKokkos can access them.
  // --- user parameters ---
  int nframes;          // frames per analysis window
  char *prefix;         // output file prefix
  int do_classical;     // 1 = also write classical thermo
  int do_normalize;     // 1 = divide extensive quantities by natom/nmol
  int units_lj;         // auto-set: 1 = LJ reduced units (dimensionless)
  // Unit-conversion factors (set in init(); unused for units_lj=1):
  double vac_to_jmol;    // native (mass*vel²) → J/mol
  double dt_to_ps;       // native timestep → ps
  double ke_to_pe_fac;   // native (mass*vel²)/atom → native PE/atom
  double pe_to_kjmol;    // native PE total → kJ/mol-equiv
  double ke_to_kjmol;    // native (mass*vel²) total → kJ/mol-equiv
  double mass_to_gmol;   // native mass/atom → g/mol
  double vol_to_angst3;  // native volume → Å³
  double e_out_scale;    // internal kJ/mol → output energy unit (1.0 most; 1/96.485 metal→eV)
  double s_out_scale;    // internal J/mol/K → output S/Cv unit (1.0 most; 1/96485 metal→eV/K)
  double lj_eps;        // LJ ε (energy scale, kcal/mol)
  double lj_sig;        // LJ σ (length scale, Å)
  double lj_mass;       // LJ m (mass scale,   g/mol)
  bool lj_params_user;  // true if user explicitly specified any LJ param
  double hbar_star;     // reduced ħ* (1.0 default; computed from lj params if provided)
  // Volume override: CONSTANT (decimal) or VARIABLE (v_<name> equal-style)
  enum { VOL_BOX = 0, VOL_CONSTANT = 1, VOL_VARIABLE = 2 };
  int    volume_style;    // VOL_BOX / VOL_CONSTANT / VOL_VARIABLE
  double user_volume;     // constant value when volume_style == VOL_CONSTANT
  char  *volume_varstr;   // variable name (no "v_" prefix) when VOL_VARIABLE
  int    volume_varindex; // resolved variable index (set in init())

  // --- HS reference + FSC options ---
  int hs_entropy_mode;    // 0 = rigorous (default), 1 = lin2003 (legacy)

  // 3PT cage-memory ΔS added to the entropy, per channel, most recent window.
  // Extensive at capture, normalised at write time.  Zero unless mode 3PT.
  // Written to .thermo as S_cage_t / S_cage_r columns.
  double s_cage_trans_last;
  double s_cage_rot_last;
  // ── mode taxonomy (1PT/2PT/3PT + refinement) ─────────────────────────────
  // User `mode` is a string (1PT|2PT|3PT, case-insensitive) parsed into req_mode;
  // `refinement` (rigorous|lin2003|desjarlais|r2pt) selects the 2PT sub-method.
  // init() resolves (req_mode, refinement) → internal dispatch label mol_mode
  // (1=1PT, 2=HS, 3=MF/des) + hs_entropy_mode + the r2pt/cage flags.  3PT implies
  // rigorous-HS + the parameter-free cage.
  enum ReqMode { REQ_UNSET = -1, REQ_1PT, REQ_2PT, REQ_3PT };
  int req_mode;                 // parsed `mode`; resolved in init()
  // 2PT values: RIGOROUS..R2PT.  3PT value: NONE.
  enum Refinement { REF_RIGOROUS = 0, REF_LIN2003, REF_DESJARLAIS, REF_R2PT,
                    REF_NONE };
  int refinement;               // sub-method; meaning depends on req_mode
  int refinement_set;           // 1 = user gave `refinement` (else mode default)
  double r2pt_delta;            // Sun 2017 δ for refinement=r2pt (default 1.5)
  int cage_entropy;             // 1 = apply 3PT translational cage (mode 3PT)
  int cage_entropy_rot;         // 1 = apply 3PT rotational cage (mode 3PT, molecular)





  // Multi-tau VACF correlator (opt-in alternative to the default FFT path).
  // Ramírez & Likhtman, JCP 133, 154103 (2010): L levels of ring buffers,
  // each sampling at S× the previous stride; log-spaced lags, low memory.
  //
  //   Correlator = 0 (FFT, default) | 1 (multi-tau)
  //   mt_M       = base block size per level     (default 32)
  //   mt_P       = inter-level overlap          (default 16)
  //   mt_S       = sample-rate reduction factor (default 2)
  //   mt_L       = number of levels, 0 = auto = ceil(log_S(nframes/M))
  enum CorrelatorKind { CORR_FFT = 0, CORR_MULTITAU = 1 };
  int correlator;
  int mt_M, mt_P, mt_S, mt_L;

  // Per-window timing accumulators (FIX_XPT_DEBUG only; ~120 bytes when off).
  double t_vac_atom, t_vac_mol, t_vac_vib, t_vac_multitau;
  double t_dos, t_thermo;
  // Per-step multi-tau push bucket.
  double t_multitau_push;
  void   reset_window_timings();

  // Sub-window thermodynamic snapshots (multi-tau only).  When `nsamples N`
  // (N>1), (N-1) intermediate analyses fire per window at iframe =
  // k·(nframes/N), each on the cumulative state without resetting
  // accumulators.  Requires nframes % nsamples == 0 and correlator=multitau.
  int nsamples;

  // Cap on frequency range written to .pwr/.vac [cm⁻¹, or 1/tau for LJ].
  // 0.0 (default) = full spectrum.  Disk-output only; the 2PT solve uses
  // the full-resolution in-memory arrays.
  double maxfreq;

  // Coarsened frequency-axis spacing for output files [cm⁻¹, or 1/tau for LJ].
  // Emitted spacing = smallest multiple of native dnu >= dnu_out (stride
  // K = max(1, round(dnu_out/dnu_real))).  Default 0.0 = native dnu.
  // Disk-output only; the 2PT solve sees the full grid.
  double dnu_out;

  // Multi-tau per-step ring-buffer state.  Allocated lazily once
  // correlator == MULTITAU and natom_buf is known.
  //   mt_n_levels       — effective L (mt_L if > 0, else auto)
  //   mt_MP             — M + P (kept once for inner loops)
  //   mt_natom_ring     — natom_buf this ring was sized for
  //   mt_v_ring[ℓ]      — (M+P) × natom × 3 doubles, ring buffer per level
  //   mt_c_sum[ℓ]       — M+P partial Σ_atoms · Σ_t   m · v(t) v(t+k)
  //   mt_c_cnt[ℓ]       — M+P sample counts for lag k at this level
  //   mt_count_seen[ℓ]  — # samples pushed into this level
  //   mt_head[ℓ]        — ring head index into mt_v_ring[ℓ]
  //   mt_down_acc[ℓ]    — natom × 3 accumulator for downsample → level ℓ+1
  //   mt_down_n[ℓ]      — # samples accumulated into mt_down_acc[ℓ]
  int mt_n_levels;
  int mt_MP;
  int mt_natom_ring;
  std::vector<std::vector<double>> mt_v_ring;
  std::vector<std::vector<double>> mt_c_sum;
  std::vector<std::vector<long>>   mt_c_cnt;
  std::vector<long>                mt_count_seen;
  std::vector<int>                 mt_head;
  std::vector<std::vector<double>> mt_down_acc;
  std::vector<int>                 mt_down_n;
  // True when vel_buf was sized to 1 frame (multi-tau, no matrix-VAC dep):
  // ibuf access sites index vel_buf[0] instead of vel_buf[iframe].
  bool mt_vel_buf_single_frame;
  // One-shot guard so the "multi-tau state not restart-persistent" warning
  // fires only once per fix (init() runs multiple times).
  bool mt_restart_warn_emitted;
  // Restart re-apply: restart() arrives before the ring/stream buffers
  // exist, so stash the raw payload and re-apply via
  // multitau_apply_restart_stash() once they're allocated (post_run()).
  std::vector<double> mt_restart_stash;   // raw restart payload (header+body)
  bool                mt_restart_pending; // stash holds unapplied state

  // Molecular per-channel multi-tau streams: trans = COM, rot = angular
  // velocity, vib = residual per-atom velocity.  Each carries its own ring +
  // accumulators sized to (n_units, 3) where n_units = nmol_group (trans/rot)
  // or natom_buf (vib).  Active when correlator=multitau and molecular mode.
  //
  // The struct is public so the NVCC CUDA stub can name it when instantiating
  // FixXPTKokkos::multitau_stream_push; only FixXPT and derived classes ever
  // construct or mutate one.
 public:
  struct MultiTauStream {
    int n_units;                                  // n_mol or n_atom
    bool is_distributed;                          // true → MPI-partial; false → global
    std::vector<std::vector<double>> v_ring;      // L × (M+P) × n_units × 3
    std::vector<std::vector<double>> c_sum;       // L × (M+P)
    std::vector<long>                count_seen;  // L
    std::vector<int>                 head;        // L
    std::vector<std::vector<double>> down_acc;    // L × n_units × 3
    std::vector<int>                 down_n;      // L
  };
 protected:
  MultiTauStream mt_trans, mt_rot, mt_vib;
  bool mt_molecular_active;
  void multitau_stream_alloc(MultiTauStream& s, int n_units, bool distributed);
  void multitau_stream_reset(MultiTauStream& s);
  // Virtual so FixXPTKokkos can override the inner-product loop with a
  // device kernel; cascade recursion + ring bookkeeping stay host-side.
  virtual void multitau_stream_push(MultiTauStream& s, int Lev,
                              const double* v_curr,
                              const double* weights,
                              const std::vector<int>* slots);
  void multitau_alloc(int natom);
  void multitau_apply_restart_stash();
  // Reconstruct per-molecule principal moments of inertia from the current
  // restart positions, for a `run 0` reanalysis of a legacy restart.
  void compute_mol_inertia_oneframe();
  void multitau_reset_window();
  void multitau_push_frame();
  void multitau_push_level(int Lev, const double* v_curr);

  int use_sim_z_mode;     // 0 = use HS-EOS Z (default), 1 = override mu_q gas-PV
                          //     Term with Z_sim = P V / (N kB T) from MD virial
                          //     (Sec V.D mitigation of HS-EOS overshoot at eta > 0.5).

  // --- molecular mode ---
  int do_molecule;      // 1 = molecular decomposition (keyword: molecule)
  int allow_mixed_molecules;  // 1 = skip the per-molecule isomorphism check
                              //     (off by default; mixed chemistries give
                              //     wrong rotational/vibrational decomposition).

  std::string symmetry;          // point-group Schoenflies symbol

  // Emit a per-block header line ("# sample x of y" for intermediate
  // snapshots, "# cumulative" otherwise) to an output file.
  void emit_block_header(FILE *f, bigint step);
  int mol_mode;         // 1=1PT, 2=2PT-HS, 3=2PT-Desjarlais (all Markovian HS).
                        // Keyword: mode (resolved from req_mode + refinement).
  int is_linear;        // linear molecule flag (keyword: linear; auto for diatomics)
  int rotsym;           // rotational symmetry number σ_rot.  Auto-derived
                        //   From `symmetry` via pg_rotsym_for_label().
                        //   Default 1 (C1 / Ci / Cs).
  int do_show_split;    // show gas/solid split for S_trans, S_rot (keyword: show_split)

  // --- molecule topology (built in init()) ---
  int nmol_group;                   // distinct molecules in group (global)
  std::vector<tagint>  mol_id_list; // sorted unique mol IDs
  std::vector<double>  molmass;     // total mass per molecule [g/mol] (global)
  std::vector<int>     mol_natom;   // atom count per molecule (global)
  std::vector<bool>    mol_is_linear; // per-molecule linear flag
  std::vector<int>     slot_to_mol; // slot_to_mol[tag-1] = mol_idx (local only)

  // Bumped when build_mol_topology() repopulates the topology arrays; lets
  // FixXPTKokkos skip re-uploading device mirrors when topology is stable.
  unsigned int mol_topology_generation;

  // DOF used in T_vac at the most recent window; surfaced in the .thermo
  // "DOF" column to expose multi-rank / SHAKE / dynamic-group anomalies.
  double dof_last_window;

  // Per-fix memory budget [GB].  0 = off.  When >0, init() auto-reduces
  // nframes to the largest power-of-2 whose buffers fit, with a warning.
  double max_memory_gb;

  // --- molecular velocity/angular-momentum buffers ---
  double ***com_vel_buf;  // [nframes][nmol_group][3]  COM velocity
  double ***omega_buf;    // [nframes][nmol_group][3]  anguv = V·(ω_body×√I) per molecule (ANGUL)
  double ***angmom_buf;   // [nframes][nmol_group][3]  angular momentum L (lab frame)
  double ***vib_vel_buf;  // [nframes][natom_buf][3]   per-atom vibrational velocity v_i−v_COM−ω×r′_i (FP64 default)
  float  ***vib_vel_buf_f; // FP32 alternate; null when buffer_precision == BUFFER_FP64
  double  **mol_inertia;  // [nmol_group][3] principal moments, window-averaged (reset each window)
  int       mol_inertia_count; // frames accumulated for mol_inertia

  // --- pressure averaging ---
  class Compute *press_compute;   // pressure compute (global or group)
  bool own_press_compute;         // true if we created it
  double press_sum;               // sum of instantaneous pressures over window
  int press_count;                // frames accumulated

  // --- energy accumulation (for E_md and Cv fluctuation) ---
  class Compute *pe_compute;      // PE compute: global scalar (all) or per-atom (subgroup)
  bool own_pe_compute;            // true if we created it
  bool pe_peratom;                // true = per-atom PE, false = global scalar
  bool pe_available;              // false = pair style does not support per-atom PE
  double E_sum;                   // sum of (KE+PE) per frame [kJ/mol or ε]
  double E_sq_sum;                // sum of (KE+PE)² per frame
  int E_count;                    // frames accumulated

  // --- bookkeeping ---
  int me, nprocs;
  int iframe;           // current frame index [0, nframes)
  bool window_idle;     // accumulate_frame bailed at iframe==0 (group globally
                        // empty, e.g. an unpopulated dynamic shell) — the window
                        // never started, so end_of_step must NOT advance iframe
  bigint nwindow;       // completed windows
  bigint step_start;    // LAMMPS step when current window began
  bool mt_run0_dump;    // set in setup() for a multitau `run 0`; post_run()
                        // emits an accumulated/restart-state analysis block
  bigint fix_dof_window;  // DOF removed by constraints at window start (consistent with group_slots)

  // --- persistent FFT scratch (allocated once in init(), reused all windows) ---
  // Plan + buffer depend only on N_fft = 2·nframes, fixed at construction.
  // Must stay persistent: per-window alloc/free poisons FFTW3's plan-cache
  // wisdom for 1×1×N degenerate plans → segfault in PPPM's KSpace destructor
  // at shutdown (fftw_forget_wisdom() does NOT fix it).
  class FFT3d *fft_vac;   // single MPI_COMM_SELF plan, size N_fft = 2·nframes
  FFT_SCALAR  *fft_buf;   // scratch (2·N_fft doubles)

  // --- velocity buffer: [nframes][natom_buf][3] ---
  int natom_buf;        // allocated slots (= max group tag, ≥ group size)
  double ***vel_buf;    // [nframes][natom_buf][3]  — FP64 storage (default)
  // Optional FP32 storage for vel_buf/vib_vel_buf (halves the largest
  // buffers).  When `buffer_precision fp32`, the FP64 pointers are null and
  // access goes through *_f.  Per-mol buffers stay FP64.
  float ***vel_buf_f;
  int     buffer_precision;       // BUFFER_FP64 (default) or BUFFER_FP32
  enum    BufferPrecision { BUFFER_FP64 = 0, BUFFER_FP32 = 1 };
  // Inline FP32/FP64 accessors (branch hoists in hot loops).
  inline double vread(int ibuf, int s, int d) const {
    return (buffer_precision == BUFFER_FP32)
         ? (double)vel_buf_f[ibuf][s][d]
         : vel_buf[ibuf][s][d];
  }
  inline void vwrite(int ibuf, int s, int d, double v) {
    if (buffer_precision == BUFFER_FP32) vel_buf_f[ibuf][s][d] = (float)v;
    else                                  vel_buf[ibuf][s][d] = v;
  }
  inline double vibread(int ibuf, int s, int d) const {
    return (buffer_precision == BUFFER_FP32)
         ? (double)vib_vel_buf_f[ibuf][s][d]
         : vib_vel_buf[ibuf][s][d];
  }
  inline void vibwrite(int ibuf, int s, int d, double v) {
    if (buffer_precision == BUFFER_FP32) vib_vel_buf_f[ibuf][s][d] = (float)v;
    else                                  vib_vel_buf[ibuf][s][d] = v;
  }
  double *mass_buf;     // [natom_buf]  atom masses (g/mol per atom)
  std::vector<int> group_slots;  // tag-1 slot indices for local group atoms

  // --- shared velocity buffer ----------------------------------
  // When multiple fix_xpt share the same (group, nevery, nframes,
  // correlator, do_molecule), only the first ("owner") allocates the
  // velocity buffers; "consumers" alias the owner's storage via
  // sync_buffer_pointers_from_owner().  The owner does the per-step push +
  // Allreduce + mol decomposition; consumers just read in their analysis.
  // Multi-tau rings (mt_*) and per-fix topology are NOT shared.
  FixXPT *buffer_owner;             // points at first matching fix (= this if owner)
  bool   owns_buffer;               // true if this instance allocates/frees buffers
  std::vector<FixXPT *> buffer_consumers;  // owner-side: fixes to sync on grow_buf
  // Virtual so FixXPTKokkos can also re-alias its device vel_buf_view from
  // the owner; CPU impl syncs the host buffer pointers.
  virtual void sync_buffer_pointers_from_owner();  // consumer-side helper

  // --- result vector ---
  static const int NVEC_MAX = 20;
  int nvec;                    // actual size_vector (10, 14, or 18)
  double result_vec[NVEC_MAX]; // S_q, A_q, E_q, Cv_q, D, f, T_vac [, mol extras], ZPE, μ_q, μ_c
  bool result_ready;

  // --- internal methods ---
  void grow_buf(int n);
  void accumulate_frame();
  void accumulate_mol_frame();   // molecular decomposition per frame

  // Virtual per-frame velocity push.  CPU impl gathers local velocities +
  // MPI_Allreduce(SUM) into vel_buf[ibuf]; FixXPTKokkos does it on device.
  // Owner-only (call site gates on owns_buffer).
  virtual void push_velocity_frame(int ibuf);

  // Per-step local→global reduction; on single-rank runs skips the costly
  // MPI_Allreduce no-op in favour of a memcpy.
  void reduce_or_copy(const double *src, double *dst, int n);
  // FP32 companion of reduce_or_copy.
  void reduce_or_copy_f(const float *src, float *dst, int n);

  // Pass 1 atom loop: COM momentum + mass-weighted unmapped position.
  // Fills the local-rank contribution into caller-sized lcom_p/lcom_r
  // (nmol_group*3, zero-init by caller); the caller MPI_Allreduces.
  virtual void compute_mol_pass1_atoms(int ibuf_vel,
                                       double *lcom_p,
                                       double *lcom_r);

  // Pass 2 atom loop: angular momentum L = Σ m·r'×v' and inertia tensor
  // I_αβ = Σ m·(|r'|²δ_αβ − r'_α r'_β).  Inputs vcom/rcom per-mol; outputs
  // lL (nm*3), lI (nm*9) are local-rank accumulators the caller Allreduces.
  virtual void compute_mol_pass2_atoms(int          ibuf_vel,
                                       const double *vcom,
                                       const double *rcom,
                                       double       *lL,
                                       double       *lI);

  // Pass 3 atom loop: per-atom vibrational velocity v_vib,i =
  // v_i − v_COM − ω×r'_i.  Output vib_local (natom_buf*3) is local-rank
  // scratch the caller Allreduces into the vib_vel_buf row.
  virtual void compute_mol_pass3_atoms(int           ibuf_vel,
                                       const double *vcom,
                                       const double *rcom,
                                       const double *omega_lab,
                                       double       *vib_local);

  // End-of-window forward-FFT loop; virtual so FixXPTKokkos can route
  // through cuFFT.  pwr is ACCUMULATED (not reset) so multiple pwr_from_*
  // calls stack into the same vector.
  virtual void pwr_from_atoms(double ***buf, const std::vector<int> &slots,
                              const double *mbuf, std::vector<double> &pwr,
                              FFT_SCALAR *fft_buf, FFT3d *fft_vac, int ns);
  // FP32 variant; called when buffer_precision == BUFFER_FP32.
  void pwr_from_atoms_f(float ***buf, const std::vector<int> &slots,
                        const double *mbuf, std::vector<double> &pwr,
                        FFT_SCALAR *fft_buf, FFT3d *fft_vac, int ns);
  virtual void pwr_from_mols(double ***buf, int nmol,
                             const std::vector<double> &wts,
                             std::vector<double> &pwr,
                             FFT_SCALAR *fft_buf, FFT3d *fft_vac, int ns);
  void run_analysis();
  void build_mol_topology();     // populate nmol_group, mol_id_list, molmass, etc.

  // Proper-rotation-subgroup order σ_rot for a Schoenflies point-group symbol
  // (the `symmetry` keyword).  Returns -1 for unknown; case-insensitive.
  // Supported: C1, Ci, Cs; Cn/Cnv/Cnh, Dn/Dnh/Dnd (n=2-6); T/Td/Th; O/Oh;
  // I/Ih; S4/S6/S8; Cinfv/Dinfh.
  static int pg_rotsym_for_label(const std::string &label);

  // --- option parsing (legacy CLI keywords AND new INI/config-block paths) ---
  // Map one (key, value) pair to internal state.  Value "" = legacy
  // presence-flag (enable); "1"/"0"/"true"/... = boolean.  error->all() on
  // unknown key or bad value.
  void set_option(const char *key, const char *value);
  // Parse key=value lines (# and ; are comments; blanks skipped) → set_option().
  void parse_kv_lines(const std::string &text);
  // Read INI file → parse_kv_lines; rank 0 reads + broadcasts.
  void parse_ini_file(const char *path);

  // --- FFT-based VAC/DOS helpers ---
  // FFT3d uses MPI_COMM_SELF (per-rank local plan) so ranks with different
  // atom counts can call compute() independently without deadlock.
  //
  // IFFT of pwr → vac[0..ns-1]; BACKWARD is unnormalized, divide by N_fft*ns.
  void vac_from_pwr(const std::vector<double> &pwr, std::vector<double> &vac,
                    FFT_SCALAR *fft_buf, FFT3d *fft_vac, int ns);
  // O(ns log ns) cosine DOS: BACKWARD FFT of zero-symmetric normalised VAC.
  void dos_from_vac(const std::vector<double> &vacc, double vnorm,
                    std::vector<double> &dos_out,
                    FFT_SCALAR *fft_buf, FFT3d *fft_vac,
                    int ns, int nused, double dos_factor);
  // Temperature from VAC(0) via equipartition.
  double T_from_vac(const std::vector<double> &vacc, int dof_comp,
                    double vel_to_Angps2) const;
  // Gas DOS at bin j: Lorentzian (mode 1/2) or Desjarlais (mode 3, Bg > 0).
  double gas_comp(int j, double dnu, double dos_j, double s0_c,
                  double f_c, double Bg_c, double nmol_c) const;
  // Build gas DOS array over all bins (calls gas_comp once per bin).
  std::vector<double> build_gas_arr(const std::vector<double> &dos_c, double dnu,
                                     double s0_c, double f_c, double Bg_c, double nmol_c);
  // --- physics: monoatomic ---
  static double search2pt(double K);

  // hs_entropy_m:  0 = rigorous (default), 1 = lin2003 (legacy +ln Z)
  static double hs_entropy(double y, double mass_per_mol,
                            double natom, double T, double V_m3,
                            int hs_entropy_m = 0);
  static double hs_entropy_lj(double y, double mass_star,
                               double natom, double T_star, double V_star,
                               double hbar_star,
                               int hs_entropy_m = 0);

  // R2PT (Sun 2017) translational entropy [k_B/atom, d=3]; refinement=r2pt.
  // dos is the extensive per-group DoS (∫=3·nmol). NaN on degenerate input.
  double r2pt_entropy(double dnu, const std::vector<double>& dos, int nused,
                      double f_delta1, double T_K, double mass_amu,
                      double vol_A3, double nmol, double delta) const;

  // 3PT cage-memory entropy correction ΔS [k_B/atom]; mode=3PT.  Form-B scalar
  // Volterra kernel + memory-excess cage.  dos_total/dos_gas are PER-ATOM (∫=3),
  // C_scalar is the clean trace/3 VACF.  Wg_override = NaN → compute trans Wg;
  // else use it (rot channel rigid-rotor weight).  0.0 on degenerate input.
  double cage_memory_entropy(double dt, const std::vector<double>& C_scalar,
                             int nvac, double dnu,
                             const std::vector<double>& dos_total,
                             const std::vector<double>& dos_gas, int nused,
                             double T_K, double mass_amu, double vol_A3,
                             double prefactor, int dimension,
                             double Wg_override, std::vector<double>* cage_out,
                             double gate_f0 = 0.01, double clip_eps = 1e-3) const;


  // ── 3PT cage low-level kernel helpers ──────────────────────────────────
  // Scalar Form-B Volterra kernel K(t) of the normalized VACF cn (cn[0]=1):
  // fills K (size nvac, zero past the cutoff) and returns the auto cutoff nK.
  // nf_run = consecutive |cn|<floor lags required before the noise-floor
  // guard truncates (1 = legacy first-lag behavior; >1 makes the guard
  // envelope-aware so oscillatory VACF nodes do not truncate the kernel).
  static int volterra_kernel_scalar(const std::vector<double>& cn, int nvac,
                                    double dt, std::vector<double>& K,
                                    int nf_run = 1);
  // First lag n>=3 with |K[n]| < alpha·|K[0]| (kernel main-lobe edge); returns
  // nK_auto when no sub-alpha lag exists in [3, nK_auto).
  static int mainlobe_cutoff(const std::vector<double>& K, int nK_auto, double alpha);
  // DC friction γ = K̃(0) = ∫₀ K dt (trapezoid endpoint weights) on K[:nk].
  static double gamma_dc(const std::vector<double>& K, int nk, double dt);



  // Hard-sphere compressibility Z = pV/(NkT) at packing fraction y.
  //   Z_CS = (1+y+y²−y³)/(1−y)³   (Carnahan-Starling)
  static double hs_compressibility(double y);

  // Simulation-derived compressibility Z_sim = P V / (N kB T) from the LAMMPS
  // virial; used when use_sim_z_mode=1 to override Z_CS in the mu_q gas-PV
  // term.  Handles atm/bar/Pa via update->unit_style (hence non-static).
  double z_sim_from_pressure(double P_native, double V_native,
                              int N, double T, int units_lj,
                              double vol_to_angst3);
  // --- physics: Desjarlais memory-function refinement (mol_mode 3) ---
  static double dawson_f(double y);
  static double sgmf_des(double nu, double s0, double f_g, double Bg, double nmol);
  static double refine_Bg_des(const std::vector<double>& pwr, double s0,
                               double dnu, double nmol, double f_g);

  // --- physics: molecular (rotational gas entropy) ---
  // Returns ws such that S_rot_gas = HSDF_rot * ws * R  [J/(mol·K)]
  // I1,I2,I3 in (g/mol)*Å²; for linear I3=0
  static double hs_entropy_rot(double y, double I1, double I2, double I3,
                                double natom_hs_rot, double T, int sigma_rot, bool linear);
  // LJ version: I_star in m_lj*sigma² units
  static double hs_entropy_rot_lj(double y, double I_star,
                                   double natom_hs_rot, double T_star,
                                   int sigma_rot, bool linear, double hbar_star);

  // --- 3×3 math utilities ---
  static void cross3(const double a[3], const double b[3], double c[3]);
  // Compute eigenvalues of 3×3 symmetric matrix via Jacobi iterations
  // Returns eigenvalues (ascending) in eig[3]; eigenvectors in cols of V[3][3]
  static void mat3_sym_eigen(const double A[3][3], double eig[3], double V[3][3]);

  // --- I/O ---
  FILE *fp;             // .thermo — thermo summary (one row per window)
  FILE *fp_pwr;         // .pwr  — power spectrum (DOS), one block per window
  FILE *fp_vac;         // .vac  — velocity autocorrelation, one block per window
  void open_file();
  void write_header();
  void write_result(bigint step,
                    double S_q, double A_q, double E_q, double Cv_q, double ZPE_q,
                    double S_c, double A_c, double E_c, double Cv_c,
                    double D, double f, double T_vac, double P_avg,
                    double E_md, int natom_group, double V_box,
                    // Molecular extras (ignored if !do_molecule)
                    double S_trans, double S_rot, double S_vib, double f_rot, double D_rot,
                    double S_trans_gas, double S_trans_solid,
                    double S_rot_gas, double S_rot_solid,
                    // Chemical potential (always present)
                    double mu_q, double mu_c);
  // Write per-window power spectrum and VAC blocks
  void write_pwr_vac(bigint step,
                     double dnu, int nused, int ns_val, double dt_vac,
                     const std::vector<double>& dos,
                     const std::vector<double>& vac,
                     const std::vector<double>& gas_mono,
                     const std::vector<double>& dos_trans,
                     const std::vector<double>& dos_rot,
                     const std::vector<double>& dos_vib,
                     const std::vector<double>& gas_trans,
                     const std::vector<double>& gas_rot,
                     const std::vector<double>& vac_trans,
                     const std::vector<double>& vac_rot,
                     const std::vector<double>& vac_vib,
                     const std::vector<double>& cage_mono,
                     const std::vector<double>& cage_trans,
                     const std::vector<double>& cage_rot);
};

}    // namespace LAMMPS_NS

#endif    // LMP_FIX_XPT_H
#endif    // FIX_CLASS
