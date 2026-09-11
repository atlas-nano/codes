.. index:: fix xpt

fix xpt command
===============

Syntax
""""""

.. code-block:: LAMMPS

   fix ID group-ID xpt Nevery Nframes prefix keyword values ...

* ID, group-ID are documented in :doc:`fix` command
* xpt = style name of this fix command
* Nevery = sample velocities every this many timesteps
* Nframes = number of frames per analysis window
* prefix = prefix for output files (*prefix*\ .thermo, *prefix*\ .pwr, *prefix*\ .vac)
* zero or more keyword/value pairs may be appended

.. parsed-literal::

   keyword = *classical* or *normalize* or *epsilon* or *sigma* or *mass* or *volume* or *molecule* or *mode* or *refinement* or *r2pt_delta* or *linear* or *symmetry* or *show_split* or *use_sim_z* or *allow_mixed_molecules* or *correlator* or *mt_M* or *mt_P* or *mt_S* or *mt_L* or *nsamples* or *maxfreq* or *dnu* or *max_memory_gb* or *buffer_precision* or *buffer_layout*
     *classical* value = none
       compute and write classical thermodynamic quantities in addition to quantum
     *normalize* value = none
       divide extensive quantities by atom count (monoatomic) or molecule count (molecular mode)
     *epsilon* value = E
       E = LJ energy scale (kcal/mol) — enables physical |hbar|\* scaling in LJ units
     *sigma* value = S
       S = LJ length scale (Angstroms) — enables physical |hbar|\* scaling in LJ units
     *mass* value = M
       M = LJ mass scale (g/mol) — enables physical |hbar|\* scaling in LJ units
     *volume* value = V
       V = positive decimal (Angstroms\ :sup:`3` or sigma\ :sup:`3`) or v_<name> equal-style variable — overrides box volume
     *molecule* value = none
       enable molecular decomposition: entropy partitioned into translational + rotational + vibrational
     *mode* value = 1PT or 2PT or 3PT
       Phase partition (case-insensitive; see description below).  Default 2PT.
       1PT = all-solid harmonic baseline (no gas); 2PT = gas + solid hard-sphere
       partition (pick the sub-method with *refinement*); 3PT = 2PT-rigorous plus
       the parameter-free memory cage phase (translational + rotational).
     *refinement* value = rigorous or lin2003 or desjarlais or r2pt (mode 2PT)
                           or none (mode 3PT)
       Mode-dependent sub-method.  For *mode* = 2PT (default rigorous = exact
       hard-sphere entropy): lin2003 = Lin-Blanco-Goddard +ln Z convention;
       desjarlais = Desjarlais Gaussian memory-function gas DoS (uses the +ln Z
       convention); r2pt = Sun 2017 revised-2PT (delta-tuned fluidicity + F_a
       sum rule).  For *mode* = 3PT the only value is *none* = the bare
       parameter-free memory cage (default markov reference).
     *r2pt_delta* value = D
       Sun 2017 delta exponent for *refinement* = r2pt (default 1.5).
     *linear* value = none
       flag all molecules as linear (auto-set for diatomics)
     *symmetry* value = sym
       sym = Schoenflies point-group symbol (default C1).  Auto-derives the
       rotational symmetry number sigma_rot from a built-in lookup table:
       C1/Ci/Cs -> 1; Cn/Cnv/Cnh -> n; Dn/Dnh/Dnd -> 2n; T/Td/Th -> 12;
       O/Oh -> 24; I/Ih -> 60; S4/S6/S8 -> n/2; Cinfv -> 1; Dinfh -> 2.
       Unknown symbols hard-error at fix-creation time.  Replaces the
       deprecated *rotsym* keyword (using *rotsym* now hard-errors).
     *show_split* value = none
       write gas/solid split columns for S_trans and S_rot
     *use_sim_z* value = 0 or 1
       chemical-potential compressibility source (default 0).  With 1, the
       PV term of :math:`\mu` uses the simulation's measured compressibility
       factor :math:`Z = PV/Nk_BT` (from the running pressure) instead of the
       hard-sphere :math:`Z_\mathrm{HS}(y)`.  Affects :math:`\mu_q`/:math:`\mu_c`
       only; entropies are unchanged.
     *allow_mixed_molecules* value = 0 or 1
       molecular-mode species check (default 0).  The per-molecule
       rotational/vibrational decomposition averages the inertia tensor and
       angular-velocity math over the molecules in the group and is only
       meaningful when they share the same chemistry (atom count, atom-type
       multiset, mass).  By default a group containing more than one species
       is rejected; set to 1 to bypass the check (S_rot / S_vib are then
       unreliable for the mixed group).
     *correlator* value = fft or multitau
       VAC algorithm.  ``fft`` (default) = uniform-lag Wiener-Khinchin via FFT3d.
       ``multitau`` = log-spaced Magatti-Schaetzel / Ramirez-Likhtman ring-buffer
       correlator (see *Multi-tau VACF* section below).  Drop-in replacement on
       every analysis path (scalar trace + mol trans/rot/vib).
     *mt_M* value = N
       Multi-tau base block size per level (default 32).
     *mt_P* value = N
       Multi-tau inter-level overlap (default 16).  Recommended pairing is
       ``mt_M 64 mt_P 64`` for <= 1% D drift on monatomic; ``mt_M 32 mt_P 16``
       (default M+P=48) for memory-tight runs.
     *mt_S* value = N
       Multi-tau per-level sample-rate reduction factor (default 2; typically 2, 4, or 8).
     *mt_L* value = N
       Multi-tau number of levels (default 0 = auto-derive
       ``ceil(log_S(nframes / M))``).  Larger L extends the lag range at log
       cost in memory.
     *nsamples* value = N
       Multi-tau only.  Emit ``N - 1`` intermediate analysis blocks per
       window at iframe = k · (nframes / N), k = 1..N-1, in addition to
       the usual end-of-window block.  Each intermediate snapshot runs
       run_analysis() against the current cumulative VAC state, so users
       can see thermo convergence within a single window.  Default 1 (no
       sub-window snapshots).  Requirements (validated in setup):
       ``correlator multitau`` and ``nframes % nsamples == 0``.
     *maxfreq* value = F
       F = maximum frequency written to all frequency-axis output
       files.  Units match each file's freq column: cm\ :sup:`-1` for
       non-LJ unit styles, 1/tau for LJ.  Default 0 = no truncation.
       Affects:

       * *prefix*\ .pwr — keeps bins ``0..floor(F/dnu)``.
       * *prefix*\ .vac — truncated to the same number of time
         samples as PWR (matching the original FFT-pair shape).
       * *prefix*\ .vac.multitau diagnostic — drops per-level
         time lags past the Nyquist threshold ``1 / (2·F·c)``.

       The in-memory dos / vac arrays driving the 2PT thermo
       (S_q, A, D, f, mu_q, gas-component fits, etc.) are
       untouched — *maxfreq* is an output-only knob.
     *dnu* value = F
       F = desired output frequency-axis spacing.  Coarsens the
       emitted spacing in *prefix*\ .pwr and .vac to the nearest
       integer multiple of the natural dnu\ :sub:`real` that is >= F.
       Stride ``K = max(1, round(F / dnu_real))`` is applied to both
       the freq axis (every K-th .pwr bin) and the time axis (every
       K-th .vac sample, so the disk pair stays FFT-shaped).  Bounded
       at the bottom by dnu\ :sub:`real`: setting F < dnu_real gives
       K = 1 (native spacing).  Units match the file's freq column:
       cm\ :sup:`-1` for non-LJ unit styles, 1/tau for LJ.  Default 0 =
       no coarsening.  Like *maxfreq* this is a disk-output knob only —
       the 2PT solve stays at full resolution.
     *max_memory_gb* value = G
       per-fix, per-rank RAM budget in GB (default 0 = off).  When > 0,
       *Nframes* is automatically reduced to the largest power of two whose
       velocity buffers fit within the budget, trading window length for
       memory.  With *buffer_layout distributed* a rank holds about 1/P of
       the group, so the same budget admits a longer window on more ranks.
     *buffer_precision* value = fp64 or fp32
       storage precision of the velocity buffers (default fp64).  *fp32*
       halves the footprint of the largest per-fix buffers at a loss of
       ~5-7 significant figures on the raw velocity samples; the
       state variables (*D*, *f*, *S(0)*, :math:`\mu`, *T_vac*) are
       typically unaffected (< 0.5%), but the heat capacity *Cv* can be
       precision-sensitive (a warning is logged).
     *buffer_layout* value = distributed or replicated
       where the frame history lives (default distributed).  *distributed*
       gives every group atom one home rank per window (whole molecules with
       *molecule*), so a rank holds about 1/P of the history; *replicated*
       keeps the whole history on every rank.  Both give the same results to
       floating-point summation order.  See the parallel memory layout
       section below.

Examples
""""""""

.. code-block:: LAMMPS

   fix xpt all xpt 5 512 argon classical normalize
   fix xpt all xpt 5 512 ar_lj normalize epsilon 0.2379 sigma 3.405 mass 39.948
   fix xpt all xpt 2 5000 water classical normalize molecule symmetry C2v mode 2PT
   fix xpt all xpt 5 512 na_liquid normalize mode 2PT refinement desjarlais
   fix xpt slab_grp xpt 5 512 slab normalize volume v_slab_vol
   fix xpt all xpt 4 2048 water classical normalize molecule symmetry C2v mode 3PT
   # disk-output controls: cap freq range + coarsen output spacing
   fix xpt all xpt 2 5000 benzene molecule symmetry D6h &
       maxfreq 4000  dnu 10

Description
"""""""""""

Compute thermodynamic properties of a group of atoms using the Two-Phase
Thermodynamics (2PT) method :ref:`(Lin 2003) <Lin2003fixxpt>`,
:ref:`(Lin 2010) <Lin2010fixxpt>`, :ref:`(Pascal 2011) <Pascal2011fixxpt>`.
Results are computed on-the-fly from the MD velocity trajectory with no
trajectory files or post-processing required.

The method partitions the vibrational density of states (DoS) derived from
the velocity autocorrelation function (VACF) into a gas-like component and a
solid-like (quantum harmonic oscillator) component.  Thermodynamic quantities
are obtained by integrating each component with quantum or classical weighting
functions.

Velocities are sampled every *Nevery* timesteps.  After *Nframes* samples are
accumulated, a full analysis window is completed and results are written.
Results are also available via the LAMMPS :doc:`thermo_style` command using
``f_ID[N]`` (1-based index; see `Output`_ below).

The unit system is auto-detected from the LAMMPS :doc:`units` command and
requires no keyword.  All eight LAMMPS unit styles are supported: ``real``,
``metal``, ``lj``, ``si``, ``cgs``, ``micro``, ``nano``, and ``electron``.

----------

**Gas DoS model: mode keyword**

The *mode* keyword controls how the gas-like component of the DoS is modeled.
It is only meaningful together with the *molecule* keyword for the translational
and rotational components.  In monoatomic mode a single *mode* analysis is
performed on the full DoS.

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - mode
     - Description
   * - 1PT
     - All-solid harmonic baseline: every DoF (translational, rotational,
       vibrational) treated as solid-like (Einstein oscillators); fluidicity
       f = 0, no gas component.
   * - 2PT
     - **Default.** Gas + solid hard-sphere partition (Lin-Blanco-Goddard):
       the translational and rotational DoS are each split into a gas-like and
       a solid-like component; vibrational DoS is solid-like.  The *refinement*
       keyword selects the entropy convention and gas-DoS model (table below).
   * - 3PT
     - 2PT-rigorous plus an explicit parameter-free memory *cage* phase
       (gas | cage | solid).  The cage is the non-Markovian memory excess of
       the Volterra friction kernel, applied to the translational and (for
       molecules) rotational channels with a per-DoF 1/d prefactor and a
       fluidicity gate.  The *refinement* keyword must be *none*.

The *refinement* keyword is **mode-dependent**.  For *mode* = 2PT it selects the
2PT sub-method:

.. list-table::
   :header-rows: 1
   :widths: 16 84

   * - refinement
     - Description
   * - rigorous
     - **Default.** Thermodynamically exact hard-sphere reference entropy.
   * - lin2003
     - Lin-Blanco-Goddard 2003 convention (adds the +ln Z compressibility term).
   * - desjarlais
     - Desjarlais Gaussian memory-function gas DoS :ref:`(Desjarlais 2013)
       <Desjarlais2013fixxpt>` (structured liquids / liquid metals with a
       backscattering shoulder; shape parameter :math:`B_g` found
       self-consistently by bisection).  Adopts the +ln Z entropy convention.
   * - r2pt
     - Sun 2017 revised-2PT: delta-tuned gas fraction (Eq A8, set by
       *r2pt_delta*) with an F_a-inclusive sum-rule gas / full-F_s solid
       partition.

For *mode* = 3PT the *refinement* keyword takes only one value:

.. list-table::
   :header-rows: 1
   :widths: 16 84

   * - refinement
     - Description
   * - none
     - **Default (and only value).** Bare parameter-free memory cage.

----------

**VACF and DoS computation**

The VACF is computed via the Wiener-Khinchin theorem using the LAMMPS FFT3d
backend (FFTW3, MKL, or KISS depending on the build):

.. math::

   \tilde{C}(\nu) = \text{FFT}[\mathbf{v}(t)]^2

.. math::

   C(\tau) = \text{IFFT}[\tilde{C}(\nu)]

The mass-weighted power spectrum is reduced across all MPI ranks via
``MPI_Allreduce``.  The one-sided DoS is obtained by a cosine transform of
the normalised VACF:

.. math::

   g(\nu_j) = \Delta t_\text{vac} \sum_{k=0}^{N_s} C_\text{norm}(k) \cos\!\left(\frac{2\pi j k}{2N_s}\right)

where :math:`N_s` = *Nframes* / 2 and :math:`\Delta t_\text{vac}` =
*Nevery* * ``timestep``.

----------

**Parallel memory layout (buffer_layout keyword)**

The frame history (per-atom velocities and, with *molecule*, the per-atom
vibrational residuals and the per-molecule translational, angular and
angular-momentum channels) is the dominant memory cost of the fix.  With the
default *buffer_layout distributed*, every group atom has one *home* rank for
the window.  At each window start the group's sorted atom list (with
*molecule*, its sorted molecule list weighted by atom count, so a molecule is
never split) is cut into P contiguous blocks of equal atom count.  Every
sampled step, each rank sends the velocities of its local group atoms (with
*molecule*, also their unwrapped positions) to their home ranks with one
``MPI_Alltoallv``.  The home rank holds complete molecules, evaluates the
translational, angular and vibrational projections locally and transforms its
own atoms and molecules; each spectrum is summed over ranks once per window.
A rank therefore stores about 1/P of the history and exchanges O(N/P) data per
frame.

*buffer_layout replicated* keeps a complete copy of the history on every rank
and assembles each frame with ``MPI_Allreduce``.  The two layouts agree to
floating-point summation order.

----------

**Sharing one velocity buffer across fixes**

The velocity buffer is the dominant memory cost of the fix, and comparing
several analyses of one trajectory is the common case.  Declaring one fix per
setting would ordinarily allocate one complete velocity history per setting.

It does not.  At the start of each ``run`` a fix checks whether an earlier
*fix xpt* is accumulating the same trajectory, and reuses its buffer if so.
The configurations must agree on *group*, *Nframes*, *Nevery*, *correlator*,
*buffer_precision*, *buffer_layout* and the *molecule* setting.  The *mode*
and *refinement* keywords are deliberately **not** part of that test, because
both act on the density of states rather than on the trajectory:

.. code-block:: LAMMPS

   fix s1 all xpt 5 4096 ar.1pt   mode 1PT
   fix s2 all xpt 5 4096 ar.rig   mode 2PT refinement rigorous
   fix s3 all xpt 5 4096 ar.lin   mode 2PT refinement lin2003
   fix s4 all xpt 5 4096 ar.des   mode 2PT refinement desjarlais
   fix s5 all xpt 5 4096 ar.r2pt  mode 2PT refinement r2pt
   fix s6 all xpt 5 4096 ar.3pt   mode 3PT

This costs one velocity history, not six.  The first fix owns the buffer (and,
with *buffer_layout distributed*, its home layout); the rest re-point at it and
skip frame accumulation, so they add no memory, no per-frame staging and no
per-frame communication.  Each still performs its own window-end analysis,
which is the part that differs between settings.  Sharing is reported once per
consumer in the log::

   FixXPT::s2-all: sharing velocity buffer with FixXPT::s1-all
                   (nevery=5, nframes=4096, FFT, do_molecule=0)

Detection runs once per ``run`` command, so fixes added or deleted between
runs are handled without user intervention.

.. note::

   Sharing applies to *correlator* = *fft* only.  The multi-tau ring buffers
   are per-fix, so a multi-tau consumer would reach the end of its window with
   no autocorrelation of its own; multi-tau fixes always own their buffers.

----------

**Thermodynamic quantities**

The fluidicity parameter *f* (fraction of DoS that is gas-like) is determined
by solving the implicit equation :ref:`(Lin 2003) <Lin2003fixxpt>`:

.. math::

   \Delta(f) = 0, \quad \Delta \equiv 6f^5/S_d^2 - 6f^4/S_d^2 + 2f^3/S_d^2 + 2f^3 - f^2 - 2f + 1

where :math:`S_d = \frac{g(0)}{N} \sqrt{\frac{\pi k_B T}{m}}` is the
normalised zero-frequency DoS.

Entropy, Helmholtz free energy, internal energy, and heat capacity are
evaluated by integrating the quantum weighting functions over the solid DoS
component.  Zero-point energy is accumulated in the same loop at no extra cost:

.. math::

   \text{ZPE}_q = \int_0^\infty g_\text{solid}(\nu)\, \frac{h\nu}{2}\, d\nu

.. note::

   The gas-component (Carnahan-Starling hard-sphere) entropy returns zero
   when the group particle count is zero — the "no gas phase" limit.  This
   guards a dynamic group that transiently empties: the Sackur-Tetrode
   :math:`\log(\lambda^3 V / N)` term diverges as :math:`N \to 0` and would
   otherwise feed a NaN into the entropy sum.  Matches the companion py-xPT
   engine's ``nmol <= 0`` behaviour.

----------

**LJ units and** |hbar|\ * **scaling**

For ``units lj`` without *epsilon*/*sigma*/*mass* keywords, the convention
:math:`\hbar^* = 1` (i.e. :math:`h^* = 2\pi`) is used.  This yields
dimensionless quantum entropy and free energy in the LJ framework.

Supplying all three physical parameters enables the reduced Planck constant:

.. math::

   \hbar^* = \frac{\hbar}{\sigma \sqrt{m_\text{atom} \varepsilon_\text{atom}}}

For Argon (:math:`\varepsilon/k_B = 119.8` K, :math:`\sigma = 3.405` Å,
:math:`m = 39.948` g/mol): :math:`\hbar^* \approx 0.02958`, enabling direct
comparison with real-unit results.

----------

**Molecular mode**

The *molecule* keyword requires ``atom_style molecular`` (or ``full``).  The
DoS is decomposed into translational (COM velocity), rotational (angular
velocity reconstructed from angular momentum and inertia tensor), and
vibrational (residual per-atom velocity) channels.  Each channel is analysed
independently and the entropy contributions are summed:

.. math::

   S = S_\text{trans} + S_\text{rot} + S_\text{vib}

The rotational inertia tensor is diagonalised via Jacobi iterations each window.
Linear molecules (including all diatomics, detected automatically) use a
two-dimensional rotor model.

The *symmetry* keyword sets the point-group Schoenflies symbol (default
``C1``) and auto-derives the rotational symmetry number
:math:`\sigma_\text{rot}` from a built-in lookup table.  For
H\ :sub:`2`\ O (``symmetry C2v``) :math:`\sigma_\text{rot} = 2`; for
benzene (``symmetry D6h``) :math:`\sigma_\text{rot} = 12`; for methane
(``symmetry Td``) :math:`\sigma_\text{rot} = 12`; etc.  The full table
covers C1, Ci, Cs, Cn/Cnv/Cnh (n=2..6), Dn/Dnh/Dnd (n=2..6), T, Td,
Th, O, Oh, I, Ih, S4, S6, S8, and the linear groups C∞v and D∞h.

The deprecated *rotsym N* keyword (which set :math:`\sigma_\text{rot}`
directly) now hard-errors with a message pointing at *symmetry*.

----------

**Volume override**

By default the instantaneous box volume is used.  The *volume* keyword overrides
this with a constant value or an equal-style LAMMPS variable:

.. code-block:: LAMMPS

   variable slab_vol equal "lx*ly*10.0"
   fix xpt slab_grp xpt 5 512 slab normalize volume v_slab_vol

This is useful for slab geometries or systems with non-trivial accessible volumes.

----------

**3PT cage (mode 3PT)**

*mode 3PT* adds an explicit parameter-free memory *cage* phase on top of
the rigorous 2PT partition.  The cage density of states is the non-Markovian
memory excess of the Volterra friction kernel,

.. math::

   \text{cage}(\nu) = \text{const}\cdot\big(F_K(\nu) - F_M(\nu)\big),
   \qquad F_K = \mathrm{Re}\!\left[\tfrac{1}{i\omega + \tilde K}\right],
   \quad F_M = \tfrac{\gamma}{\gamma^2 + \omega^2},

where :math:`\tilde K(\omega)` is the Fourier transform of the (Form-B) scalar
Volterra kernel :math:`K(t)` and :math:`\gamma = \tilde K(0)`.  The cage entropy
correction is :math:`\Delta S = p\,g(f)\int \text{cage}\,(1-w)\,(W_g - W_s)\,d\nu`
with per-DoF prefactor :math:`p = 1/d`, a fluidicity gate
:math:`g(f) = f^2/(f^2 + f_0^2)`, and the harmonic high-pass
:math:`w(\nu) = \nu^2/(\nu^2 + \nu_c^2)`.  It is applied to the translational
channel and, for molecules, the rotational channel (using the free rigid-rotor
gas weight).  The gas/solid weights :math:`W_g, W_s` are a physical-units
construction (cm\ :sup:`-1` frequencies, Kelvin, the de Broglie thermal
wavelength), so the cage is **skipped in LJ reduced units** (a one-time warning
is logged and *mode 3PT* reduces to its *mode 2PT refinement rigorous* base);
all non-LJ unit styles are supported.  3PT reproduces the py-xPT post-processor
to within 0.05 % (liquid argon) / 0.3 % (water solvent).  The cage density of
states is written to the *cage* columns of *prefix*\ .pwr, and the cage entropy
correction that was folded into the entropy is reported in the *S_cage_t* /
*S_cage_r* columns of *prefix*\ .thermo.

.. note::

   **Low-frequency sign structure** (verified on MB-pol water at 298 K).
   :math:`\text{cage}(0) = 0` is exact by construction
   (:math:`F_K(0) = F_M(0) = 1/\gamma` — the diffusive pole belongs to the
   2PT gas component), and the unclipped excess is a zero-sum redistribution
   (:math:`\int (F_K - F_M)\,d\omega = 0`).  The *translational* channel has
   no low-frequency clipping: :math:`\gamma` is band-scale, the excess is
   positive from :math:`\nu = 0^+` and the compensating deficit sits above
   the band.  The *rotational* channel clips everything below the libration
   band (:math:`\nu_+ \approx 319` cm\ :sup:`-1` at 298 K):
   :math:`\gamma_{\rm rot} \gg` band makes :math:`F_M` essentially flat, so
   the sub-band mobility deficit — spectral weight trapped by the cage and
   pushed up into the librations — is negative excess.  The deficit is the
   caging seen from below.

The bare cage integrates the friction kernel to an automatic cutoff (the last
reliable lag of the guarded Volterra inversion).  A *smooth* spurious kernel
tail past the main lobe — coherent, so undetected by the inversion's noise/swing
guards — would inflate the friction :math:`\gamma = \tilde K(0)` and over-count
the cage (the SPC/E-water failure mode: :math:`\gamma` ~4×, translational cage
2.66 → 1.36 J/mol/K).  A built-in safeguard compares :math:`\gamma` at the auto
cutoff with :math:`\gamma` at the main-lobe cutoff (first
:math:`|K| < 0.02\,|K(0)|`); if they differ by more than a factor of two it
falls back to the truncation-robust main-lobe cutoff and logs a warning.

----------

**Multi-tau VACF correlator (correlator keyword)**

The default *correlator fft* path stores a
:math:`N_\text{frames} \times N_\text{atom} \times 3` velocity history
plus per-channel molecular buffers (spread over the ranks, about 1/P each,
with *buffer_layout distributed*), and computes the VAC via the
Wiener-Khinchin FFT pipeline.  Fine for short runs and small systems;
the memory bottleneck on long supercooled-water trajectories, multi-ns
slabs, and 10 k-atom molecular systems.

Setting *correlator multitau* switches to a small ring-buffer
correlator that emits log-spaced VAC lags, then interp-merges them
onto the uniform-lag grid that ``dos_from_vac`` expects — downstream
analysis paths are unchanged.  Implementation follows
Magatti-Schaetzel / Ramirez-Likhtman (per-step ring buffer with
averaging downsample to next level; biased estimator preserves L = 1
bit-exactness vs FFT).

Knobs:

- *mt_M*: base block size per level (default 32).
- *mt_P*: inter-level overlap (default 16).  Recommended pairing
  ``mt_M 64 mt_P 64`` (M + P = 128) gives <= 1 % D drift on
  monatomic and sub-percent on per-channel molecular entropies;
  ``mt_M 32 mt_P 16`` (default M + P = 48) is fine for memory-tight
  runs.
- *mt_S*: sample-rate reduction factor between levels (default 2).
- *mt_L*: number of levels (default 0 = auto-derive
  :math:`\lceil \log_S(N_\text{frames}/M) \rceil`).

Coverage: all four VAC channels — scalar trace, mol trans (COM), mol rot
(angular velocity), mol vib (residual) — are routed through the same
algorithm.  When
``mt_M + mt_P >= nframes`` the multi-tau path is bit-exact vs FFT on
every thermo column (used as a regression sanity test).

Typical memory win on 200-mol SPC/Ew x 10 k frames: ~240 MB FFT path
-> ~30 MB multi-tau.  On 10 k-atom systems several GB -> ~150 MB
(15-30 x).

----------

**Restart reanalysis (run 0)**

The multi-tau correlator state (the accumulated VAC, plus the molecular
per-molecule inertia) is written to :doc:`binary restart files <restart>`,
so a finished — or in-progress — trajectory can be **re-analysed without
re-running the MD**.  Re-declare the same *fix xpt* after
:doc:`read_restart <read_restart>` and issue a bare ``run 0``: the fix
harvests the restored correlator state and immediately writes a
*prefix*\ .thermo / .pwr / .vac block (no integration).  A multi-week
trajectory re-analyses in seconds, which is useful for applying analysis-code
fixes to an existing restart.

*mode* is **not** part of the restart-match key — it only controls how the
mode-agnostic accumulated VAC/DoS is partitioned at analysis time.  Several
fixes with different IDs and different *mode* values can therefore be
re-declared against one restart and emitted from a single ``run 0`` (e.g. one
``mode 2PT`` and one ``mode 3PT``).

For the restore to take (otherwise the saved state is silently dropped with
``Unused restart file global fix info`` and a trivial all-zero block results):

- The **fix ID must match** the ID used in the run that wrote the restart —
  LAMMPS keys restart fix state by ID.  The output *prefix* is independent.
- **nframes must equal** the value used in the original run; the multi-tau
  level count is auto-derived from it and is a restart-shape match key.  A
  mismatch logs ``restart multi-tau shape mismatch ... discarding state``.
- The *correlator multitau* and *mt_** settings must match the original fix.

The restart holds the correlator sums reduced over all ranks, so reanalysis
is exact on any number of ranks, independent of the rank count that wrote it.
A restart written by fix-xpt 1.0.0 holds only rank 0's partial sums; it is
restored on one rank and refused, with a log message, on more.

KOKKOS (``xpt/kk``) note: a restart written on another machine may store the
fix style unstripped (``xpt/kk``).  Re-run **without** ``-sf kk`` (so the live
style is not stripped to ``xpt`` and stops matching the stored string), keep
KOKKOS active with ``-k on``, and request the integrator explicitly with
``run_style verlet/kk``.

.. code-block:: LAMMPS

   read_restart   prod.restart
   pair_style     ...            # re-declare force field
   pair_coeff     ...            # (read_restart must precede pair_coeff)
   run_style      verlet/kk      # only when running KOKKOS without -sf kk
   fix  m2 all xpt 4 25000000 out.2pt molecule symmetry C2v mode 2PT &
        correlator multitau mt_M 64 mt_P 64 mt_S 2
   fix  m3 all xpt 4 25000000 out.3pt molecule symmetry C2v mode 3PT &
        correlator multitau mt_M 64 mt_P 64 mt_S 2
   run 0

Restarts written before molecular-inertia persistence are reconstructed from
the restart geometry at ``run 0`` (one snapshot, a close approximation of the
window-averaged inertia for near-rigid molecules).

----------

**Output files**

Three files are written with prefix *prefix*:

``prefix.thermo`` — thermodynamic summary, one row per analysis window.
The file header contains a ``# Units:`` line describing the output units for
the current unit style, and classical/ZPE column labels carry inline unit
tags (e.g. ``S_c(eV/K)`` for metal, ``A_c(kJ/mol)`` for real).
Column order:

.. parsed-literal::

   Step  S_q  A_q  E_q  Cv_q  D  fluidicity  T_vac  Natom  V  P_avg  E_md
     [if classical: S_c  A_c  E_c  Cv_c]
     [if molecule:  S_trans  S_rot  S_vib  f_rot  D_rot]
     [if show_split: S_trans_gas  S_trans_sol  S_rot_gas  S_rot_sol]
     mu_q  mu_c  ZPE_q

``prefix.pwr`` — power spectrum (density of states), one block per window.

``prefix.vac`` — velocity autocorrelation, one block per window.

----------

.. _fix-2pt-restart:

Restart, fix_modify, output, run start/stop, minimize info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

No information is written to :doc:`binary restart files <restart>`.

The :doc:`fix_modify` *energy* and *virial* options are not supported.

This fix computes a global vector of length 10 (monoatomic), 14 (molecular
mode), or 18 (molecular mode + *show_split*).  The vector can be accessed by
various :doc:`output commands <Howto_output>`.  The values in the vector are
"intensive" in the sense that they are per-atom or per-molecule when *normalize*
is specified; otherwise they are extensive (total for the group).

The vector components (1-based thermo index) are listed in the table below.
All values are zero until the first analysis window is complete.

**Monoatomic mode (10 elements):**

.. list-table::
   :header-rows: 1
   :widths: 8 20 28 20 20

   * - Index
     - Quantity
     - Unit (real / si / cgs / …)
     - Unit (metal)
     - Unit (lj)
   * - 1
     - S_q — quantum entropy
     - J mol\ :sup:`-1` K\ :sup:`-1`
     - eV K\ :sup:`-1`
     - k\ :sub:`B` atom\ :sup:`-1`
   * - 2
     - A_q — quantum Helmholtz free energy
     - kJ mol\ :sup:`-1`
     - eV
     - :math:`\varepsilon` atom\ :sup:`-1`
   * - 3
     - E_q — quantum internal energy
     - kJ mol\ :sup:`-1`
     - eV
     - :math:`\varepsilon` atom\ :sup:`-1`
   * - 4
     - Cv_q — quantum heat capacity
     - J mol\ :sup:`-1` K\ :sup:`-1`
     - eV K\ :sup:`-1`
     - k\ :sub:`B` atom\ :sup:`-1`
   * - 5
     - D — self-diffusivity
     - cm\ :sup:`2` s\ :sup:`-1`
     - cm\ :sup:`2` s\ :sup:`-1`
     - :math:`\sigma^2 \tau^{-1}`
   * - 6
     - f — fluidicity
     - —
     - —
     - —
   * - 7
     - T_vac — temperature from VAC(0)
     - K
     - K
     - :math:`\varepsilon k_B^{-1}`
   * - 8
     - mu_q — quantum chemical potential
     - kJ mol\ :sup:`-1`
     - eV
     - :math:`\varepsilon` atom\ :sup:`-1`
   * - 9
     - mu_c — classical chemical potential
     - kJ mol\ :sup:`-1`
     - eV
     - :math:`\varepsilon` atom\ :sup:`-1`
   * - 10
     - ZPE_q — zero-point energy
     - kJ mol\ :sup:`-1`
     - eV
     - :math:`\varepsilon` atom\ :sup:`-1`

**Molecular mode (14 elements, indices 1–7 same as above):**

.. list-table::
   :header-rows: 1
   :widths: 8 20 28 20

   * - Index
     - Quantity
     - Unit (real / si / cgs / …)
     - Unit (metal)
   * - 8
     - S_trans — translational entropy
     - J mol\ :sup:`-1` K\ :sup:`-1`
     - eV K\ :sup:`-1`
   * - 9
     - S_rot — rotational entropy
     - J mol\ :sup:`-1` K\ :sup:`-1`
     - eV K\ :sup:`-1`
   * - 10
     - S_vib — vibrational entropy
     - J mol\ :sup:`-1` K\ :sup:`-1`
     - eV K\ :sup:`-1`
   * - 11
     - f_rot — rotational fluidicity
     - —
     - —
   * - 12
     - mu_q — quantum chemical potential
     - kJ mol\ :sup:`-1`
     - eV
   * - 13
     - mu_c — classical chemical potential
     - kJ mol\ :sup:`-1`
     - eV
   * - 14
     - ZPE_q — zero-point energy
     - kJ mol\ :sup:`-1`
     - eV

**Molecular mode + show_split (18 elements, ZPE_q always last):**

.. list-table::
   :header-rows: 1
   :widths: 8 35 24 16

   * - Index
     - Quantity
     - Unit (real / si / cgs / …)
     - Unit (metal)
   * - 12
     - S_trans_gas — translational gas-component entropy
     - J mol\ :sup:`-1` K\ :sup:`-1`
     - eV K\ :sup:`-1`
   * - 13
     - S_trans_solid — translational solid-component entropy
     - J mol\ :sup:`-1` K\ :sup:`-1`
     - eV K\ :sup:`-1`
   * - 14
     - S_rot_gas — rotational gas-component entropy
     - J mol\ :sup:`-1` K\ :sup:`-1`
     - eV K\ :sup:`-1`
   * - 15
     - S_rot_solid — rotational solid-component entropy
     - J mol\ :sup:`-1` K\ :sup:`-1`
     - eV K\ :sup:`-1`
   * - 16
     - mu_q — quantum chemical potential
     - kJ mol\ :sup:`-1`
     - eV
   * - 17
     - mu_c — classical chemical potential
     - kJ mol\ :sup:`-1`
     - eV
   * - 18
     - ZPE_q — zero-point energy
     - kJ mol\ :sup:`-1`
     - eV

.. note::

   ZPE_q is always the last element of the output vector, regardless of which
   optional columns are active.  In molecular mode, ZPE_q is the sum of the
   solid-component zero-point energies from the translational, rotational, and
   vibrational channels, each evaluated at its own component temperature.

This fix is not invoked during :doc:`energy minimization <minimize>`.

No information is available for the *run start/stop* option.

Protocol notes
""""""""""""""

Two LAMMPS-deck details affect every *fix xpt* run.  Vanilla LAMMPS defaults
will silently produce wrong thermodynamics unless addressed.

**1. Suppress center-of-mass drift — REQUIRED for accurate D.**  Add
:doc:`fix momentum <fix_momentum>` to every 2PT-driving script:

.. code-block:: LAMMPS

   fix mom all momentum 100 linear 1 1 1

Without this, COM drift over multi-ns NVE/NVT trajectories inflates *D* by
1-3 orders of magnitude — *fix xpt* has no internal mechanism to subtract
group-COM motion from VAC(0).

**2. Match the thermostat's DoF convention to fix xpt — REQUIRED for small
flexible solute groups.**  *fix xpt* infers the group temperature from
equipartition on VAC(0) using

.. math::

   \text{DOF} = \text{dim} \cdot N - \text{fix\_dof}, \qquad
   T_\text{vac} = \frac{\text{vac}[0]}{R \cdot \text{DOF}}

(constraint DoF subtracted; the COM dim is **not** subtracted).
LAMMPS's default :doc:`compute temp <compute_temp>` — which
:doc:`fix nvt/npt <fix_nh>` thermostats target — uses
``DOF = dim*N - dim``.  For a flexible group with no SHAKE, the two
conventions differ by :math:`(N - 1)/N`, so *T_vac* reads systematically
below the thermostat target by that ratio.

This is invisible for *N* ≥ ~ 100 (< 1 % bias) but severe for small
solute groups: an 11-atom flexible solute at *T* = 298 K reports
*T_vac* ≈ 271 K → ~ 9 % bias propagated into *S_q*, *A_q*, *μ_q*,
*ZPE_q*.

Workaround (per group, until the upcoming *dof_convention* keyword
lands): pair the thermostat with a custom temperature compute that
suppresses the COM-DoF subtraction, then attach via *fix_modify*:

.. code-block:: LAMMPS

   compute        tsolu solute temp
   compute_modify tsolu extra/dof 0          # drop the dim COM subtraction
   fix            nvtSolu solute nvt temp ${T} ${T} 50.0
   fix_modify     nvtSolu temp tsolu

Repeat for the solvent group (cosmetic for large *N*, but keeps the
two-group convention consistent).  Especially important for solvation
free-energy cycles where each leg's absolute thermo enters
:math:`\Delta G_\text{solv}`.

Restrictions
""""""""""""

This fix is part of the XPT package.  It is only enabled if LAMMPS was
built with that package.  See the :doc:`Build package <Build_package>` page
for more info.

The fix requires that LAMMPS was built with FFT support (FFTW3, MKL, or the
built-in KISS FFT).

The *molecule* keyword requires ``atom_style molecular`` or ``atom_style full``.

The fix follows atoms by ID and weights them by per-type mass, so it requires
atom IDs (``atom_modify id yes``, the default) and per-type masses (not
per-atom ``rmass``).

The Desjarlais memory-function model (*mode 2PT refinement desjarlais*) is valid for
both monoatomic and molecular modes.  It incurs a small overhead from the
bisection search for :math:`B_g` (at most 80 iterations per window per component).

.. note::

   Using *volume v_name* requires the named variable to be an equal-style
   LAMMPS variable defined before the fix is declared.  The variable is
   evaluated once per analysis window at the time of writing results.

Related commands
""""""""""""""""

:doc:`fix ave/time <fix_ave_time>`, :doc:`fix ave/correlate <fix_ave_correlate>`,
:doc:`compute vacf <compute_vacf>`, :doc:`thermo_style <thermo_style>`

Default
"""""""

The option defaults are *mode* = ``2PT``, *refinement* = ``rigorous``,
*symmetry* = ``C1`` (:math:`\sigma_\text{rot}` = 1 derived from the symmetry
lookup), *correlator* = ``fft`` (with *mt_M* = 32, *mt_P* = 16,
*mt_S* = 2, *mt_L* = 0 = auto when *correlator multitau* is selected),
*buffer_precision* = ``fp64`` and *buffer_layout* = ``distributed``.
The *classical*, *normalize*, *molecule*, *linear*, and *show_split* flags
are off.  The volume is taken from the simulation box.

----------

.. _Lin2003fixxpt:

**(Lin 2003)** S. T. Lin, M. Blanco and W. A. Goddard, J. Chem. Phys., 119,
11792 (2003).

.. _Lin2010fixxpt:

**(Lin 2010)** S. T. Lin, P. K. Maiti and W. A. Goddard, J. Phys. Chem. B,
114, 8191 (2010).

.. _Pascal2011fixxpt:

**(Pascal 2011)** T. A. Pascal, S. T. Lin and W. A. Goddard, Phys. Chem. Chem.
Phys., 13, 169 (2011).

.. _Desjarlais2013fixxpt:

**(Desjarlais 2013)** M. P. Desjarlais, Phys. Rev. E, 88, 062145 (2013).

.. |hbar| replace:: ħ
