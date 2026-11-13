# Quantities, decisions and limitations

## Measured-strain harmonic response

Let the measured axial engineering strain be
`epsilon_z(t) = (L_z(t)-L_z0)/L_z0` and let the tensile-positive stress be
`sigma = -P × 0.101325 MPa/atm` for LAMMPS `units real`.
For `theta = 2 pi f t` and integer harmonic n, projection gives

```
epsilon_hat_n = 2 mean(epsilon sin(n theta)) + i 2 mean(epsilon cos(n theta))
sigma_hat_n   = 2 mean(sigma   sin(n theta)) + i 2 mean(sigma   cos(n theta))
M_1* = sigma_hat_1 / epsilon_hat_1 = M_1' + i M_1''.
```

The quotient uses the actual measured strain amplitude **and phase**. Under
`epsilon=A sin(theta)`, positive loss means stress contains `+A M'' cos(theta)`;
the first-harmonic work density per cycle is `pi A² M''` (MPa = MJ/m³).
For nonsinusoidal strain, full spectral work contains contributions from each
matching stress/strain harmonic; `pi A² M''` alone is no longer the complete work.

Uniform integer-cycle projection equals least squares with a constant and the
same sine/cosine basis. Adding a linear trend changes the statistical model;
its overlap with finite-window sinusoids can alter H1. This is exposed as a
diagnostic rather than hidden baseline subtraction. Sampling flags and residuals
are numerical diagnostics, not physical validation. Harmonic magnitudes are
positive and can contain noise; H2/H1 or H3/H1 point estimates alone are not
significance tests or proof of nonlinear mechanisms.

## Boundary conditions and tensor conversion

Fixed lateral strains give **M***, the constrained longitudinal response.
Using simultaneous normal stresses, define

```
D* = [sigma_zz* - (sigma_xx*+sigma_yy*)/2] / epsilon_zz*
G* = D*/2
K* = M* - 4 G*/3
E* = 9 K* G* / (3 K* + G*)
nu* = (3 K* - 2 G*) / [2(3 K* + G*)].
```

These are exact identities for a linear, isotropic material. Applied to finite
amplitude H1 they are **isotropic-equivalent candidates**. Isotropy, linearity,
finite-age stability, adequate precision and nonzero denominators must be
qualified separately. M precision does not automatically transfer to G/E,
because tensor subtraction can amplify uncertainty. No experimental static
Poisson ratio is silently substituted.

Total LAMMPS pressure includes kinetic and virial terms; `compute pressure NULL
virial` omits the kinetic contribution. With the prescribed sign convention,
both are retained separately. The difference can be dominated by an isotropic
kinetic bulk-storage term under specific isothermal conditions, but is not
assumed invariant with temperature, frequency, streaming or thermostat choices.

For a homogeneous continuum with `exp(i omega t-i k x)` convention,
`k=omega sqrt(rho/M*)`, phase speed is `omega/Re(k)` and attenuation is
`-Im(k)`. These finite-loss formulas are preferable to treating
`sqrt(M'/rho)` as exact. Homogeneous driven strain is a q→0 observable;
Brillouin experiments use finite acoustic wavevectors and require matched
temperature, pressure, density, morphology, frequency and deformation observable.

## Finite preparation and runtime

Rg is the equal-backbone-site radius of gyration of each chain,
`Rg² = mean_j |r_j - r_center|²`; Re is the magnitude of the vector joining
the two backbone ends. Reported `mean_Rg` is the mean of each chain's Rg,
distinct from `sqrt(mean(Rg²))`. Backbone connectivity is used to unwrap
periodic chains. Neither Rg nor Re has a polymer-independent target value.
Dependence on chain length, chemistry and preparation history is expected.
The bond-orientation tensor is `Q=(3/2)mean(u uᵀ)-(1/2)I`; its maximum
eigenvalue is a descriptive anisotropy measure, not a crystallinity fraction.

Preparation checks compare consecutive fixed-temperature windows. Their
tolerances must name a quantity and purpose; they are not universal physical
constants. NVT density is imposed, while NPT density may continue changing.
Apparent chain immobility can mean an arrested nonequilibrium state.
Fixed preparation is therefore followed by diagnostics and, where justified,
bounded additional conditioning; no global equilibration certificate is issued.

Runtime diagnostics average joint sine/cosine stress/strain coefficients within
nonoverlapping physical-time blocks, then resample those block vectors together.
The default 200 ps block is an inherited diagnostic choice, **not** a measured
universal decorrelation time. Intervals are conditional on one history and the
block-resampling model. They do not include independent-packing uncertainty.
Early/late contrasts use disjoint halves, avoiding the automatic agreement of
heavily overlapping cumulative windows. A small increment from one extra cycle
can occur even during slow drift because that cycle's weight shrinks as 1/N.

Explicit precision targets compare the larger 95% interval reach with a chosen
fraction of |estimate|. Consistency targets require the whole early/late
difference interval to lie inside the chosen tolerance. Neither a failed
containment test nor an ordinary zero-containing difference interval establishes
aging or stationarity. No targets means descriptive analysis only. At least six
blocks is an implementation safeguard for finite-window qualification, not a
general guarantee of bootstrap calibration. Weak/signed quantities need absolute
error criteria or an explicit unresolved designation.

## TTS and experimental anchors

Horizontal TTS uses `f_reduced = f × a_T`. This package compares independent
storage/loss shifts against one shared shift using only measured overlap.
No overlap is extrapolated. Separate numerical shifts can reveal failure of a
single thermorheologically simple description, but do not uniquely identify
its physical cause. Noise, morphology/history differences and model inadequacy
must also be considered. The diagnostic does not claim uncertainty-calibrated
TTS acceptance.

An edge maximum is censored: its true peak may lie outside the measured range.
A maximum of M'' versus temperature at one frequency is a different slice
through the response surface and does not identify an absolute relaxation time.
A tan(delta) maximum and an M'' maximum need not coincide. None is assigned
automatically to an alpha relaxation, melting or Tg.

With explicit WLF anchors,

```
log10(f/f_ref) = C1 (T-T_ref) / [C2 + (T-T_ref)]
T = T_ref + C2 log10(f/f_ref) / [C1-log10(f/f_ref)].
```

The same observable/relaxation criterion must be represented at both anchors.
For example, the convention tau_g=100 s with f=1/tau gives f_ref=0.01 Hz;
the omega tau=1 convention instead gives f_ref=1/(2 pi ×100) Hz.
Constants defined at Tg+50 K require a consistently transformed reference
frequency as well as the shifted temperature. This command is an anchored
sensitivity demonstration and never an experiment-free Tg prediction.
