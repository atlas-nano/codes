# Shielding-kernel comparison — the gas-phase TIP4P-FQ monomer

A self-contained, two-run example showing that the shielding kernel is a **run-time setting**
whose effect can be measured in isolation. It is the source of the kernel-comparison table in the
samQEq method paper.

## What is here

| File | Role |
|---|---|
| `in.gas_slater` | monomer solved with the exact Slater kernel |
| `in.gas_cbrt` | the same monomer, same parameters, cube-root kernel |
| `water4.mol` | the TIP4P-geometry monomer template |
| `water4_gas_slater.param` | the parameter file, **shared by both decks** |
| `log.slater`, `log.cbrt` | reference output, for comparison against your own |

## Running it

```
LMP=<your samQEq-enabled lmp>
$LMP -in in.gas_slater -log log.slater
$LMP -in in.gas_cbrt -log log.cbrt
grep -hE "GAS_SLATER|GAS_CBRT" log.slater log.cbrt
```

Both are single-point solves on one molecule in vacuum; they finish in seconds.

## Expected output

```
GAS_SLATER qM=-0.88870 qH1=0.44435 qH2=0.44435 mu_D=1.86112 pe_eV=-4.396359
GAS_CBRT qM=0.91109 qH1=-0.45555 qH2=-0.45555 mu_D=1.90801 pe_eV=0.859385
```

| kernel | q_M (e) | q_H (e) | mu (D) | pe (eV) | vs published 1.85 D |
|---|---|---|---|---|---|
| `slater`, using Rick's own zeta | -0.88870 | +0.44435 | **1.86112** | -4.396359 | +0.6% |
| `cbrt`, same numbers read as gamma | **+0.91109** | -0.45555 | 1.90801 | **+0.859385** | +3.1%, **charges inverted** |

The `pe` column includes the on-site term $\sum_i(\chi_i q_i + \frac{1}{2}\eta_i q_i^2)$, which
counts toward `pe` and `etotal` by default. It exerts no force and contributes no virial, but it is
part of the conserved quantity; `fix_modify <id> energy no` excludes it. Of note, the inverted cube-root solve is driven to a **positive** total energy, which the
charge signs alone do not show.

## What the two runs differ by

One substantive line. The full `diff` shows two hunks:

```
$ diff in.gas_slater in.gas_cbrt
27c27
< fix_modify chg shield slater 2s 1 3
---
> fix_modify chg shield cbrt
37c37
< print "GAS_SLATER qM=..."
---
> print "GAS_CBRT qM=..."
```

Line 27 is the kernel selection and is the only physical difference. Line 37 changes nothing but
the label on the print statement, so the two logs can be told apart. Molecule, parameter file,
solve tolerance and every other setting are identical.

## What it shows

The dipole magnitudes are close. The **charge distribution is sign-inverted**. With
chi_M = 7.47 eV the M site is the most electronegative in the model and must carry negative
charge, which `slater` gives and `cbrt` does not.

The cause is the definiteness of the operator. Read as inverse shielding lengths, Rick's Slater
exponents put the cube-root kernel far above the on-site hardness at contact:

| pair | cbrt J(0) | eta | J/eta |
|---|---|---|---|
| H-M | 32.96 eV | 15.31 eV | 2.15 |
| M-M | 44.36 eV | 16.11 eV | 2.75 |
| H-H | 24.49 eV | 15.31 eV | 1.60 |

Contact values alone do not decide stability: samQEq's native water (`examples/water_native/`,
gamma = 1.11 A^-1, J(0) = 15.98 eV, eta_O = 15.11 eV) has J(0)/eta_O = 1.06 and is stable. What decides
it is the operator restricted to the neutral subspace, at the molecule's actual geometry. For the three
solved sites here (H, H, M), the cube-root operator with Rick's exponents has eigenvalues -2.18 and
+5.98 eV/e^2: it is indefinite, the stationary point is a saddle rather than a minimum, and the solve
inverts. The native monomer's eigenvalues are +2.26 and +7.79 eV/e^2, because at the O-H distance the
coupling has fallen to 11.99 eV, below eta. (Both are the charge-equilibration limit of the solve,
computed from the parameter files and geometries.)

**The point is not that `cbrt` is inferior.** A Slater exponent and an inverse shielding length
are parameters of *different kernels*. Reusing one number under the other is not a small
approximation but a different model, and here a qualitatively wrong one. That is exactly why the
kernel is exposed as a setting rather than welded into the code: the substitution is easy to make
by accident and impossible to detect from the dipole magnitude alone.

## Related

The three parameter interpretations are also exercised in `../tip4pfq_liquid/`, which holds the
liquid-phase decks and the `water4_liquid_*.param` variants.
