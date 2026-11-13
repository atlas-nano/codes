# Reproduction routes

## Reduced scientific evidence

`dmax reproduce-data --output paper_source_tables` verifies the bundled reduced
CSV checksums and exports the tables with the source manifest. Then run
`dmax reproduce-figures --data paper_source_tables --output paper_figures`
to invoke the portable, source-bound plotting script. This route
reproduces reported reduced numerical evidence without regenerating atomistic
trajectories. Figure generation must preserve operator, amplitude, chemistry,
architecture, packing, preparation ancestry and the retained time aperture.
Temporal intervals and sample scatter across independent packings are different
quantities and must retain their original labels.

## New MD realizations of the workflow

The `dmax init`, `build`, `plan`, `run`, `analyze`, `aggregate`, `plot-summary`
and `tts-diagnostics` commands in the README implement the portable workflow.
The starting-coordinate manifest pins seeds, caps, topology, force-field assets
and executable versions where available. All generated input and execution
receipts are retained. New independently built packings statistically reproduce
a protocol; they do not recreate historical binary checkpoint bit patterns.
Coordinates, LAMMPS/MPI build and floating-point ordering can change trajectories.

The publication's 432-history paper grid was measured at **1% strain**, not the
new 2% candidate. It used **2.4 ns** complete records with **400–2400 ps retained**,
0.5 fs integration and 2 fs synchronous acquisition. To plan an archival-style
measurement set explicitly set `measurement.strain_amplitudes` to `[0.01]`,
`measurement.runtime.policy` to `"fixed_duration"` and
`measurement.runtime.maximum_ps` to `2400` before `dmax plan`; analyze with
`--start-ps 400`. Preserve each case's actual temperature/frequency, parameter
cutoff, neighbor policy and finite preparation history from the accompanying
table/manifests rather than substituting a universal new preset.

No case is upgraded to 2% by changing its label. The independent amplitude
controls at 300/400 K are a distinct finite-amplitude study. A 15-cycle control
does not automatically satisfy a long-record temporal precision criterion.

## Scientific acceptance

Numerical tests and successful LAMMPS exits establish implementation checks,
not the physical validity of a polymer model. The paper reports constrained
longitudinal response, conditional tensor conversions, finite-amplitude/age
limits, and matched-observable experimental comparisons. Recreating a plot does
not create a new equilibrium, isotropy, linearity, alpha-relaxation or Tg result.
Frequency-edge maxima remain censored and master-curve/Tg extrapolation is not
enabled automatically. Raw trajectories/checkpoints and complete historical
allocation evidence remain in the external research archive; the lightweight
package contains source tables and portable protocol inputs, not that archive.
