# DMAx reduced paper data and portable figure reproduction

This bundle reproduces the revised-paper plots from frozen reduced measurements. It does not rerun molecular dynamics or refit trajectories, shifts, relaxation laws or confidence intervals. The numerical tables preserve source identity, available seeds, loading/operator/window fields and scientific exclusion flags.

Run from any working directory, using Python with NumPy, pandas and Matplotlib:

```sh
python reproduce_figures.py --output ./reproduced_figures
```

For an exported copy of the data:

```sh
python reproduce_figures.py --data ./exported_paper_data --output ./reproduced_figures
```

`--data` must contain `sources/` and `source_manifest.json`. The runner verifies every source CSV checksum before plotting. No site-specific paths, scheduler access or raw trajectories are required. `manifest.json` lists all distributable bundle members; validation rendering directories are deliberately excluded.

## Figure roles

- Main Figure 1: `Figure01_preparation`.
- Main Figure 2: `fig02_constrained_M_heatmaps_and_H1`.
- Main Figure 3: `fig03_native_temperature_modern`.
- Main Figure 4: `PDMS_shared_clock_counterexample` (saved separate versus shared shifts, without new fitting).
- Main Figure 5: `fig05_amplitude_modern_T300` and `fig05_amplitude_modern_T400`.
- Main Figure 6: `fig06_runtime_precision`.
- Main Figure 7: `fig07_architecture_density_storage`.
- Main Figure 8: `Figure08_workflow`.
- Main Figure 9: `fig09_sampling_component_offsets`.
- Main Figure 10: `fig10_packing_uncertainty`.
- Main Figure 11: `Figure11_native_acoustic_comparison`.
- Supplementary harmonic assessment: `figS_amplitude_harmonics_modern`.
- Supplementary loss-frequency/edge-censor audit: `figS_native_frequency_loss_censoring`, with its 144-coordinate mean/SD table. All 24 packing-mean and 72 packing-specific loss-frequency curves have sampled maxima at 500 GHz.
- Supplementary all-system candidate-shift diagnostic: `Figure04_TTS_diagnostics` (distinct from main Figure 4; intended Fig. S11a).
- Supplementary six-pilot dense-reference sampling diagnostic: `Figure09_sampling` (distinct from main Figure 9; intended Fig. S7b).

All figures have PNG and PDF companions. The runner also writes the reduced plot tables and `reproduction_report.json`; figure styling can depend on the installed Matplotlib/font version. `validation_report.json` records a successful run and numerical comparison with the frozen plot tables.

## Measurement scope and limits

The primary grid comprises 432 histories: six polymer architectures, four material-specific temperatures, six native frequencies (10–500 GHz), and three prepared packings. These measured histories used 1% strain, 0.5 fs integration, 2 fs stress acquisition, 2.4 ns driven duration and the retained 400–2400 ps window. The proposed 2% operating amplitude is a separate candidate assessment, not the grid input or a universally validated linear-response default.

The amplitude experiment includes 252 modern independent-packing histories and 126 historical velocity-seed histories, at seven amplitudes and 300/400 K. Their apertures differ: modern records use the first 15 cycles (0–600 ps), historical controls use the first 50 cycles (0–2000 ps). `amplitude_histories378.csv` selects total stress, 200 ps uncertainty blocks and no discarded cycles; it preserves this difference explicitly. Do not pool packing and velocity-seed populations, temperatures or observation windows. `amplitude_summary126.csv` contains the existing 126 condition means, not 126 raw runs.

Temperature-context data contain 155 histories (144 modern and 11 legacy-reference records). Room-temperature phase, packing repeatability and the native-curve censoring table are included. The runtime-prefix tables retain 80/200/400 ps block sensitivities and saved warning flags. The logging offsets are correlated downsamplings of the same paths, not independent confidence intervals. Historical anchor, density-breakpoint and PDMS peak-status tables support the M1/SI response and retain their historical context; they are not new primary-grid measurements.

The directly measured fixed-lateral axial response is the constrained longitudinal M*, not Young's E*. The tensor table includes conditional isotropic conversions and their existing validity warnings. The plotted H1 ellipses are zero-offset reconstructions from mean complex M*, not raw trajectories, and the geometric ellipse major axis is not E*. Between-packing sample SD is not a 95% population confidence interval. Temporal block intervals do not certify equilibrium or lack of aging. No shared complex TTS law, predictive Tg, common interior loss-tangent peak, universal linear-strain window or shorter stationary production duration is established by these exports.

The acoustic comparison is an observable-level, finite-loss local-continuum mapping. Its Sato density/speed scale is not a certified matched-frequency complex-modulus target. The 7.15 GHz imposed-density condition retains its high conditioning pressure and failed precision/distortion flags. All these limitations remain in the tables and captions.

## Provenance and sanitization

`source_manifest.json` records every local source CSV SHA256, row count, columns, original source basename and full original source SHA256. Intermediate reduced exports also record their upstream source SHA where available. Absolute storage locations embedded in string fields were replaced by basenames; available case IDs, seeds, source hashes, numerical values and scientific flags are retained. The bundle contains reduced tables rather than original atomic coordinates, trajectory files or full simulation inputs. It therefore reproduces the paper's reduced data/figures; exact MD replay requires the separately archived inputs and source configurations.
