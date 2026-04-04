# Validation And Benchmarking

This directory defines a reproducible validation workflow for comparing the repository's differentiable Gaussian-beam solver against Bellhop on two benchmark cases:

- `munk`: canonical deep-water Munk-profile validation
- `dickins`: Dickins seamount range-dependent bathymetry case

Related documentation:

- `docs/solver_architecture.md`
- `docs/api_usage_guide.md`
- `src/running.md`

## Scope

The goal is to support validation artifacts for:

- full TL field comparisons
- TL-vs-range slices at fixed receiver depths
- error metrics against Bellhop reference fields
- solver runtime benchmarking

## Directory layout

- `cases.py`: benchmark case definitions and sampling grids
- `export_bellhop_inputs.py`: exports case manifests, SSP tables, bathymetry tables, and Bellhop input templates
- `import_bellhop_reference.py`: imports Bellhop `.shd` outputs into the CSV reference format used by the benchmark runner
- `read_shd_bin.py`: Bellhop binary shade-file reader
- `run_benchmarks.py`: runs the JAX solver, loads Bellhop reference grids if available, computes metrics, and writes figures/reports
- `metrics.py`: quantitative comparison metrics
- `reference_data/`: place Bellhop-exported reference grids here
- `results/`: generated JAX outputs, figures, and reports

## Case definitions

### `munk`

- Frequency: `50 Hz`
- Source depth: `1000 m`
- Flat bottom at `5000 m`
- Canonical Munk sound-speed profile

### `dickins`

- Frequency: `230 Hz`
- Source depth: `18 m`
- Dickins seamount bathymetry from the repository boundary model
- Default repository SSP

## Bellhop export workflow

1. Export benchmark inputs:

```bash
python validation/export_bellhop_inputs.py --case all
```

2. Use the files in `validation/bellhop_inputs/<case>/` to build and run the corresponding Bellhop cases.

Files written per case:

- `<case>.env`
- `<case>.json`
- `<case>_ssp.csv`
- `<case>_bathy.csv`

The `.env` files are repository-generated templates intended to standardize the case setup. They should be checked against the Bellhop version and Acoustics Toolbox conventions used in your validation environment before publication.

3. Import Bellhop `.shd` results into the validation reference directory:

```bash
python validation/import_bellhop_reference.py --case all
```

This writes:

- `validation/reference_data/<case>/rr_grid_m.csv`
- `validation/reference_data/<case>/rz_grid_m.csv`
- `validation/reference_data/<case>/tl_field_db.csv`

The reference field is stored as a `Nz x Nr` CSV with TL in dB.

## Running the benchmark suite

```bash
python validation/run_benchmarks.py --case all
```

To let the solver replace the requested beam count with its Bellhop-style recommended beam count:

```bash
python validation/run_benchmarks.py --case all --auto-beam-count
```

Outputs are written to `validation/results/<case>/`:

- `tl_field_db.csv`
- `*_jax_tl_field.png`
- `*_comparison_tl_field.png` when Bellhop reference data are available
- `*_slice_<depth>m.png`
- `*_report.json`
- `*_report.md`

## Metrics reported

For the full TL field and each requested TL-vs-range slice:

- RMSE
- MAE
- max absolute error
- 95th-percentile absolute error
- bias
- Pearson correlation

The report also includes solver runtime for the JAX solve.
It also records:

- requested beam count
- actual beam count used by the solver
- Bellhop-style recommended beam count
- whether auto beam count was enabled
- min/max launch amplitude from the source beam pattern normalization

## Validation guidance

For validation, the minimum standard should be:

- compare the full TL shade field against Bellhop on identical grids
- compare multiple TL-vs-range slices, not just one receiver depth
- report both local errors and aggregate field metrics
- report runtime and hardware context
- note any Bellhop options used for beam type, bottom loss, and surface/bottom reflection physics
- state explicitly whether the comparison is coherent TL, incoherent TL, or arrivals-derived TL

## Current limitations

- Boundary loss models and full Bellhop-equivalent field accumulation are still incomplete, so discrepancies are expected even after the benchmark pipeline is populated with Bellhop outputs.
- Source beam patterns are supported in the validation path, but the current benchmark cases still use the Bellhop default omnidirectional pattern unless case metadata are extended with explicit pattern tables.
