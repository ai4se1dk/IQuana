# Changelog

All notable changes to this project are documented in this file.

## [0.6.0] - unreleased

This release adds 2D data and magnetic (polarised) SANS
([#21](https://github.com/ai4se1dk/IQuana/issues/21)). 1D analyses behave as
before, and analysis files written by 0.5 keep their fit results.

### Added

- **2D data.** `load_data()` and `set_data()` accept 2D detector images.
  `plot_model()` and `plot_results()` draw data, model and residual
  heatmaps; `fit()`, `fit_bayesian()`, `calculate()`, `save_results()`,
  `save_analysis()` and `report()` work on them. `set_q_range()` selects an
  annulus in |Q|. The fitter's `is_2d` property says which kind is loaded.
- Pixel masks for 2D data: `mask_ring()`, `mask_box()`, `mask_invalid()`,
  `set_mask()`, `reset_mask()` and `get_mask()`. The selection is fitter
  state, never written to the dataset's own mask, and is saved with the
  analysis. The pixel at |Q| = 0 is always excluded.
- `set_data(data, mask_convention=...)` for 2D objects whose mask uses the
  other convention.
- 2D resolution: `'data'` uses the radial and azimuthal widths, `'pinhole'`
  sets both to `dq_over_q·|Q|`; `set_resolution(..., accuracy=...)` sets the
  2D oversampling.
- **Magnetism.** `enable_magnetism(beam=..., efficiency_in=..., efficiency_out=...)`,
  `supports_magnetism()`, `get_magnetic_parameters()` and
  `is_magnetism_enabled()` expose sasmodels' magnetic parameters for 2D
  data, with beam presets for unpolarised, SANSPOL and polarisation-analysis
  experiments. A fit refuses a polarisation channel whose magnetisation can
  reach zero (sasmodels switches kernels there) and reports magnetic
  parameters the model does not respond to.
- `examples.simulate_2d()`, and the 2D examples `p123_2d`, `sans2d_2d`,
  `sanspol_plus`, `sanspol_minus` and `sanspol_unpol`.
- A fit warns when orientation parameters vary on 1D data, where they have no
  effect.
- `notebooks/magnetic_2d_fitting.ipynb` and `examples/magnetic_2d_example.py`.

### Changed

- The analysis file schema is version 3. It adds `data.dimension`, a `mask`
  section and, only when used, `configuration.magnetism` and
  `resolution.accuracy`. Versions 1 and 2 load as before, with their results.
- `examples.simulate()` rejects magnetic parameters, which it used to accept
  and silently ignore.
- `MultiFitter` refuses 2D data with a clear message before adding it.
- The minimum versions are now `sasmodels>=1.0.12` and `sasdata>=0.11`, the
  oldest the tests run against.

## [0.5.0] - 2026-10-28

This release fixes three silent fitting errors. Fits that use a structure
factor, the LMFit engine, or polydispersity can give different results from
0.4. Check the **Changed** and **Fixed** sections before you upgrade.

### Added

- Fits check every varying parameter before starting. A fit is refused, with
  the parameter named, if the parameter starts outside its bounds or if its
  bounds meet (`min == max`) or cross (`min > max`). The check runs for both
  engines and for `fit_bayesian()`.
  ([#1](https://github.com/ai4se1dk/IQuana/pull/1))
- `fit(engine='lmfit', method='leastsq')` warns when a fitted value ends up
  outside its bounds. ([#1](https://github.com/ai4se1dk/IQuana/pull/1))
- `set_pd_param()` warns when it is asked for a width or `vary=True` after
  polydispersity was switched off with `enable_polydispersity(False)`.
  ([#16](https://github.com/ai4se1dk/IQuana/pull/16))
- An integration test suite runs every `examples/*_example.py` script end to
  end. Run it with `pixi run test-integration` or
  `pytest tests/integration -m integration`. A GitHub workflow runs it on
  Linux, Windows and macOS for release pull requests, every Sunday, and on
  demand. ([#14](https://github.com/ai4se1dk/IQuana/pull/14))
- `simulated_sans_data.csv` and `simulated_sans_data_with_resolution.csv` in
  `tests/data`, used by the notebooks and examples.

### Changed

- **The LMFit engine's default method is now `'least_squares'`**, which keeps
  parameters inside their bounds. It was `'leastsq'`, which ignores bounds.
  Pass `method='leastsq'` to get the old behaviour.
  ([#1](https://github.com/ai4se1dk/IQuana/pull/1))
- **`set_pd_param()` switches polydispersity on** when it is given a width
  above zero or `vary=True`. A separate `enable_polydispersity(True)` call is
  only needed to switch it back on after switching it off.
  ([#16](https://github.com/ai4se1dk/IQuana/pull/16))
- Under the `'dI'` weighting mode, a missing or empty `dI` column is treated
  as all zeros, the same way sasdata represents a file without `dI`.
  ([#12](https://github.com/ai4se1dk/IQuana/pull/12))
- Analysis files saved by 0.4 that use a structure factor load with the
  `radius_effective_mode` kernel switch corrected (see **Fixed**) and with a
  warning. Their saved fit result is not restored, because the old kernel
  computed it; refit after loading.
  ([#13](https://github.com/ai4se1dk/IQuana/pull/13))
- The notebooks and the test helpers no longer add the repository to
  `sys.path`. Install the package (`pip install -e .` or `pixi install`)
  before running them. ([#2](https://github.com/ai4se1dk/IQuana/pull/2))
- The API reference for `ParameterManager` and the polydispersity constants
  (`PD_DEFAULTS`, `PD_DISTRIBUTION_TYPES`) is generated from the source code,
  so it lists the current method names.
  ([#3](https://github.com/ai4se1dk/IQuana/pull/3))

### Removed

- `run_tests.py`. Run the tests with `pytest` or `pixi run test`.
  ([#2](https://github.com/ai4se1dk/IQuana/pull/2))

### Fixed

- **A structure factor ignored the `radius_effective` parameter.** sasmodels'
  product kernel has its own `radius_effective_mode` switch, and its default
  computes the effective radius from the form factor. IQuana now sets the
  switch so `radius_effective` is used, both when it is free
  (`'unconstrained'`) and when it is linked to `radius` (`'link_radius'`).
  Results differ most for `'unconstrained'` and for non-spherical form
  factors. ([#13](https://github.com/ai4se1dk/IQuana/pull/13),
  [#9](https://github.com/ai4se1dk/IQuana/issues/9))
- **`set_pd_param()` settings were silently left out of the fit** until
  `enable_polydispersity()` was called.
  ([#16](https://github.com/ai4se1dk/IQuana/pull/16),
  [#9](https://github.com/ai4se1dk/IQuana/issues/9))
- The LMFit engine could return values outside the bounds set with
  `set_param()`, such as a negative radius.
  ([#1](https://github.com/ai4se1dk/IQuana/pull/1))
- The LMFit engine failed with a `TypeError` from sasmodels on data that has
  no `dI` column (`dy=None`). It now fits such data with unit weights and a
  warning, as it already did for zero uncertainties.
  ([#12](https://github.com/ai4se1dk/IQuana/pull/12))
- Removing a structure factor did not restore whether polydispersity had been
  switched on or off by the user, which could produce a misleading
  "switched off" warning.

## [0.4.0] - 2026-10-23

- First release as IQuana, carried over from SANS-fitter 0.4.

[0.5.0]: https://github.com/ai4se1dk/IQuana/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/ai4se1dk/IQuana/releases/tag/v0.4.0
