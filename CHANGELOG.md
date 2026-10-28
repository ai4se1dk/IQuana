# Changelog

All notable changes to this project are documented in this file.

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
  the repository root, used by the notebooks and examples.

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
