# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-05-12

### Changed

- Renamed package from `gdx` to `gdalxarray` for PyPI publication. The original
  `gdx` name is taken on PyPI by an unrelated GAMS Data Exchange project.
- Repository moved from `mdsumner/gdx` to `hypertidy/gdalxarray`.
- Python import path is now `from gdalxarray import GDALBackendEntrypoint`.
- Build backend switched from setuptools to hatchling; `setup.py` removed.
- Packaging modernised to PEP 621 / PEP 639 standards.
- `requires-python` bumped to `>=3.10` to match `xarray>=2025.6`.

### Added

- CF datetime decoding for time coordinates using `units` from
  `MDArray.GetUnit()` and `calendar` attribute.
- Backend arrays accessible via `ds['var'].encoding['gdal_backend']` for
  debugging and introspection.
- GDAL dataset and group objects retained in `ds.encoding['gdal_dataset']`
  and `ds.encoding['gdal_group']` to keep `MDArray` methods functional.
- Entry point registration so `xr.open_dataset(..., engine="gdal")` works.

### Fixed

- Slice index parsing where `0` was incorrectly treated as `None` due to
  Python's falsy evaluation (`k.start or 0` → `k.start if k.start is not None else 0`).
- Re-enabled `AdviseRead` for chunk-aligned prefetching on remote datasets.

## [0.1.0] - 2026-01-20

Initial release as `gdx`.

### Added

- GDAL backend for xarray, supporting both Classic and Multidimensional APIs.
- `chunks={}` uses native block sizes from GDAL's `GetBlockSize()`, aligning
  Dask chunks with storage chunks for efficient reads.
- `multidim=True` is the default for `open_dataset()`.

### Fixed

- Dask lazy loading for remote Zarr datasets. Zero-sized slice requests (used
  by Dask for `_meta` inference) no longer hang or attempt full array allocation.
- Slice start/stop of `0` now parsed correctly.

[Unreleased]: https://github.com/hypertidy/gdalxarray/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/hypertidy/gdalxarray/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/hypertidy/gdalxarray/releases/tag/v0.1.0