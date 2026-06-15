# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Classic-raster open (`multidim=False`) on a file with no bands but
  subdatasets now raises a `ValueError` listing the available
  subdataset paths and pointing at `multidim=True`. Previously this
  produced a confusing empty 512×512 stub Dataset. Mirrors what
  `terra::rast()` does on the R side: shows the user the structure
  they actually have and how to choose. (#issue-classic-raster-helpful)

## [0.3.0] - 2026-06-15

### Added

- `band_as_dim` parameter on `open_dataset` (classic raster mode, default `True`).
  Bands now become an xarray `band` dimension on a single `band_data` DataArray —
  the rioxarray-compatible idiom — letting `da.mean(dim="band")`, `da.sel(band=...)`
  and similar xarray operations work naturally. Pass `band_as_dim=False` for the
  legacy per-band-variable layout, which is preferable when bands carry
  semantically distinct quantities.
- New `GDALMultiBandArray` BackendArray reading via
  `dataset.ReadAsArray(band_list=...)`, letting GDAL handle BIP/BIL/BSQ
  interleaving internally — typically faster than iterating per-band.
- Lazy-by-default for classic raster mode: `chunks=None` now returns a
  `LazilyIndexedArray`-wrapped Dataset rather than eagerly reading.
- `ds.encoding["source"]` and `ds.encoding["gdal_driver"]` provenance strings
  (serialization-safe replacements for the live GDAL objects previously stashed
  in encoding).
- `[tool.ruff]` config + `.pre-commit-config.yaml` for lint and format.

### Changed

- `gdal.UseExceptions()` moved from module import to
  `GDALBackendEntrypoint.__init__`, guarded with `gdal.GetUseExceptions()`.
  Scopes the side effect to actual backend use.
- `GDALMultiDimArray` now holds optional `_parent_dataset` and `_parent_group`
  references internally to keep the underlying `mdarray` valid for the dataset's
  lifetime. Previously stashed in `ds.encoding`, which broke `to_netcdf()` and
  other serialization paths.
- Per-band metadata (description, nodata, scale, offset) in `band_as_dim` mode
  is attached as scalar attrs when uniform across bands, or as `band`-dim
  coordinates when it varies.
- Multidim group navigation accepts nested paths (`"/a/b/c"`) with tolerant
  leading/trailing slash handling.
- Unsupported codecs on individual MDArrays (e.g. `numcodecs.pcodec`) are now
  logged at warning level and the array skipped, instead of aborting the
  open. This lets you open stores like Earthmover's public ERA5 Icechunk
  store and access the readable coordinate/mask variables.

### Fixed

- `Dataset.ReadAsArray` call in `GDALMultiBandArray` now uses `xsize`/`ysize`
  (the Dataset API) rather than `win_xsize`/`win_ysize` (which are the Band API).
- Reverse-slice canonicalisation across all axes in all three BackendArray
  classes — fixes failures when xarray sends `slice(stop, start, 1)` for
  selections on decreasing coordinates (very common in atmospheric data with
  latitude 90 → -90).
- `GDALBackendArray.__dask_tokenize__` no longer references a non-existent
  `self.dataset` attribute. Was latent (only fired on certain Dask graph
  hashing paths) but would have raised AttributeError when it did.
- `AdviseRead` now skips both tiny reads (no benefit) and huge reads (which
  blow up on sharded stores reporting shard-sized blocks); bounded
  `CACHE_SIZE` between 4 MB and 512 MB.

### Removed

- Live GDAL objects no longer stored in `ds.encoding`. Use
  `ds.encoding["source"]` and `ds.encoding["gdal_driver"]` (strings) for
  introspection. GDAL refs are now held inside the BackendArrays themselves.

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

[Unreleased]: https://github.com/hypertidy/gdalxarray/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/hypertidy/gdalxarray/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/hypertidy/gdalxarray/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/hypertidy/gdalxarray/releases/tag/v0.1.0