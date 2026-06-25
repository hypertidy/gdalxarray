# Changelog

## [dev]

### Fixed

- CF decoding (`scale_factor`, `add_offset`, `_FillValue`) now applies
  automatically when reading. Previously gdalxarray exposed GDAL's
  scale/offset/nodata under non-CF attribute names, which prevented
  xarray's CF decoder from applying value transformation and fill-value
  masking. The result was raw integer values and sentinel-valued land
  pixels in scaled scientific data (OISST, CMEMS, ERA5, etc.) — silently
  wrong. The fix renames attributes to CF-standard names in both
  classic-raster and multidim modes, makes the backend signature accept
  xarray's decoder kwargs, and explicitly applies CF decoding before
  returning the Dataset.
  
  Breaking for users explicitly accessing `ds.attrs["scale"]`,
  `ds.attrs["offset"]`, or `ds.attrs["nodata"]`. These are now
  `ds.encoding["scale_factor"]`, `ds.encoding["add_offset"]`, and
  `ds.encoding["_FillValue"]` after decoding (or in `ds.attrs[...]`
  before decoding if `mask_and_scale=False` is passed).

## [0.4.0] - 2026-06-16

### Added

- `gdalxarray.warp()` — build lazy warp-VRT recipes for reprojection,
  regridding, GCP/RPC/geolocation-array transformation, cutline clipping,
  and resampling. Returns a VRT XML string that composes with the
  `gdalxarray` engine: only the bytes the consumer reads are
  materialised. Common args (`crs`, `bbox`, `shape`, `resolution`,
  `resampling`, `nodata`) plus a full escape hatch for any
  `gdal.WarpOptions` keyword.
  
### Changed

- Classic-raster open (`multidim=False`) on a file with no bands but
  subdatasets now raises a `ValueError` listing the available
  subdataset paths and pointing at `multidim=True`. Previously this
  produced an empty 512x512 stub Dataset. 

## [0.3.0] - 2026-06-15

### Added

- `band_as_dim` parameter on `open_dataset` (classic raster mode, default `True`).
  Bands now become an xarray `band` dimension on a single `band_data` DataArray. 
  Pass `band_as_dim=False` for per-band-variable layout. 
- New `GDALMultiBandArray` BackendArray reading via
  `dataset.ReadAsArray(band_list=...)`, letting GDAL handle BIP/BIL/BSQ
  interleaving internally. 
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
  classes - fixes failures when xarray sends `slice(stop, start, 1)` for
  selections on decreasing coordinates (very common in atmospheric data with
  latitude 90 -> -90).
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
  Python's falsy evaluation (`k.start or 0` -> `k.start if k.start is not None else 0`).
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
