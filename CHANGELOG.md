# Changelog

## [dev]

### Fixed

- Outer and boolean indexers now work on all read paths, and stepped or
  reversed slices return correct data. Selections built from integer or
  boolean arrays, e.g.
  `isel(time=(ds.time.dt.month == 6) & (ds.time.dt.day == 27))`,
  previously raised
  `IndexError: Unsupported index type: <class 'numpy.ndarray'>`; the
  three BackendArray classes now declare `IndexingSupport.BASIC` via
  xarray's `explicit_indexing_adapter`, which decomposes fancy indexers
  into a covering basic read plus an in-memory numpy step. At the same
  time the basic reads themselves were completed: the classic-raster
  paths previously ignored slice steps entirely (a stepped `isel`
  returned the wrong shape or wrong rows), every path read
  negative-step slices forward without reversing the result, and a
  reversed band slice (`isel(band=slice(None, None, -1))`) silently
  returned an empty array. Classic paths serve stepped/reversed slices
  by reading the smallest contiguous covering window and reordering in
  numpy; the multidim path keeps native strided reads and flips
  negative-step axes after the read; band lists pass arbitrary order
  natively. Negative integer indices are also supported everywhere,
  raw int/slice keys on the backend arrays keep working (wrapped
  as BasicIndexer), and the eager coordinate load at open time
  goes through the raw read path.
  (Empty integer-array indexers remain an upstream xarray limitation:
  `_decompose_outer_indexer` crashes before backend code runs.)

- The multidim path is now safe under dask's threaded scheduler (#34).
  `GDALMultiDimArray` previously read through a single shared `mdarray`
  handle from every dask worker thread; GDAL handles are not
  thread-safe per-handle, so threaded computes over chunked multidim
  data were a data race (crash or corruption). Reads now resolve a
  per-thread handle lazily (`gdal.OpenEx` + `OpenMDArrayFromFullname`
  per worker thread), the same pattern the classic-raster classes
  already used: GDAL is thread-safe across handles, drivers take their
  own global locks internally, and osgeo releases the GIL during I/O,
  so threaded reads are both safe and genuinely parallel. The
  constructing thread reuses the already-open handle, so the non-dask
  path pays no reopen cost. `__dask_tokenize__` no longer touches a
  live handle (it can be called from any thread), and instances now
  pickle by dropping handles and reopening on first read, which makes
  the multidim path usable under distributed schedulers.

- Empty label selections no longer raise or silently return wrong data
  (#32). pandas emits positional slices with stop < start and a positive
  step for an empty label selection on a descending coordinate, e.g.
  `sel(latitude=slice(-16.93, -16.87))` on a north-up grid. All three
  BackendArray classes previously "flipped" such slices into forward
  reads: when the flipped window fell out of bounds GDAL raised
  `RuntimeError: arrayStartIdx + (count-1) * arrayStep >= size`, and when
  it stayed in bounds the read silently returned real data for a
  selection that should be empty. Slices that are empty under Python
  slice semantics now return correctly shaped zero-size arrays without
  touching GDAL. (A reverse read is expressed as step < 0, never as
  stop < start with a positive step.)

- A leading `~` in plain filesystem paths is now expanded (#33). GDAL
  does not perform tilde expansion, so `~/file.tif` previously failed
  with a misleading "not recognized as a supported file format" error.
  Expansion applies in `open_dataset`, in `guess_can_open` (so engine
  auto-detection also works for `~` paths), and in `warp()`. Only
  `os.path.expanduser` is applied, and only to strings starting with
  `~`: `abspath`/`normpath` are deliberately avoided because they
  corrupt GDAL URIs such as `/vsicurl/https://...` (collapsing the
  double slash) and `ZARR:`/`NETCDF:` connection strings. All other
  dsns pass through byte-identical.

- The `mask_and_scale` keyword is now honoured (#29). It was previously
  accepted by `open_dataset` and silently ignored; `mask_and_scale=False`
  now returns raw values with the CF attributes (`scale_factor`,
  `add_offset`, `_FillValue`) left in `attrs`, in both classic layouts
  and multidim mode.

- Unscaled data keeps its native dtype (#29). The `band_as_dim=False`
  path previously attached identity `scale_factor=1.0` and
  `add_offset=0.0` to every variable (GDAL's Python bindings return
  1.0/0.0 for unset scale/offset), which made CF decoding promote plain
  unscaled Byte/Int16 imagery to float64. Only informative values are
  recorded now.

- Bands with heterogeneous scale/offset/nodata are surfaced honestly
  (#29). With `band_as_dim=True`, per-band values were previously
  exposed as coordinates *named* `scale_factor`/`add_offset`/`_FillValue`,
  names CF decoding cannot consume on a per-band coordinate: the dataset
  looked decoded while the values stayed raw. Such values are now
  provided as `band_scale_factor`/`band_add_offset`/`band_nodata`
  coordinates with a `UserWarning` pointing at `band_as_dim=False`,
  which decodes each band independently. Attributes that are
  homogeneous across bands still decode (a shared nodata masks even
  when scales differ).

  Breaking for users reading the old per-band coordinate names
  `scale_factor`/`add_offset`/`_FillValue` on mixed-band datasets.

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

### Changed

- Added silencing for HDF5 messages. 

- Chunking is delegated to xarray's managed path (#31, closes #30). The
  backend no longer constructs dask collections itself: all variables
  open as lazy arrays, native GDAL block sizes are recorded in each data
  variable's `encoding["preferred_chunks"]`, and `chunks=` is applied
  once at the end of `open_dataset` via
  `Dataset.chunk(name_prefix="gdalxarray-", token=<sha256 of the dsn>)`,
  after CF decoding and coordinate assembly. The chunked array class is
  therefore created by the same chunk manager xarray later uses to
  recognise it, which fixes "Could not find a Chunk Manager which
  recognises type ..." errors under dask's expression-array migration,
  and the explicit token means xarray only ever tokenizes strings, so
  the unpicklable GDAL handles are never pickled or tokenized.

  Consequences: dask is now an optional dependency (install with
  `gdalxarray[dask]` or have dask present when passing `chunks=`); dask
  layer names change from `gdal-multiband-{path}` style to deterministic
  `gdalxarray-{variable}-{token}`; dimensions not named in an explicit
  `chunks=` mapping keep a single chunk (previously `band` defaulted to
  a chunk size of 1; `chunks={}` still yields per-band chunks via the
  recorded native block sizes); and CF decoders now always operate on
  lazy arrays rather than dask collections.

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

- usable tokenize method in this frisky world
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
