<!-- README.md is generated from README.Rmd. Please edit that file -->

# gdalxarray

<!-- badges: start -->
[![PyPI](https://img.shields.io/pypi/v/gdalxarray.svg)](https://pypi.org/project/gdalxarray/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
<!-- badges: end -->

An xarray backend powered directly by GDAL.

```python
import xarray as xr
ds = xr.open_dataset(path_or_uri, engine="gdalxarray")
```

`gdalxarray` is a thin bridge between GDAL's reading capabilities- classic
raster, multidimensional, and any of the virtualized stores GDAL knows about-
and xarray's labelled-array model. Lazy by default, optionally Dask-chunked,
with native CRS and CF time handling.

## Installation

GDAL has no usable PyPI wheels, so `pip install gdalxarray` alone is not
enough. You need a working `osgeo.gdal` Python binding first. The
recommended paths, in order of friction:

- **conda-forge** (cross-platform, just works): `mamba install -c conda-forge gdal`
- **Docker** image with GDAL preinstalled (e.g. `ghcr.io/hypertidy/gdal-r-python:latest`)
- **System package manager** (apt, brew) plus matching system Python bindings

Then `pip install gdalxarray` for the engine itself. See
[**INSTALL.md**](https://hypertidy.github.io/gdalxarray/install/) for the full guide, including troubleshooting
for NumPy ABI mismatches and Python version pinning.

## Why this exists (vs rioxarray)

[rioxarray](https://corteva.github.io/rioxarray/) is an xarray accessor
and backend built on
[rasterio](https://rasterio.readthedocs.io/), which wraps GDAL with its
own Python conventions. For straightforward 2D/3D raster work it's the
mature, widely-used choice- the `da.rio.reproject(...)` accessor pattern
is well-known and well-tested.

`gdalxarray` goes directly to `osgeo.gdal`, with no `rasterio` layer
in between. That choice matters in a few specific cases:

- **GDAL's multidimensional API is exposed natively**- N-D arrays
  with named dimensions, not just (y, x) rasters with optional bands
- **Any GDAL virtualization composes**- `/vsicurl/`, `/vsis3/`,
  `vrt://`, `ZARR:`, `NETCDF:`, classic VRT, multidim VRT
- **Codec and driver support tracks GDAL** rather than whatever
  rasterio re-exposes- Zarr v3, Icechunk, kerchunk-Parquet stores,
  GRIB, HDF4/5 multidim- all readable via the GDAL drivers

For a single GeoTIFF or a STAC item, rioxarray is usually a better fit.
For multidim cloud-native datasets, virtualized Zarr/Icechunk stores, or
anything where you want GDAL itself to be the source of truth,
`gdalxarray` puts you closer to the metal.

## Three core usage modes

The package has three ways to open a dataset, and almost everything else
is a composition of these with GDAL virtual paths.

### 1. Classic raster, bands as a dimension (default)

For multispectral imagery, image stacks, and anything where bands are
interchangeable axes. Produces a single `band_data` DataArray with
dims `(band, y, x)`- the rioxarray-compatible layout.

```python
import xarray as xr

ds = xr.open_dataset("image.tif", engine="gdalxarray", multidim=False)
ds["band_data"]
# <xarray.DataArray 'band_data' (band: 3, y: 1024, x: 1024)>
#   ...

# xarray idioms work as expected:
mean_image = ds["band_data"].mean(dim="band")
just_nir = ds["band_data"].sel(band=4)
```

### 2. Classic raster, bands as separate variables

For multiband rasters where each band carries a semantically distinct
quantity (e.g. a NetCDF translated to multiband GeoTIFF where bands are
different physical variables). Each band becomes a separate data variable
named after its description.

```python
ds = xr.open_dataset(
    "multivariable.tif",
    engine="gdalxarray",
    multidim=False,
    band_as_dim=False,
)
ds
# <xarray.Dataset>
# Data variables:
#     temperature  (y, x) float32
#     salinity     (y, x) float32
#     density      (y, x) float32
```

### 3. Multidim- N-D arrays with named dimensions

For datasets with their own dimension/coordinate structure: HDF5,
NetCDF, multidim VRT, GRIB, Zarr (v2 and v3). Produces a Dataset whose
dims and coords come from the source.

```python
ds = xr.open_dataset("dataset.nc", engine="gdalxarray", multidim=True)
ds["temperature"].sel(time="2024-06", level=500).isel(latitude=slice(100, 200))
```

`multidim=True` is the default for `engine="gdalxarray"`.

## Composing with GDAL virtual paths

The three modes above combine with GDAL's virtualization layers to cover
nearly every cloud-native and remote-data scenario. None of these
require any code changes in `gdalxarray`- they're just different paths:

| Prefix or syntax           | What it does                                              |
|----------------------------|-----------------------------------------------------------|
| `/vsicurl/<url>`           | HTTP/HTTPS-served files                                   |
| `/vsis3/<bucket>/<key>`    | S3 (anonymous via `AWS_NO_SIGN_REQUEST=YES`)              |
| `/vsigs/...`               | Google Cloud Storage                                      |
| `vrt://<path>?<options>`   | Inline classic-raster VRT- subdataset selection, resampling, ... |
| `NETCDF:<path>:<var>`      | Pick a subdataset from a NetCDF                           |
| `ZARR:"<path>":/<array>`   | Open one array of a Zarr store as a classic raster        |
| Classic VRT (`.vrt`)       | XML file referencing other sources                        |
| Multidim VRT (`.vrt`)      | N-D version, layered over NetCDF/HDF/Zarr sources         |

A few illustrative compositions:

```python
# Public COG over HTTPS:
xr.open_dataset(
    "/vsicurl/https://example.com/data.tif",
    engine="gdalxarray", multidim=False,
)

# All variables of a CMEMS NetCDF on S3:
xr.open_dataset(
    "NETCDF:/vsis3/bucket/path/file.nc",
    engine="gdalxarray", multidim=True,
)

# A multidim VRT as a labelled coordinate-aware view over a raw NetCDF:
xr.open_dataset("study_area.vrt", engine="gdalxarray", multidim=True)
```

## Which mode for which format?

As a rough guide, `multidim=True` is the natural fit for formats whose
own data model is N-dimensional with named axes:

- NetCDF (3 and 4)
- HDF5 / HDF4
- Multidim VRT
- GRIB / GRIB2
- Zarr (v2 and v3)
- Icechunk (where supported by your GDAL build)

`multidim=False` is the natural fit for image-like formats:

- GeoTIFF (including COG)
- JPEG, PNG, JPEG2000
- ERDAS Imagine (.img)
- Classic VRT files
- Anything GDAL identifies as a 2D-with-bands raster


## Status

Active development. The API has settled but small changes are
possible before 1.0. See [`CHANGELOG.md`](https://hypertidy.github.io/gdalxarray/changelog/) for what's
landed and the [issue tracker](https://github.com/hypertidy/gdalxarray/issues)
for what's next.

For worked examples against real cloud-native data
(BRAN2023 ocean reanalysis, ECMWF AIFS forecasts, CMEMS sea level,
NOAA OISST), see [`docs/cookbook.md`](https://hypertidy.github.io/gdalxarray/cookbook/).


## License

Apache-2.0.
