# gdalxarray cookbook

Recipes for opening real cloud-native and remote datasets. These exercise
the full composition of `gdalxarray`'s three open modes with GDAL's
virtual-path layer. Many are dataset-specific - paths, regions, codec
support, and auth requirements change over time.

For an introduction to the package itself see the [README](../README.md).

## Network and config setup

Most recipes here read from anonymous public buckets or HTTPS-served
files. A few common configs:

```python
import os

# Public AWS buckets requiring anonymous read:
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("AWS_REGION", "us-west-2")  # adjust per bucket

# Skip HEAD probes for /vsicurl/ - cleaner request patterns:
os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "NO")
```

These can also be set per-call via `gdal.SetConfigOption(...)` from the
`osgeo.gdal` module; environment variables are easier to make
reproducible in notebooks.

## NCI THREDDS (rate-limited HTTPS source)

Some institutional servers - notably NCI - rate-limit aggressive parallel
reads. The recipe is to cap GDAL's read parallelism:

```python
from osgeo import gdal
gdal.SetConfigOption("GDAL_NUM_THREADS", "4")
gdal.SetConfigOption("CPL_VSIL_CURL_USE_HEAD", "NO")
```

`GDAL_NUM_THREADS=ALL_CPUS` (the default for some operations) can produce
truncated reads that look like data corruption but are actually
HTTP-level throttling. `4` is a defensible default for `/vsicurl/`
against institutional servers.

---

## NOAA OISST sea surface temperature

A single daily file over plain HTTPS, no auth.

```python
import xarray as xr

ds = xr.open_dataset(
    "/vsicurl/https://www.ncei.noaa.gov/data/sea-surface-temperature-"
    "optimum-interpolation/v2.1/access/avhrr/202501/"
    "oisst-avhrr-v02r01.20250103.nc",
    engine="gdalxarray", multidim=True,
)
```

Compact daily product, useful for quick-look examples and as a
non-throttled HTTPS sanity check.

---

## CMEMS sea level via virtualized Zarr (Pawsey)

Copernicus Marine Service NRT sea-level product, virtualized as Zarr
behind a Pawsey HTTPS proxy.

```python
import xarray as xr

url = (
    "ZARR:\"/vsicurl/https://s3.waw3-1.cloudferro.com/mdl-arco-time-045/"
    "arco/SEALEVEL_GLO_PHY_L4_MY_008_047/cmems_obs-sl_glo_phy-ssh_my_"
    "allsat-l4-duacs-0.125deg_P1D_202411/timeChunked.zarr\""
)
ds = xr.open_dataset(url, engine="gdalxarray", multidim=True)
ds["adt"].sel(time=slice("2024-06-01", "2024-06-10")).mean(dim="time").values
```

The `ZARR:"..."` prefix wraps a `/vsicurl/` HTTPS URL - GDAL's
virtualization composes through.

---

## BRAN2023 ocean reanalysis via kerchunk-Parquet

The Australian Bureau of Meteorology + CSIRO + AAD ocean reanalysis,
~135 TB of NetCDF on NCI THREDDS, virtualized as Zarr via a
kerchunk-Parquet reference manifest hosted at Pawsey. Per-variable
yearly manifests; daily 4-D ocean fields.

```python
from osgeo import gdal
import xarray as xr

gdal.SetConfigOption("GDAL_NUM_THREADS", "4")
gdal.SetConfigOption("CPL_VSIL_CURL_USE_HEAD", "NO")

url = (
    "vrt:///vsicurl/https://projects.pawsey.org.au/aad-index/"
    "vzarr/ocean_temp_2023.parq"
)
ds = xr.open_dataset(
    url, engine="gdalxarray", multidim=True,
    drop_variables=["Time_bnds", "DT", "average_DT", "average_T1", "average_T2"],
)

# Date-string slicing over a virtualized 3 TB store, in microseconds:
ds.sel(Time="2010-01-30").temp.values
```

The store opens in seconds; `.values` triggers the actual byte-range
reads via kerchunk's reference manifest.

---

## ECMWF AIFS forecast via Icechunk on S3

ECMWF's AIFS AI weather forecasts, published as an Icechunk store on
anonymous S3 by [Earthmover/dynamical.org](https://dynamical.org/).
~14 TB across 21 variables, 6-hourly initialisations, CC-BY-4.0.

```python
import os
import xarray as xr

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("AWS_REGION", "us-west-2")

ds = xr.open_dataset(
    "/vsis3/dynamical-ecmwf-aifs-single/ecmwf-aifs-single-forecast/"
    "v0.1.0.icechunk",
    engine="gdalxarray", multidim=True,
)

# Single-point wind at Hobart, first forecast hour of first init time:
hobart_u10 = (
    ds["wind_u_10m"]
    .isel(init_time=0, lead_time=0)
    .sel(latitude=-42.9, longitude=147.3, method="nearest")
    .values
)
```

Requires a GDAL build with the Icechunk driver compiled in. The Icechunk
driver is in active development; check your GDAL build's capabilities
with `gdalinfo --formats | grep -i icechunk`.

---

## Earthmover ERA5 (Icechunk on S3, partially readable)

ECMWF ERA5 reanalysis published by Earthmover as an Icechunk store.
Coordinate and mask variables open cleanly; data variables currently
fail because they use the `numcodecs.pcodec` codec which GDAL doesn't
yet support.

```python
import os
import xarray as xr

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("AWS_REGION", "us-east-1")

# Note the nested group syntax via the /vsiicechunk/ VFS:
url = (
    "/vsiicechunk/{/vsis3/earthmover-icechunk-era5/icechunkV2}"
    "/pressure/spatial"
)
ds = xr.open_dataset(url, engine="gdalxarray", multidim=True)

# `ds` will be a Dataset with all dimensions and coordinate variables,
# but reading data variables raises:
#   RuntimeError: Unsupported codec: numcodecs.pcodec
```

This is a `gdalxarray` limitation downstream of a GDAL one. Tracked
upstream at [osgeo/gdal](https://github.com/osgeo/gdal) - when GDAL
adds pcodec, this works without changes here.

---

## Single 2D slice of a multidim Icechunk store

GDAL supports a composable VRT-style path syntax for picking a single
2D face out of an N-D Icechunk store, usable from any classic GDAL tool:

```python
import xarray as xr

# init_time=0, lead_time=0, latitude  longitude of one variable:
url = (
    'ZARR:"/vsiicechunk/{/vsis3/dynamical-ecmwf-aifs-single/'
    'ecmwf-aifs-single-forecast/v0.1.0.icechunk}":/wind_u_10m:{0}:{0}'
)
ds = xr.open_dataset(url, engine="gdalxarray", multidim=False)
# Dataset with shape (1, 721, 1440) - band_data dim is 1, y is lat, x is lon
```

This composition lets `gdal_translate`, `gdal_warp`, etc. consume
specific forecast frames as if they were GeoTIFFs.

---

## GEBCO bathymetry (Cloud-Optimised GeoTIFF over HTTPS)

A 14 GB COG of global bathymetry, served via Pawsey's HTTPS endpoint.
Classic raster mode; `gdalxarray` will lazily read only the windows
you ask for.

```python
import xarray as xr

ds = xr.open_dataset(
    "/vsicurl/https://projects.pawsey.org.au/idea-gebco-tif/GEBCO_2024.tif",
    engine="gdalxarray", multidim=False,
)
# Window around Hobart:
hobart_box = ds["band_data"].sel(
    x=slice(146.5, 148.5),
    y=slice(-42.5, -43.5),
)
```

---

## Patterns worth knowing

**Composing virtual paths:** all GDAL virtual paths can be nested.
`ZARR:"/vsis3/bucket/...store.zarr"` opens a Zarr inside an S3 bucket.
`ZARR:"/vsicurl/https://.../store.zarr"` opens one served over HTTPS.
`ZARR:"/vsiicechunk/{/vsis3/.../v0.icechunk}"` opens an Icechunk via
its virtual filesystem.

**Picking subdatasets:** `NETCDF:path:var` and `vrt://path?sd_name=var`
both work for picking a single subdataset from a multi-variable file.
The `vrt://` form composes with other GDAL options (resampling, output
size).

**Checking driver name post-open:** `ds.encoding["gdal_driver"]` is
the driver GDAL used. Useful for distinguishing Zarr from Icechunk from
NetCDF when the same data is accessible multiple ways.

**Skip a known-bad variable at open:** `drop_variables=["X", "Y"]`.
Faster than open-then-drop because the BackendArray is never built.
