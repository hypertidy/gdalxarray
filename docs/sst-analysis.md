# A one-day-per-year SST record from R, via gdalxarray

Status: parked draft, 2026-07-11. Captured as motivation for a deeper
document (tutorial or paper section); see the tracking issue. Not yet
reviewed for publication; numbers below are from an interactive session.

## The example

One query, from R, against a remote store, returns a 45-year sea surface
temperature record for a patch of ocean off eastern Tasmania:

```r
library(DBI); library(xrdbi)

dsn <- "/vsizip//vsicurl/https://github.com/mdsumner/xrdbi/releases/download/latest/oisst-mdim.vrt.zip/oisst-mdim.vrt"
Sys.setenv("GDAL_DISABLE_READDIR_ON_OPEN" = "TRUE",
           "AWS_NO_SIGN_REQUEST" = "YES")

con <- dbConnect(xarray(), dsn,
                 engine = "gdalxarray", chunks = reticulate::dict())

dq <- dbGetQuery(con, paste0(
  "ds.sst.sel(lat=slice(-44, -43), lon=slice(147, 148))",
  ".isel(time=(ds.time.dt.month == 6) & (ds.time.dt.day == 27))"
))
```

Result: 720 rows (16 pixels x 45 years), June 27 of every year from
1982 to 2026. A plain `plot(dq$sst)` already shows the signal: the
2020s cluster sits roughly 1.5 degrees C above the early-1980s
baseline. The Tasman Sea east of Tasmania is one of the fastest
warming ocean regions in the world, and the query reproduces that
from a cold start in a few seconds.

The per-year reduction pushes the compute into the query itself:

```r
dq <- dbGetQuery(con, paste0(
  "ds.sst.sel(lat=slice(-44, -43), lon=slice(147, 148))",
  ".isel(time=(ds.time.dt.month == 6) & (ds.time.dt.day == 27))",
  ".groupby('time.year').mean(['zlev','lat','lon'])"
))
plot(dq$year, dq$sst, type = "b")
abline(lm(sst ~ year, dq), lty = 2)
coef(lm(sst ~ year, dq))[2] * 10   # degrees C per decade
```

## Why this example earns a deeper writeup

Every clause of that query exercises a distinct layer of the stack,
and each layer was recently fixed or built:

- The dsn is a recipe, not a payload: a GDAL multidim VRT, inside a
  zip, on a GitHub release, read over HTTP range requests
  (`/vsizip//vsicurl/`). No download step exists.
- `chunks = reticulate::dict()` maps to `chunks={}`: native GDAL block
  sizes, chunked through xarray's managed path (issue #31), computed
  on dask's threaded scheduler over per-thread GDAL handles
  (issue #34).
- `isel(time=(ds.time.dt.month == 6) & ...)` is a boolean outer
  indexer, decomposed by `explicit_indexing_adapter` into a covering
  basic read plus a numpy step (indexing-completeness fix; this exact
  query was the failing report).
- `sel(lat=slice(-44, -43), ...)` relies on correct label-slice
  semantics on descending coordinates (issue #32).
- The values arrive masked and scaled exactly once (issue #29).
- R -> DBI -> reticulate -> xarray -> gdalxarray -> GDAL -> curl:
  the "dumb format, smart engine" argument end to end.

## Outline for the deeper document (to develop)

1. The one-liner and the plot: hook with the climate signal.
2. Anatomy of the dsn: VRT as recipe, vsizip/vsicurl composition,
   why no ETL step exists.
3. Anatomy of the query: label vs positional selection, boolean time
   indexing, where the covering-read decomposition sends bytes.
4. Lazy vs chunked: chunks=None vs chunks={} vs explicit, what
   parallelises (per-thread handles, GIL release), and what does not
   (netCDF global lock).
5. Verification: same result lazy vs threaded (all.equal), and cost
   accounting (requests, bytes, wall time) at three scales.
6. Scaling up: wider boxes, all-June, full-series climatology; when
   to move to Icechunk/kerchunk reference stores instead.
7. The science sidebar: Tasman Sea warming, one-day-per-year sampling
   caveats, pointer to proper anomaly methodology.

## Loose ends to resolve before publishing

- Pin exact wall-times and request counts for the three query sizes.
- all.equal check lazy vs chunks={} on the full 720-row query.
- Confirm HDF5-DIAG silencing patch behaviour on worker threads.
- Decide venue: docs/cookbook, blog post, or xrdbi vignette (or all,
  with this file as the shared source).
