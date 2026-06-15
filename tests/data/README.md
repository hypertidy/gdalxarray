# Test data fixtures

## `foo_5dimensional.nc`

Downloaded from the GDAL autotest suite at
<https://github.com/OSGeo/gdal/raw/refs/heads/master/autotest/gdrivers/data/netcdf/foo_5dimensional.nc>

A small NetCDF (~4 KB) with five dimensions (station, time, x, y, z) and
three arrays (time, xx, temperature). No coordinate variables or CF
metadata of its own.

This file is fetched once and committed to the repo, so tests run offline.
To refresh:

```bash
curl -L -o tests/data/foo_5dimensional.nc \
  https://github.com/OSGeo/gdal/raw/refs/heads/master/autotest/gdrivers/data/netcdf/foo_5dimensional.nc
```

## `foo5.vrt`

A GDAL multidim VRT layered on top of `foo_5dimensional.nc` to add:

* synthetic coordinate arrays for x, y, z, station
* CF time metadata so `time` decodes to `datetime64`
* a deliberately *decreasing* y coordinate (40 -> -40) so the
  reverse-slice canonicalization path gets exercised by `sel()`

The VRT references its source file relatively, so the pair must live in
the same directory.
