"""Network-gated integration tests against real public cloud-native data.

Run with::

    pytest -m network

By default these are skipped (see ``addopts = "-m 'not network'"`` in
pyproject.toml). The tests open real public Zarr / Icechunk / NetCDF
stores, so they require network access AND, in some cases, GDAL config
options (anonymous AWS, etc.) which are set in this module.
"""

from __future__ import annotations

import os

import pytest
import xarray as xr

pytestmark = pytest.mark.network


@pytest.fixture(scope="module", autouse=True)
def _anonymous_s3():
    """Anonymous S3 access for the public buckets these tests touch."""
    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
    yield


def test_oisst_netcdf_over_vsicurl():
    """NOAA OISST daily file via /vsicurl/ - small, fast, public."""
    url = (
        "/vsicurl/https://www.ncei.noaa.gov/data/sea-surface-temperature-"
        "optimum-interpolation/v2.1/access/avhrr/202501/"
        "oisst-avhrr-v02r01.20250103.nc"
    )
    ds = xr.open_dataset(url, engine="gdalxarray", multidim=True)
    assert "sst" in ds.data_vars or "time" in ds.coords


# def test_aifs_icechunk_on_s3():
#    """ECMWF AIFS forecast Icechunk store - confirms /vsiicechunk/ works."""
#    os.environ.setdefault("AWS_REGION", "us-west-2")
#    url = "/vsis3/dynamical-ecmwf-aifs-single/ecmwf-aifs-single-forecast/v0.1.0.icechunk"
#    ds = xr.open_dataset(url, engine="gdalxarray", multidim=True)
#    assert "init_time" in ds.coords
#    assert "latitude" in ds.coords
#    assert ds["latitude"].size == 721

# import numpy as np
# def test_aifs_lazy_isel():
#    """One-element isel from AIFS should be tiny and fast."""
#    os.environ.setdefault("AWS_REGION", "us-west-2")
#    url = "/vsis3/dynamical-ecmwf-aifs-single/ecmwf-aifs-single-forecast/v0.1.0.icechunk"
#    ds = xr.open_dataset(url, engine="gdalxarray", multidim=True)
#    val = (
#        ds["wind_u_10m"]
#        .isel(init_time=0, lead_time=0)
#        .sel(latitude=-42.9, longitude=147.3, method="nearest")
#        .values
#    )
#    assert val.dtype == np.float32
#    assert np.isfinite(val)
