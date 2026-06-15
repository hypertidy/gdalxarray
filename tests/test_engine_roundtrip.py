"""Tests that exercise xarray's plugin discovery end-to-end.

These confirm that ``xr.open_dataset(path, engine="gdalxarray", ...)`` works,
which is the path the average user takes (rather than constructing the
entrypoint directly).
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def test_engine_open_classic_raster(synthetic_geotiff):
    """xr.open_dataset on a multiband GeoTIFF using the registered engine."""
    ds = xr.open_dataset(synthetic_geotiff, engine="gdalxarray", multidim=False)
    assert "band_data" in ds.data_vars
    assert ds["band_data"].dims == ("band", "y", "x")
    assert ds.sizes == {"band": 3, "y": 80, "x": 100}


def test_engine_open_multidim_vrt(foo5_vrt):
    """xr.open_dataset on a multidim VRT using the registered engine."""
    ds = xr.open_dataset(foo5_vrt, engine="gdalxarray", multidim=True)
    assert "temperature" in ds.data_vars
    assert ds["time"].dtype == np.dtype("datetime64[ns]")


def test_engine_kwargs_passthrough(synthetic_geotiff):
    """Engine-level kwargs (band_as_dim, multidim) make it through."""
    ds = xr.open_dataset(
        synthetic_geotiff,
        engine="gdalxarray",
        multidim=False,
        band_as_dim=False,
    )
    # band_as_dim=False produces per-band variables
    assert "band_1" in ds.data_vars
    assert "band_data" not in ds.data_vars


def test_engine_drop_variables(foo5_vrt):
    """drop_variables works through the engine."""
    ds = xr.open_dataset(
        foo5_vrt,
        engine="gdalxarray",
        multidim=True,
        drop_variables=["xx"],
    )
    assert "xx" not in ds.data_vars
    assert "temperature" in ds.data_vars
