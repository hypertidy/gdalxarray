"""Shared pytest fixtures for gdalxarray tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal


@pytest.fixture(autouse=True, scope="session")
def _enable_gdal_exceptions():
    if not gdal.GetUseExceptions():
        gdal.UseExceptions()


DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def foo5_vrt() -> str:
    """Path to the extended 5-D multidim VRT fixture.

    Requires ``tests/data/foo_5dimensional.nc`` (download once with curl).
    Skips the test if the source NetCDF is missing.
    """
    nc = DATA_DIR / "foo_5dimensional.nc"
    vrt = DATA_DIR / "foo5.vrt"
    if not nc.exists():
        pytest.skip(
            f"missing {nc}. Download with:\n"
            f"  curl -L -o {nc} https://github.com/OSGeo/gdal/raw/"
            f"refs/heads/master/autotest/gdrivers/data/netcdf/foo_5dimensional.nc"
        )
    return str(vrt)


@pytest.fixture(scope="session")
def foo5_zarr() -> str:
    """Path to the zarr with scale offset and nodata."""
    z = DATA_DIR / "foo5_scale_offset_nodata.zarr"
    return str(z)


@pytest.fixture(scope="session")
def synthetic_geotiff(tmp_path_factory) -> str:
    """A small multiband GeoTIFF written once per session.

    Three bands, distinct constant values, tiled with 32x32 blocks. Geotransform
    has a negative y-step so y values run north-down - same orientation as the
    bulk of real-world data, and exercises the reverse-slice fix on slices.
    """
    from osgeo import gdal

    path = str(tmp_path_factory.mktemp("data") / "synthetic.tif")
    ds = gdal.GetDriverByName("GTiff").Create(
        path,
        100,  # xsize
        80,  # ysize
        3,  # bands
        gdal.GDT_Float32,
        options=["TILED=YES", "BLOCKXSIZE=32", "BLOCKYSIZE=32"],
    )
    # Origin upper-left, 0.1 degree per pixel, y decreasing
    ds.SetGeoTransform([100.0, 0.1, 0.0, 30.0, 0.0, -0.1])
    ds.SetProjection("EPSG:4326")
    for i in range(3):
        band = ds.GetRasterBand(i + 1)
        band.SetDescription(f"band_{i + 1}")
        band.SetNoDataValue(-9999.0)
        # Fill each band with a recognisable constant
        band.WriteArray(np.full((80, 100), 10.0 * (i + 1), dtype=np.float32))
    ds = None  # close / flush
    return path


@pytest.fixture(scope="session")
def synthetic_geotiff_with_scale(tmp_path_factory):
    import numpy as np
    from osgeo import gdal

    path = str(tmp_path_factory.mktemp("data") / "scaled.tif")
    ds = gdal.GetDriverByName("GTiff").Create(path, 100, 80, 1, gdal.GDT_Int16)
    ds.SetGeoTransform([100.0, 0.1, 0.0, 30.0, 0.0, -0.1])
    ds.SetProjection("EPSG:4326")
    band = ds.GetRasterBand(1)
    band.SetScale(0.01)
    band.SetOffset(0.0)
    band.SetNoDataValue(-999)
    arr = np.full((80, 100), 1000, dtype=np.int16)
    arr[0:10, :] = -999  # land row
    band.WriteArray(arr)
    ds = None
    return path
