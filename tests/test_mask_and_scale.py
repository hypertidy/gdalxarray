"""Mask/scale/nodata handling (issue #29).

Covers: the mask_and_scale keyword actually being honoured; unscaled
data keeping its native dtype (no spurious identity scale_factor); and
heterogeneous per-band scale/offset/nodata being surfaced honestly
(raw values, non-CF coordinate names, a warning) rather than appearing
decoded while staying raw.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from osgeo import gdal

from gdalxarray import GDALBackendEntrypoint


@pytest.fixture
def backend():
    return GDALBackendEntrypoint()


@pytest.fixture
def plain_byte_tif(tmp_path):
    """Two Byte bands, no nodata/scale/offset metadata at all."""
    path = str(tmp_path / "plain.tif")
    ds = gdal.GetDriverByName("GTiff").Create(path, 4, 3, 2, gdal.GDT_Byte)
    ds.SetGeoTransform((-0.5, 1.0, 0.0, 3.5, 0.0, -1.0))
    for i in range(2):
        band = ds.GetRasterBand(i + 1)
        band.SetDescription(f"b{i + 1}")
        band.WriteArray(
            (np.arange(12, dtype=np.uint8) + 1).reshape(3, 4) * (i + 1)
        )
    ds = None
    return path


@pytest.fixture
def hetero_scale_tif(tmp_path):
    """Two Int16 bands sharing nodata but with different scales."""
    path = str(tmp_path / "hetero.tif")
    ds = gdal.GetDriverByName("GTiff").Create(path, 4, 3, 2, gdal.GDT_Int16)
    ds.SetGeoTransform((-0.5, 1.0, 0.0, 3.5, 0.0, -1.0))
    arr = np.full((3, 4), 1000, dtype=np.int16)
    arr[0, :] = -999
    for i, scale in enumerate((0.01, 0.1)):
        band = ds.GetRasterBand(i + 1)
        band.SetDescription(f"b{i + 1}")
        band.SetNoDataValue(-999.0)
        band.SetScale(scale)
        band.WriteArray(arr)
    ds = None
    return path


# ---------------------------------------------------------------------
# no spurious identity scaling (dtype preservation)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("band_as_dim", [True, False])
def test_unscaled_keeps_native_dtype(backend, plain_byte_tif, band_as_dim):
    ds = backend.open_dataset(
        plain_byte_tif, multidim=False, band_as_dim=band_as_dim
    )
    name = "band_data" if band_as_dim else "b1"
    assert ds[name].dtype == np.uint8, ds[name].dtype
    assert "scale_factor" not in ds[name].attrs
    assert "scale_factor" not in ds[name].encoding


# ---------------------------------------------------------------------
# mask_and_scale keyword honoured
# ---------------------------------------------------------------------


def test_mask_and_scale_default_applies(backend, synthetic_geotiff_with_scale):
    ds = backend.open_dataset(synthetic_geotiff_with_scale, multidim=False)
    v = ds["band_data"].values
    assert np.issubdtype(v.dtype, np.floating)
    assert np.isnan(v[0, 0:10, :]).all()
    np.testing.assert_allclose(v[0, 10:, :], 10.0)  # 1000 * 0.01, once
    assert ds["band_data"].encoding.get("scale_factor") == 0.01


def test_mask_and_scale_false_returns_raw(backend, synthetic_geotiff_with_scale):
    ds = backend.open_dataset(
        synthetic_geotiff_with_scale, multidim=False, mask_and_scale=False
    )
    v = ds["band_data"].values
    assert v.dtype == np.int16, v.dtype
    assert (v[0, 0:10, :] == -999).all()
    assert ds["band_data"].attrs.get("scale_factor") == 0.01
    assert ds["band_data"].attrs.get("_FillValue") == -999.0


def test_mask_and_scale_via_xr_open_dataset(synthetic_geotiff_with_scale):
    """The keyword must also work through xr.open_dataset plumbing."""
    ds = xr.open_dataset(
        synthetic_geotiff_with_scale,
        engine="gdalxarray",
        multidim=False,
        mask_and_scale=False,
    )
    assert ds["band_data"].dtype == np.int16


# ---------------------------------------------------------------------
# heterogeneous bands
# ---------------------------------------------------------------------


def test_heterogeneous_warns_and_stays_raw(backend, hetero_scale_tif):
    with pytest.warns(UserWarning, match="band_as_dim=False"):
        ds = backend.open_dataset(hetero_scale_tif, multidim=False)
    bd = ds["band_data"]
    assert "band_scale_factor" in ds.coords
    np.testing.assert_array_equal(
        ds["band_scale_factor"].values, [0.01, 0.1]
    )
    # never half-decoded: no CF names lingering on the variable
    assert "scale_factor" not in bd.attrs
    assert "scale_factor" not in bd.encoding
    # homogeneous nodata still masks; heterogeneous scale left raw
    v = bd.values
    assert np.isnan(v[:, 0, :]).all()
    np.testing.assert_allclose(v[:, 1:, :], 1000.0)


def test_heterogeneous_decodes_per_band_as_vars(backend, hetero_scale_tif):
    ds = backend.open_dataset(
        hetero_scale_tif, multidim=False, band_as_dim=False
    )
    np.testing.assert_allclose(ds["b1"].values[1:, :], 10.0)
    np.testing.assert_allclose(ds["b2"].values[1:, :], 100.0)
    assert np.isnan(ds["b1"].values[0, :]).all()
