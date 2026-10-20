"""Dtype mapping, NaN nodata, and classic-array pickling.

Covers: declared dtypes matching what GDAL actually returns for the
newer integer/half types; NaN nodata shared by every band being treated
as uniform (not heterogeneous); classic-raster backend arrays pickling
(distributed schedulers).
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pytest
import xarray as xr
from osgeo import gdal

from gdalxarray.backend import GDALBackendArray, GDALMultiBandArray


def _make_tif(path, gdal_type, nbands=1, nodata=None, options=None):
    ds = gdal.GetDriverByName("GTiff").Create(
        str(path), 4, 3, nbands, gdal_type, options=options or []
    )
    ds.SetGeoTransform((0.0, 1.0, 0.0, 3.0, 0.0, -1.0))
    if nodata is not None:
        for b in range(1, nbands + 1):
            ds.GetRasterBand(b).SetNoDataValue(nodata)
    ds = None
    return str(path)


@pytest.mark.parametrize("type_name", ["Int8", "Int64", "UInt64", "Float16"])
def test_declared_dtype_matches_read(tmp_path, type_name):
    gdal_type = getattr(gdal, f"GDT_{type_name}", None)
    if gdal_type is None:
        pytest.skip(f"GDT_{type_name} not available in this GDAL")
    try:
        path = _make_tif(tmp_path / f"{type_name}.tif", gdal_type)
    except RuntimeError as e:
        pytest.skip(f"GTiff cannot write {type_name}: {e}")
    ds = xr.open_dataset(path, engine="gdalxarray", multidim=False)
    assert ds.band_data.dtype == ds.band_data.values.dtype
    assert ds.band_data.dtype != np.float32


def test_nan_nodata_all_bands_is_uniform(tmp_path):
    path = _make_tif(tmp_path / "nan.tif", gdal.GDT_Float32, nbands=2, nodata=float("nan"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds = xr.open_dataset(path, engine="gdalxarray", multidim=False)
    assert not [w for w in caught if "differing" in str(w.message)]
    assert "band_nodata" not in ds.coords
    assert np.isnan(ds.band_data.encoding["_FillValue"])


@pytest.mark.parametrize("cls", [GDALBackendArray, GDALMultiBandArray])
def test_classic_arrays_pickle_after_read(tmp_path, cls):
    path = _make_tif(tmp_path / "p.tif", gdal.GDT_Byte, nbands=2)
    arr = cls(path)
    key = (slice(0, 2), slice(0, 2)) if cls is GDALBackendArray else (0, slice(0, 2), slice(0, 2))
    before = arr[key]  # populates the per-thread handle
    restored = pickle.loads(pickle.dumps(arr))
    np.testing.assert_array_equal(restored[key], before)
