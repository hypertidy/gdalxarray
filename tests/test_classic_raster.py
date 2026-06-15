"""Tests for the classic raster open path using a synthetic GeoTIFF."""

from __future__ import annotations

from pathlib import Path

import dask.array as da
import numpy as np
import pytest

from gdalxarray import GDALBackendEntrypoint


@pytest.fixture
def backend():
    return GDALBackendEntrypoint()


# -- band_as_dim=True (default) -------------------------------------------


def test_band_as_dim_shape(backend, synthetic_geotiff):
    ds = backend.open_dataset(synthetic_geotiff, multidim=False)
    assert "band_data" in ds.data_vars
    assert ds["band_data"].dims == ("band", "y", "x")
    assert ds.sizes == {"band": 3, "y": 80, "x": 100}


def test_band_as_dim_values(backend, synthetic_geotiff):
    ds = backend.open_dataset(synthetic_geotiff, multidim=False)
    arr = ds["band_data"].values  # (3, 80, 100)
    np.testing.assert_array_equal(arr[0], np.full((80, 100), 10.0))
    np.testing.assert_array_equal(arr[1], np.full((80, 100), 20.0))
    np.testing.assert_array_equal(arr[2], np.full((80, 100), 30.0))


def test_band_as_dim_isel_band(backend, synthetic_geotiff):
    ds = backend.open_dataset(synthetic_geotiff, multidim=False)
    single = ds["band_data"].isel(band=1).values  # (80, 100)
    assert single.shape == (80, 100)
    np.testing.assert_array_equal(single, np.full((80, 100), 20.0))


def test_band_as_dim_sel_band_range(backend, synthetic_geotiff):
    ds = backend.open_dataset(synthetic_geotiff, multidim=False)
    pair = ds["band_data"].sel(band=[1, 3]).values  # bands 1 and 3
    assert pair.shape == (2, 80, 100)
    np.testing.assert_array_equal(pair[0], 10.0)
    np.testing.assert_array_equal(pair[1], 30.0)


def test_band_as_dim_mean(backend, synthetic_geotiff):
    """The xarray idiom that bands-as-dim unlocks."""
    ds = backend.open_dataset(synthetic_geotiff, multidim=False)
    mean = ds["band_data"].mean(dim="band").values
    np.testing.assert_array_equal(mean, np.full((80, 100), 20.0))


def test_band_descriptions_attached(backend, synthetic_geotiff):
    ds = backend.open_dataset(synthetic_geotiff, multidim=False)
    descs = ds["band_data"].attrs["descriptions"]
    assert descs == ["band_1", "band_2", "band_3"]


def test_uniform_nodata_as_scalar_attr(backend, synthetic_geotiff):
    """All bands share nodata=-9999, so it should be a scalar attr."""
    ds = backend.open_dataset(synthetic_geotiff, multidim=False)
    assert ds["band_data"].attrs["nodata"] == -9999.0


# -- band_as_dim=False (legacy per-variable layout) -----------------------


def test_band_as_vars_layout(backend, synthetic_geotiff):
    ds = backend.open_dataset(synthetic_geotiff, multidim=False, band_as_dim=False)
    assert "band_data" not in ds.data_vars
    for name in ("band_1", "band_2", "band_3"):
        assert name in ds.data_vars
        assert ds[name].dims == ("y", "x")


def test_band_as_vars_values(backend, synthetic_geotiff):
    ds = backend.open_dataset(synthetic_geotiff, multidim=False, band_as_dim=False)
    np.testing.assert_array_equal(ds["band_1"].values, 10.0)
    np.testing.assert_array_equal(ds["band_2"].values, 20.0)
    np.testing.assert_array_equal(ds["band_3"].values, 30.0)


# -- Lazy / Dask behaviour ------------------------------------------------


def test_chunks_none_lazy_not_dask(backend, synthetic_geotiff):
    ds = backend.open_dataset(synthetic_geotiff, multidim=False)
    assert not isinstance(ds["band_data"].data, da.Array)
    assert not isinstance(ds["band_data"].variable._data, np.ndarray)


def test_chunks_empty_dict_dask(backend, synthetic_geotiff):
    ds = backend.open_dataset(synthetic_geotiff, multidim=False, chunks={})
    assert isinstance(ds["band_data"].data, da.Array)


def test_chunks_explicit_dask(backend, synthetic_geotiff):
    ds = backend.open_dataset(
        synthetic_geotiff,
        multidim=False,
        chunks={"y": 32, "x": 32},
    )
    assert isinstance(ds["band_data"].data, da.Array)


# -- Indexing edge cases --------------------------------------------------


def test_isel_x_y_lazy(backend, synthetic_geotiff):
    """isel should be lazy across spatial dims too."""
    ds = backend.open_dataset(synthetic_geotiff, multidim=False)
    sub = ds["band_data"].isel(y=slice(10, 20), x=slice(30, 40))
    arr = sub.values
    assert arr.shape == (3, 10, 10)


def test_sel_decreasing_lat(backend, synthetic_geotiff):
    """Geotransform has negative y-step -> y values decreasing.

    A sel slice from a larger to a smaller y coordinate triggers the
    reverse-slice fix in GDALMultiBandArray._raw_indexing_method.
    """
    ds = backend.open_dataset(synthetic_geotiff, multidim=False)
    # Top half of the raster (larger y -> smaller y)
    sub = ds["band_data"].sel(y=slice(30.0, 22.0))
    arr = sub.values
    # Each band should still equal its constant value
    assert arr.ndim == 3
    np.testing.assert_array_equal(arr[0], 10.0)


# -- CRS / spatial coords -------------------------------------------------


def test_crs_attached(backend, synthetic_geotiff):
    ds = backend.open_dataset(synthetic_geotiff, multidim=False)
    # Either a CRSIndex coordinate, or proj-aware attrs - minimally there
    # must be SOMETHING describing CRS.
    has_crs = (
        "crs" in ds.coords
        or "spatial_ref" in ds.coords
        or ds["band_data"].attrs.get("crs") is not None
    )
    assert has_crs


def test_provenance_in_encoding(backend, synthetic_geotiff):
    ds = backend.open_dataset(synthetic_geotiff, multidim=False)
    assert ds.encoding.get("source") == synthetic_geotiff
    assert ds.encoding.get("gdal_driver") == "GTiff"


# -- Stub-Dataset policy --------------------------------------------------


def test_subdataset_file_raises_helpful_error(backend, foo5_vrt):
    """A multidim source opened in classic mode should refuse, not return a stub.

    Three possible error paths, all acceptable:

    * GDAL itself refuses the file ("not recognized as being in a supported
      file format") - this is the case for multidim-only formats like a
      multidim VRT, where the classic driver can't open it at all.
    * The file opens but reports zero bands and subdataset entries, which
      gdalxarray catches and raises a ValueError listing the subdatasets
      and suggesting multidim=True.
    * The file opens but has no bands and no subdatasets - a separate
      ValueError.

    What must NOT happen: the silent empty 512x512 stub Dataset.
    """
    import pytest

    with pytest.raises((ValueError, RuntimeError)) as excinfo:
        backend.open_dataset(foo5_vrt, multidim=False)
    msg = str(excinfo.value).lower()
    assert any(
        kw in msg
        for kw in (
            "multidim=true",
            "subdataset",
            "no raster bands",
            "could not open",
            "not recognized",  # GDAL's own driver-refusal message
            "not recognised",  # en_GB spelling, just in case
        )
    ), f"unexpected error message: {excinfo.value!r}"


def test_netcdf_classic_lists_subdatasets(backend):
    """Helpful-error branch: NetCDF with subdatasets gets a useful refusal."""
    nc_path = Path(__file__).parent / "data" / "foo_5dimensional.nc"
    if not nc_path.exists():
        pytest.skip("foo_5dimensional.nc not present")
    with pytest.raises(ValueError) as excinfo:
        backend.open_dataset(str(nc_path), multidim=False)
    msg = str(excinfo.value)
    assert "subdataset" in msg.lower()
    assert "multidim=True" in msg
