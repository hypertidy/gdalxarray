"""Tests for the multidim open path against the foo5 VRT fixture."""

from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from gdalxarray import GDALBackendEntrypoint


@pytest.fixture
def backend():
    return GDALBackendEntrypoint()


def test_open_basic_shape(backend, foo5_vrt):
    ds = backend.open_dataset(foo5_vrt, multidim=True)
    assert isinstance(ds, xr.Dataset)
    # Dimensions inherited from the VRT
    assert dict(ds.sizes) == {
        "station": 10,
        "time": 10,
        "x": 2,
        "y": 2,
        "z": 2,
    }


def test_coordinates_present(backend, foo5_vrt):
    ds = backend.open_dataset(foo5_vrt, multidim=True)
    for coord in ("time", "x", "y", "z", "station"):
        assert coord in ds.coords, f"missing coord: {coord}"


def test_y_is_decreasing(backend, foo5_vrt):
    """Y coordinate decreases (40, -40) - the reverse-slice case."""
    ds = backend.open_dataset(foo5_vrt, multidim=True)
    y = ds["y"].values
    assert y[0] > y[-1], f"expected decreasing y, got {y}"
    np.testing.assert_array_equal(y, [40.0, -40.0])


def test_time_decoded_to_datetime(backend, foo5_vrt):
    """CF time units -> datetime64 via decode_cf_datetime.

    The VRT uses a synthetic time array [0, 1, ..., 9] with unit
    'days since 2020-01-01', so the decoded values are deterministic.
    """
    ds = backend.open_dataset(foo5_vrt, multidim=True)
    assert ds["time"].dtype == np.dtype("datetime64[ns]")
    assert ds["time"].size == 10
    assert ds["time"].values[0] == np.datetime64("2020-01-01")
    assert ds["time"].values[-1] == np.datetime64("2020-01-10")


def test_data_variables(backend, foo5_vrt):
    ds = backend.open_dataset(foo5_vrt, multidim=True)
    assert "temperature" in ds.data_vars
    assert "xx" in ds.data_vars
    # Coordinates aren't data variables
    assert "time" not in ds.data_vars
    assert "station" not in ds.data_vars


def test_isel_does_not_read_eagerly(backend, foo5_vrt):
    """isel should produce a lazy view, not trigger a read."""
    ds = backend.open_dataset(foo5_vrt, multidim=True)
    sub = ds["temperature"].isel(time=slice(0, 3))
    # Underlying data wrapper should be xarray's LazilyIndexedArray, not a
    # materialised ndarray.
    assert not isinstance(sub.variable._data, np.ndarray)


def test_values_materialises(backend, foo5_vrt):
    ds = backend.open_dataset(foo5_vrt, multidim=True)
    arr = ds["temperature"].isel(time=slice(0, 3)).values
    # temperature shape is (x, y, z, time, station) = (2, 2, 2, 10, 10);
    # slicing time -> (2, 2, 2, 3, 10)
    assert arr.shape == (2, 2, 2, 3, 10)
    assert arr.dtype == np.float64


def test_sel_decreasing_y(backend, foo5_vrt):
    """sel on a decreasing coordinate exercises the reverse-slice fix."""
    ds = backend.open_dataset(foo5_vrt, multidim=True)
    # Pick the value at lat=40 (first row) and lat=-40 (second row)
    north = ds["temperature"].sel(y=40.0).values
    south = ds["temperature"].sel(y=-40.0).values
    # Same shape after collapsing the y axis: (x, z, time, station)
    assert north.shape == (2, 2, 10, 10)
    assert south.shape == (2, 2, 10, 10)


def test_sel_reverse_slice(backend, foo5_vrt):
    """sel(y=slice(40, -40)) on decreasing coord - the canonical repro."""
    ds = backend.open_dataset(foo5_vrt, multidim=True)
    sub = ds["temperature"].sel(y=slice(40.0, -40.0))
    # Both rows included; shape on y axis is 2
    arr = sub.values
    assert arr.shape == (2, 2, 2, 10, 10)


def test_drop_variables(backend, foo5_vrt):
    ds = backend.open_dataset(foo5_vrt, multidim=True, drop_variables=["xx"])
    assert "xx" not in ds.data_vars
    assert "temperature" in ds.data_vars


def test_chunks_empty_dict_gives_dask(backend, foo5_vrt):
    """chunks={} uses native block sizes and returns a Dask-backed Dataset."""
    ds = backend.open_dataset(foo5_vrt, multidim=True, chunks={})
    assert isinstance(ds["temperature"].data, da.Array)


def test_chunks_none_is_lazy_not_dask(backend, foo5_vrt):
    """chunks=None returns lazy but not Dask."""
    ds = backend.open_dataset(foo5_vrt, multidim=True)
    assert not isinstance(ds["temperature"].data, da.Array)
    # Still lazy - slicing must not materialise yet
    assert not isinstance(ds["temperature"].variable._data, np.ndarray)


def test_provenance_in_encoding(backend, foo5_vrt):
    ds = backend.open_dataset(foo5_vrt, multidim=True)
    assert ds.encoding.get("source") == foo5_vrt
    # GDAL's driver name for multidim VRT - accept several possible spellings
    driver = ds.encoding.get("gdal_driver", "")
    assert "VRT" in driver, f"expected VRT in driver name, got {driver!r}"


def test_no_live_objects_in_encoding(backend, foo5_vrt):
    """ds.encoding should be serialization-safe (no live GDAL handles)."""
    ds = backend.open_dataset(foo5_vrt, multidim=True)
    for key, val in ds.encoding.items():
        # Strings, primitives, or None - never GDAL Python proxy objects
        assert val is None or isinstance(val, str | int | float | bool), (
            f"encoding[{key!r}] is a {type(val).__name__}, not serialization-safe"
        )
