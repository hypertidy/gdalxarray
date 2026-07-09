"""Empty-selection slice semantics (issue #32) and tilde expansion (#33).

pandas emits positional slices with stop < start (and positive step) for
empty label selections on descending coordinates, e.g. slice(3, 0) from
Index([-16.875, -16.9, -16.925]).slice_indexer(-16.93, -16.87). Under
Python slice semantics these are empty; the backend must return a
zero-sized array rather than flipping them into forward reads.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import xarray as xr
from osgeo import gdal

from gdalxarray import GDALBackendEntrypoint
from gdalxarray.backend import _expand_tilde


@pytest.fixture
def backend():
    return GDALBackendEntrypoint()


@pytest.fixture
def descending_tif(tmp_path):
    """A 3x4 float32 GeoTIFF with a north-up (descending y) geotransform."""
    path = str(tmp_path / "descending.tif")
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(path, 4, 3, 2, gdal.GDT_Float32)
    # y runs 10 -> 7 at the pixel centres (descending), x runs 0 -> 3
    ds.SetGeoTransform((-0.5, 1.0, 0.0, 10.5, 0.0, -1.0))
    ds.GetRasterBand(1).WriteArray(np.arange(12, dtype=np.float32).reshape(3, 4))
    ds.GetRasterBand(2).WriteArray(np.arange(12, 24, dtype=np.float32).reshape(3, 4))
    ds = None
    return path


# ---------------------------------------------------------------------
# classic raster paths
# ---------------------------------------------------------------------


@pytest.mark.parametrize("band_as_dim", [True, False])
def test_classic_empty_label_sel_descending_y(backend, descending_tif, band_as_dim):
    ds = backend.open_dataset(descending_tif, multidim=False, band_as_dim=band_as_dim)
    ycoord = ds["y"].values
    assert ycoord[0] > ycoord[-1], "fixture must have descending y"
    # label range given in ascending order is empty on a descending coord
    sel = ds.sel(y=slice(ycoord[-1] - 2.0, ycoord[-1] - 1.0))
    assert sel.sizes["y"] == 0
    for var in sel.data_vars:
        vals = sel[var].values
        assert vals.shape[sel[var].dims.index("y")] == 0
    sel.to_dataframe()  # must not raise (issue #32 trigger)


@pytest.mark.parametrize("band_as_dim", [True, False])
def test_classic_empty_isel(backend, descending_tif, band_as_dim):
    ds = backend.open_dataset(descending_tif, multidim=False, band_as_dim=band_as_dim)
    for indexer in (slice(0, 0), slice(2, 1), slice(3, 0)):
        sel = ds.isel(x=indexer)
        assert sel.sizes["x"] == 0
        for var in sel.data_vars:
            assert 0 in sel[var].values.shape


def test_classic_empty_band_isel(backend, descending_tif):
    ds = backend.open_dataset(descending_tif, multidim=False, band_as_dim=True)
    sel = ds.isel(band=slice(2, 0))
    assert sel.sizes["band"] == 0
    assert sel["band_data"].values.shape[0] == 0


def test_classic_valid_reads_unchanged(backend, descending_tif):
    ds = backend.open_dataset(descending_tif, multidim=False, band_as_dim=True)
    expected = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.testing.assert_array_equal(
        ds["band_data"].isel(band=0, y=slice(1, 3), x=slice(0, 2)).values,
        expected[1:3, 0:2],
    )


# ---------------------------------------------------------------------
# multidim path (uses the shared foo5 fixture: y is [40, -40], descending)
# ---------------------------------------------------------------------


def test_multidim_empty_label_sel_descending_y(backend, foo5_vrt):
    ds = backend.open_dataset(foo5_vrt, multidim=True)
    y = ds["y"].values
    assert y[0] > y[-1]
    # ascending-order label range on a descending coord: pandas emits
    # a positional slice with stop < start; selection is empty
    name = next(iter(ds.data_vars))
    sel = ds[name].sel(y=slice(-50.0, 50.0))
    assert sel.sizes["y"] == 0
    vals = sel.values
    assert vals.shape[sel.dims.index("y")] == 0
    sel.to_dataframe()  # must not raise (issue #32 trigger)


def test_multidim_empty_isel(backend, foo5_vrt):
    ds = backend.open_dataset(foo5_vrt, multidim=True)
    name = next(iter(ds.data_vars))
    var = ds[name]
    dim = var.dims[-1]
    for indexer in (slice(0, 0), slice(2, 0)):
        sel = var.isel({dim: indexer})
        assert sel.sizes[dim] == 0
        assert 0 in sel.values.shape


# ---------------------------------------------------------------------
# tilde expansion (issue #33)
# ---------------------------------------------------------------------


def test_expand_tilde_expands_home():
    assert _expand_tilde("~/x.tif") == os.path.join(os.path.expanduser("~"), "x.tif")


@pytest.mark.parametrize(
    "dsn",
    [
        "/vsicurl/https://example.com/a.tif",
        'ZARR:"/vsicurl/https://example.com/s.zarr":/sst',
        "NETCDF:file.nc:var",
        "vrt://a.tif?bands=1",
        "/vsis3/bucket/key.tif",
        "relative/path.tif",
        "/absolute/path.tif",
    ],
)
def test_expand_tilde_passes_uris_through_byte_identical(dsn):
    assert _expand_tilde(dsn) is dsn


def test_expand_tilde_non_string_passthrough():
    assert _expand_tilde(42) == 42
    assert _expand_tilde(None) is None


def test_open_dataset_tilde_path(backend, descending_tif, monkeypatch, tmp_path):
    """A dsn starting with ~ opens after HOME is pointed at its directory."""
    monkeypatch.setenv("HOME", os.path.dirname(descending_tif))
    monkeypatch.delenv("USERPROFILE", raising=False)
    tilde_path = "~/" + os.path.basename(descending_tif)
    ds = backend.open_dataset(tilde_path, multidim=False)
    assert isinstance(ds, xr.Dataset)


def test_guess_can_open_tilde_path(backend, descending_tif, monkeypatch):
    monkeypatch.setenv("HOME", os.path.dirname(descending_tif))
    monkeypatch.delenv("USERPROFILE", raising=False)
    tilde_path = "~/" + os.path.basename(descending_tif)
    assert backend.guess_can_open(tilde_path)
