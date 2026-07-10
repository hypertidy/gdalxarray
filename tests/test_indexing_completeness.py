"""Indexing completeness: outer/boolean indexers, stepped and negative
slices (the xrdbi report: isel(time=(ds.time.dt.month == 6) & ...)
raised IndexError: Unsupported index type: <class 'numpy.ndarray'>).

All three BackendArray classes now declare IndexingSupport.BASIC via
explicit_indexing_adapter, so xarray decomposes fancy indexers into a
covering basic read plus a numpy step; stepped and reversed slices are
correct on all paths. Ground truth is the fully loaded dataset, so no
fixture semantics are assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from gdalxarray import GDALBackendEntrypoint


@pytest.fixture
def backend():
    return GDALBackendEntrypoint()


def _indexer_matrix(sizes):
    """Indexers over the first two dims of a variable, sized to fit."""
    d0, d1 = list(sizes)[:2]
    n0, n1 = sizes[d0], sizes[d1]
    third = np.zeros(n0, dtype=bool)
    third[:: max(1, n0 // 3)] = True
    return [
        {d0: np.array([0, n0 - 1])},
        {d0: np.array([n0 - 1, 0])},          # unordered
        {d0: third},                           # boolean
        {d0: slice(0, n0, 2)},                 # stepped
        {d0: slice(None, None, -1)},           # reversed
        {d0: slice(n0 - 1, 0, -2)},            # reversed + stepped
        {d0: np.array([0, n0 - 1]), d1: slice(0, n1, 2)},
        {d0: 0, d1: np.array([n1 - 1, 0])},
        {d0: -1},                              # negative int
        {d0: slice(2, 2)},                     # empty slice
    ]


def _assert_matches(lazy_da, loaded_da):
    for idx in _indexer_matrix(lazy_da.sizes):
        got = lazy_da.isel(idx).values
        want = loaded_da.isel(idx).values
        np.testing.assert_array_equal(got, want, err_msg=repr(idx))


@pytest.mark.parametrize("band_as_dim", [True, False])
def test_classic_indexing_matrix(backend, synthetic_geotiff, band_as_dim):
    ds = backend.open_dataset(
        synthetic_geotiff, multidim=False, band_as_dim=band_as_dim
    )
    for name in ds.data_vars:
        _assert_matches(ds[name], ds[name].compute())


def test_band_dim_fancy_band_keys(backend, synthetic_geotiff):
    ds = backend.open_dataset(synthetic_geotiff, multidim=False)
    da = ds["band_data"]
    ref = da.compute()
    for idx in (
        {"band": np.array([2, 0])},
        {"band": np.array([True, False, True])},
        {"band": slice(None, None, -1)},   # silently EMPTY before this fix
        {"band": 1, "y": slice(None, None, -1), "x": slice(0, 100, 3)},
    ):
        np.testing.assert_array_equal(
            da.isel(idx).values, ref.isel(idx).values, err_msg=repr(idx)
        )


def test_multidim_indexing_matrix(backend, foo5_vrt):
    ds = backend.open_dataset(foo5_vrt, multidim=True)
    name = next(iter(ds.data_vars))
    _assert_matches(ds[name], ds[name].compute())


def test_multidim_boolean_isel(backend, foo5_vrt):
    """The reported failure shape: isel with a boolean array."""
    ds = backend.open_dataset(foo5_vrt, multidim=True)
    name = next(iter(ds.data_vars))
    da = ds[name]
    dim = da.dims[0]
    mask = np.zeros(da.sizes[dim], dtype=bool)
    mask[-1] = True
    got = da.isel({dim: mask})
    assert got.sizes[dim] == 1
    np.testing.assert_array_equal(
        got.values, da.compute().isel({dim: mask}).values
    )
