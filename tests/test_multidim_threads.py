"""Thread safety of GDALMultiDimArray (issue #34).

GDAL handles are not thread-safe per-handle but GDAL is thread-safe
across handles, so reads resolve a per-thread handle via
``_get_mdarray`` (the same pattern as the classic-raster classes).
These tests verify the mechanism directly and run a real threaded
dask compute over many small chunks.
"""

from __future__ import annotations

import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from gdalxarray import GDALBackendEntrypoint
from gdalxarray.backend import GDALMultiDimArray

da = pytest.importorskip("dask.array")


@pytest.fixture
def backend():
    return GDALBackendEntrypoint()


def _first_mdarray(path):
    from osgeo import gdal

    ds = gdal.OpenEx(path, gdal.OF_MULTIDIM_RASTER | gdal.GA_ReadOnly)
    root = ds.GetRootGroup()
    name = root.GetMDArrayNames()[0]
    return ds, root, root.OpenMDArray(name)


def test_per_thread_handles_distinct(foo5_vrt):
    ds, root, mdarray = _first_mdarray(foo5_vrt)
    arr = GDALMultiDimArray(
        mdarray, filename=foo5_vrt, _parent_dataset=ds, _parent_group=root
    )
    seed = arr._get_mdarray()

    def grab(_):
        return arr._get_mdarray()

    with ThreadPoolExecutor(max_workers=3) as pool:
        handles = list(pool.map(grab, range(3)))
    ids = {id(h) for h in handles} | {id(seed)}
    # 3 pool threads reopen; the constructing thread reuses the seed
    assert len(ids) >= 2
    assert all(h is not seed for h in handles)


def test_concurrent_reads_correct(foo5_vrt):
    ds, root, mdarray = _first_mdarray(foo5_vrt)
    arr = GDALMultiDimArray(
        mdarray, filename=foo5_vrt, _parent_dataset=ds, _parent_group=root
    )
    key = tuple(slice(None) for _ in arr.shape)
    expected = arr._raw_indexing_method(key)

    def read(_):
        return arr._raw_indexing_method(key)

    with ThreadPoolExecutor(max_workers=8) as pool:
        for got in pool.map(read, range(32)):
            np.testing.assert_array_equal(got, expected)


def test_threaded_dask_compute_matches_lazy(backend, foo5_vrt):
    lazy = backend.open_dataset(foo5_vrt, multidim=True)
    chunked = backend.open_dataset(foo5_vrt, multidim=True, chunks={})
    name = next(iter(chunked.data_vars))
    got = (
        chunked[name]
        .mean()
        .compute(scheduler="threads", num_workers=8)
    )
    expected = float(lazy[name].mean())
    assert float(got) == pytest.approx(expected)


def test_pickle_roundtrip_reads(foo5_vrt):
    ds, root, mdarray = _first_mdarray(foo5_vrt)
    arr = GDALMultiDimArray(
        mdarray, filename=foo5_vrt, _parent_dataset=ds, _parent_group=root
    )
    key = tuple(slice(None) for _ in arr.shape)
    expected = arr._raw_indexing_method(key)
    arr2 = pickle.loads(pickle.dumps(arr))
    np.testing.assert_array_equal(arr2._raw_indexing_method(key), expected)
    assert arr.__dask_tokenize__() == arr2.__dask_tokenize__()
