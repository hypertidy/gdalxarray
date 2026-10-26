"""OUTER indexing on the multidim path: sparse selections read only what
they select.

Before, the multidim array declared BASIC support, so xarray turned an
integer or boolean array indexer into one covering slab (first selected
index to last) and thinned it in memory. Over a daily mosaic, picking one
day a year read every day in between. Now array indexers reach the
backend, which reads runs of nearby indices (adjacent, or in the same
storage block) and stitches them together.

Ground truth is a numpy array written to a small chunked Zarr store; read
counts come from wrapping ``_read_mdim``.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from osgeo import gdal

from gdalxarray import backend as gxb

SHAPE = (40, 6, 8)  # time, y, x
BLOCK = (1, 3, 4)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("outer") / "cube.zarr")
    ds = gdal.GetDriverByName("Zarr").CreateMultiDimensional(path)
    rg = ds.GetRootGroup()
    dims = [
        rg.CreateDimension(n, None, None, s) for n, s in zip(("time", "y", "x"), SHAPE, strict=True)
    ]
    for d in dims:
        coord = rg.CreateMDArray(d.GetName(), [d], gdal.ExtendedDataType.Create(gdal.GDT_Float64))
        coord.Write(np.arange(d.GetSize(), dtype=np.float64))
        d.SetIndexingVariable(coord)
    arr = rg.CreateMDArray(
        "v",
        dims,
        gdal.ExtendedDataType.Create(gdal.GDT_Float32),
        [f"BLOCKSIZE={','.join(map(str, BLOCK))}"],
    )
    truth = np.arange(np.prod(SHAPE), dtype=np.float32).reshape(SHAPE)
    arr.Write(truth)
    ds = None
    return path, truth


@pytest.fixture
def reads(monkeypatch):
    log = []
    orig = gxb.GDALMultiDimArray._read_mdim

    def counting(self, key, starts, counts, steps):
        log.append((tuple(starts), tuple(counts), tuple(steps)))
        return orig(self, key, starts, counts, steps)

    monkeypatch.setattr(gxb.GDALMultiDimArray, "_read_mdim", counting)
    return log


def _open(path):
    return xr.open_dataset(path, engine="gdalxarray", mask_and_scale=False)["v"]


SELECTIONS = [
    {"time": np.array([0, 2])},
    {"time": np.array([2, 0])},  # unordered
    {"time": np.array([5, 5, 1])},  # duplicates
    {"time": np.array([-1, 0])},  # negative
    {"time": np.arange(40) % 7 == 3},  # boolean, sparse
    {"time": np.array([3, 30]), "y": np.array([0, 5]), "x": slice(1, 7, 2)},
    {"time": 4, "y": np.array([5, 0, 2])},
    {"time": np.array([1, 2, 3]), "x": np.array([7, 0])},
    {"time": slice(None, None, -3), "y": np.array([1])},
    {"y": np.array([4, 1]), "x": 3},
]


@pytest.mark.parametrize("sel", SELECTIONS, ids=[str(i) for i in range(len(SELECTIONS))])
def test_outer_selection_matches_numpy(store, sel):
    path, truth = store
    expect = xr.DataArray(truth, dims=("time", "y", "x")).isel(sel).values
    got = _open(path).isel(sel).values
    np.testing.assert_array_equal(got, expect)


def test_sparse_selection_reads_only_selected(store, reads):
    path, truth = store
    da = _open(path)
    reads.clear()
    got = da.isel(time=[0, 2]).values
    np.testing.assert_array_equal(got, truth[[0, 2]])
    # two one-step reads, not one covering read of three
    assert sorted(r[0][0] for r in reads) == [0, 2]
    assert all(r[1][0] == 1 for r in reads)


def test_one_day_a_year_style_selection(store, reads):
    path, truth = store
    da = _open(path)
    idx = np.array([1, 9, 18, 26, 35])  # irregular stride, like 27 June each year
    reads.clear()
    got = da.isel(time=idx).values
    np.testing.assert_array_equal(got, truth[idx])
    elems = sum(int(np.prod(c)) for _, c, _ in reads)
    assert elems == idx.size * SHAPE[1] * SHAPE[2]  # nothing read between selections


def test_adjacent_and_same_block_indices_share_a_read(store, reads):
    path, truth = store
    da = _open(path)
    reads.clear()
    np.testing.assert_array_equal(da.isel(time=[4, 5, 6]).values, truth[[4, 5, 6]])
    assert len(reads) == 1
    reads.clear()
    # y has block size 3: rows 0 and 2 share a block, row 5 does not
    np.testing.assert_array_equal(da.isel(y=[0, 2, 5]).values, truth[:, [0, 2, 5]])
    assert len(reads) == 2


def test_read_count_is_bounded(store, reads, monkeypatch):
    path, truth = store
    monkeypatch.setattr(gxb.GDALMultiDimArray, "MAX_OUTER_READS", 4)
    da = _open(path)
    t = np.arange(0, 40, 4)  # 10 separate runs
    y = np.array([0, 5])  # 2 runs
    reads.clear()
    np.testing.assert_array_equal(da.isel(time=t, y=y).values, truth[t][:, y])
    assert len(reads) <= 4


def test_dask_path_unchanged(store):
    pytest.importorskip("dask")
    path, truth = store
    da = xr.open_dataset(path, engine="gdalxarray", mask_and_scale=False, chunks={})["v"]
    np.testing.assert_array_equal(da.isel(time=[3, 1]).values, truth[[3, 1]])
