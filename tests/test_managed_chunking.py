"""Managed chunking via Dataset.chunk (issue #31).

The backend never constructs dask collections itself: variables open as
lazy arrays, CF decoding runs on the lazy arrays, and chunking is
delegated to Dataset.chunk with an explicit token. The chunked array
class is therefore created by the same chunk manager xarray later uses
to recognise it, and the (unpicklable) GDAL handles are never tokenized.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

da = pytest.importorskip("dask.array")

from gdalxarray import GDALBackendEntrypoint  # noqa: E402


@pytest.fixture
def backend():
    return GDALBackendEntrypoint()


def test_preferred_chunks_recorded(backend, synthetic_geotiff):
    ds = backend.open_dataset(synthetic_geotiff, multidim=False)
    pref = ds["band_data"].encoding.get("preferred_chunks")
    assert pref is not None
    assert set(pref) == {"band", "y", "x"}
    assert pref["band"] == 1


def test_explicit_chunks_values_roundtrip(backend, synthetic_geotiff):
    lazy = backend.open_dataset(synthetic_geotiff, multidim=False)
    chunked = backend.open_dataset(
        synthetic_geotiff, multidim=False, chunks={"y": 32, "x": 32}
    )
    arr = chunked["band_data"].data
    assert isinstance(arr, da.Array)
    np.testing.assert_array_equal(
        chunked["band_data"].values, lazy["band_data"].values
    )


def test_chunk_layer_names_deterministic(backend, synthetic_geotiff):
    n1 = backend.open_dataset(
        synthetic_geotiff, multidim=False, chunks={"y": 32, "x": 32}
    )["band_data"].data.name
    n2 = backend.open_dataset(
        synthetic_geotiff, multidim=False, chunks={"y": 32, "x": 32}
    )["band_data"].data.name
    assert n1 == n2
    assert n1.startswith("gdalxarray-")


def test_native_chunks_match_blocksize(backend, synthetic_geotiff):
    ds = backend.open_dataset(synthetic_geotiff, multidim=False, chunks={})
    arr = ds["band_data"].data
    assert isinstance(arr, da.Array)
    pref = ds["band_data"].encoding["preferred_chunks"]
    assert arr.chunks[0][0] == pref["band"]
    assert arr.chunks[1][0] == min(pref["y"], ds.sizes["y"])
    assert arr.chunks[2][0] == min(pref["x"], ds.sizes["x"])


def test_multidim_chunked_computes(backend, foo5_vrt):
    lazy = backend.open_dataset(foo5_vrt, multidim=True)
    chunked = backend.open_dataset(foo5_vrt, multidim=True, chunks={})
    name = next(iter(chunked.data_vars))
    assert isinstance(chunked[name].data, da.Array)
    np.testing.assert_array_equal(chunked[name].values, lazy[name].values)


def test_chunked_pickles_and_tokenizes(backend, synthetic_geotiff):
    """The dask graph must not capture live GDAL handles (issue #31)."""
    from dask.base import tokenize

    ds = backend.open_dataset(
        synthetic_geotiff, multidim=False, chunks={"y": 32, "x": 32}
    )
    # strict tokenization of the collection must succeed
    t1 = tokenize(ds["band_data"].data)
    t2 = tokenize(ds["band_data"].data)
    assert t1 == t2
