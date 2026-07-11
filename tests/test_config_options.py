"""GDAL configuration options via open_dataset (issue #25).

Options are applied thread-locally around the open and around every
read (including per-thread reopens under dask, issue #34), then unset,
and participate in dask tokenization. These tests wrap the real
``gdal.config_options`` with a recorder to verify the plumbing without
depending on any option's observable side effect.
"""

from __future__ import annotations

import numpy as np
import pytest

from gdalxarray import GDALBackendEntrypoint
from gdalxarray import backend as gxb


@pytest.fixture
def backend():
    return GDALBackendEntrypoint()


@pytest.fixture
def recorder(monkeypatch):
    """Record every config_options invocation, delegating to the real one."""
    calls = []
    real = gxb.gdal.config_options

    def wrapper(opts, thread_local=True):
        calls.append((dict(opts), thread_local))
        return real(opts, thread_local=thread_local)

    monkeypatch.setattr(gxb.gdal, "config_options", wrapper)
    return calls


CFG = {"GDAL_HTTP_MAX_RETRY": "3", "SOME_NUMBER": 7}
CFG_STR = {"GDAL_HTTP_MAX_RETRY": "3", "SOME_NUMBER": "7"}


def test_open_applies_options_thread_local(backend, synthetic_geotiff, recorder):
    backend.open_dataset(
        synthetic_geotiff, multidim=False, config_options=CFG
    )
    assert recorder, "config context never entered during open"
    assert all(opts == CFG_STR for opts, _ in recorder)
    assert all(tl for _, tl in recorder), "must request thread_local scoping"


def test_reads_reapply_options(backend, synthetic_geotiff, recorder):
    ds = backend.open_dataset(
        synthetic_geotiff, multidim=False, config_options=CFG
    )
    n_open = len(recorder)
    _ = ds["band_data"].values
    assert len(recorder) > n_open, "read did not re-enter the config context"
    assert all(opts == CFG_STR for opts, _ in recorder[n_open:])


def test_no_options_no_context(backend, synthetic_geotiff, recorder):
    ds = backend.open_dataset(synthetic_geotiff, multidim=False)
    _ = ds["band_data"].values
    assert recorder == [], "config context entered without options"


def test_values_identical_with_and_without(backend, synthetic_geotiff):
    plain = backend.open_dataset(synthetic_geotiff, multidim=False)
    opted = backend.open_dataset(
        synthetic_geotiff, multidim=False,
        config_options={"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"},
    )
    np.testing.assert_array_equal(
        plain["band_data"].values, opted["band_data"].values
    )


def test_multidim_reads_under_options(backend, foo5_vrt, recorder):
    ds = backend.open_dataset(foo5_vrt, multidim=True, config_options=CFG)
    name = next(iter(ds.data_vars))
    n_open = len(recorder)
    _ = ds[name].values
    assert len(recorder) > n_open
    assert all(opts == CFG_STR for opts, _ in recorder[n_open:])


def test_tokenize_includes_options(backend, synthetic_geotiff):
    da = pytest.importorskip("dask.array")
    from dask.base import tokenize

    a = backend.open_dataset(
        synthetic_geotiff, multidim=False,
        chunks={"y": 32, "x": 32}, config_options=CFG,
    )["band_data"].data
    b = backend.open_dataset(
        synthetic_geotiff, multidim=False,
        chunks={"y": 32, "x": 32}, config_options=CFG,
    )["band_data"].data
    c = backend.open_dataset(
        synthetic_geotiff, multidim=False,
        chunks={"y": 32, "x": 32}, config_options={"OTHER": "1"},
    )["band_data"].data
    assert isinstance(a, da.Array)
    assert tokenize(a) == tokenize(b)
    assert tokenize(a) != tokenize(c)
