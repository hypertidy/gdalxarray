"""Cheap smoke tests that don't need GDAL data files.

These always run and catch the worst kind of bug: 'package doesn't import'.
"""

from __future__ import annotations


def test_import():
    import gdalxarray

    assert hasattr(gdalxarray, "__version__")
    assert gdalxarray.__version__ != "0.0.0+unknown", (
        "version stamping failed — check [tool.hatch.version] path and "
        "__version__ literal in __init__.py"
    )


def test_entrypoint_class_importable():
    from gdalxarray import GDALBackendEntrypoint

    assert GDALBackendEntrypoint is not None


def test_engine_registered():
    """Confirm xarray's plugin discovery sees gdalxarray."""
    import xarray as xr

    assert "gdalxarray" in xr.backends.list_engines()


def test_entrypoint_metadata_present():
    """Importlib.metadata sees the entry point — what xarray uses internally."""
    from importlib.metadata import entry_points

    eps = entry_points(group="xarray.backends")
    names = [ep.name for ep in eps]
    assert "gdalxarray" in names


def test_backend_constructible():
    """Instantiating the entrypoint is what enables gdal.UseExceptions."""
    from gdalxarray import GDALBackendEntrypoint

    backend = GDALBackendEntrypoint()
    assert backend.available
