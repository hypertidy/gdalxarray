"""gdalxarray — an xarray backend powered by GDAL."""

# Friendly error when GDAL bindings are missing — gdalxarray cannot work without
# them and there is no fallback (osgeo.gdal has no PyPI wheels).
try:
    from osgeo import gdal as _gdal  # noqa: F401  (used by .backend, re-exported for env check)
except ImportError as e:
    raise ImportError(
        "gdalxarray requires the GDAL Python bindings (osgeo.gdal), "
        "which are not installable via pip alone. Install GDAL through "
        "conda-forge, your system package manager, or use a Docker image "
        "such as ghcr.io/hypertidy/gdal-r-python:latest. "
        "See https://github.com/hypertidy/gdalxarray/blob/main/INSTALL.md "
        "for details."
    ) from e

# Single source of truth for the version string. Hatchling parses this line
# at build time via [tool.hatch.version] path = "gdalxarray/__init__.py".
__version__ = "0.4.0"

from .backend import GDALBackendEntrypoint
from .warp import warp

__all__ = ["GDALBackendEntrypoint", "__version__", "warp"]
