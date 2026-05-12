from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("gdalxarray")
except PackageNotFoundError:
    # package is not installed (running from source checkout)
    __version__ = "0.0.0+unknown"


try:
    from osgeo import gdal
except ImportError as e:
    raise ImportError(
        "gdalxarray requires the GDAL Python bindings (osgeo.gdal), "
        "which are not installable via pip alone. Install GDAL through "
        "conda-forge, your system package manager, or use a Docker image "
        "such as ghcr.io/hypertidy/gdal-r-python:latest. "
        "See https://github.com/hypertidy/gdalxarray/blob/main/INSTALL.md "
        "for details."
    ) from e


from .backend import GDALBackendEntrypoint
