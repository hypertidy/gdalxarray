"""Lazy warp-VRT recipe construction.

The :func:`warp` function builds a description of a warp operation —
target CRS, resolution, bounds, GCP/RPC/geoloc transformer, cutline,
etc. — and returns it as VRT XML text. The result composes with
:class:`GDALBackendEntrypoint`::

    >>> import gdalxarray
    >>> import xarray as xr
    >>> vrt = gdalxarray.warp(source, crs="+proj=laea")
    >>> ds = xr.open_dataset(vrt, engine="gdalxarray", multidim=False)

No pixels are read or written by ``warp()`` — the VRT is a recipe.
Bytes flow only when the consumer (xarray, gdalwarp, gdal_translate)
asks for a window.
"""

from __future__ import annotations

import warnings

from osgeo import gdal

from .backend import _expand_tilde

# Options that gdal.Warp accepts but that do not affect the data and
# do not survive into the serialised VRT. Passing these to warp() with
# format="VRT" is a no-op; we warn so users aren't misled.
_RUNTIME_ONLY_OPTS = frozenset(
    {
        "multithread",
        "warpMemoryLimit",
        "creationOptions",
        "callback",
        "callback_data",
    }
)


def warp(
    source,
    *,
    crs=None,
    bbox=None,
    shape=None,
    resolution=None,
    resampling="near",
    nodata=None,
    src_nodata=None,
    bands=None,
    **warp_options,
):
    """Build a warped-VRT recipe string.

    Wraps :func:`gdal.Warp` with ``format="VRT"`` to produce a textual
    description of a reprojection / regrid / warp. The result is a
    self-contained VRT XML string that downstream tools can read lazily.

    The named keyword arguments cover the common cases. Any further
    :func:`gdal.WarpOptions` keyword can be passed as an additional
    keyword argument and is forwarded verbatim — see GDAL's
    `gdal.WarpOptions` documentation for the full list (transformer
    options for RPC/GCPs/geolocation arrays, cutlines, alpha-band
    handling, working type, etc.).

    Parameters
    ----------
    source : str
        GDAL-recognised source path or URI. Anything ``gdal.Open`` can
        handle: local file, ``/vsicurl/``, ``/vsis3/``, ``vrt://``,
        ``ZARR:"..."``, ``NETCDF:...:var``, etc.
    crs : str or int, optional
        Target CRS as PROJ string, WKT, or EPSG code. Maps to GDAL's
        ``dstSRS``.
    bbox : sequence of 4 floats, optional
        Output bounding box ``(xmin, ymin, xmax, ymax)`` in target CRS
        units. Maps to GDAL's ``outputBounds``.
    shape : tuple of 2 ints, optional
        Output ``(width, height)`` in pixels. Maps to GDAL's ``width``
        and ``height``.
    resolution : float or tuple of 2 floats, optional
        Output resolution. A scalar means isotropic ``(res, res)``. A
        tuple is ``(xres, yres)``. Maps to GDAL's ``xRes`` / ``yRes``.
    resampling : str, default ``"near"``
        Resampling algorithm: ``"near"``, ``"bilinear"``, ``"cubic"``,
        ``"cubicspline"``, ``"lanczos"``, ``"average"``, ``"mode"``,
        ``"min"``, ``"max"``, ``"med"``, ``"q1"``, ``"q3"``, ``"sum"``,
        ``"rms"``. Maps to GDAL's ``resampleAlg``.
    nodata : float, optional
        Output nodata value. Maps to GDAL's ``dstNodata``.
    src_nodata : float, optional
        Override the source nodata value. Maps to GDAL's ``srcNodata``.
    bands : list of int, optional
        Subset of source bands to warp (1-based). Maps to GDAL's
        ``srcBands``.
    **warp_options
        Any other keyword argument is forwarded to :func:`gdal.WarpOptions`
        verbatim. Common power-user options: ``rpc=True``,
        ``tps=True``, ``geoloc=True``, ``transformerOptions=[...]``,
        ``cutlineWKT="POLYGON(...)"``, ``cropToCutline=True``,
        ``coordinateOperation="+proj=..."``.

    Returns
    -------
    str
        VRT XML describing the warped output. Suitable for
        ``xr.open_dataset(..., engine="gdalxarray")``, for any GDAL
        tool that accepts a path, or for writing to a ``.vrt`` file as
        a portable, durable warp recipe.

    Notes
    -----
    Runtime-only options (``multithread``, ``warpMemoryLimit``,
    ``creationOptions``, ``callback``, ``callback_data``) are not
    serialised into the VRT description — they affect *how* a warp is
    executed, not *what* the warp produces. Passing them to ``warp()``
    emits a :class:`UserWarning` and they are dropped. To control
    runtime behaviour for the eventual read, set GDAL config options
    on the consumer side, e.g.::

        gdal.SetConfigOption("GDAL_NUM_THREADS", "4")
        ds = xr.open_dataset(vrt, engine="gdalxarray")

    Examples
    --------
    Reproject anything to Lambert Azimuthal Equal Area, lazily::

        vrt = warp(src, crs="+proj=laea")
        ds = xr.open_dataset(vrt, engine="gdalxarray", multidim=False)

    Specific target grid::

        vrt = warp(
            src,
            crs="EPSG:3577",
            bbox=(-2000000, -5000000, 2000000, -1000000),
            resolution=1000,
            resampling="bilinear",
        )

    Drone imagery with GCPs, warped via thin-plate splines::

        vrt = warp(src, crs="EPSG:32755", tps=True)

    Satellite Level-1B product with RPCs and a DEM::

        vrt = warp(
            src,
            crs="EPSG:4326",
            rpc=True,
            transformerOptions=["RPC_DEM=/path/to/dem.tif"],
        )
    """
    # Detect runtime-only options that won't affect the VRT.
    ignored = set(warp_options) & _RUNTIME_ONLY_OPTS
    if ignored:
        warnings.warn(
            f"Warp options {sorted(ignored)} have no effect with format='VRT' "
            f"(they are runtime hints, not serialised into the VRT). Configure "
            f"them on the consumer side via gdal.SetConfigOption or environment "
            f"variables before opening the resulting VRT.",
            UserWarning,
            stacklevel=2,
        )
        warp_options = {k: v for k, v in warp_options.items() if k not in ignored}

    # Start from the escape-hatch passthroughs.
    kwargs = dict(warp_options)

    # Translate named convenience kwargs into GDAL's argument names.
    # setdefault: an explicit GDAL form (e.g. dstSRS=...) wins over crs=.
    if crs is not None:
        kwargs.setdefault("dstSRS", crs)
    if bbox is not None:
        kwargs.setdefault("outputBounds", tuple(bbox))
    if shape is not None:
        kwargs.setdefault("width", int(shape[0]))
        kwargs.setdefault("height", int(shape[1]))
    if resolution is not None:
        if hasattr(resolution, "__len__"):
            xres, yres = resolution[0], resolution[1]
        else:
            xres, yres = resolution, resolution
        kwargs.setdefault("xRes", xres)
        kwargs.setdefault("yRes", yres)
    if resampling != "near":
        kwargs.setdefault("resampleAlg", resampling)
    if nodata is not None:
        kwargs.setdefault("dstNodata", nodata)
    if src_nodata is not None:
        kwargs.setdefault("srcNodata", src_nodata)
    if bands is not None:
        kwargs.setdefault("srcBands", list(bands))

    # Force VRT output — this is what makes the result a recipe.
    kwargs["format"] = "VRT"

    if isinstance(source, str):
        source = _expand_tilde(source)
    wds = gdal.Warp(destNameOrDestDS="", srcDSOrSrcDSTab=source, **kwargs)
    if wds is None:
        raise ValueError(f"gdal.Warp failed for {source!r}")
    try:
        vrt = wds.GetMetadata("xml:VRT")[0]
    finally:
        wds = None  # release C-level resources
    return vrt
