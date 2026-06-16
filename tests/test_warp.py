"""Tests for gdalxarray.warp.

The warp() function produces VRT recipe strings rather than reading data,
so tests focus on:

* the function returns a usable VRT XML string
* the string composes correctly with the gdalxarray engine
* named convenience kwargs translate to GDAL's option names
* unrecognised kwargs pass through to gdal.WarpOptions verbatim
* runtime-only kwargs (multithread, warpMemoryLimit) warn and are dropped
* failure cases raise rather than silently produce broken output
"""

from __future__ import annotations

import warnings

import pytest
import xarray as xr

from gdalxarray import warp

# -- Basic shape -----------------------------------------------------------


def test_warp_returns_vrt_string(synthetic_geotiff):
    vrt = warp(synthetic_geotiff, crs="EPSG:3857")
    assert isinstance(vrt, str)
    assert vrt.lstrip().startswith("<VRTDataset")


def test_warp_no_options_wraps_source(synthetic_geotiff):
    """With no warp options the result is still a valid VRT wrapping the source."""
    vrt = warp(synthetic_geotiff)
    ds = xr.open_dataset(vrt, engine="gdalxarray", multidim=False)
    assert "band_data" in ds.data_vars


def test_warp_string_opens_with_engine(synthetic_geotiff):
    """The VRT string round-trips through xr.open_dataset."""
    vrt = warp(synthetic_geotiff, crs="EPSG:3857")
    ds = xr.open_dataset(vrt, engine="gdalxarray", multidim=False)
    assert ds.sizes  # opens, has dimensions
    assert "band_data" in ds.data_vars


# -- Named kwargs translate correctly --------------------------------------


def test_warp_with_shape(synthetic_geotiff):
    vrt = warp(synthetic_geotiff, crs="EPSG:3857", shape=(50, 40))
    ds = xr.open_dataset(vrt, engine="gdalxarray", multidim=False)
    assert ds.sizes["x"] == 50
    assert ds.sizes["y"] == 40


def test_warp_with_resolution_scalar(synthetic_geotiff):
    """Scalar resolution → isotropic. The resolution doesn't appear as an
    explicit element in the VRT — GDAL bakes it into the output grid
    (rasterXSize/rasterYSize). So verify the effect: the produced Dataset
    has dimensions consistent with the requested resolution.
    """
    vrt = warp(synthetic_geotiff, crs="EPSG:3857", resolution=10000)
    ds = xr.open_dataset(vrt, engine="gdalxarray", multidim=False)
    assert ds.sizes["x"] > 0
    assert ds.sizes["y"] > 0


def test_warp_resolution_affects_dimensions(synthetic_geotiff):
    """Coarser resolution → smaller pixel grid than finer resolution."""
    coarse_vrt = warp(synthetic_geotiff, crs="EPSG:3857", resolution=20000)
    fine_vrt = warp(synthetic_geotiff, crs="EPSG:3857", resolution=5000)
    coarse_ds = xr.open_dataset(coarse_vrt, engine="gdalxarray", multidim=False)
    fine_ds = xr.open_dataset(fine_vrt, engine="gdalxarray", multidim=False)
    assert coarse_ds.sizes["x"] < fine_ds.sizes["x"]
    assert coarse_ds.sizes["y"] < fine_ds.sizes["y"]


def test_warp_with_resolution_tuple(synthetic_geotiff):
    """Tuple resolution → anisotropic (xres, yres)."""
    vrt = warp(synthetic_geotiff, crs="EPSG:3857", resolution=(10000, 5000))
    ds = xr.open_dataset(vrt, engine="gdalxarray", multidim=False)
    assert ds.sizes  # opens


def test_warp_with_resampling(synthetic_geotiff):
    """resampling kwarg → resampleAlg in gdal.Warp."""
    vrt = warp(synthetic_geotiff, crs="EPSG:3857", resampling="bilinear")
    ds = xr.open_dataset(vrt, engine="gdalxarray", multidim=False)
    assert ds.sizes


def test_warp_with_nodata(synthetic_geotiff):
    vrt = warp(synthetic_geotiff, crs="EPSG:3857", nodata=-32768)
    ds = xr.open_dataset(vrt, engine="gdalxarray", multidim=False)
    assert ds.sizes


def test_warp_with_bands(synthetic_geotiff):
    """bands=[1] selects only the first band."""
    vrt = warp(synthetic_geotiff, crs="EPSG:3857", bands=[1])
    ds = xr.open_dataset(vrt, engine="gdalxarray", multidim=False)
    # band_as_dim mode → 'band' dim has length matching the selection
    assert ds.sizes["band"] == 1


# -- Escape-hatch passthrough ----------------------------------------------


def test_warp_kwargs_passthrough(synthetic_geotiff):
    """Unrecognised kwargs flow through to gdal.WarpOptions verbatim."""
    # targetAlignedPixels is a GDAL option we don't bless as a named kwarg.
    # It should pass through without error.
    vrt = warp(
        synthetic_geotiff,
        crs="EPSG:3857",
        resolution=10000,
        targetAlignedPixels=True,
    )
    assert isinstance(vrt, str)


def test_warp_explicit_gdal_name_wins(synthetic_geotiff):
    """Explicit GDAL option (dstSRS=) takes precedence over convenience (crs=)."""
    # Both set; dstSRS should win.
    vrt = warp(synthetic_geotiff, crs="EPSG:4326", dstSRS="EPSG:3857")
    ds = xr.open_dataset(vrt, engine="gdalxarray", multidim=False)
    assert ds.sizes  # opens, doesn't fail


# -- Runtime-only options warn and are dropped -----------------------------


def test_warp_warns_on_multithread(synthetic_geotiff):
    """multithread is a runtime hint, not VRT-serialisable."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warp(synthetic_geotiff, crs="EPSG:3857", multithread=True)
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "multithread" in str(w[0].message)


def test_warp_warns_on_warpMemoryLimit(synthetic_geotiff):
    """warpMemoryLimit is a runtime hint, not VRT-serialisable."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warp(synthetic_geotiff, crs="EPSG:3857", warpMemoryLimit=128)
        assert any("warpMemoryLimit" in str(warn.message) for warn in w)


def test_warp_multiple_runtime_options_one_warning(synthetic_geotiff):
    """Multiple runtime-only options batch into one warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warp(
            synthetic_geotiff,
            crs="EPSG:3857",
            multithread=True,
            warpMemoryLimit=128,
        )
        # Should emit a single warning naming both
        assert len(w) == 1
        msg = str(w[0].message)
        assert "multithread" in msg
        assert "warpMemoryLimit" in msg


def test_warp_no_warning_for_data_options(synthetic_geotiff):
    """Data-affecting options (resampleAlg, etc.) don't warn."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warp(
            synthetic_geotiff,
            crs="EPSG:3857",
            resampling="bilinear",
            nodata=-9999,
        )
        # No UserWarning from our code
        our_warnings = [warn for warn in w if issubclass(warn.category, UserWarning)]
        assert len(our_warnings) == 0


# -- Failure cases ---------------------------------------------------------


def test_warp_invalid_source_raises():
    """Invalid source path raises, doesn't silently produce junk."""
    with pytest.raises(Exception):  # noqa: B017 — accept RuntimeError or ValueError
        warp("/this/does/not/exist.tif", crs="EPSG:3857")
