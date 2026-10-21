"""CF-1.8 geometry containers pass through the multidim path (issue #11).

A geometry container is a 0-d variable whose attributes point at ragged
node arrays. GDAL's multidim listing hides 0-d arrays by default and folds
grid_mapping into the SRS, so without special handling the container never
reaches xarray and ``ds.xvec.decode_cf()`` cannot build geometries or a CRS.

Fixtures are written with GDAL's own netCDF writers so the base test
environment needs no netCDF4 or xvec; the xvec round-trip tests are skipped
when xvec is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from osgeo import gdal, osr

pytestmark = pytest.mark.skipif(
    gdal.GetDriverByName("netCDF") is None, reason="GDAL netCDF driver not available"
)


def _srs(epsg: int) -> osr.SpatialReference:
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return srs


def _write(group, name, dims, values, dtype):
    arr = group.CreateMDArray(name, dims, gdal.ExtendedDataType.Create(dtype))
    arr.Write(np.asarray(values))
    return arr


def _attr_str(obj, name, value):
    obj.CreateAttribute(name, [], gdal.ExtendedDataType.CreateString()).WriteString(value)


@pytest.fixture(scope="module")
def xvec_style_nc(tmp_path_factory) -> str:
    """Geometry cube laid out as xvec.encode_cf writes it.

    Three features (square, triangle, two-part multipolygon) on a
    ``geometry`` dimension with ``variable_name`` set, crossed with 4 times,
    EPSG:4326. Values are 0..11 in (geometry, time) order.
    """
    path = str(tmp_path_factory.mktemp("cfgeom") / "xvec_style.nc")
    drv = gdal.GetDriverByName("netCDF")
    ds = drv.CreateMultiDimensional(path)
    rg = ds.GetRootGroup()

    xs = [1, 1, 0, 0, 1, 2, 3, 3, 2, 6, 6, 5, 5, 6, 8, 8, 7, 7, 8]
    ys = [0, 1, 1, 0, 0, 0, 0, 2, 0, 5, 6, 6, 5, 5, 7, 8, 8, 7, 7]
    d_geom = rg.CreateDimension("geometry", None, None, 3)
    d_part = rg.CreateDimension("part", None, None, 4)
    d_node = rg.CreateDimension("node", None, None, len(xs))
    d_time = rg.CreateDimension("time", None, None, 4)

    _write(rg, "node_count", [d_geom], np.array([5, 4, 10], "i4"), gdal.GDT_Int32)
    _write(rg, "part_node_count", [d_part], np.array([5, 4, 5, 5], "i4"), gdal.GDT_Int32)
    x = _write(rg, "x", [d_node], np.array(xs, "f8"), gdal.GDT_Float64)
    y = _write(rg, "y", [d_node], np.array(ys, "f8"), gdal.GDT_Float64)
    for arr in (x, y):
        arr.SetSpatialRef(_srs(4326))

    t = _write(rg, "time", [d_time], np.arange(4, dtype="f8"), gdal.GDT_Float64)
    t.SetUnit("days since 2020-01-01")

    v = _write(
        rg, "value", [d_geom, d_time], np.arange(12, dtype="f4").reshape(3, 4), gdal.GDT_Float32
    )
    _attr_str(v, "geometry", "geometry_container")

    gc = rg.CreateMDArray("geometry_container", [], gdal.ExtendedDataType.Create(gdal.GDT_Int32))
    for k, val in {
        "geometry_type": "polygon",
        "node_count": "node_count",
        "part_node_count": "part_node_count",
        "node_coordinates": "x y",
        "variable_name": "geometry",
    }.items():
        _attr_str(gc, k, val)
    ds = None
    return path


@pytest.fixture(scope="module")
def gdal_vector_nc(tmp_path_factory) -> str:
    """CF-1.8 geometry netCDF as GDAL's vector writer produces it (EPSG:3031).

    GDAL's layout differs from xvec's: no ``variable_name``, the instance
    dimension is the node_count dimension, and the grid_mapping sits on the
    container rather than on the node arrays.
    """
    tmp = tmp_path_factory.mktemp("cfgeom_ogr")
    src = gdal.GetDriverByName("Memory").Create("", 0, 0, 0, gdal.GDT_Unknown)
    lyr = src.CreateLayer("polys", _srs(3031), gdal.ogr.wkbMultiPolygon)
    for wkt in (
        "MULTIPOLYGON(((0 0,1000 0,1000 1000,0 1000,0 0)))",
        "MULTIPOLYGON(((2000 0,3000 0,3000 500,2000 0)))",
    ):
        f = gdal.ogr.Feature(lyr.GetLayerDefn())
        f.SetGeometry(gdal.ogr.CreateGeometryFromWkt(wkt))
        lyr.CreateFeature(f)
    path = str(tmp / "ogr_polys.nc")
    gdal.VectorTranslate(path, src, format="netCDF")
    return path


def _open(path):
    return xr.open_dataset(path, engine="gdalxarray", multidim=True)


def test_container_reaches_xarray(xvec_style_nc):
    ds = _open(xvec_style_nc)
    assert "geometry_container" in ds.variables
    gc = ds["geometry_container"]
    assert gc.ndim == 0
    assert gc.attrs["geometry_type"] == "polygon"
    assert gc.attrs["node_coordinates"] == "x y"
    assert gc.attrs["variable_name"] == "geometry"


def test_no_placeholder_index_on_geometry_dim(xvec_style_nc):
    ds = _open(xvec_style_nc)
    # a 0..n placeholder would block cf_xarray from decoding into `geometry`
    assert "geometry" not in ds.coords
    # other dims without coordinate variables keep their placeholder index
    np.testing.assert_array_equal(ds["part"].values, np.arange(4))


def test_container_gets_grid_mapping(xvec_style_nc):
    ds = _open(xvec_style_nc)
    gm = ds["geometry_container"].attrs["grid_mapping"]
    wkt = ds[gm].attrs["crs_wkt"]
    assert osr.SpatialReference(wkt=wkt).GetAuthorityCode(None) == "4326"


def test_gdal_writer_container_and_crs(gdal_vector_nc):
    ds = _open(gdal_vector_nc)
    containers = [v for v in ds.data_vars.values() if "geometry_type" in v.attrs]
    assert len(containers) == 1
    gm = containers[0].attrs["grid_mapping"]
    wkt = ds[gm].attrs["crs_wkt"]
    assert osr.SpatialReference(wkt=wkt).GetAuthorityCode(None) == "3031"


def test_plain_netcdf_unchanged(foo5_vrt):
    # files without geometry containers get no extra variables
    ds = _open(foo5_vrt)
    assert not [v for v in ds.variables.values() if "geometry_type" in v.attrs]
    assert "spatial_ref" not in ds.variables


def test_xvec_decode_round_trip(xvec_style_nc):
    pytest.importorskip("xvec")
    shapely = pytest.importorskip("shapely")
    dec = _open(xvec_style_nc).xvec.decode_cf()

    expected = [
        shapely.Polygon([(1, 0), (1, 1), (0, 1), (0, 0), (1, 0)]),
        shapely.Polygon([(2, 0), (3, 0), (3, 2), (2, 0)]),
        shapely.MultiPolygon(
            [
                shapely.Polygon([(6, 5), (6, 6), (5, 6), (5, 5), (6, 5)]),
                shapely.Polygon([(8, 7), (8, 8), (7, 8), (7, 7), (8, 7)]),
            ]
        ),
    ]
    got = list(dec["geometry"].values)
    assert all(g.equals(e) for g, e in zip(got, expected, strict=True))
    assert dec.xindexes["geometry"].crs.to_epsg() == 4326
    np.testing.assert_array_equal(dec["value"].values, np.arange(12, dtype="f4").reshape(3, 4))
    # geometry-labelled selection works on the decoded cube
    sq = dec["value"].sel(geometry=expected[0])
    np.testing.assert_array_equal(sq.values, [0, 1, 2, 3])
