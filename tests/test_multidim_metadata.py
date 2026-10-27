"""Metadata the multidim path carries into xarray: CRS, time encoding, units,
dimension types. Fixture: a vrtstack-style multidim VRT stacking three
packed GeoTIFFs along time (the GHRSST MUR COG layout)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from osgeo import gdal

WKT_LATLON = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,'
    'AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,'
    'AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
    'AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'
)


@pytest.fixture
def stack_vrt(tmp_path):
    ny, nx, nt = 4, 6, 3
    tifs = []
    for i in range(nt):
        p = tmp_path / f"sst_{i}.tif"
        d = gdal.GetDriverByName("GTiff").Create(str(p), nx, ny, 1, gdal.GDT_Int16)
        d.SetGeoTransform([140.0, 0.01, 0, -40.0, 0, -0.01])
        d.SetProjection(WKT_LATLON)
        b = d.GetRasterBand(1)
        b.SetNoDataValue(-32768)
        b.WriteArray((np.arange(ny * nx).reshape(ny, nx) - 11000 + 100 * i).astype("int16"))
        d = None
        tifs.append(p)
    srcs = "\n".join(
        f"""      <Source>
        <SourceFilename>{p}</SourceFilename>
        <SourceBand>1</SourceBand>
        <SourceTranspose>-1,0,1</SourceTranspose>
        <DestSlab offset="{i},0,0"/>
      </Source>"""
        for i, p in enumerate(tifs)
    )
    xml = f"""<VRTDataset>
  <Group name="/">
    <Dimension name="time" size="{nt}" type="TEMPORAL" indexingVariable="time"/>
    <Dimension name="y" size="{ny}" type="HORIZONTAL_Y" direction="NORTH" indexingVariable="y"/>
    <Dimension name="x" size="{nx}" type="HORIZONTAL_X" direction="EAST" indexingVariable="x"/>
    <Array name="time">
      <DataType>Float64</DataType>
      <DimensionRef ref="time"/>
      <Unit>days since 1981-01-01 12:00:00</Unit>
      <RegularlySpacedValues start="16000" increment="1"/>
    </Array>
    <Array name="y">
      <DataType>Float64</DataType>
      <DimensionRef ref="y"/>
      <Unit>degrees_north</Unit>
      <RegularlySpacedValues start="-40.005" increment="-0.01"/>
    </Array>
    <Array name="x">
      <DataType>Float64</DataType>
      <DimensionRef ref="x"/>
      <RegularlySpacedValues start="140.005" increment="0.01"/>
    </Array>
    <Array name="sst">
      <DataType>Int16</DataType>
      <DimensionRef ref="time"/>
      <DimensionRef ref="y"/>
      <DimensionRef ref="x"/>
      <SRS dataAxisToSRSAxisMapping="2,3">{WKT_LATLON.replace('"', "&quot;")}</SRS>
      <Unit>celsius</Unit>
      <NoDataValue>-32768</NoDataValue>
      <Offset>25</Offset>
      <Scale>0.001</Scale>
{srcs}
    </Array>
  </Group>
</VRTDataset>
"""
    out = tmp_path / "stack.vrt"
    out.write_text(xml)
    return str(out)


def test_crs_attached_like_classic_path(stack_vrt):
    ds = xr.open_dataset(stack_vrt, engine="gdalxarray")
    assert "crs" in ds.coords
    assert ds.proj.crs.to_epsg() == 4326  # xproj CRSIndex, as the classic path
    assert "crs_wkt" in ds["crs"].attrs  # CF spelling
    assert ds["sst"].attrs.get("grid_mapping") == "crs"


def test_time_keeps_cf_encoding(stack_vrt):
    ds = xr.open_dataset(stack_vrt, engine="gdalxarray")
    assert ds.time.dtype.kind == "M"
    assert ds.time.encoding["units"] == "days since 1981-01-01 12:00:00"
    assert ds.time.encoding["calendar"] == "standard"
    assert str(ds.time.values[0]).startswith("2024-10-22T12:00")
    # Re-encoding with the stashed encoding reproduces the stored values.
    from xarray.coding.times import encode_cf_datetime

    num, _, _ = encode_cf_datetime(
        ds.time.values, ds.time.encoding["units"], ds.time.encoding["calendar"]
    )
    np.testing.assert_array_equal(num, [16000, 16001, 16002])


def test_units_on_data_and_coords(stack_vrt):
    ds = xr.open_dataset(stack_vrt, engine="gdalxarray")
    assert ds["sst"].attrs["units"] == "celsius"
    assert ds["y"].attrs["units"] == "degrees_north"


def test_dimension_types_as_cf_axis(stack_vrt):
    ds = xr.open_dataset(stack_vrt, engine="gdalxarray")
    assert ds["time"].attrs["axis"] == "T"
    assert ds["y"].attrs["axis"] == "Y"
    assert ds["x"].attrs["axis"] == "X"


def test_values_still_decoded(stack_vrt):
    ds = xr.open_dataset(stack_vrt, engine="gdalxarray")
    v = ds["sst"].isel(time=0, y=0, x=slice(0, 2)).values
    np.testing.assert_allclose(v, [-11000 * 0.001 + 25, -10999 * 0.001 + 25])
    assert ds["sst"].encoding["dtype"] == np.dtype("int16")
