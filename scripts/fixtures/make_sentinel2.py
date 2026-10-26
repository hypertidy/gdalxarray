#!/usr/bin/env python3
"""A real Sentinel-2 fixture for the Open modes guide page.

Builds docs/fixtures/sentinel2/:

    sentinel2-hobart-<date>.tif   four bands (blue, green, red, nir) of one
                                  Sentinel-2 L2A scene over Hobart, cropped
                                  and averaged to 20 m, uint16 COG
    provenance.json               the STAC item, asset URLs, licence note

The scene is chosen by searching the public Earth Search STAC API for the
least cloudy summer scene over Hobart on MGRS tile 55GEN; once built, the
item id is recorded and the fixture is kept unless --refresh is given. Each
band stays in the scene's stored digital numbers, with the scale and offset
from the STAC item written to the band, so a reader decodes reflectance.

Data: Copernicus Sentinel-2 Level-2A, via Earth Search (Element 84) on AWS
Open Data. Contains modified Copernicus Sentinel data.

Usage:
    python scripts/fixtures/make_sentinel2.py [--refresh] [--item ID]
"""

import argparse
import hashlib
import json
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

from osgeo import gdal, osr

gdal.UseExceptions()

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "fixtures" / "sentinel2"

STAC = "https://earth-search.aws.element84.com/v1"
COLLECTIONS = ["sentinel-2-c1-l2a", "sentinel-2-l2a"]
HOBART_LONLAT = [147.18, -42.93, 147.36, -42.82]
SEASON = "2024-11-01T00:00:00Z/2025-03-31T23:59:59Z"
TILE = "55GEN"
EPSG = 32755  # WGS 84 / UTM zone 55S, the tile's own CRS
# kunanyi / Mount Wellington to the Derwent estuary, 15 km x 12.5 km
BOUNDS = (515000.0, 5243000.0, 530000.0, 5255500.0)
RES = 20.0
BANDS = ["blue", "green", "red", "nir"]  # 10 m bands, STAC asset keys

UA = "gdalxarray-fixtures (+https://github.com/hypertidy/gdalxarray)"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def request(url, body=None, attempts=5):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"User-Agent": UA, "Accept": "application/geo+json, application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    for i in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code not in (403, 429, 500, 502, 503, 504) or i == attempts - 1:
                raise RuntimeError(f"HTTP {e.code} from {url}: {e.read()[:300]!r}") from e
            time.sleep(15 * 2**i)


def tile_of(item):
    p = item["properties"]
    code = p.get("grid:code") or p.get("s2:mgrs_tile") or ""
    if not code and "mgrs:utm_zone" in p:
        code = f"{p['mgrs:utm_zone']}{p.get('mgrs:latitude_band', '')}{p.get('mgrs:grid_square', '')}"
    return code.replace("MGRS-", "")


def find_item(item_id=None):
    for coll in COLLECTIONS:
        if item_id:
            try:
                return coll, request(f"{STAC}/collections/{coll}/items/{item_id}")
            except RuntimeError:
                continue
        body = {
            "collections": [coll],
            "bbox": HOBART_LONLAT,
            "datetime": SEASON,
            "query": {"eo:cloud_cover": {"lt": 5}},
            "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
            "limit": 50,
        }
        res = request(f"{STAC}/search", body)
        items = [f for f in res.get("features", []) if tile_of(f) == TILE]
        if items:
            items.sort(key=lambda f: (f["properties"].get("eo:cloud_cover", 100),
                                      f["properties"]["datetime"]))
            return coll, items[0]
    raise RuntimeError(f"no item found ({item_id or 'search'}) in {COLLECTIONS}")


def band_scaling(asset):
    rb = (asset.get("raster:bands") or [{}])[0]
    return rb.get("scale"), rb.get("offset"), rb.get("nodata")


def existing():
    prov = OUT / "provenance.json"
    if not prov.exists():
        return None
    p = json.loads(prov.read_text())
    f = OUT / p.get("file", "")
    if f.exists() and sha256(f) == p.get("sha256") and tuple(p.get("bounds", [])) == BOUNDS:
        return p
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refresh", action="store_true", help="search again and rebuild")
    ap.add_argument("--item", help="use this STAC item id")
    args = ap.parse_args()

    kept = existing()
    if kept and not args.refresh and not args.item:
        print(f"  sentinel2: keeping {kept['file']} ({kept['item']['id']})")
        return

    coll, item = find_item(args.item)
    p = item["properties"]
    print(f"  sentinel2: {item['id']} ({coll}), cloud {p.get('eo:cloud_cover')}%")

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(EPSG)
    width = int(round((BOUNDS[2] - BOUNDS[0]) / RES))
    height = int(round((BOUNDS[3] - BOUNDS[1]) / RES))
    mem = gdal.GetDriverByName("MEM").Create("", width, height, len(BANDS), gdal.GDT_UInt16)
    mem.SetGeoTransform([BOUNDS[0], RES, 0, BOUNDS[3], 0, -RES])
    mem.SetSpatialRef(srs)

    assets = {}
    for i, key in enumerate(BANDS, start=1):
        asset = item["assets"][key]
        href = asset["href"]
        if href.startswith("s3://"):
            bucket, _, path = href[5:].partition("/")
            href = f"https://{bucket}.s3.us-west-2.amazonaws.com/{path}"
        scale, offset, nodata = band_scaling(asset)
        one = gdal.Warp(
            "", "/vsicurl/" + href, format="MEM", dstSRS=f"EPSG:{EPSG}",
            outputBounds=BOUNDS, xRes=RES, yRes=RES, resampleAlg="average",
            outputType=gdal.GDT_UInt16, srcNodata=nodata if nodata is not None else 0,
            dstNodata=0,
        )
        band = mem.GetRasterBand(i)
        band.WriteArray(one.GetRasterBand(1).ReadAsArray())
        band.SetDescription(key)
        band.SetNoDataValue(0)
        if scale is not None:
            band.SetScale(scale)
        if offset is not None:
            band.SetOffset(offset)
        assets[key] = {"href": href, "scale": scale, "offset": offset, "nodata": nodata,
                       "eo:bands": asset.get("eo:bands")}
        print(f"    {key}: {href.rsplit('/', 1)[-1]}")

    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    date = p["datetime"][:10]
    tif = OUT / f"sentinel2-hobart-{date}.tif"
    gdal.Translate(str(tif), mem, format="COG",
                   creationOptions=["COMPRESS=DEFLATE", "PREDICTOR=YES", "BLOCKSIZE=256",
                                    "OVERVIEWS=AUTO", "INTERLEAVE=PIXEL"])

    prov = {
        "dataset": "Copernicus Sentinel-2 Level-2A surface reflectance",
        "day": date,
        "license": "Copernicus Sentinel data terms (free and open); "
                   "attribution: contains modified Copernicus Sentinel data",
        "item": {"id": item["id"], "collection": coll, "stac": f"{STAC}/collections/{coll}/items/{item['id']}",
                 "datetime": p["datetime"], "eo:cloud_cover": p.get("eo:cloud_cover"),
                 "platform": p.get("platform"), "tile": tile_of(item)},
        "assets": assets,
        "bounds": list(BOUNDS),
        "crs": f"EPSG:{EPSG}",
        "processing": (
            f"blue, green, red and nir (10 m) cropped to {BOUNDS} and averaged to {RES:g} m; "
            "stored digital numbers kept, with each asset's scale and offset written to its band"
        ),
        "file": tif.name,
        "sha256": sha256(tif),
        "made_with": {"script": "scripts/fixtures/make_sentinel2.py", "gdal": gdal.__version__},
    }
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(f"done: {tif.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
