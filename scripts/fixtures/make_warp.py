#!/usr/bin/env python3
"""Real-data fixtures for the Warping guide page, and their GDAL oracles.

Builds three directories under docs/fixtures/:

    seaice/   one day of Antarctic sea-ice concentration on its native
              25 km polar stereographic grid (EPSG:3412), as a small COG
    gebco/    global bathymetry and topography decimated to 0.5 degree,
              as a small int16 COG
    warp/     target grids over those two sources, and what GDAL's
              warper writes for each (the pinned oracle), with manifest.json

Each source directory gets a provenance.json recording the URL fetched,
its checksum, and the licence and citation the source itself publishes
(copied from the source's own metadata, not written here).

Sources:
- Sea ice: NOAA/NSIDC Climate Data Record of Passive Microwave Sea Ice
  Concentration, version 6, southern hemisphere daily, served by NOAA
  CoastWatch ERDDAP (dataset nsidcG02202v6sh1day).
- Bathymetry: GEBCO grid, from the cloud-optimised GeoTIFF published at
  data.source.coop/ausantarctic/gebco, read through its overviews.

Usage:
    python scripts/fixtures/make_warp.py
    python scripts/fixtures/make_warp.py --local SEAICE.tif GEBCO.tif   # testing
"""

import datetime as dt
import hashlib
import json
import shutil
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "docs" / "fixtures"

SEAICE_DAY = "2025-09-20"
SEAICE_ERDDAP = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/nsidcG02202v6sh1day"
SEAICE_EPSG = 3412  # NSIDC Sea Ice Polar Stereographic South (Hughes 1980 ellipsoid)

GEBCO_YEAR = "2026"
GEBCO_BASE = "https://data.source.coop/ausantarctic/gebco"
GEBCO_SHAPE = (720, 360)  # 0.5 degree
# The STAC item on source.coop carries no licence or citation, so these are
# recorded from GEBCO's own pages (linked in provenance.json).
GEBCO_TERMS = {
    "license": "public domain; acknowledge the source (GEBCO Grid terms of use)",
    "doi": "10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa",
    "citation": (
        "GEBCO Bathymetric Compilation Group 2026 (2026). The GEBCO_2026 Grid - a "
        "continuous terrain model for oceans and land at 15 arc-second intervals. "
        "NERC EDS British Oceanographic Data Centre NOC."
    ),
    "not_for": "navigation or any other purpose involving safety at sea",
    "terms_url": "https://www.gebco.net/data-products/gridded-bathymetry/terms-of-use",
    "grid_url": "https://www.gebco.net/data-products-gridded-bathymetry-data/gebco2026-grid",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# Some servers (ERDDAP among them) refuse Python's default User-Agent.
UA = "gdalxarray-fixtures (+https://github.com/hypertidy/gdalxarray)"


def urlopen(url, timeout=300, attempts=5):
    """urlopen with a User-Agent, retrying refusals and server errors with back-off."""
    import time
    import urllib.error

    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for i in range(attempts):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code not in (403, 429, 500, 502, 503, 504) or i == attempts - 1:
                raise RuntimeError(f"HTTP {e.code} from {url}") from e
            wait = 15 * 2**i
            print(f"  HTTP {e.code} from {url}; retrying in {wait} s")
            time.sleep(wait)


def download(url, dst):
    with urlopen(url) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f)
    return dst


def cog(src_ds, path, nodata=None):
    opts = [
        "COMPRESS=ZSTD",
        "PREDICTOR=YES",
        "BLOCKSIZE=128",
        "OVERVIEWS=AUTO",
        "OVERVIEW_RESAMPLING=AVERAGE",
    ]
    if nodata is not None:
        src_ds.GetRasterBand(1).SetNoDataValue(nodata)
    gdal.Translate(str(path), src_ds, format="COG", creationOptions=opts)


# ---- sea ice ---------------------------------------------------------------


def make_seaice(tmp):
    out = FIX / "seaice"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    query = f"cdr_seaice_conc[({SEAICE_DAY}T00:00:00Z)][][]"
    url = f"{SEAICE_ERDDAP}.nc?" + urllib.parse.quote(query, safe="()=:,")
    nc = download(url, Path(tmp) / "seaice.nc")
    print(f"  sea ice: {nc.stat().st_size:,} bytes from ERDDAP")

    md = gdal.OpenEx(str(nc), gdal.OF_MULTIDIM_RASTER).GetRootGroup()
    attrs = {a.GetName(): a.Read() for a in md.GetAttributes()}
    arr = md.OpenMDArray("cdr_seaice_conc")
    dims = [d.GetName() for d in arr.GetDimensions()]
    vals = arr.ReadAsArray()  # (time, y, x)
    y = md.OpenMDArray("y").ReadAsArray()
    x = md.OpenMDArray("x").ReadAsArray()
    assert dims[-2:] == ["y", "x"], dims
    conc = np.asarray(vals, dtype=np.float64).reshape(len(y), len(x))
    if y[0] < y[-1]:  # north-up
        conc, y = conc[::-1], y[::-1]

    # Concentration is 0..1; values above 1 are the record's flags (pole hole,
    # lake, coast, land, missing). Keep the concentration, mask the flags.
    flags = conc > 1
    data = np.where(flags, np.nan, conc).astype(np.float32)

    res = float(x[1] - x[0])
    gt = [float(x[0]) - res / 2, res, 0.0, float(y[0]) + res / 2, 0.0, -res]
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(SEAICE_EPSG)
    mem = gdal.GetDriverByName("MEM").Create("", len(x), len(y), 1, gdal.GDT_Float32)
    mem.SetGeoTransform(gt)
    mem.SetSpatialRef(srs)
    mem.GetRasterBand(1).WriteArray(data)
    mem.GetRasterBand(1).SetDescription("sea ice concentration (fraction)")
    tif = out / f"seaice-sh-{SEAICE_DAY}.tif"
    cog(mem, tif, nodata=float("nan"))

    keep = (
        "title",
        "institution",
        "license",
        "doi",
        "id",
        "naming_authority",
        "references",
        "creator_name",
        "publisher_name",
        "summary",
        "grid_mapping_srid",
        "proj4",
        "grid_mapping_GeoTransform",
    )
    prov = {
        "dataset": attrs.get("title", "NOAA/NSIDC sea ice concentration CDR"),
        "day": SEAICE_DAY,
        "url": url,
        "download_sha256": sha256(nc),
        "source_metadata": {k: attrs[k] for k in keep if k in attrs},
        "processing": (
            "cdr_seaice_conc for one day, flag values (> 1) set to NaN, written as a "
            f"float32 COG on the source grid (EPSG:{SEAICE_EPSG}, {res:g} m)"
        ),
        "files": {tif.name: sha256(tif)},
        "geotransform": gt,
        "flagged_cells": int(flags.sum()),
        "made_with": {"script": "scripts/fixtures/make_warp.py", "gdal": gdal.__version__},
    }
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, default=str) + "\n")
    return tif


# ---- bathymetry ------------------------------------------------------------


def make_gebco(tmp):
    out = FIX / "gebco"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    tif_url = f"{GEBCO_BASE}/GEBCO_{GEBCO_YEAR}.tif"
    stac_url = f"{GEBCO_BASE}/GEBCO_{GEBCO_YEAR}.stac-item.json"
    stac = json.loads(urlopen(stac_url, timeout=120).read())

    src = gdal.Open("/vsicurl/" + tif_url)
    w, h = GEBCO_SHAPE
    small = gdal.Translate(
        "", src, format="MEM", width=w, height=h, resampleAlg="average", outputType=gdal.GDT_Int16
    )
    # Translate's decimated geotransform carries float noise (a top edge of
    # 89.99999999999999); snap it to the exact half-degree grid. With the
    # noise, a warped VRT of this grid to south polar stereographic fails to
    # read ("stere: Invalid latitude") where an exact 90 reads fine.
    exact = (-180.0, 360.0 / w, 0.0, 90.0, 0.0, -180.0 / h)
    noise = max(abs(a - b) for a, b in zip(small.GetGeoTransform(), exact, strict=True))
    assert noise < 1e-9, small.GetGeoTransform()
    small.SetGeoTransform(exact)
    tif = out / f"gebco-{GEBCO_YEAR}-0.5deg.tif"
    cog(small, tif)
    print(f"  gebco: {w} x {h} from {src.RasterXSize} x {src.RasterYSize}")

    props = stac.get("properties", {})
    prov = {
        "dataset": f"GEBCO {GEBCO_YEAR} Grid",
        "url": tif_url,
        "stac_item": stac_url,
        "source_metadata": {
            k: v
            for k, v in props.items()
            if k
            in (
                "title",
                "description",
                "license",
                "sci:doi",
                "sci:citation",
                "providers",
                "datetime",
                "created",
            )
        }
        | ({"license": stac["license"]} if "license" in stac else {}),
        "links": [
            lk for lk in stac.get("links", []) if lk.get("rel") in ("license", "cite-as", "about")
        ],
        "processing": (
            f"averaged to {w} x {h} (0.5 degree) through the COG overviews, int16 metres; "
            "geotransform snapped to the exact half-degree grid"
        ),
        "publisher_terms": GEBCO_TERMS,
        "license": GEBCO_TERMS["license"],
        "doi": GEBCO_TERMS["doi"],
        "files": {tif.name: sha256(tif)},
        "source_size": [src.RasterXSize, src.RasterYSize],
        "made_with": {"script": "scripts/fixtures/make_warp.py", "gdal": gdal.__version__},
    }
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, default=str) + "\n")
    return tif


# ---- warp oracles ------------------------------------------------------------

CASES = [
    # name, source, target CRS, geotransform, width, height, overview
    ("seaice-to-4326", "seaice", "EPSG:4326", [-180.0, 0.5, 0, -50.0, 0, -0.5], 720, 80, "none"),
    (
        "seaice-to-3031",
        "seaice",
        "EPSG:3031",
        [-4e6, 25000.0, 0, 4.4e6, 0, -25000.0],
        320,
        336,
        "none",
    ),
    (
        "seaice-to-4326-ovr",
        "seaice",
        "EPSG:4326",
        [-180.0, 2.0, 0, -50.0, 0, -1.0],
        180,
        40,
        "auto",
    ),
    ("gebco-to-3031", "gebco", "EPSG:3031", [-4e6, 25000.0, 0, 4e6, 0, -25000.0], 320, 320, "none"),
    (
        "gebco-to-eqearth",
        "gebco",
        "+proj=eqearth",
        [-17.3e6, 69200.0, 0, 8.4e6, 0, -69135.8],
        500,
        243,
        "none",
    ),
    (
        "gebco-to-ortho-hobart",
        "gebco",
        "+proj=ortho +lat_0=-42.9 +lon_0=147.3",
        [-6.4e6, 40000.0, 0, 6.4e6, 0, -40000.0],
        320,
        320,
        "none",
    ),
    ("gebco-to-4326-ovr", "gebco", "EPSG:4326", [-180.0, 3.6, 0, 90.0, 0, -3.6], 100, 50, "auto"),
]
METHODS = ["near", "bilinear", "cubic"]


def warp_direct(src_path, crs, gt, w, h, method, overview):
    """gdal.Warp into memory: single-threaded, error threshold 0.125, nodata kept."""
    s = gdal.Open(str(src_path))
    band = s.GetRasterBand(1)
    nd = band.GetNoDataValue()
    srs = osr.SpatialReference()
    srs.SetFromUserInput(crs)
    dst = gdal.GetDriverByName("MEM").Create("", w, h, 1, band.DataType)
    dst.SetGeoTransform(gt)
    dst.SetProjection(srs.ExportToWkt())
    if nd is not None:
        dst.GetRasterBand(1).SetNoDataValue(nd)
    dst.GetRasterBand(1).Fill(nd if nd is not None else 0)
    gdal.Warp(
        dst,
        s,
        options=gdal.WarpOptions(
            resampleAlg=method,
            srcNodata=nd,
            dstNodata=nd,
            errorThreshold=0.125,
            multithread=False,
            overviewLevel="AUTO" if overview != "none" else "NONE",
            warpOptions=["INIT_DEST=NO_DATA"] if nd is not None else [],
        ),
    )
    return dst.ReadAsArray()


def make_oracles(sources):
    out = FIX / "warp"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    cases = []
    for name, src, crs, gt, w, h, ovr in CASES:
        arrays = {m: warp_direct(sources[src], crs, gt, w, h, m, ovr) for m in METHODS}
        f = f"{name}.npz"
        np.savez_compressed(out / f, **arrays)
        cases.append(
            {
                "name": name,
                "source": f"{src}/{sources[src].name}",
                "crs": crs,
                "geotransform": gt,
                "width": w,
                "height": h,
                "overview": ovr,
                "error_threshold": 0.125,
                "file": f,
                "resampling": METHODS,
                "dtype": str(arrays[METHODS[0]].dtype),
            }
        )
    man = {
        "gdal": gdal.__version__,
        "made": dt.date.today().isoformat(),
        "script": "scripts/fixtures/make_warp.py",
        "cases": cases,
    }
    (out / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    print(f"  warp: {len(cases)} cases x {len(METHODS)} methods")


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--local",
        nargs=2,
        metavar=("SEAICE_TIF", "GEBCO_TIF"),
        help="testing only: use these rasters instead of downloading",
    )
    args = ap.parse_args()
    if args.local:
        sources = {}
        for key, path in zip(("seaice", "gebco"), args.local, strict=True):
            d = FIX / key
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True)
            sources[key] = d / Path(path).name
            shutil.copyfile(path, sources[key])
            (d / "provenance.json").write_text(
                json.dumps(
                    {
                        "dataset": f"local stand-in for {key} ({Path(path).name}), testing only",
                        "made_with": {"script": "scripts/fixtures/make_warp.py --local"},
                    }
                )
                + "\n"
            )
    else:
        with tempfile.TemporaryDirectory() as tmp:
            sources = {"seaice": make_seaice(tmp), "gebco": make_gebco(tmp)}
    make_oracles(sources)
    total = sum(
        p.stat().st_size
        for d in ("seaice", "gebco", "warp")
        for p in (FIX / d).rglob("*")
        if p.is_file()
    )
    print(f"done: {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
