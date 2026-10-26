#!/usr/bin/env python3
"""Real NOAA OISST v2.1 fixtures for the docs guide pages.

Builds docs/fixtures/oisst/ from three consecutive daily files of NOAA
OISST v2.1 (0.25 degree daily sea surface temperature), as published:

    nc/oisst-avhrr-v02r01.YYYYMMDD.nc   the daily netCDF files, unchanged
    oisst.vrt                           `gdal mdim mosaic` over them
    refs_<array>.parquet                byte table: one row per chunk
    oisst.zarr/                         the same days, written by xarray as Zarr v3
    provenance.json                     sources, checksums, licence, versions

The byte table is read from the files with GDAL (MDArray.GetRawBlockInfo):
for each chunk, its index along each dimension, the file it lives in
(relative to the fixture directory), the byte offset and size, and the
codec as GDAL reports it.

The Zarr store is what xarray writes for the decoded days with the
source's own packing (int16, scale 0.01, _FillValue -999), so it carries
whatever fill-value convention the current xarray/zarr stack uses.

Data: NOAA NCEI, OISST v2.1, doi:10.25921/RE9P-PT57, licence CC0-1.0.
Reference: Huang et al. (2021), J. Climate, doi:10.1175/JCLI-D-20-0166.1.

Usage:
    python scripts/fixtures/make_oisst.py                # download from NOAA
    python scripts/fixtures/make_oisst.py --from DIR     # use local copies (testing)
"""

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "fixtures" / "oisst"
DAYS = ["20250101", "20250102", "20250103"]
ARRAYS = ["sst", "ice"]
BASE = (
    "https://noaa-cdr-sea-surface-temp-optimum-interpolation-pds.s3.amazonaws.com/"
    "data/v2.1/avhrr/{ym}/oisst-avhrr-v02r01.{day}.nc"
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(days, src_dir):
    nc = OUT / "nc"
    nc.mkdir(parents=True)
    sources = []
    for day in days:
        name = f"oisst-avhrr-v02r01.{day}.nc"
        dst = nc / name
        if src_dir:
            hits = list(Path(src_dir).rglob(name))
            if not hits:
                sys.exit(f"{name} not found under {src_dir}")
            shutil.copyfile(hits[0], dst)
            url = f"file:{hits[0]}"
        else:
            url = BASE.format(ym=day[:6], day=day)
            with urllib.request.urlopen(url, timeout=120) as r, open(dst, "wb") as f:
                shutil.copyfileobj(r, f)
        sources.append(
            {
                "day": day,
                "url": url,
                "file": f"nc/{name}",
                "bytes": dst.stat().st_size,
                "sha256": sha256(dst),
            }
        )
        print(f"  {name}  {dst.stat().st_size:,} bytes")
    return sources


def mosaic(files):
    from osgeo import gdal

    vrt = OUT / "oisst.vrt"
    gdal.Run("mdim", "mosaic", input=[str(f) for f in files], output=str(vrt))
    # gdal mdim mosaic records sources relative to the VRT; check, so the
    # fixture directory can move.
    text = vrt.read_text()
    if str(OUT) in text:
        vrt.write_text(text.replace(str(OUT) + "/", ""))
    return vrt


def byte_table(files, array):
    """One row per chunk of `array` in each file, from GDAL's raw block info."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from osgeo import gdal

    rows = []
    dims = None
    for t, f in enumerate(files):
        a = gdal.OpenEx(str(f), gdal.OF_MULTIDIM_RASTER).GetRootGroup().OpenMDArray(array)
        dims = [d.GetName() for d in a.GetDimensions()]
        shape = [d.GetSize() for d in a.GetDimensions()]
        chunks = a.GetBlockSize()
        nchunks = [-(-s // c) for s, c in zip(shape, chunks, strict=True)]
        for local in np.ndindex(*nchunks):
            try:
                info = a.GetRawBlockInfo(list(local))
            except RuntimeError:
                # An unallocated chunk: GDAL raises rather than reporting
                # it. Record it as absent only if it reads as all fill.
                start = [i * c for i, c in zip(local, chunks, strict=True)]
                n = [min(c, s - o) for c, s, o in zip(chunks, shape, start, strict=True)]
                block = a.ReadAsArray(array_start_idx=start, count=n)
                if not np.all(block == a.GetNoDataValue()):
                    raise
                info = None
            idx = list(local)
            idx[0] += t  # one day per file: the file's time index
            present = info is not None and info.GetOffset() is not None and info.GetSize() > 0
            rows.append(
                {
                    **dict(zip(dims, idx, strict=True)),
                    "present": present,
                    "path": f"nc/{Path(f).name}",
                    "offset": int(info.GetOffset()) if present else 0,
                    "size": int(info.GetSize()) if present else 0,
                    "info": "; ".join(info.GetInfo() or []) if present else "",
                }
            )
    table = pa.Table.from_pylist(rows)
    out = OUT / f"refs_{array}.parquet"
    pq.write_table(table, out, compression="zstd")
    return {"file": out.name, "rows": len(rows), "chunks": chunks, "dims": dims}


def zarr_copy(files):
    """The days as xarray writes them to Zarr v3, with the source's packing."""
    import xarray as xr

    parts = [xr.open_dataset(str(f), engine="gdalxarray")[ARRAYS] for f in files]
    ds = xr.concat(parts, "time")
    # gdalxarray leaves `calendar` in the decoded time's attrs (xarray's
    # own backends move it to encoding), which blocks writing it back.
    cal = ds["time"].attrs.pop("calendar", "standard")
    ds["time"].encoding.update(units="days since 1978-01-01 12:00:00", calendar=cal)
    ds.attrs = {
        "title": "NOAA OISST v2.1, three days, as xarray writes them to Zarr v3",
        "source": "doi:10.25921/RE9P-PT57",
        "license": "CC0-1.0",
    }
    enc = {
        v: {
            "dtype": "int16",
            "scale_factor": 0.01,
            "add_offset": 0.0,
            "_FillValue": -999,
            "chunks": (1, 1, 360, 720),
        }
        for v in ARRAYS
    }
    out = OUT / "oisst.zarr"
    ds.to_zarr(out, mode="w", zarr_format=3, encoding=enc, consolidated=False)
    return {"file": out.name, "arrays": ARRAYS, "chunks": [1, 1, 360, 720]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from", dest="src", help="directory holding the daily .nc files")
    ap.add_argument("--days", nargs="+", default=DAYS)
    args = ap.parse_args()

    import xarray as xr
    import zarr
    from osgeo import gdal

    import gdalxarray

    gdal.UseExceptions()
    shutil.rmtree(OUT, ignore_errors=True)
    print(f"writing {OUT}")
    sources = fetch(args.days, args.src)
    files = [OUT / s["file"] for s in sources]
    mosaic(files)
    tables = [byte_table(files, a) for a in ARRAYS]
    store = zarr_copy(files)

    prov = {
        "dataset": "NOAA 0.25-degree Daily Optimum Interpolation Sea Surface Temperature (OISST), Version 2.1",
        "doi": "10.25921/RE9P-PT57",
        "reference": "Huang, B. et al. (2021) J. Climate 34, 2923-2939, doi:10.1175/JCLI-D-20-0166.1",
        "license": "CC0-1.0",
        "retrieved": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d"),
        "sources": sources,
        "products": {"vrt": "oisst.vrt", "byte_tables": tables, "zarr": store},
        "made_with": {
            "script": "scripts/fixtures/make_oisst.py",
            "gdal": gdal.__version__,
            "xarray": xr.__version__,
            "zarr": zarr.__version__,
            "gdalxarray": gdalxarray.__version__,
        },
    }
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"done: {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
