#!/usr/bin/env python3
"""Check that each remote store in docs/stores.yml still opens with GDAL.

Writes docs/_stores_status.json, which the site's executed pages read to
decide whether to run their remote cells (see docs/_common.py).

Each store is opened in its own subprocess with a timeout, with the
store's `env` applied, as a multidim or classic dataset per its
`multidim` flag. Only metadata is touched: this checks the address,
not every chunk behind it.

Usage:
    python scripts/check_stores.py              # check all, write status
    python scripts/check_stores.py --strict     # also exit 1 when a
                                                # pinned/permanent store fails
    python scripts/check_stores.py --only cmems-sla oisst-ncei
"""

import argparse
import datetime as dt
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
TIMEOUT = 90


def _open(entry, q):
    try:
        for k, v in (entry.get("env") or {}).items():
            os.environ[k] = str(v)
        os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
        from osgeo import gdal

        gdal.UseExceptions()
        gdal.PushErrorHandler("CPLQuietErrorHandler")
        need = [
            n for n in entry.get("needs", [])
            if (n.startswith("/vsi") and n not in gdal.GetFileSystemsPrefixes())
            or (not n.startswith("/vsi") and gdal.GetDriverByName(n) is None)
        ]
        if need:
            q.put({"ok": False, "skipped": True,
                   "error": "GDAL lacks " + ", ".join(need)})
            return
        if entry.get("multidim", True):
            ds = gdal.OpenEx(entry["dsn"], gdal.OF_MULTIDIM_RASTER)
            names = ds.GetRootGroup().GetMDArrayNames() or []
            summary = f"{len(names)} arrays"
        else:
            ds = gdal.Open(entry["dsn"])
            summary = f"{ds.RasterXSize} x {ds.RasterYSize} x {ds.RasterCount}"
        q.put({"ok": True, "driver": ds.GetDriver().ShortName,
               "summary": summary})
    except Exception as e:  # noqa: BLE001
        q.put({"ok": False, "error": str(e).splitlines()[0][:300]})


def check(entry):
    q = mp.Queue()
    p = mp.Process(target=_open, args=(entry, q))
    p.start()
    p.join(TIMEOUT)
    if p.is_alive():
        p.terminate()
        return {"ok": False, "error": f"timed out after {TIMEOUT} s"}
    return q.get() if not q.empty() else {"ok": False, "error": "no result"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()

    stores = yaml.safe_load((DOCS / "stores.yml").read_text())["stores"]
    out_file = DOCS / "_stores_status.json"
    status = json.loads(out_file.read_text()) if out_file.exists() else {}
    from osgeo import gdal

    failed_hard = []
    for s in stores:
        if args.only and s["key"] not in args.only:
            continue
        if not s.get("used_in"):
            status[s["key"]] = {"ok": False, "skipped": True,
                                "error": "not used by any page"}
            continue
        r = check(s)
        r["checked"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        r["gdal"] = gdal.__version__
        status[s["key"]] = r
        mark = "ok  " if r["ok"] else ("skip" if r.get("skipped") else "FAIL")
        print(f"{mark} {s['key']:26s} {r.get('summary') or r.get('error', '')}")
        if (not r["ok"] and not r.get("skipped")
                and s.get("permanence") in ("pinned", "permanent")):
            failed_hard.append(s["key"])

    out_file.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    if args.strict and failed_hard:
        print("pinned/permanent stores failing: " + ", ".join(failed_hard))
        sys.exit(1)


if __name__ == "__main__":
    main()
