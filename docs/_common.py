"""Shared helpers for the executed pages of the gdalxarray site.

Every page starts with

    from _common import *

which gives it:

- ``fixture(rel)``: a path to a small test store from wxylib/fixtures,
  pinned to one commit (``FIXTURES_SHA``). Uses a local clone in
  ``docs/_fixtures`` when there is one (CI makes it), otherwise the
  same commit over ``/vsicurl/`` from raw.githubusercontent.com.
- ``store(key)``: the registry entry for a remote store from
  ``stores.yml``.
- ``live(key)``: whether that store answered the last liveness check
  (``_stores_status.json``, written by ``scripts/check_stores.py``)
  and whether this GDAL build has what the store needs. Pages guard
  their remote reads with it, so a store that has gone away gives a
  callout instead of a failed build.
- ``status_callout(key)``: markdown describing the store, its
  permanence class and its last check, for ``#| output: asis`` cells.
- ``versions()``: GDAL / xarray / gdalxarray versions this page was
  rendered with.
"""

import json
import os
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
from osgeo import gdal

import gdalxarray

gdal.UseExceptions()
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*PROJ database.*")

# Quiet, cheap remote access for every page.
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "NO")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
os.environ.setdefault("PROJ_NETWORK", "OFF")
# A remote read must fail rather than hang the whole site build.
os.environ.setdefault("GDAL_HTTP_CONNECTTIMEOUT", "20")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
os.environ.setdefault("GDAL_HTTP_LOW_SPEED_TIME", "30")
os.environ.setdefault("GDAL_HTTP_LOW_SPEED_LIMIT", "1")

# Per-cell wall-clock limit (seconds) for cells run under Jupyter. A cell
# that overruns raises TimeoutError, which an `error: true` gallery cell
# shows in place; a guide cell fails the build, as it should.
CELL_TIMEOUT = int(os.environ.get("GDALXARRAY_DOCS_CELL_TIMEOUT", "600"))


def _install_cell_timeout():
    import signal

    try:
        ip = get_ipython()  # noqa: F821
    except NameError:
        return
    if getattr(ip, "_gdalxarray_timeout", False):
        return

    def _alarm(signum, frame):
        raise TimeoutError(f"cell exceeded {CELL_TIMEOUT} s")

    signal.signal(signal.SIGALRM, _alarm)
    ip.events.register("pre_run_cell", lambda *a: signal.alarm(CELL_TIMEOUT))
    ip.events.register("post_run_cell", lambda *a: signal.alarm(0))
    ip._gdalxarray_timeout = True
    signal.alarm(CELL_TIMEOUT)


_install_cell_timeout()

DOCS = Path(__file__).resolve().parent
FIXTURES_REPO = "wxylib/fixtures"
FIXTURES_SHA = "c3adea57de88b76964125d7abf37329417d124e2"
_LOCAL = DOCS / "_fixtures"


def fixture(rel, remote=False):
    """Path to a wxylib fixture, local clone first, pinned raw URL otherwise."""
    if not remote and (_LOCAL / rel).exists():
        return str(_LOCAL / rel)
    return (
        "/vsicurl/https://raw.githubusercontent.com/"
        f"{FIXTURES_REPO}/{FIXTURES_SHA}/{rel}"
    )


def fixture_bytes(rel):
    """Raw bytes of a fixture file, through GDAL's VSI so either path works."""
    p = fixture(rel)
    st = gdal.VSIStatL(p)
    if st is None:
        raise FileNotFoundError(p)
    f = gdal.VSIFOpenL(p, "rb")
    try:
        return gdal.VSIFReadL(1, st.size, f)
    finally:
        gdal.VSIFCloseL(f)


def fixture_json(rel):
    return json.loads(fixture_bytes(rel))


def fixture_array(rel, dtype):
    """A raw little-endian oracle .bin from the fixtures, as a flat array."""
    return np.frombuffer(fixture_bytes(rel), dtype=np.dtype(dtype).newbyteorder("<"))


def fixture_url(rel):
    """Browsable GitHub URL of a fixture at the pinned commit."""
    return f"https://github.com/{FIXTURES_REPO}/tree/{FIXTURES_SHA}/{rel}"


# ---- real-data fixtures owned by this repository (docs/fixtures) -----------
# Small cuts of public data, each built by a script in scripts/fixtures/ that
# records its sources in provenance.json (issue #41).

DATA = DOCS / "fixtures"
GDALXARRAY_REPO = "hypertidy/gdalxarray"


def _data_commit(rel):
    """The commit that last changed a fixture, so remote reads are pinned.

    Needs full history (the docs workflow checks out with fetch-depth 0);
    falls back to main.
    """
    import subprocess

    try:
        sha = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", str(DATA / rel)],
            cwd=DOCS, capture_output=True, text=True, check=True,
        ).stdout.strip()
        return sha or "main"
    except (OSError, subprocess.CalledProcessError):
        return "main"


def data(rel, remote=False):
    """Path to a docs fixture; with remote=True, the same commit over /vsicurl/."""
    if not remote:
        return str(DATA / rel)
    top = rel.split("/")[0]
    return (
        "/vsicurl/https://raw.githubusercontent.com/"
        f"{GDALXARRAY_REPO}/{_data_commit(top)}/docs/fixtures/{rel}"
    )


def data_json(rel):
    return json.loads((DATA / rel).read_text())


def provenance(top):
    """Markdown line naming a fixture's source, from its provenance.json."""
    p = data_json(f"{top}/provenance.json")
    meta = p.get("source_metadata", {})
    doi = p.get("doi") or meta.get("doi") or meta.get("sci:doi")
    lic = p.get("license") or meta.get("license")
    bits = [p["dataset"]]
    if doi:
        doi = str(doi).removeprefix("https://doi.org/").removeprefix("doi:")
        bits.append(f"[doi:{doi}](https://doi.org/{doi})")
    if lic:
        bits.append(f"licence: {lic}")
    if "sources" in p:
        bits.append("days " + ", ".join(s["day"] for s in p["sources"]))
    elif "day" in p:
        bits.append(f"day {p['day']}")
    script = p.get("made_with", {}).get("script") or "scripts/fixtures/"
    print(f"Data: {'; '.join(bits)}. Built by `{script}`; details in "
          f"[`docs/fixtures/{top}/provenance.json`](https://github.com/{GDALXARRAY_REPO}"
          f"/blob/main/docs/fixtures/{top}/provenance.json).")


with open(DOCS / "stores.yml") as f:
    STORES = {s["key"]: s for s in yaml.safe_load(f)["stores"]}

_STATUS_FILE = DOCS / "_stores_status.json"
STATUS = json.loads(_STATUS_FILE.read_text()) if _STATUS_FILE.exists() else {}


def store(key):
    return STORES[key]


def _gdal_has(need):
    if need.startswith("/vsi"):
        return need in gdal.GetFileSystemsPrefixes()
    return gdal.GetDriverByName(need) is not None


def missing_capabilities(key):
    return [n for n in store(key).get("needs", []) if not _gdal_has(n)]


def live(key):
    """True when the store was reachable at the last check and GDAL can read it."""
    if os.environ.get("GDALXARRAY_DOCS_OFFLINE"):
        return False
    if missing_capabilities(key):
        return False
    st = STATUS.get(key)
    return bool(st and st.get("ok"))


PERMANENCE = {
    "pinned": "Pinned: a fixed commit or immutable object, will not change.",
    "permanent": "Permanent: an archive the provider commits to keeping.",
    "versioned": "Versioned: stable until the provider retires this version.",
    "rolling": "Rolling: the address is stable, the contents change over time.",
    "temporary": "Temporary: a scratch or transfer location, expected to go.",
    "unknown": "Not yet classified.",
}


def status_callout(key):
    s = store(key)
    st = STATUS.get(key, {})
    miss = missing_capabilities(key)
    skipped = os.environ.get("GDALXARRAY_DOCS_OFFLINE")
    if skipped:
        kind, head = "warning", "Not run in this build"
        why = (skipped if len(skipped) > 1 else "remote cells were switched off for this build") + "."
    elif miss:
        kind, head = "warning", "Not run in this build"
        why = "this GDAL lacks " + ", ".join(f"`{m}`" for m in miss) + "."
    elif st.get("ok"):
        kind, head = "note", "Live"
        why = f"answered the store check at {st.get('checked', '?')}."
    elif st:
        kind, head = "warning", "Not run in this build"
        why = f"the store did not answer at {st.get('checked', '?')}: {st.get('error', '')}"
    else:
        kind, head = "warning", "Not run in this build"
        why = "no store check has been run (see `scripts/check_stores.py`)."
    lines = [
        f"::: {{.callout-{kind} collapse=true}}",
        f"## {head}: {s['title']}",
        "",
        f"- Address: `{s['dsn']}`",
        f"- Provider: {s.get('provider', '')}",
        f"- {PERMANENCE[s.get('permanence', 'unknown')]}",
        f"- Status: {why}",
        ":::",
        "",
    ]
    print("\n".join(lines))


def versions():
    import importlib.metadata as md

    out = {
        "GDAL": gdal.__version__,
        "xarray": xr.__version__,
        "numpy": np.__version__,
        "gdalxarray": gdalxarray.__version__,
    }
    try:
        out["rasterix"] = md.version("rasterix")
    except md.PackageNotFoundError:
        pass
    return out


def footer():
    v = versions()
    print("[Rendered with " + ", ".join(f"{k} {x}" for k, x in v.items()) + ".]{.footer-versions}")


__all__ = [
    "np", "xr", "gdal", "gdalxarray", "fixture", "fixture_url",
    "fixture_bytes", "fixture_json", "fixture_array", "store",
    "data", "data_json", "provenance", "DOCS",
    "live", "status_callout", "versions", "footer", "missing_capabilities",
    "STORES", "STATUS", "FIXTURES_SHA", "PERMANENCE",
]
