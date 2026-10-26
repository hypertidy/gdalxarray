#!/usr/bin/env python3
"""Execute each gallery page on its own, under a hard time limit.

Gallery pages read remote stores, and a GDAL read blocked inside libcurl
cannot be interrupted from Python. So each page is rendered in its own
process group and killed outright if it overruns. A page that overruns
(or fails) is rendered again with its remote cells switched off, so it
still appears, with a callout saying why.

Results land in docs/_freeze (the gallery uses `freeze: auto`), so the
full `quarto render` that follows reuses them instead of re-reading the
stores.

Prints a timing table, and appends it to $GITHUB_STEP_SUMMARY when set.
Exits 1 only if a page fails even with its remote cells off, which means
a bug in the page rather than a slow store.

Usage (from the repository root):
    python scripts/render_gallery.py [--timeout 300] [page ...]
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"


OFFLINE_TIMEOUT = 300  # a render with remote cells off cannot wait on the network


def kill_tagged(tag):
    """Kill every process whose environment carries this render's tag.

    Quarto starts the Jupyter kernel in a session of its own, so killing
    the render's process group leaves the kernel (still blocked in a
    remote read) running. The tag is inherited by everything the render
    starts, so it finds exactly those processes. Linux only (/proc).
    """
    needle = f"GDALXARRAY_RENDER_TAG={tag}".encode()
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/environ", "rb") as f:
                if needle in f.read().split(b"\0"):
                    os.kill(int(pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass


def render(page, timeout, extra_env=None):
    tag = uuid.uuid4().hex
    env = dict(os.environ, GDALXARRAY_RENDER_TAG=tag, **(extra_env or {}))
    t0 = time.monotonic()
    proc = subprocess.Popen(
        ["quarto", "render", f"gallery/{page}.qmd", "--no-execute-daemon"],
        cwd=DOCS,
        env=env,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
        status = "ok" if proc.returncode == 0 else "failed"
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        kill_tagged(tag)
        out, _ = proc.communicate()
        status = "timed out"
        for leftover in (DOCS / "gallery").glob(f"{page}.quarto_ipynb*"):
            leftover.unlink()  # intermediate notebook a killed render leaves
    return status, time.monotonic() - t0, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=300, help="seconds per page")
    ap.add_argument("pages", nargs="*")
    args = ap.parse_args()

    pages = args.pages or sorted(
        p.stem for p in (DOCS / "gallery").glob("*.qmd") if p.stem != "index"
    )
    rows, broken = [], []
    for page in pages:
        status, secs, out = render(page, args.timeout)
        note = ""
        if status != "ok":
            print(f"--- {page}: {status} after {secs:.0f} s; last output:")
            print("\n".join(out.splitlines()[-25:]))
            reason = (
                f"this page exceeded its {args.timeout} s limit in the site build"
                if status == "timed out"
                else "this page failed against the live stores in the site build"
            )
            s2, secs2, out2 = render(
                page, max(args.timeout, OFFLINE_TIMEOUT), {"GDALXARRAY_DOCS_OFFLINE": reason}
            )
            note = f"re-rendered without remote cells: {s2} ({secs2:.0f} s)"
            if s2 != "ok":
                broken.append(page)
                print("\n".join(out2.splitlines()[-25:]))
        rows.append((page, status, secs, note))
        print(f"{page:12s} {status:10s} {secs:6.0f} s  {note}", flush=True)

    table = ["| gallery page | live render | seconds | note |", "|---|---|---|---|"]
    table += [f"| {p} | {s} | {t:.0f} | {n} |" for p, s, t, n in rows]
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write("\n".join(table) + "\n\n")
    print("\n".join(table))
    sys.exit(1 if broken else 0)


if __name__ == "__main__":
    main()
