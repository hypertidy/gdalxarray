"""Quarto pre-render step: wxylib fixtures at the pinned commit.

Fetches one commit of wxylib/fixtures into docs/_fixtures (shallow,
about 36 MB). Pages fall back to reading the same commit over
/vsicurl/ if this fails, so it is best-effort.

Two other steps are separate, run before `quarto render`:

    python -m quartodoc build              # API reference pages
    python ../scripts/check_stores.py      # remote store liveness

(see .github/workflows/docs.yml and docs/README.md).
"""

import subprocess
from pathlib import Path

DOCS = Path(__file__).resolve().parent
FIX = DOCS / "_fixtures"
SHA = "c3adea57de88b76964125d7abf37329417d124e2"


def git(*a):
    return subprocess.run(["git", *a], cwd=FIX, check=True,
                          capture_output=True, text=True).stdout.strip()


try:
    if not (FIX / ".git").exists():
        FIX.mkdir(exist_ok=True)
        git("init", "-q")
        git("remote", "add", "origin", "https://github.com/wxylib/fixtures.git")
    try:
        have = git("rev-parse", "HEAD")
    except subprocess.CalledProcessError:
        have = ""
    if have != SHA:
        git("fetch", "-q", "--depth", "1", "origin", SHA)
        git("checkout", "-q", "FETCH_HEAD")
    print(f"fixtures at {SHA[:10]}")
except Exception as e:  # noqa: BLE001
    print(f"fixtures: local clone failed ({e}); pages will use /vsicurl/")
