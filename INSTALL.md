# Installing gdalxarray

`gdalxarray` is a pure-Python package and is installed with `pip`. It depends on
the GDAL Python bindings (`osgeo.gdal`), which are **not** installable via pip
alone - GDAL is a C++ library and its Python bindings need a matching system
library, development headers, and a compiler.

This is the same situation faced by `osgeo.gdal` itself, `rasterio` (which
sidesteps it by shipping binary wheels), and other GDAL-based tools. We
recommend installing GDAL through one of the channels below, then `pip install
gdalxarray` on top.

## Quick start: Docker

The fastest path to a working environment is a container that already has GDAL.

### hypertidy gdal-r-python (recommended for development)

```bash
docker run --rm -it ghcr.io/hypertidy/gdal-r-python:latest bash
# inside the container:
pip install gdalxarray
```

This image is the reference development environment for `gdalxarray`. It
includes GDAL 3.12+, the Python bindings, numpy 2.x, and the standard hypertidy
toolchain. Tests pass here; if something works elsewhere but not here, it's a
bug.

### Pangeo notebook

```bash
docker run --rm -it  quay.io/pangeo/pangeo-notebook:latest bash
pip install gdalxarray
```

The Pangeo image is a known-working profile (verified with GDAL 3.12.3 and
numpy 2.4.6) and ships JupyterLab.

### osgeo/gdal

For a GDAL-only base image:

```bash
docker run --rm -it ghcr.io/osgeo/gdal:ubuntu-small-latest bash

## example using venv
apt-get update && apt-get install -y python3-pip python3-venv
python3 -m venv .venv
. .venv/bin/activate
pip install gdalxarray
```

## conda / mamba / pixi

conda-forge has well-maintained GDAL builds for Linux, macOS, and Windows.

This was tried on ubuntu.

```bash
DEBIAN_FRONTEND=noninteractive apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y wget python3


wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
## interactive step requires input
bash Miniforge3-Linux-x86_64.sh
bash
mamba create -n gdalxarray -c conda-forge python gdal numpy xarray dask
mamba activate gdalxarray
pip install gdalxarray
```

Or with `pixi` (see https://github.com/hypertidy/gdalxarray/issues/28)

A minimum-viable `environment.yml`:

```yaml
name: gdalxarray
channels:
  - conda-forge
dependencies:
  - python>=3.10
  - gdal>=3.12
  - libgdal-netcdf
  - libgdal-hdf5
  - numpy
  - xarray>=2025.6
  - dask
  - affine
  - xproj
  - pip
  - pip:
    - gdalxarray
    - rasterix
```

The `libgdal-*` subpackages enable extra format drivers (NetCDF, HDF5). The base
`gdal` package on conda-forge is split for size; add subpackages for the formats
you need. The Zarr driver is included in the base.

## System package manager

If your system has a recent enough GDAL (>= 3.12), use the system package and
install the Python bindings separately:

```bash
# Ubuntu / Debian
sudo apt install gdal-bin libgdal-dev python3-gdal
pip install gdalxarray

# Fedora
sudo dnf install gdal gdal-devel gdal-python3
pip install gdalxarray

# macOS (Homebrew)
brew install gdal
pip install gdal=="$(gdal-config --version).*"
pip install gdalxarray
```

The Homebrew path requires `pip install gdal` to compile against the brew GDAL,
which usually works. Linux distributions ship `python3-gdal` precompiled.

## What gdalxarray pulls in via pip

Once GDAL is on the system, `pip install gdalxarray` installs only pure-Python
dependencies:

- `numpy`
- `xarray>=2025.6`
- `dask` (optional: only needed when opening with `chunks=`;
  install as `gdalxarray[dask]`)
- `affine`
- `rasterix`
- `xproj`

None of these need compilation.

## Verifying the install

```python
import gdalxarray
from osgeo import gdal
print(f"gdalxarray {gdalxarray.__version__}, GDAL {gdal.__version__}")

import xarray as xr
print("gdalxarray" in xr.backends.list_engines())  # should be True
```

## Troubleshooting

**`ImportError: gdalxarray requires the GDAL Python bindings ...`**
GDAL isn't installed in this Python environment. Install it through one of
the channels above before installing `gdalxarray`.

**`unrecognized engine 'gdalxarray'`**
`gdalxarray` is importable but its xarray backend entry point isn't
registered. Most commonly this means the package was used directly from a
source checkout without `pip install`. Run `pip install -e .` from the repo
root (for development) or `pip install gdalxarray` (for use).

**GDAL version too old**
Some `gdalxarray` features (notably `gdal mdim mosaic`) require GDAL >= 3.12.
Check with `python -c "from osgeo import gdal; print(gdal.__version__)"`.

**numpy ABI mismatch**
Mixing pip-installed GDAL with conda numpy (or vice versa) can produce
import errors complaining about numpy ABI. Stay within a single channel
(all conda, or all pip on top of system GDAL).

## Reporting installation problems

If none of the above paths work for your platform, please open an issue at
<https://github.com/hypertidy/gdalxarray/issues> with:

- OS and version
- Python version
- `gdal-config --version` (if GDAL is installed)
- `pip list | grep -i gdal`
- The full error output
