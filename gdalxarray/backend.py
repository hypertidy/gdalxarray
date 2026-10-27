"""GDAL backend for xarray.

Three BackendArray classes wrap GDAL's three reading modes:

* ``GDALBackendArray``   - a single band of a classic raster
* ``GDALMultiBandArray`` - all bands of a classic raster as a (band, y, x) cube
* ``GDALMultiDimArray``  - a multidimensional array via GDAL's multidim API

``GDALBackendEntrypoint`` is registered as the ``gdalxarray`` xarray engine and
dispatches to one of two open paths depending on the ``multidim`` flag.
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import logging
import os
import threading
import warnings

import numpy as np
import xarray as xr
from affine import Affine
from osgeo import gdal
from rasterix import RasterIndex
from xarray.backends import BackendArray, BackendEntrypoint
from xarray.coding.times import decode_cf_datetime
from xarray.core import indexing
from xproj import CRSIndex  # noqa: F401  (registers xproj accessor)

logger = logging.getLogger(__name__)


# GDAL data type -> numpy type, matching what ReadAsArray returns
# (osgeo.gdal_array's table). Newer types are looked up by name so older
# bindings without them still import; unknown types fall back to float32.
# Not imported from gdal_array itself: that would make an ABI-mismatched
# numpy fail at gdalxarray import rather than at read time.
_GDAL_NUMPY_DTYPES = {
    getattr(gdal, f"GDT_{name}"): np_type
    for name, np_type in (
        ("Byte", np.uint8),
        ("Int8", np.int8),
        ("UInt16", np.uint16),
        ("Int16", np.int16),
        ("UInt32", np.uint32),
        ("Int32", np.int32),
        ("UInt64", np.uint64),
        ("Int64", np.int64),
        ("Float16", np.float16),
        ("Float32", np.float32),
        ("Float64", np.float64),
        ("CInt16", np.complex64),
        ("CInt32", np.complex64),
        ("CFloat16", np.complex64),
        ("CFloat32", np.complex64),
        ("CFloat64", np.complex128),
    )
    if hasattr(gdal, f"GDT_{name}")
}


# GDAL dimension types -> CF ``axis`` values.
_DIM_TYPE_AXIS = {
    "HORIZONTAL_X": "X",
    "HORIZONTAL_Y": "Y",
    "VERTICAL": "Z",
    "TEMPORAL": "T",
}


def _is_time_coord(array_name: str, attrs: dict, units: str | None) -> bool:
    """Recognise a time coordinate via CF conventions or a 'units since' string."""
    if attrs.get("axis") == "T":
        return True
    if attrs.get("standard_name") == "time":
        return True
    if array_name.lower() == "time":
        return True
    return bool(units and " since " in units)


def _attach_mdim_crs(ds: xr.Dataset, srs_wkt: dict) -> xr.Dataset:
    """Attach the md arrays' spatial reference, as the classic path does.

    Same shape as the classic raster path (an xproj ``CRSIndex`` on a scalar
    ``crs`` coordinate), plus the CF spelling (``crs_wkt`` on that coordinate,
    ``grid_mapping="crs"`` on each data variable that carries the SRS) so
    readers that follow CF conventions (rioxarray, writers that emit VRT or
    netCDF) find it too.

    Arrays with differing SRSs cannot share one CRS index: the most common
    one is attached and the others are reported in a warning.
    """
    if not srs_wkt or "crs" in ds.variables:
        return ds
    counts: dict[str, int] = {}
    for w in srs_wkt.values():
        counts[w] = counts.get(w, 0) + 1
    wkt = max(counts, key=counts.get)
    others = sorted(v for v, w in srs_wkt.items() if w != wkt)
    if others:
        warnings.warn(
            f"arrays {others} have a different spatial reference from the "
            "others; only the most common one is attached as the dataset CRS",
            UserWarning,
            stacklevel=4,
        )
    ds = ds.proj.assign_crs(crs=wkt).proj.write_crs_info()
    for v, w in srs_wkt.items():
        if w == wkt and v in ds.data_vars:
            ds[v].attrs.setdefault("grid_mapping", "crs")
    return ds


def _hidden_geometry_containers(group, listed: list[str]) -> list[str]:
    """Names of CF geometry container variables GDAL hides by default (issue #11).

    A CF-1.8 geometry container is a 0-d variable whose attributes
    (``geometry_type``, ``node_count``, ``node_coordinates``, ...) describe
    ragged node arrays elsewhere in the file. GDAL's multidim listing omits
    0-d arrays unless ``SHOW_ALL=YES`` is passed, so without this the
    container never reaches xarray and ``ds.xvec.decode_cf()`` (cf_xarray)
    cannot find it. Only containers are added back: other hidden arrays,
    such as grid-mapping variables GDAL has folded into the SRS, stay hidden
    so existing output is unchanged.
    """
    try:
        all_names = group.GetMDArrayNames(["SHOW_ALL=YES"])
    except (RuntimeError, TypeError):
        return []
    seen = set(listed)
    found = []
    for name in all_names or []:
        if name in seen:
            continue
        try:
            arr = group.OpenMDArray(name)
        except RuntimeError:
            continue
        if arr is None or arr.GetDimensionCount() != 0:
            continue
        if any(a.GetName() == "geometry_type" for a in arr.GetAttributes()):
            found.append(name)
    return found


def _restore_geometry_grid_mapping(group, listed, data_vars, coords) -> None:
    """Give each CF geometry container a ``grid_mapping`` xvec can decode.

    The CRS is the SRS GDAL reports on the container itself, or failing that
    on its first node coordinate array (writers differ in which carries the
    grid_mapping), and is written as ``crs_wkt`` on a 0-d coordinate. The
    variable keeps the file's own grid-mapping name when the file has exactly
    one hidden grid-mapping variable, otherwise it is called ``spatial_ref``.
    """
    hidden_gm = []
    try:
        for name in group.GetMDArrayNames(["SHOW_ALL=YES"]) or []:
            if name in listed:
                continue
            arr = group.OpenMDArray(name)
            if arr is not None and any(
                a.GetName() == "grid_mapping_name" for a in arr.GetAttributes()
            ):
                hidden_gm.append(name)
    except (RuntimeError, TypeError):
        pass
    gm_name = hidden_gm[0] if len(hidden_gm) == 1 else "spatial_ref"

    for name, var in data_vars.items():
        if var.ndim != 0 or "geometry_type" not in var.attrs or "grid_mapping" in var.attrs:
            continue
        node_coords = str(var.attrs.get("node_coordinates", "")).split()
        srs = None
        for candidate in [name] + node_coords[:1]:
            try:
                arr = group.OpenMDArray(candidate)
            except RuntimeError:
                arr = None
            srs = arr.GetSpatialRef() if arr is not None else None
            if srs is not None:
                break
        if srs is None:
            continue
        wkt = srs.ExportToWkt(["FORMAT=WKT2_2019"])
        existing = coords.get(gm_name)
        if existing is None:
            coords[gm_name] = xr.DataArray(0, attrs={"crs_wkt": wkt})
        elif getattr(existing, "attrs", {}).get("crs_wkt") != wkt:
            continue  # name taken by a different CRS; leave this container alone
        var.attrs["grid_mapping"] = gm_name


def _expand_tilde(dsn):
    """Expand a leading ``~`` in a plain filesystem path (issue #33).

    GDAL does not perform tilde expansion, so ``~/file.tif`` fails with a
    misleading "not recognized as a supported file format" error. Only
    ``os.path.expanduser`` is applied, and only when the string starts with
    ``~``: running ``abspath``/``normpath`` here (as xarray's
    ``_normalize_path`` does) would corrupt GDAL URIs such as
    ``/vsicurl/https://...`` by collapsing the double slash, and would
    mangle ``ZARR:``/``NETCDF:`` connection strings.
    """
    if isinstance(dsn, str) and dsn.startswith("~"):
        return os.path.expanduser(dsn)
    return dsn


def _gdal_config_context(config_options):
    """Thread-locally scoped GDAL configuration options (issue #25).

    Returns a context manager applying ``config_options`` for the
    duration of a GDAL call block on the *current thread only*, so
    concurrent dask workers with different stores never see each
    other's options and nothing leaks to unrelated work sharing the
    thread pool. Values are stringified. Empty/None is a no-op.
    """
    if not config_options:
        return contextlib.nullcontext()
    opts = {str(k): str(v) for k, v in config_options.items()}
    try:
        return gdal.config_options(opts, thread_local=True)
    except TypeError:
        # older bindings without the thread_local parameter
        return gdal.config_options(opts)


_HDF5_SILENCED = threading.local()


def _silence_hdf5_diagnostics_this_thread():
    """Best-effort: disable HDF5 error auto-printing on this thread.

    HDF5 error stacks are per-thread in threadsafe builds, and GDAL /
    netcdf-c disable auto-printing only on the thread that initialized
    the library. Per-thread reopens (issue #34) therefore enter HDF5
    from threads where printing is still enabled, and harmless probes
    for absent optional attributes spew HDF5-DIAG stacks to stderr.
    This calls H5Eset_auto2(H5E_DEFAULT, NULL, NULL) via ctypes on the
    libhdf5 already loaded in-process; real failures still surface as
    GDAL exceptions. No-op if libhdf5 is absent or the call fails.
    """
    if getattr(_HDF5_SILENCED, "done", False):
        return
    _HDF5_SILENCED.done = True
    try:
        import ctypes
        import ctypes.util

        lib = None
        for name in (
            "libhdf5.so.310",
            "libhdf5.so.200",
            "libhdf5.so.103",
            "libhdf5_serial.so.103",
            "libhdf5.so",
            ctypes.util.find_library("hdf5"),
        ):
            if not name:
                continue
            try:
                lib = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if lib is None:
            return
        fn = lib.H5Eset_auto2
        fn.argtypes = [ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p]
        fn.restype = ctypes.c_int
        fn(0, None, None)  # 0 == H5E_DEFAULT: this thread's stack
    except Exception:  # never let diagnostics hygiene break a read
        pass


def _axis_read_plan(key, size):
    """Plan a contiguous 1-D GDAL read for an int or slice key.

    Returns ``(read_start, read_count, rel_index, out_len, squeeze)``.
    Classic-raster reads have no stride, so stepped or reversed slices
    read the smallest contiguous covering window and reorder afterwards
    with ``rel_index`` (positions relative to ``read_start``, in output
    order); ``rel_index`` is None for ascending step-1 slices.
    ``read_count == 0`` marks an empty selection (issue #32) that must
    never reach GDAL. Negative ints and slice bounds follow Python
    semantics via ``slice.indices``.
    """
    if isinstance(key, int | np.integer):
        k = int(key)
        if k < 0:
            k += size
        if not 0 <= k < size:
            raise IndexError(f"index {key} out of bounds for axis of size {size}")
        return k, 1, None, 1, True
    if isinstance(key, slice):
        idx = np.arange(*key.indices(size))
        n = int(idx.size)
        if n == 0:
            return 0, 0, None, 0, False
        step = key.step if key.step is not None else 1
        if step == 1:
            return int(idx[0]), n, None, n, False
        lo = int(idx.min())
        return lo, int(idx.max()) - lo + 1, idx - lo, n, False
    raise IndexError(f"Unsupported index type: {type(key)}")


# ---------------------------------------------------------------------------
# Classic-raster, single band
# ---------------------------------------------------------------------------


class GDALBackendArray(BackendArray):
    """Wrap one GDAL raster band as an xarray BackendArray of shape (y, x)."""

    def __init__(self, filename: str, band_index: int = 1, config_options=None):
        self.filename = filename
        self._config_options = dict(config_options or {})
        self.band_index = band_index
        self._local = threading.local()
        logger.debug("filename: %s", filename)

        ds = gdal.Open(filename, gdal.GA_ReadOnly)
        if ds is None:
            raise ValueError(f"Could not open {filename}")
        band = ds.GetRasterBand(band_index)

        self._shape = (ds.RasterYSize, ds.RasterXSize)
        self._dtype = self._gdal_to_numpy_dtype(band.DataType)
        self._block_size = band.GetBlockSize()
        ds = None  # release

    def _get_band(self):
        if not hasattr(self._local, "ds"):
            self._local.ds = gdal.Open(self.filename, gdal.GA_ReadOnly)
            self._local.band = self._local.ds.GetRasterBand(self.band_index)
        return self._local.band

    def __getstate__(self):
        state = self.__dict__.copy()
        # Per-thread handles are not picklable; reopen lazily after unpickling.
        state["_local"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._local = threading.local()

    def __dask_tokenize__(self):
        return (
            type(self).__name__,
            self.filename,
            self.band_index,
            tuple(sorted(self._config_options.items())),
        )

    @staticmethod
    def _gdal_to_numpy_dtype(gdal_dtype):
        return _GDAL_NUMPY_DTYPES.get(gdal_dtype, np.float32)

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return np.dtype(self._dtype)

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def size(self):
        return int(np.prod(self._shape))

    def __getitem__(self, key):
        # xarray decomposes outer/vectorized indexers (integer or boolean
        # arrays, e.g. isel(time=ds.time.dt.month == 6)) into a covering
        # BASIC read here plus an in-memory numpy step on its side.
        # Raw int/slice keys (direct use outside xarray's lazy wrappers)
        # are wrapped as BasicIndexer for backward compatibility.
        if not isinstance(key, indexing.ExplicitIndexer):
            key = indexing.BasicIndexer(key if isinstance(key, tuple) else (key,))
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key):
        """Read (y, x) from a tuple of ints and slices.

        Stepped and reversed slices read the smallest contiguous
        covering window (band reads have no stride) and reorder in
        numpy via the plan's relative index.
        """
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) < 2:
            key = key + (slice(None),) * (2 - len(key))
        if len(key) > 2:
            raise IndexError(f"Expected at most 2D index, got {len(key)}D")

        y0, ny, y_rel, y_len, squeeze_y = _axis_read_plan(key[0], self.shape[0])
        x0, nx, x_rel, x_len, squeeze_x = _axis_read_plan(key[1], self.shape[1])

        # Zero-sized read: emitted by Dask for _meta inference and by
        # empty label selections (issue #32); never reaches GDAL
        if ny == 0 or nx == 0:
            shape = []
            if not squeeze_y:
                shape.append(y_len)
            if not squeeze_x:
                shape.append(x_len)
            return np.empty(shape, dtype=self._dtype)

        logger.debug("read: yoff=%s xoff=%s ysize=%s xsize=%s", y0, x0, ny, nx)
        with _gdal_config_context(self._config_options):
            band = self._get_band()
            data = band.ReadAsArray(
                xoff=x0,
                yoff=y0,
                win_xsize=nx,
                win_ysize=ny,
            )
        if y_rel is not None:
            data = data[y_rel, :]
        if x_rel is not None:
            data = data[:, x_rel]

        if squeeze_y and squeeze_x:
            return data[0, 0]
        if squeeze_y:
            return data[0, :]
        if squeeze_x:
            return data[:, 0]
        return data


# ---------------------------------------------------------------------------
# Classic-raster, all bands as one (band, y, x) array
# ---------------------------------------------------------------------------


class GDALMultiBandArray(BackendArray):
    """All bands of a GDAL dataset exposed as one 3D (band, y, x) array.

    Used by ``_open_raster`` when ``band_as_dim=True`` (the default). Reading
    multiple bands at once via ``dataset.ReadAsArray(band_list=...)`` lets
    GDAL handle BIP/BIL/BSQ interleaving internally - typically faster than
    iterating per-band.
    """

    def __init__(self, filename: str, band_indices: list[int] | None = None, config_options=None):
        self.filename = filename
        self._config_options = dict(config_options or {})
        self._local = threading.local()
        logger.debug("multiband filename: %s", filename)

        ds = gdal.Open(filename, gdal.GA_ReadOnly)
        if ds is None:
            raise ValueError(f"Could not open {filename}")

        num_bands = ds.RasterCount
        if band_indices is None:
            band_indices = list(range(1, num_bands + 1))
        self.band_indices = list(band_indices)

        band1 = ds.GetRasterBand(self.band_indices[0])
        self._shape = (len(self.band_indices), ds.RasterYSize, ds.RasterXSize)
        self._dtype = GDALBackendArray._gdal_to_numpy_dtype(band1.DataType)
        self._block_size = band1.GetBlockSize()
        ds = None

    def _get_dataset(self):
        if not hasattr(self._local, "ds"):
            self._local.ds = gdal.Open(self.filename, gdal.GA_ReadOnly)
        return self._local.ds

    def __getstate__(self):
        state = self.__dict__.copy()
        # Per-thread handles are not picklable; reopen lazily after unpickling.
        state["_local"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._local = threading.local()

    def __dask_tokenize__(self):
        return (
            type(self).__name__,
            self.filename,
            tuple(self.band_indices),
            tuple(sorted(self._config_options.items())),
        )

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return np.dtype(self._dtype)

    @property
    def ndim(self):
        return 3

    @property
    def size(self):
        return int(np.prod(self._shape))

    def __getitem__(self, key):
        # xarray decomposes outer/vectorized indexers (integer or boolean
        # arrays, e.g. isel(time=ds.time.dt.month == 6)) into a covering
        # BASIC read here plus an in-memory numpy step on its side.
        # Raw int/slice keys (direct use outside xarray's lazy wrappers)
        # are wrapped as BasicIndexer for backward compatibility.
        if not isinstance(key, indexing.ExplicitIndexer):
            key = indexing.BasicIndexer(key if isinstance(key, tuple) else (key,))
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key):
        """Read via ``dataset.ReadAsArray`` with an explicit band_list.

        The band dimension supports arbitrary order natively (GDAL
        honours band_list order, so reversed or stepped band slices and
        integer-array band keys need no post-processing); y/x use
        covering-window reads with a numpy reorder for stepped or
        reversed slices.
        """
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) < 3:
            key = key + (slice(None),) * (3 - len(key))
        if len(key) > 3:
            raise IndexError(f"Expected at most 3D index, got {len(key)}D")

        b_idx, y_idx, x_idx = key
        nbands = len(self.band_indices)
        squeeze_b = False

        if isinstance(b_idx, int | np.integer):
            b = int(b_idx)
            if b < 0:
                b += nbands
            if not 0 <= b < nbands:
                raise IndexError(f"band index {b_idx} out of bounds for {nbands} bands")
            positions = [b]
            squeeze_b = True
        elif isinstance(b_idx, slice):
            positions = list(range(*b_idx.indices(nbands)))
        elif isinstance(b_idx, list | np.ndarray):
            positions = [int(i) + nbands if int(i) < 0 else int(i) for i in b_idx]
        else:
            raise IndexError(f"Unsupported band index type: {type(b_idx)}")
        band_list = [self.band_indices[i] for i in positions]

        y0, ny, y_rel, y_len, squeeze_y = _axis_read_plan(y_idx, self._shape[1])
        x0, nx, x_rel, x_len, squeeze_x = _axis_read_plan(x_idx, self._shape[2])

        # Zero-sized read: emitted by Dask for _meta inference and by
        # empty label selections (issue #32); never reaches GDAL
        if ny == 0 or nx == 0 or len(band_list) == 0:
            shape = []
            if not squeeze_b:
                shape.append(len(band_list))
            if not squeeze_y:
                shape.append(y_len)
            if not squeeze_x:
                shape.append(x_len)
            return np.empty(shape, dtype=self._dtype)

        logger.debug(
            "read: bands=%s yoff=%s xoff=%s ysize=%s xsize=%s",
            band_list,
            y0,
            x0,
            ny,
            nx,
        )
        with _gdal_config_context(self._config_options):
            ds = self._get_dataset()
            data = ds.ReadAsArray(
                xoff=x0,
                yoff=y0,
                xsize=nx,
                ysize=ny,
                band_list=band_list,
            )

        # ReadAsArray returns 2D for a single band, 3D for multiple - normalise
        if data.ndim == 2:
            data = data[np.newaxis, :, :]

        if y_rel is not None:
            data = data[:, y_rel, :]
        if x_rel is not None:
            data = data[:, :, x_rel]

        axes_to_squeeze = []
        if squeeze_b:
            axes_to_squeeze.append(0)
        if squeeze_y:
            axes_to_squeeze.append(1)
        if squeeze_x:
            axes_to_squeeze.append(2)
        if axes_to_squeeze:
            data = np.squeeze(data, axis=tuple(axes_to_squeeze))
        return data


# ---------------------------------------------------------------------------
# Multidim
# ---------------------------------------------------------------------------


class GDALMultiDimArray(BackendArray):
    """Wrap a GDAL MDArray as an N-D xarray BackendArray.

    Thread safety (issue #34): GDAL handles are not thread-safe
    per-handle, but GDAL is thread-safe across handles (drivers such as
    netCDF/HDF5 take their own global locks internally). Reads therefore
    go through a per-thread handle resolved lazily in ``_get_mdarray``,
    mirroring ``GDALBackendArray._get_band``: each dask worker thread
    reopens the store once and reuses its own handle, and since osgeo
    releases the GIL during I/O those reads genuinely parallelize.

    The constructing thread is seeded with the already-open handle, so
    the non-dask path pays no reopen cost. Parent dataset/group
    references keep that seed handle valid after the open-time variables
    drop out of scope (a multidim MDArray is a view into its parent).

    Instances pickle by dropping live handles and thread-locals; the
    unpickling side reopens from ``filename``/``fullname`` on first
    read, which makes the class usable under distributed schedulers.
    """

    def __init__(
        self, mdarray, filename=None, _parent_dataset=None, _parent_group=None, config_options=None
    ):
        self._filename = filename
        self._config_options = dict(config_options or {})
        self._fullname = mdarray.GetFullName()  # e.g. "/group/array"
        self._local = threading.local()
        # Seed this thread with the open handle (no reopen on first read).
        self._local.mdarray = mdarray
        # Keep parent objects alive - without these refs the seed mdarray
        # can be invalidated when the user-facing ds goes out of scope.
        self._parent_dataset = _parent_dataset
        self._parent_group = _parent_group

        dims = mdarray.GetDimensions()
        self._shape = tuple(dim.GetSize() for dim in dims)

        gdal_dtype = mdarray.GetDataType().GetNumericDataType()
        self._dtype = GDALBackendArray._gdal_to_numpy_dtype(gdal_dtype)
        self._chunks = tuple(mdarray.GetBlockSize())

    def _get_mdarray(self):
        """Per-thread MDArray handle, reopened lazily off-thread."""
        if not hasattr(self._local, "mdarray"):
            _silence_hdf5_diagnostics_this_thread()
            if self._filename is None:
                raise RuntimeError(
                    "GDALMultiDimArray was constructed without a filename; "
                    "reads from other threads or after unpickling need one "
                    "to reopen the store (issue #34)"
                )
            ds = gdal.OpenEx(self._filename, gdal.OF_MULTIDIM_RASTER | gdal.GA_ReadOnly)
            if ds is None:
                raise ValueError(f"Could not reopen {self._filename}")
            root = ds.GetRootGroup()
            md = root.OpenMDArrayFromFullname(self._fullname)
            if md is None:
                raise ValueError(f"Could not open array {self._fullname!r} in {self._filename}")
            # Keep the whole chain alive for this thread's lifetime.
            self._local.ds = ds
            self._local.root = root
            self._local.mdarray = md
        return self._local.mdarray

    def __getstate__(self):
        state = self.__dict__.copy()
        # Live GDAL handles and thread-locals are not picklable; the
        # receiving side reopens from filename/fullname on first read.
        state["_local"] = None
        state["_parent_dataset"] = None
        state["_parent_group"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._local = threading.local()

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return np.dtype(self._dtype)

    def __dask_tokenize__(self):
        # Strings only: tokenize can be called from any thread, so it
        # must never touch a live GDAL handle (issue #34).
        return (
            type(self).__name__,
            self._filename,
            self._fullname,
            self._shape,
            str(self._dtype),
            self._chunks,
            tuple(sorted(self._config_options.items())),
        )

    # Upper bound on the separate reads one outer selection may issue.
    # Beyond it, the axes with the most runs fall back to one covering read.
    MAX_OUTER_READS = 512

    def __getitem__(self, key):
        # OUTER support: integer or boolean arrays (e.g.
        # isel(time=ds.time.dt.month == 6)) reach _raw_indexing_method as
        # arrays, which it reads as runs of nearby indices instead of one
        # covering slab. Vectorized (pointwise) indexers are decomposed by
        # xarray into an outer read here plus an in-memory numpy step.
        # Raw int/slice keys (direct use outside xarray's lazy wrappers)
        # are wrapped as BasicIndexer for backward compatibility.
        if not isinstance(key, indexing.ExplicitIndexer):
            key = indexing.BasicIndexer(key if isinstance(key, tuple) else (key,))
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.OUTER,
            self._raw_indexing_method,
        )

    def _runs(self, axis, uniq):
        """Group sorted unique indices on one axis into read runs.

        Neighbouring indices share a run when they are adjacent or fall in
        the same storage block along this axis, so a run never costs a
        block read that a per-index read would not also cost.
        """
        block = self._chunks[axis] if axis < len(self._chunks) else 0
        block = block or self.shape[axis]
        runs = []
        lo = 0
        for j in range(1, len(uniq) + 1):
            if j == len(uniq) or not (
                uniq[j] - uniq[j - 1] == 1 or uniq[j] // block == uniq[j - 1] // block
            ):
                runs.append((lo, j))  # positions [lo, j) in uniq
                lo = j
        return runs

    def _read_outer(self, key):
        """Read an outer selection with integer-array axes as runs.

        Each array axis is reduced to its sorted unique indices, grouped
        into runs (see ``_runs``); one basic read is made per combination
        of runs across array axes, and the selected positions are taken
        from each read. Order and duplicates are restored at the end.
        """
        arr_axes, uniqs, inverses, runs = [], {}, {}, {}
        for i, k in enumerate(key):
            if isinstance(k, np.ndarray):
                k = np.flatnonzero(k) if k.dtype == bool else k.astype(np.int64)
                k = np.where(k < 0, k + self.shape[i], k)
                if k.size and (k.min() < 0 or k.max() >= self.shape[i]):
                    raise IndexError(
                        f"index out of bounds for dimension {i} of size {self.shape[i]}"
                    )
                uniqs[i], inverses[i] = np.unique(k, return_inverse=True)
                runs[i] = self._runs(i, uniqs[i])
                arr_axes.append(i)

        # Keep the number of reads bounded: collapse the busiest axes to a
        # single covering run until the combination count fits.
        while int(np.prod([len(runs[i]) for i in arr_axes])) > self.MAX_OUTER_READS:
            worst = max(arr_axes, key=lambda i: len(runs[i]))
            runs[worst] = [(0, len(uniqs[worst]))]

        # Output axes: every key axis except integer-indexed ones.
        out_axis = {}
        n = 0
        for i, k in enumerate(key):
            if not isinstance(k, int | np.integer):
                out_axis[i] = n
                n += 1

        out = None
        for combo in itertools.product(*(runs[i] for i in arr_axes)):
            sub = list(key)
            for i, (lo, hi) in zip(arr_axes, combo, strict=True):
                sub[i] = slice(int(uniqs[i][lo]), int(uniqs[i][hi - 1]) + 1)
            block = self._raw_indexing_method(tuple(sub))
            place = [slice(None)] * block.ndim
            for i, (lo, hi) in zip(arr_axes, combo, strict=True):
                local = uniqs[i][lo:hi] - uniqs[i][lo]
                if local.size != block.shape[out_axis[i]]:
                    block = np.take(block, local, axis=out_axis[i])
                place[out_axis[i]] = slice(lo, hi)
            if out is None:
                shape = list(block.shape)
                for i in arr_axes:
                    shape[out_axis[i]] = len(uniqs[i])
                out = np.empty(shape, dtype=block.dtype)
            out[tuple(place)] = block

        for i in arr_axes:
            inv = inverses[i]
            if not (len(inv) == len(uniqs[i]) and np.array_equal(inv, np.arange(len(inv)))):
                out = np.take(out, inv, axis=out_axis[i])
        return out

    def _raw_indexing_method(self, key):
        """Read data from GDAL multidim array."""
        if not isinstance(key, tuple):
            key = (key,)

        if any(isinstance(k, np.ndarray) for k in key):
            if any(isinstance(k, np.ndarray) and k.size == 0 for k in key):
                shape = [
                    (k.size if isinstance(k, np.ndarray) else len(range(*k.indices(self.shape[i]))))
                    for i, k in enumerate(key)
                    if not isinstance(k, int | np.integer)
                ]
                return np.empty(shape, dtype=self._dtype)
            return self._read_outer(key)

        starts = []
        counts = []
        steps = []
        squeeze_dims = []

        flip_axes = []
        for i, k in enumerate(key):
            if isinstance(k, slice):
                # slice.indices gives Python slice semantics: clamped
                # bounds, and empty ranges (count 0) for e.g. the
                # slice(3, 0) pandas emits for an empty label selection
                # on a descending coordinate (issue #32).
                start, stop, step = k.indices(self.shape[i])
                count = len(range(start, stop, step))
                if count and step < 0:
                    # MDArray strided reads ascend: read the same index
                    # set forward and flip this axis after the read.
                    start = start + (count - 1) * step
                    step = -step
                    flip_axes.append(i)
            elif isinstance(k, int | float | np.integer | np.floating):
                start = int(k)
                if start < 0:
                    start += self.shape[i]
                if not 0 <= start < self.shape[i]:
                    raise IndexError(
                        f"index {k} out of bounds for dimension {i} of size {self.shape[i]}"
                    )
                count = 1
                step = 1
                squeeze_dims.append(i)
            else:
                raise IndexError(f"Unsupported index type: {type(k)}")

            starts.append(start)
            counts.append(count)
            steps.append(step)

        # Zero-sized read: emitted by Dask for _meta inference and by
        # empty label selections normalised above (issue #32)
        if any(c == 0 for c in counts):
            shape = [c for i, c in enumerate(counts) if i not in squeeze_dims]
            return np.empty(shape, dtype=self._dtype)

        # Resolve the per-thread handle once per read (issue #34): every
        # GDAL call below must go through it, never a shared handle. The
        # config context (issue #25) covers the reopen and all GDAL calls.
        with _gdal_config_context(self._config_options):
            data = self._read_mdim(key, starts, counts, steps)

        for ax in flip_axes:
            data = np.flip(data, axis=ax)

        # Squeeze out integer-indexed dimensions
        for dim_idx in reversed(squeeze_dims):
            data = np.squeeze(data, axis=dim_idx)

        return data

    def _read_mdim(self, key, starts, counts, steps):
        mdarray = self._get_mdarray()

        # AdviseRead is a prefetch hint. Compute a reasonable CACHE_SIZE for it,
        # bounded above to avoid GDAL choking on absurd values when sharded
        # stores report a shard-sized block (see #issue-AdviseRead-cache).
        block = np.array(mdarray.GetBlockSize())
        for i in range(len(block)):
            if block[i] == 0:
                block[i] = self.shape[i]

        read_elems = int(np.prod(counts))
        read_bytes = int(self._dtype().itemsize * read_elems)
        # Skip AdviseRead on tiny reads (no benefit) or huge reads (likely shard).
        MIN_BYTES = 4 * 1024 * 1024  # 4 MB
        MAX_BYTES = 512 * 1024 * 1024  # 512 MB
        if MIN_BYTES < read_bytes < MAX_BYTES:
            cache_size = min(int(read_bytes * 1.2), MAX_BYTES)
            try:
                mdarray.AdviseRead(
                    array_start_idx=starts,
                    count=counts,
                    options=[f"CACHE_SIZE={cache_size}"],
                )
            except RuntimeError as e:
                logger.debug("AdviseRead failed (%s); proceeding with direct read", e)

        logger.debug(
            "starts=%s counts=%s steps=%s shape=%s chunks=%s",
            starts,
            counts,
            steps,
            self.shape,
            self._chunks,
        )

        data = mdarray.ReadAsArray(
            array_start_idx=starts,
            count=counts,
            array_step=steps,
        )

        return data


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


class GDALBackendEntrypoint(BackendEntrypoint):
    """xarray backend for reading geospatial files via GDAL."""

    available = True
    description = "Open geospatial datasets with GDAL (classic and multidim APIs)"
    url = "https://github.com/hypertidy/gdalxarray"

    def __init__(self):
        # Opt the process into GDAL Python exceptions on first entrypoint use,
        # rather than at module import - keeps the side-effect scoped to actual
        # use of the backend. Guard against repeated calls.
        if not gdal.GetUseExceptions():
            gdal.UseExceptions()

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        chunks=None,
        multidim=True,
        group=None,
        band_as_dim=True,
        config_options=None,
        mask_and_scale=None,
        decode_times=None,
        decode_timedelta=None,
        use_cftime=None,
        concat_characters=None,
        decode_coords=None,
    ) -> xr.Dataset:
        """Open a dataset using GDAL.

        Parameters
        ----------
        filename_or_obj : str
            Path or GDAL-recognised URI (``/vsicurl/``, ``/vsis3/``,
            ``vrt://``, ``ZARR:``, ``NETCDF:``, ``/vsiicechunk/...``).
        drop_variables : iterable of hashable, optional
            Variables to omit from the returned Dataset.
        chunks : dict, optional
            Chunk sizes for Dask arrays. ``None`` (default) returns a lazy
            non-Dask Dataset; ``{}`` uses GDAL's native block sizes;
            an explicit mapping like ``{"y": 256, "x": 256}`` is honoured
            (dimensions not named keep a single chunk). Chunking is
            delegated to ``Dataset.chunk`` so chunked arrays are created
            by xarray's active chunk manager rather than by this backend
            (issue #31); dask is only required when chunks is not None.
        multidim : bool, default True
            If True, use GDAL's multidimensional API
            (``OpenEx`` + ``OF_MULTIDIM_RASTER``). If False, use the classic
            raster API.
        group : str, optional
            Group path for multidim datasets (e.g. ``"/group/subgroup"``).
            Leading/trailing slashes are tolerated.
        band_as_dim : bool, default True
            Classic-raster only. If True, bands become an xarray ``band``
            dimension on a single ``band_data`` DataArray (the rioxarray-
            compatible idiom). If False, each band becomes a separate data
            variable named after its description (or ``band_N``). The True
            default suits multispectral imagery; False is preferable when
            bands carry semantically distinct quantities. Bands reporting
            differing nodata/scale/offset cannot be CF-decoded as one
            variable: values are returned raw with per-band metadata in
            band_nodata/band_scale_factor/band_add_offset coordinates and
            a warning is emitted (use ``band_as_dim=False`` to decode each
            band independently).
        config_options : dict, optional
            GDAL configuration options, e.g.
            ``{"GDAL_HTTP_MAX_RETRY": "3", "GDAL_HTTP_RETRY_DELAY": "5"}``.
            Applied thread-locally around the open and around every
            subsequent read, including per-thread reopens under dask
            (issue #34) and reads on distributed workers after
            unpickling, then unset again, so concurrent opens with
            different options do not interfere. Values are stringified.
            Not applied during engine auto-detection (guess_can_open).
        mask_and_scale : bool, optional
            If True (default), apply CF mask/scale decoding: nodata becomes
            NaN and scale_factor/add_offset are applied, with the original
            values recorded in each variable's ``encoding``. If False,
            return raw values with the CF attributes left in ``attrs``.
        """
        filename_or_obj = _expand_tilde(filename_or_obj)
        # xarray passes mask_and_scale=None to mean "backend default"; the
        # default here matches xarray's own (apply CF mask/scale decoding).
        mask_and_scale = True if mask_and_scale is None else bool(mask_and_scale)
        config_options = dict(config_options or {})
        with _gdal_config_context(config_options):
            if multidim:
                ds = self._open_multidim(
                    filename_or_obj,
                    group,
                    drop_variables,
                    mask_and_scale,
                    config_options,
                )
            else:
                ds = self._open_raster(
                    filename_or_obj,
                    drop_variables,
                    band_as_dim=band_as_dim,
                    mask_and_scale=mask_and_scale,
                    config_options=config_options,
                )
        return self._maybe_chunk_dataset(ds, chunks, filename_or_obj, config_options)

    @staticmethod
    def _maybe_chunk_dataset(ds, chunks, filename_or_obj, config_options=None):
        """Delegate chunking to xarray's managed path (issue #31).

        The backend never constructs dask collections directly: variables
        are opened as lazy arrays and chunked here via ``Dataset.chunk``,
        so the chunked array class is produced by the same chunk manager
        that xarray later uses to recognise it. A deterministic token
        derived from the dsn keeps dask layer names stable across sessions
        without tokenizing the underlying (unpicklable) GDAL handles:
        with an explicit token, xarray's ``_maybe_chunk`` only tokenizes
        strings. Chunking runs after CF decoding and coordinate assembly,
        so decoders operate on lazy arrays and never need to recognise a
        chunked array class.
        """
        if chunks is None:
            return ds
        try:
            import dask  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "opening with chunks= requires dask, which is an optional "
                "dependency: install it with 'pip install gdalxarray[dask]' "
                "or add dask to the environment"
            ) from e
        # The explicit token becomes the dask layer name and short-circuits
        # tokenization of the wrapped arrays (issue #31), so it must carry
        # everything that affects read results: the dsn AND the config
        # options (issue #25) - otherwise same-dsn opens with different
        # options would deduplicate to one graph.
        token_src = str(filename_or_obj) + repr(sorted((config_options or {}).items()))
        token = hashlib.sha256(token_src.encode("utf-8", "replace")).hexdigest()[:16]
        if chunks == {}:
            # Native GDAL block sizes, recorded per variable at open time
            # in encoding["preferred_chunks"]. First-wins on any dim shared
            # by variables reporting different block sizes.
            merged = {}
            for var in ds.data_vars.values():
                for dim, size in (var.encoding.get("preferred_chunks") or {}).items():
                    merged.setdefault(dim, size)
            chunks = merged
        return ds.chunk(dict(chunks), name_prefix="gdalxarray-", token=token)

    # ------------------------------------------------------------------
    # Classic-raster path
    # ------------------------------------------------------------------

    def _open_raster(
        self,
        filename_or_obj,
        drop_variables,
        band_as_dim=True,
        mask_and_scale=True,
        config_options=None,
    ):
        """Open using GDAL's classic raster API."""
        logger.debug("filename_or_obj: %s", filename_or_obj)
        dataset = gdal.Open(filename_or_obj, gdal.GA_ReadOnly)
        if dataset is None:
            raise ValueError(f"Could not open {filename_or_obj} with GDAL")

        # If the file presents no raster bands but has subdatasets, the user
        # has opened a multidim source in classic mode. Refuse with a helpful
        # listing rather than silently returning an empty 512x512 stub
        # Dataset (which GDAL's default raster info produces for these files).
        if dataset.RasterCount == 0:
            subdatasets = dataset.GetSubDatasets()
            if subdatasets:
                lines = [f"\n  {path}\n      {desc}" for path, desc in subdatasets]
                raise ValueError(
                    f"{filename_or_obj!r} has no raster bands at the top level "
                    f"but contains {len(subdatasets)} subdataset(s). "
                    f"Use multidim=True to read the full hierarchy as an xarray "
                    f"Dataset, or pass one of the subdataset paths below as the "
                    f"filename to read a single 2D view in classic mode:" + "".join(lines)
                )
            raise ValueError(
                f"{filename_or_obj!r} opens but has no raster bands and no "
                f"subdatasets. Cannot be read as a classic raster."
            )

        geotransform = dataset.GetGeoTransform()
        index = RasterIndex.from_transform(
            Affine.from_gdal(*geotransform),
            width=dataset.RasterXSize,
            height=dataset.RasterYSize,
        )
        projection = dataset.GetProjection()
        driver_name = dataset.GetDriver().GetDescription() if dataset.GetDriver() else None
        num_bands = dataset.RasterCount

        if band_as_dim:
            ds = self._raster_as_band_dim(
                filename_or_obj,
                dataset,
                drop_variables,
                num_bands,
                mask_and_scale,
                config_options,
            )
        else:
            ds = self._raster_as_vars(
                filename_or_obj,
                dataset,
                drop_variables,
                num_bands,
                mask_and_scale,
                config_options,
            )

        # Spatial coords and CRS for both layouts
        ds = ds.assign_coords(xr.Coordinates.from_xindex(index))
        if len(projection) > 0:
            ds = ds.proj.assign_crs(crs=projection)

        # Serialization-safe provenance hints in encoding (strings only).
        ds.encoding["source"] = filename_or_obj
        if driver_name:
            ds.encoding["gdal_driver"] = driver_name
        return ds

    def _raster_as_band_dim(
        self,
        filename_or_obj,
        dataset,
        drop_variables,
        num_bands,
        mask_and_scale=True,
        config_options=None,
    ):
        """Bands collapsed into a single ``band`` dimension on ``band_data``."""
        band_indices = list(range(1, num_bands + 1))
        descriptions = []
        nodatas = []
        scales = []
        offsets = []
        for band_idx in band_indices:
            band = dataset.GetRasterBand(band_idx)
            descriptions.append(band.GetDescription() or f"band_{band_idx}")
            nodatas.append(band.GetNoDataValue())
            scales.append(band.GetScale() if band.GetScale() is not None else 1.0)
            offsets.append(band.GetOffset() if band.GetOffset() is not None else 0.0)

        backend_array = GDALMultiBandArray(
            filename_or_obj, band_indices, config_options=config_options
        )
        data = indexing.LazilyIndexedArray(backend_array)

        # Native block sizes for chunks={} (band chunk of 1 matches GDAL's
        # per-band block model); consumed by _maybe_chunk_dataset.
        block_size = dataset.GetRasterBand(1).GetBlockSize()  # [x, y]
        preferred_chunks = {
            "band": 1,
            "y": block_size[1] if block_size[1] > 0 else dataset.RasterYSize,
            "x": block_size[0] if block_size[0] > 0 else dataset.RasterXSize,
        }

        attrs = {"descriptions": descriptions}
        coords = {"band": np.array(band_indices)}
        heterogeneous = []

        # _FillValue: scalar if all bands agree, per-band coord otherwise.
        # Skip if all bands have no nodata.
        # np.unique collapses NaN, unlike a set of distinct float('nan') objects
        if all(n is None for n in nodatas):
            pass
        elif None not in nodatas and np.unique(np.array(nodatas, dtype=float)).size == 1:
            attrs["_FillValue"] = nodatas[0]
        else:
            # Mixed or partially-set. Deliberately NOT named _FillValue:
            # CF decoding cannot consume a per-band coordinate, so using
            # the CF name would make the dataset look decoded while the
            # values stay raw (issue #29).
            coords["band_nodata"] = (
                "band",
                np.array([n if n is not None else np.nan for n in nodatas]),
            )
            heterogeneous.append("nodata")

        # scale_factor: skip default 1.0, scalar if all agree, per-band otherwise.
        if all(s == 1.0 for s in scales):
            pass
        elif len(set(scales)) == 1:
            attrs["scale_factor"] = scales[0]
        else:
            coords["band_scale_factor"] = ("band", np.array(scales))
            heterogeneous.append("scale")

        # add_offset: skip default 0.0, scalar if all agree, per-band otherwise.
        if all(o == 0.0 for o in offsets):
            pass
        elif len(set(offsets)) == 1:
            attrs["add_offset"] = offsets[0]
        else:
            coords["band_add_offset"] = ("band", np.array(offsets))
            heterogeneous.append("offset")

        if heterogeneous:
            warnings.warn(
                f"Bands of {filename_or_obj!r} report differing "
                f"{'/'.join(heterogeneous)} values. CF decoding cannot apply "
                "per-band values to a single band_data variable, so those "
                "values are left raw, with the per-band metadata available "
                "in the band_nodata/band_scale_factor/band_add_offset "
                "coordinates. Open with band_as_dim=False to decode each "
                "band independently.",
                UserWarning,
                stacklevel=4,
            )

        da_obj = xr.DataArray(
            data,
            dims=["band", "y", "x"],
            coords=coords,
            attrs=attrs,
            name="band_data",
        )
        da_obj.encoding["preferred_chunks"] = preferred_chunks

        if drop_variables and "band_data" in drop_variables:
            return xr.Dataset()

        ds = xr.Dataset({"band_data": da_obj})
        ds = xr.decode_cf(ds, decode_times=False, mask_and_scale=mask_and_scale)
        return ds

    def _raster_as_vars(
        self,
        filename_or_obj,
        dataset,
        drop_variables,
        num_bands,
        mask_and_scale=True,
        config_options=None,
    ):
        """Each band as a separate (y, x) data variable."""
        data_vars = {}

        for band_idx in range(1, num_bands + 1):
            band = dataset.GetRasterBand(band_idx)
            band_name = band.GetDescription() or f"band_{band_idx}"

            if drop_variables and band_name in drop_variables:
                continue

            backend_array = GDALBackendArray(
                filename_or_obj, band_idx, config_options=config_options
            )
            logger.debug("band: %i", band_idx)
            data = indexing.LazilyIndexedArray(backend_array)

            block_size = band.GetBlockSize()  # [x, y]
            preferred_chunks = {
                "y": block_size[1] if block_size[1] > 0 else dataset.RasterYSize,
                "x": block_size[0] if block_size[0] > 0 else dataset.RasterXSize,
            }

            # Only record CF attributes that carry information: GDAL's
            # Python bindings return 1.0/0.0 for unset scale/offset, and a
            # spurious identity scale_factor makes decode_cf promote every
            # variable to float64 even for plain unscaled imagery.
            band_attrs = {}
            nodata = band.GetNoDataValue()
            if nodata is not None:
                band_attrs["_FillValue"] = nodata
            scale = band.GetScale()
            if scale is not None and scale != 1.0:
                band_attrs["scale_factor"] = scale
            offset = band.GetOffset()
            if offset is not None and offset != 0.0:
                band_attrs["add_offset"] = offset

            data_vars[band_name] = xr.DataArray(data, dims=["y", "x"], attrs=band_attrs)
            data_vars[band_name].encoding["preferred_chunks"] = preferred_chunks

        ds = xr.Dataset(data_vars)
        ds = xr.decode_cf(ds, decode_times=False, mask_and_scale=mask_and_scale)
        return ds

    # ------------------------------------------------------------------
    # Multidim path
    # ------------------------------------------------------------------

    def _open_multidim(
        self, filename_or_obj, group, drop_variables, mask_and_scale=True, config_options=None
    ):
        """Open using GDAL's multidimensional API."""
        dataset = gdal.OpenEx(filename_or_obj, gdal.OF_MULTIDIM_RASTER | gdal.GA_ReadOnly)
        if dataset is None:
            raise ValueError(f"Could not open {filename_or_obj} with GDAL multidim API")

        root_group = dataset.GetRootGroup()
        if root_group is None:
            raise ValueError(f"No root group found in {filename_or_obj}")

        driver_name = dataset.GetDriver().GetDescription() if dataset.GetDriver() else None

        # Navigate to the requested group; handle None, "", "/", "a/b/c", "/a/b/c"
        parts = [p for p in (group or "").strip("/").split("/") if p]
        target_group = root_group
        for part in parts:
            target_group = target_group.OpenGroup(part)
            if target_group is None:
                raise ValueError(f"Group component {part!r} not found in path {group!r}")

        array_names = target_group.GetMDArrayNames()
        array_names += _hidden_geometry_containers(target_group, array_names)

        data_vars = {}
        coords = {}
        time_encoding = {}  # coord name -> CF encoding of a decoded time axis
        srs_wkt = {}  # data var name -> WKT of the array's spatial reference
        dim_axis = {}  # dim name -> CF axis from GDAL's dimension type
        synthesized = set()  # dims given a placeholder np.arange index
        geometry_dims = set()  # dims a CF geometry container will decode into
        has_geometry = False  # any CF geometry container present

        for array_name in array_names:
            if drop_variables and array_name in drop_variables:
                continue

            try:
                mdarray = target_group.OpenMDArray(array_name)
            except RuntimeError as e:
                # e.g. unsupported codec (numcodecs.pcodec) - skip rather than abort
                logger.warning("Skipping array %r: %s", array_name, e)
                continue
            if mdarray is None:
                continue

            dims = mdarray.GetDimensions()
            dim_names = [dim.GetName() or f"dim_{i}" for i, dim in enumerate(dims)]
            for dim, dim_name in zip(dims, dim_names, strict=False):
                axis = _DIM_TYPE_AXIS.get(dim.GetType() or "")
                if axis:
                    dim_axis.setdefault(dim_name, axis)

            # Backend array holds refs to parent dataset/group so the mdarray
            # stays valid for the lifetime of the resulting xarray Dataset.
            backend_array = GDALMultiDimArray(
                mdarray,
                filename=filename_or_obj,
                _parent_dataset=dataset,
                _parent_group=target_group,
                config_options=config_options,
            )
            data = indexing.LazilyIndexedArray(backend_array)

            # Native block sizes for chunks={}; consumed by
            # _maybe_chunk_dataset. Zero entries mean "whole dimension".
            block_size = mdarray.GetBlockSize()
            preferred_chunks = {
                dim_names[i]: (int(b) if b > 0 else dims[i].GetSize())
                for i, b in enumerate(block_size)
            }

            attrs = {}
            for attr in mdarray.GetAttributes():
                attr_name = attr.GetName()
                attr_value = attr.Read()
                if attr_value is not None:
                    attrs[attr_name] = attr_value

            # Synthesise CF attributes from GDAL's direct accessors. The NetCDF
            # multidim driver exposes these via GetScale()/GetOffset()/GetNoDataValueAsDouble()
            # rather than as CF-named attribute entries, so we surface them
            # under the names xarray's CF decoder expects.
            scale = mdarray.GetScale()
            if scale is not None and scale != 1.0:
                attrs.setdefault("scale_factor", scale)

            offset = mdarray.GetOffset()
            if offset is not None and offset != 0.0:
                attrs.setdefault("add_offset", offset)

            nodata = mdarray.GetNoDataValueAsDouble()
            if nodata is not None:
                attrs.setdefault("_FillValue", nodata)

            is_coord = any(dim.GetName() == array_name for dim in dims)

            if is_coord and len(dim_names) == 1:
                # Load eagerly for index variables; go through the raw
                # method (the adapter __getitem__ expects xarray
                # ExplicitIndexer keys, not raw slices)
                coord_data = backend_array._raw_indexing_method(
                    tuple(slice(None) for _ in backend_array.shape)
                )
                units = mdarray.GetUnit()
                if _is_time_coord(array_name, attrs, units) and units and " since " in units:
                    calendar = attrs.pop("calendar", "standard")
                    raw_dtype = coord_data.dtype
                    coord_data = decode_cf_datetime(coord_data, units, calendar)
                    # Keep the CF encoding, as xarray's own decoder does, so
                    # writers can reproduce the stored values exactly.
                    time_encoding[array_name] = {
                        "units": units,
                        "calendar": calendar,
                        "dtype": raw_dtype,
                    }
                elif units:
                    attrs.setdefault("units", units)
                coords[array_name] = xr.DataArray(coord_data, dims=dim_names, attrs=attrs)
            else:
                unit = mdarray.GetUnit()
                if unit:
                    attrs.setdefault("units", unit)
                srs = mdarray.GetSpatialRef()
                if srs is not None:
                    srs_wkt[array_name] = srs.ExportToWkt(["FORMAT=WKT2_2019"])
                data_vars[array_name] = xr.DataArray(data, dims=dim_names, attrs=attrs)
                data_vars[array_name].encoding["preferred_chunks"] = preferred_chunks
                # Create simple index coordinate for any dimension without one
                for dim, dim_name in zip(dims, dim_names, strict=False):
                    if dim_name not in coords and dim_name not in data_vars:
                        coords[dim_name] = np.arange(dim.GetSize())
                        synthesized.add(dim_name)
                if len(dim_names) == 0 and "geometry_type" in attrs:
                    has_geometry = True
                    if attrs.get("variable_name"):
                        geometry_dims.add(attrs["variable_name"])

        # A placeholder 0..n index on a geometry dimension would block
        # cf_xarray from decoding the container into that coordinate (issue
        # #11), so leave such dimensions without an index.
        for dim_name in geometry_dims & synthesized:
            del coords[dim_name]

        # GDAL folds CF grid_mapping variables into the SRS and strips the
        # grid_mapping attribute, which loses the CRS xvec reads when it
        # decodes a geometry container. Restore that link for containers only.
        if has_geometry:
            _restore_geometry_grid_mapping(target_group, array_names, data_vars, coords)

        # Group-level attributes
        group_attrs = {}
        for attr in target_group.GetAttributes():
            attr_name = attr.GetName()
            attr_value = attr.Read()
            if attr_value is not None:
                group_attrs[attr_name] = attr_value

        ds = xr.Dataset(data_vars, coords=coords, attrs=group_attrs)

        # Serialization-safe provenance hints (strings only). Live GDAL refs
        # are held inside the backend arrays themselves - no live objects in
        # encoding (where to_netcdf() etc. would trip over them).
        ds.encoding["source"] = filename_or_obj
        if driver_name:
            ds.encoding["gdal_driver"] = driver_name

        ds = xr.decode_cf(ds, decode_times=False, mask_and_scale=mask_and_scale)

        for name, enc in time_encoding.items():
            if name in ds.variables:
                ds[name].encoding.update(enc)
        for dim_name, axis in dim_axis.items():
            if dim_name in ds.coords and dim_name not in synthesized:
                ds[dim_name].attrs.setdefault("axis", axis)
        ds = _attach_mdim_crs(ds, srs_wkt)
        return ds

    # ------------------------------------------------------------------
    # Engine discovery
    # ------------------------------------------------------------------

    def guess_can_open(self, filename_or_obj):
        """Conservative heuristic for xarray's engine auto-discovery."""
        if not isinstance(filename_or_obj, str):
            return False
        try:
            ds = gdal.Open(_expand_tilde(filename_or_obj), gdal.GA_ReadOnly)
        except Exception:
            return False
        if ds is None:
            return False
        ds = None
        return True
