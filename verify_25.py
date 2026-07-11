"""Verify #25: GDAL config options applied thread-locally around the
open and around every read, including per-thread reopens and reads
after unpickling; included in dask tokens; never leaked.

The stub gdal.config_options records the active option dict in a
thread-local; every fake GDAL open/read snapshots what was active on
its thread at call time.
"""
import pickle
import sys
import threading
import types
import numpy as np

osgeo = types.ModuleType("osgeo")
gdal = types.ModuleType("osgeo.gdal")
for i, name in enumerate(
    ["GDT_Byte", "GDT_UInt16", "GDT_Int16", "GDT_UInt32", "GDT_Int32",
     "GDT_Float32", "GDT_Float64", "GDT_CInt16", "GDT_CInt32",
     "GDT_CFloat32", "GDT_CFloat64", "GA_ReadOnly", "OF_MULTIDIM_RASTER"]
):
    setattr(gdal, name, i)
gdal.GetUseExceptions = lambda: True
gdal.UseExceptions = lambda: None

_ACTIVE = threading.local()


def active():
    return dict(getattr(_ACTIVE, "opts", {}))


class _Ctx:
    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        self.prev = active()
        _ACTIVE.opts = {**self.prev, **self.opts}

    def __exit__(self, *exc):
        _ACTIVE.opts = self.prev


def config_options(opts, thread_local=True):
    assert thread_local, "backend must request thread-local scoping"
    return _Ctx(opts)


gdal.config_options = config_options

SEEN = []  # (event, thread_id, active_options)
_SEEN_LOCK = threading.Lock()


def see(event):
    with _SEEN_LOCK:
        SEEN.append((event, threading.get_ident(), active()))


class FakeDim:
    def __init__(self, n):
        self.n = n

    def GetSize(self):
        return self.n


class FakeDataType:
    def GetNumericDataType(self):
        return gdal.GDT_Float64


class FakeMDArray:
    def __init__(self, data):
        self.data = data

    def GetFullName(self):
        return "/v"

    def GetDimensions(self):
        return [FakeDim(n) for n in self.data.shape]

    def GetDataType(self):
        return FakeDataType()

    def GetBlockSize(self):
        return [1, self.data.shape[1]]

    def AdviseRead(self, **k):
        pass

    def ReadAsArray(self, array_start_idx, count, array_step):
        see("md_read")
        ix = tuple(slice(s, s + c * st, st) for s, c, st in
                   zip(array_start_idx, count, array_step))
        return self.data[ix].copy()


class _Root:
    def __init__(self, data):
        self.data = data

    def OpenMDArrayFromFullname(self, fullname):
        see("md_reopen")
        return FakeMDArray(self.data)


class _MDDs:
    def __init__(self, data):
        self.data = data

    def GetRootGroup(self):
        return _Root(self.data)


MDIM = {}
gdal.OpenEx = lambda fn, *a, **k: (
    see("openex"), _MDDs(MDIM[fn]) if fn in MDIM else None
)[1]


class FakeBand:
    def __init__(self, data):
        self.data = data
        self.DataType = gdal.GDT_Float64

    def GetBlockSize(self):
        return [self.data.shape[1], 1]

    def ReadAsArray(self, xoff, yoff, win_xsize, win_ysize):
        see("band_read")
        return self.data[yoff:yoff + win_ysize,
                         xoff:xoff + win_xsize].copy()


class FakeClassicDataset:
    def __init__(self, data):
        self.data = data
        self.RasterYSize, self.RasterXSize = data.shape
        self.RasterCount = 1

    def GetRasterBand(self, i):
        b = FakeBand(self.data)
        b.GetDescription = lambda: "b1"
        b.GetNoDataValue = lambda: None
        b.GetScale = lambda: 1.0
        b.GetOffset = lambda: 0.0
        return b

    def GetGeoTransform(self):
        return (-0.5, 1.0, 0.0, float(self.RasterYSize) + 0.5, 0.0, -1.0)

    def GetProjection(self):
        return ""

    def GetDriver(self):
        return None

    def GetSubDatasets(self):
        return []

    def ReadAsArray(self, xoff, yoff, xsize, ysize, band_list):
        see("band_read")
        out = self.data[yoff:yoff + ysize, xoff:xoff + xsize].copy()
        return out  # single band: 2D, normalised by the backend


CLASSIC = {}
gdal.Open = lambda fn, *a, **k: (
    see("open"), CLASSIC.get(fn)
)[1]

osgeo.gdal = gdal
sys.modules["osgeo"] = osgeo
sys.modules["osgeo.gdal"] = gdal

sys.path.insert(0, "/home/claude/gxcheck")
from gdalxarray import backend  # noqa: E402
import xarray as xr  # noqa: E402
from xarray.core import indexing as xri  # noqa: E402

DATA = np.arange(64, dtype=np.float64).reshape(16, 4)
MDIM["s.nc"] = DATA
CFG = {"GDAL_HTTP_MAX_RETRY": "3", "SOME_INT": 7}
CFG_STR = {"GDAL_HTTP_MAX_RETRY": "3", "SOME_INT": "7"}

results = []


def check(name, fn):
    SEEN.clear()
    try:
        fn()
        results.append(("PASS", name))
    except Exception as e:
        results.append(("FAIL", f"{name}: {type(e).__name__}: {e}"))


def make_md(cfg=CFG):
    seed = FakeMDArray(DATA)
    return backend.GDALMultiDimArray(seed, filename="s.nc",
                                     config_options=cfg)


def t_reads_see_options_across_threads():
    md = make_md()
    var = xr.DataArray(xri.LazilyIndexedArray(md), dims=("t", "x"))
    var.encoding["preferred_chunks"] = {"t": 1, "x": 4}
    ds = xr.Dataset({"v": var})
    out = backend.GDALBackendEntrypoint._maybe_chunk_dataset(ds, {}, "s.nc")
    out["v"].mean().compute(scheduler="threads", num_workers=6)
    reads = [s for s in SEEN if s[0] == "md_read"]
    assert len(reads) >= 16
    assert all(opts == CFG_STR for _, _, opts in reads), reads[:2]
    threads = {t for _, t, _ in reads}
    assert len(threads) >= 2, "expected reads across multiple threads"
    reopens = [s for s in SEEN if s[0] in ("openex", "md_reopen")]
    assert reopens and all(o == CFG_STR for _, _, o in reopens)


def t_no_leak_after_reads():
    md = make_md()
    md._raw_indexing_method((slice(0, 2), slice(None)))
    assert active() == {}, "options leaked on the calling thread"


def t_no_config_default_path():
    md = make_md(cfg=None)
    md._raw_indexing_method((slice(0, 2), slice(None)))
    reads = [s for s in SEEN if s[0] == "md_read"]
    assert reads and all(o == {} for _, _, o in reads)


def t_pickle_carries_options():
    md2 = pickle.loads(pickle.dumps(make_md()))
    md2._raw_indexing_method((slice(0, 2), slice(None)))
    reads = [s for s in SEEN if s[0] == "md_read"]
    assert reads and all(o == CFG_STR for _, _, o in reads)


def t_tokenize_includes_options():
    a = make_md().__dask_tokenize__()
    b = make_md().__dask_tokenize__()
    c = make_md(cfg={"OTHER": "1"}).__dask_tokenize__()
    d = make_md(cfg=None).__dask_tokenize__()
    assert a == b and a != c and a != d


def t_collection_token_includes_options():
    # tokenize of the MANAGED collection: the explicit layer name
    # short-circuits the array's __dask_tokenize__, so the layer token
    # itself must differ when config options differ (repo-test parity)
    from dask.base import tokenize

    def coll(cfg):
        md = make_md(cfg)
        var = xr.DataArray(xri.LazilyIndexedArray(md), dims=("t", "x"))
        var.encoding["preferred_chunks"] = {"t": 4, "x": 4}
        ds = xr.Dataset({"v": var})
        out = backend.GDALBackendEntrypoint._maybe_chunk_dataset(
            ds, {}, "s.nc", cfg
        )
        return out["v"].data

    a, b, c = coll(CFG), coll(CFG), coll({"OTHER": "1"})
    assert tokenize(a) == tokenize(b)
    assert tokenize(a) != tokenize(c), "layer token must include options"
    assert a.name != c.name


def t_open_dataset_end_to_end_classic():
    CLASSIC["c.tif"] = FakeClassicDataset(DATA)
    ep = backend.GDALBackendEntrypoint()
    ds = ep.open_dataset("c.tif", multidim=False, config_options=CFG)
    opens = [s for s in SEEN if s[0] == "open"]
    assert opens and all(o == CFG_STR for _, _, o in opens), opens
    ds["band_data"].values if "band_data" in ds else next(
        iter(ds.data_vars.values())
    ).values
    reads = [s for s in SEEN if s[0] == "band_read"]
    assert reads and all(o == CFG_STR for _, _, o in reads)
    assert active() == {}


check("dask threaded reads + per-thread reopens see options",
      t_reads_see_options_across_threads)
check("options never leak outside the context", t_no_leak_after_reads)
check("no-config path stays a no-op", t_no_config_default_path)
check("pickle roundtrip carries options to reads", t_pickle_carries_options)
check("tokenize includes config options", t_tokenize_includes_options)
check("managed collection token includes options",
      t_collection_token_includes_options)
check("open_dataset end-to-end: open and classic reads under options",
      t_open_dataset_end_to_end_classic)

fails = 0
for status, name in results:
    print(f"{status}  {name}")
    fails += status == "FAIL"
print(f"\n{len(results) - fails}/{len(results)} passed")
sys.exit(1 if fails else 0)
