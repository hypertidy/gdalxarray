"""GDAL backend for xarray.

Three BackendArray classes wrap GDAL's three reading modes:

* ``GDALBackendArray``   - a single band of a classic raster
* ``GDALMultiBandArray`` - all bands of a classic raster as a (band, y, x) cube
* ``GDALMultiDimArray``  - a multidimensional array via GDAL's multidim API

``GDALBackendEntrypoint`` is registered as the ``gdalxarray`` xarray engine and
dispatches to one of two open paths depending on the ``multidim`` flag.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Hashable, Iterable

import dask.array as da
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


def _is_time_coord(array_name: str, attrs: dict, units: str | None) -> bool:
    """Recognise a time coordinate via CF conventions or a 'units since' string."""
    if attrs.get("axis") == "T":
        return True
    if attrs.get("standard_name") == "time":
        return True
    if array_name.lower() == "time":
        return True
    return bool(units and " since " in units)


# ---------------------------------------------------------------------------
# Classic-raster, single band
# ---------------------------------------------------------------------------


class GDALBackendArray(BackendArray):
    """Wrap one GDAL raster band as an xarray BackendArray of shape (y, x)."""

    def __init__(self, filename: str, band_index: int = 1):
        self.filename = filename
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

    def __dask_tokenize__(self):
        return (type(self).__name__, self.filename, self.band_index)

    @staticmethod
    def _gdal_to_numpy_dtype(gdal_dtype):
        dtype_map = {
            gdal.GDT_Byte: np.uint8,
            gdal.GDT_UInt16: np.uint16,
            gdal.GDT_Int16: np.int16,
            gdal.GDT_UInt32: np.uint32,
            gdal.GDT_Int32: np.int32,
            gdal.GDT_Float32: np.float32,
            gdal.GDT_Float64: np.float64,
            gdal.GDT_CInt16: np.complex64,
            gdal.GDT_CInt32: np.complex64,
            gdal.GDT_CFloat32: np.complex64,
            gdal.GDT_CFloat64: np.complex128,
        }
        return dtype_map.get(gdal_dtype, np.float32)

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
        logger.debug("backend __getitem__ key type=%s key=%r", type(key).__name__, key)

        from xarray.core import indexing as xr_indexing

        if isinstance(
            key, (xr_indexing.BasicIndexer, xr_indexing.OuterIndexer, xr_indexing.VectorizedIndexer)
        ):
            key = key.tuple

        if isinstance(key, tuple):
            return self._raw_indexing_method(key)
        else:
            return self._raw_indexing_method((key,))

    def _raw_indexing_method(self, key):
        """Read data from GDAL using basic indexing."""
        if not isinstance(key, tuple):
            key = (key,)

        # Pad to 2D
        if len(key) < 2:
            key = key + (slice(None),) * (2 - len(key))
        if len(key) > 2:
            raise IndexError(f"Expected at most 2D index, got {len(key)}D")

        y_idx, x_idx = key
        squeeze_y = squeeze_x = False

        # y indexer
        if isinstance(y_idx, (int, np.integer)):
            y_start, y_size = int(y_idx), 1
            squeeze_y = True
        elif isinstance(y_idx, slice):
            y_start = y_idx.start if y_idx.start is not None else 0
            y_stop = y_idx.stop if y_idx.stop is not None else self.shape[0]
            y_step = y_idx.step if y_idx.step is not None else 1
            if y_step > 0 and y_stop < y_start:
                y_start, y_stop = y_stop, y_start + 1
            elif y_step < 0:
                y_start, y_stop, y_step = y_stop + 1, y_start + 1, -y_step
            y_size = y_stop - y_start
        else:
            raise IndexError(f"Unsupported y index type: {type(y_idx)}")

        # x indexer
        if isinstance(x_idx, (int, np.integer)):
            x_start, x_size = int(x_idx), 1
            squeeze_x = True
        elif isinstance(x_idx, slice):
            x_start = x_idx.start if x_idx.start is not None else 0
            x_stop = x_idx.stop if x_idx.stop is not None else self.shape[1]
            x_step = x_idx.step if x_idx.step is not None else 1
            if x_step > 0 and x_stop < x_start:
                x_start, x_stop = x_stop, x_start + 1
            elif x_step < 0:
                x_start, x_stop, x_step = x_stop + 1, x_start + 1, -x_step
            x_size = x_stop - x_start
        else:
            raise IndexError(f"Unsupported x index type: {type(x_idx)}")

        # Zero-sized slice (Dask uses these for _meta)
        if y_size == 0 or x_size == 0:
            shape = []
            if not squeeze_y:
                shape.append(y_size)
            if not squeeze_x:
                shape.append(x_size)
            return np.empty(shape, dtype=self._dtype)

        band = self._get_band()
        logger.debug("read: yoff=%s xoff=%s ysize=%s xsize=%s", y_start, x_start, y_size, x_size)
        data = band.ReadAsArray(
            xoff=x_start,
            yoff=y_start,
            win_xsize=x_size,
            win_ysize=y_size,
        )

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

    def __init__(self, filename: str, band_indices: list[int] | None = None):
        self.filename = filename
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

    def __dask_tokenize__(self):
        return (type(self).__name__, self.filename, tuple(self.band_indices))

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
        logger.debug("multiband __getitem__ key type=%s key=%r", type(key).__name__, key)

        from xarray.core import indexing as xr_indexing

        if isinstance(
            key, (xr_indexing.BasicIndexer, xr_indexing.OuterIndexer, xr_indexing.VectorizedIndexer)
        ):
            key = key.tuple

        if isinstance(key, tuple):
            return self._raw_indexing_method(key)
        else:
            return self._raw_indexing_method((key,))

    def _raw_indexing_method(self, key):
        """Read via ``dataset.ReadAsArray`` with an explicit band_list."""
        if len(key) < 3:
            key = key + (slice(None),) * (3 - len(key))
        if len(key) > 3:
            raise IndexError(f"Expected at most 3D index, got {len(key)}D")

        b_idx, y_idx, x_idx = key
        squeeze_b = squeeze_y = squeeze_x = False

        # band indexer -> list of 1-based GDAL band numbers
        if isinstance(b_idx, (int, np.integer)):
            band_list = [self.band_indices[int(b_idx)]]
            squeeze_b = True
        elif isinstance(b_idx, slice):
            b_start = b_idx.start if b_idx.start is not None else 0
            b_stop = b_idx.stop if b_idx.stop is not None else self._shape[0]
            b_step = b_idx.step if b_idx.step is not None else 1
            if b_step > 0 and b_stop < b_start:
                b_start, b_stop = b_stop, b_start + 1
            elif b_step < 0:
                b_start, b_stop, b_step = b_stop + 1, b_start + 1, -b_step
            band_list = [self.band_indices[i] for i in range(b_start, b_stop, b_step)]
        elif isinstance(b_idx, (list, np.ndarray)):
            band_list = [self.band_indices[int(i)] for i in b_idx]
        else:
            raise IndexError(f"Unsupported band index type: {type(b_idx)}")

        # y indexer
        if isinstance(y_idx, (int, np.integer)):
            y_start, y_size = int(y_idx), 1
            squeeze_y = True
        elif isinstance(y_idx, slice):
            y_start = y_idx.start if y_idx.start is not None else 0
            y_stop = y_idx.stop if y_idx.stop is not None else self._shape[1]
            y_step = y_idx.step if y_idx.step is not None else 1
            if y_step > 0 and y_stop < y_start:
                y_start, y_stop = y_stop, y_start + 1
            elif y_step < 0:
                y_start, y_stop, y_step = y_stop + 1, y_start + 1, -y_step
            y_size = y_stop - y_start
        else:
            raise IndexError(f"Unsupported y index type: {type(y_idx)}")

        # x indexer
        if isinstance(x_idx, (int, np.integer)):
            x_start, x_size = int(x_idx), 1
            squeeze_x = True
        elif isinstance(x_idx, slice):
            x_start = x_idx.start if x_idx.start is not None else 0
            x_stop = x_idx.stop if x_idx.stop is not None else self._shape[2]
            x_step = x_idx.step if x_idx.step is not None else 1
            if x_step > 0 and x_stop < x_start:
                x_start, x_stop = x_stop, x_start + 1
            elif x_step < 0:
                x_start, x_stop, x_step = x_stop + 1, x_start + 1, -x_step
            x_size = x_stop - x_start
        else:
            raise IndexError(f"Unsupported x index type: {type(x_idx)}")

        if y_size == 0 or x_size == 0 or len(band_list) == 0:
            shape = []
            if not squeeze_b:
                shape.append(len(band_list))
            if not squeeze_y:
                shape.append(y_size)
            if not squeeze_x:
                shape.append(x_size)
            return np.empty(shape, dtype=self._dtype)

        ds = self._get_dataset()
        logger.debug(
            "multiband read: bands=%s yoff=%s xoff=%s ysize=%s xsize=%s",
            band_list,
            y_start,
            x_start,
            y_size,
            x_size,
        )

        data = ds.ReadAsArray(
            xoff=x_start,
            yoff=y_start,
            xsize=x_size,
            ysize=y_size,
            band_list=band_list,
        )

        # ReadAsArray returns 2D for a single band, 3D for multiple - normalise
        if data.ndim == 2:
            data = data[np.newaxis, :, :]

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

    Holds optional references to the parent dataset and group so that the
    underlying ``mdarray`` remains valid even after the open-time variables
    drop out of scope (a multidim MDArray is a view into its parent).
    """

    def __init__(self, mdarray, _parent_dataset=None, _parent_group=None):
        self.mdarray = mdarray
        # Keep parent objects alive - without these refs the mdarray can be
        # invalidated when the user-facing ds goes out of scope mid-session.
        self._parent_dataset = _parent_dataset
        self._parent_group = _parent_group

        dims = mdarray.GetDimensions()
        self._shape = tuple(dim.GetSize() for dim in dims)

        gdal_dtype = mdarray.GetDataType().GetNumericDataType()
        self._dtype = GDALBackendArray._gdal_to_numpy_dtype(gdal_dtype)
        self._chunks = tuple(mdarray.GetBlockSize())

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return np.dtype(self._dtype)

    def __dask_tokenize__(self):
        return (type(self).__name__, id(self.mdarray))

    def __getitem__(self, key):
        logger.debug("multidim __getitem__ key type=%s key=%r", type(key).__name__, key)

        from xarray.core import indexing as xr_indexing

        if isinstance(
            key, (xr_indexing.BasicIndexer, xr_indexing.OuterIndexer, xr_indexing.VectorizedIndexer)
        ):
            key = key.tuple

        if not isinstance(key, tuple):
            key = (key,)
        return self._raw_indexing_method(key)

    def _raw_indexing_method(self, key):
        """Read data from GDAL multidim array."""
        if not isinstance(key, tuple):
            key = (key,)

        starts = []
        counts = []
        steps = []
        squeeze_dims = []

        for i, k in enumerate(key):
            if isinstance(k, slice):
                start = k.start if k.start is not None else 0
                stop = k.stop if k.stop is not None else self.shape[i]
                step = k.step if k.step is not None else 1
                if step > 0 and stop < start:
                    # xarray canonicalises reverse-slice on decreasing coords as
                    # slice(stop<start) - read forward, let xarray flip display
                    start, stop = stop, start + 1
                elif step < 0:
                    start, stop, step = stop + 1, start + 1, -step
                count = (stop - start + step - 1) // step
            elif isinstance(k, (int, float, np.integer, np.floating)):
                start = int(k)
                count = 1
                step = 1
                squeeze_dims.append(i)
            else:
                raise IndexError(f"Unsupported index type: {type(k)}")

            starts.append(start)
            counts.append(count)
            steps.append(step)

        # Zero-sized slice (Dask uses these for _meta inference)
        if any(c == 0 for c in counts):
            shape = [c for i, c in enumerate(counts) if i not in squeeze_dims]
            return np.empty(shape, dtype=self._dtype)

        # AdviseRead is a prefetch hint. Compute a reasonable CACHE_SIZE for it,
        # bounded above to avoid GDAL choking on absurd values when sharded
        # stores report a shard-sized block (see #issue-AdviseRead-cache).
        block = np.array(self.mdarray.GetBlockSize())
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
                self.mdarray.AdviseRead(
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

        data = self.mdarray.ReadAsArray(
            array_start_idx=starts,
            count=counts,
            array_step=steps,
        )

        # Squeeze out integer-indexed dimensions
        for dim_idx in reversed(squeeze_dims):
            data = np.squeeze(data, axis=dim_idx)

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
        filename_or_obj: str,
        *,
        drop_variables: Iterable[Hashable] | None = None,
        chunks: dict[Hashable, int] | None = None,
        multidim: bool = True,
        group: str | None = None,
        band_as_dim: bool = True,
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
            an explicit mapping like ``{"y": 256, "x": 256}`` is honoured.
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
            bands carry semantically distinct quantities.
        """
        if multidim:
            return self._open_multidim(filename_or_obj, chunks, group, drop_variables)
        return self._open_raster(filename_or_obj, chunks, drop_variables, band_as_dim=band_as_dim)

    # ------------------------------------------------------------------
    # Classic-raster path
    # ------------------------------------------------------------------

    def _open_raster(self, filename_or_obj, chunks, drop_variables, band_as_dim=True):
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
                filename_or_obj, dataset, chunks, drop_variables, num_bands
            )
        else:
            ds = self._raster_as_vars(filename_or_obj, dataset, chunks, drop_variables, num_bands)

        # Spatial coords and CRS for both layouts
        ds = ds.assign_coords(xr.Coordinates.from_xindex(index))
        if len(projection) > 0:
            ds = ds.proj.assign_crs(crs=projection)

        # Serialization-safe provenance hints in encoding (strings only).
        ds.encoding["source"] = filename_or_obj
        if driver_name:
            ds.encoding["gdal_driver"] = driver_name
        return ds

    def _raster_as_band_dim(self, filename_or_obj, dataset, chunks, drop_variables, num_bands):
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

        backend_array = GDALMultiBandArray(filename_or_obj, band_indices)

        if chunks is not None:
            if chunks == {}:
                block_size = dataset.GetRasterBand(1).GetBlockSize()  # [x, y]
                y_chunk = block_size[1] if block_size[1] > 0 else dataset.RasterYSize
                x_chunk = block_size[0] if block_size[0] > 0 else dataset.RasterXSize
                chunk_tuple = (1, y_chunk, x_chunk)
                logger.debug("multiband chunks (band, y, x)=%s", chunk_tuple)
            else:
                chunk_tuple = (
                    chunks.get("band", 1),
                    chunks.get("y", -1),
                    chunks.get("x", -1),
                )
            data = da.from_array(
                backend_array,
                chunks=chunk_tuple,
                name=f"gdal-multiband-{filename_or_obj}",
                asarray=False,
            )
        else:
            data = indexing.LazilyIndexedArray(backend_array)

        attrs = {"descriptions": descriptions}
        coords = {"band": np.array(band_indices)}

        nodatas_set = set(n for n in nodatas if n is not None)
        if len(nodatas_set) == 1 and all(n is not None for n in nodatas):
            attrs["nodata"] = nodatas[0]
        elif any(n is not None for n in nodatas):
            coords["nodata"] = (
                "band",
                np.array([n if n is not None else np.nan for n in nodatas]),
            )

        if all(s == 1.0 for s in scales):
            pass
        elif len(set(scales)) == 1:
            attrs["scale"] = scales[0]
        else:
            coords["scale"] = ("band", np.array(scales))

        if all(o == 0.0 for o in offsets):
            pass
        elif len(set(offsets)) == 1:
            attrs["offset"] = offsets[0]
        else:
            coords["offset"] = ("band", np.array(offsets))

        da_obj = xr.DataArray(
            data,
            dims=["band", "y", "x"],
            coords=coords,
            attrs=attrs,
            name="band_data",
        )

        if drop_variables and "band_data" in drop_variables:
            return xr.Dataset()
        return xr.Dataset({"band_data": da_obj})

    def _raster_as_vars(self, filename_or_obj, dataset, chunks, drop_variables, num_bands):
        """Each band as a separate (y, x) data variable."""
        data_vars = {}

        for band_idx in range(1, num_bands + 1):
            band = dataset.GetRasterBand(band_idx)
            band_name = band.GetDescription() or f"band_{band_idx}"

            if drop_variables and band_name in drop_variables:
                continue

            backend_array = GDALBackendArray(filename_or_obj, band_idx)
            logger.debug("band: %i", band_idx)

            if chunks is not None:
                if chunks == {}:
                    block_size = dataset.GetRasterBand(1).GetBlockSize()  # [x, y]
                    y_chunk = block_size[1] if block_size[1] > 0 else dataset.RasterYSize
                    x_chunk = block_size[0] if block_size[0] > 0 else dataset.RasterXSize
                    chunk_tuple = (y_chunk, x_chunk)
                    logger.debug("shape=(y,x), chunks=(y,x)=%s", chunk_tuple)
                else:
                    chunk_tuple = (chunks.get("y", -1), chunks.get("x", -1))
                data = da.from_array(
                    backend_array,
                    chunks=chunk_tuple,
                    name=f"gdal-{filename_or_obj}-{band_name}",
                    asarray=False,
                )
            else:
                data = indexing.LazilyIndexedArray(backend_array)

            band_attrs = {
                "nodata": band.GetNoDataValue(),
                "scale": band.GetScale() or 1.0,
                "offset": band.GetOffset() or 0.0,
            }
            band_attrs = {k: v for k, v in band_attrs.items() if v is not None}

            data_vars[band_name] = xr.DataArray(data, dims=["y", "x"], attrs=band_attrs)

        return xr.Dataset(data_vars)

    # ------------------------------------------------------------------
    # Multidim path
    # ------------------------------------------------------------------

    def _open_multidim(self, filename_or_obj, chunks, group, drop_variables):
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

        data_vars = {}
        coords = {}

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

            # Backend array holds refs to parent dataset/group so the mdarray
            # stays valid for the lifetime of the resulting xarray Dataset.
            backend_array = GDALMultiDimArray(
                mdarray, _parent_dataset=dataset, _parent_group=target_group
            )

            if chunks is not None:
                if chunks == {}:
                    block_size = mdarray.GetBlockSize()
                    dim_sizes = [dim.GetSize() for dim in dims]
                    chunk_tuple = tuple(
                        b if b > 0 else dim_sizes[i] for i, b in enumerate(block_size)
                    )
                else:
                    chunk_tuple = tuple(chunks.get(dim_name, -1) for dim_name in dim_names)
                data = da.from_array(
                    backend_array,
                    chunks=chunk_tuple,
                    name=f"gdal-multidim-{filename_or_obj}-{array_name}",
                    asarray=False,
                )
            else:
                data = indexing.LazilyIndexedArray(backend_array)

            attrs = {}
            for attr in mdarray.GetAttributes():
                attr_name = attr.GetName()
                attr_value = attr.Read()
                if attr_value is not None:
                    attrs[attr_name] = attr_value

            is_coord = any(dim.GetName() == array_name for dim in dims)

            if is_coord and len(dim_names) == 1:
                # Load eagerly for index variables
                coord_data = backend_array[:]
                units = mdarray.GetUnit()
                if _is_time_coord(array_name, attrs, units):
                    calendar = attrs.get("calendar", "standard")
                    if units:
                        coord_data = decode_cf_datetime(coord_data, units, calendar)
                coords[array_name] = xr.DataArray(coord_data, dims=dim_names, attrs=attrs)
            else:
                data_vars[array_name] = xr.DataArray(data, dims=dim_names, attrs=attrs)
                # Create simple index coordinate for any dimension without one
                for dim, dim_name in zip(dims, dim_names, strict=False):
                    if dim_name not in coords and dim_name not in data_vars:
                        coords[dim_name] = np.arange(dim.GetSize())

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
        return ds

    # ------------------------------------------------------------------
    # Engine discovery
    # ------------------------------------------------------------------

    def guess_can_open(self, filename_or_obj):
        """Conservative heuristic for xarray's engine auto-discovery."""
        if not isinstance(filename_or_obj, str):
            return False
        try:
            ds = gdal.Open(filename_or_obj, gdal.GA_ReadOnly)
        except Exception:
            return False
        if ds is None:
            return False
        ds = None
        return True
