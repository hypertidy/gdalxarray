import xarray as xr
from xarray.backends import BackendEntrypoint, BackendArray
from xarray.core import indexing
from xarray.coding.times import decode_cf_datetime

from collections.abc import Hashable, Iterable

import numpy as np
import dask.array as da
from osgeo import gdal
gdal.UseExceptions()
from typing import Iterable, Optional, Any
import threading

##https://gist.github.com/mdsumner/911c181467abb2c91d08544a94d8510a
from affine import Affine
from rasterix import RasterIndex
#https://xarray.dev/blog/flexible-indexing#xprojcrsindex
from xproj import CRSIndex

import logging
logger = logging.getLogger(__name__)

def _is_time_coord(array_name, attrs, units):
    """Check if this is a time coordinate using CF conventions."""
    if attrs.get('axis') == 'T':
        return True
    if attrs.get('standard_name') == 'time':
        return True
    if array_name.lower() == 'time':
        return True
    if units and ' since ' in units:
        return True
    return False

class GDALBackendArray(BackendArray):
    """Wrapper around GDAL dataset that implements xarray's BackendArray interface."""
    
    def __init__(self, filename, band_index=1):
        self.filename = filename
        self.band_index = band_index
        self._local = threading.local()
        logger.debug("filename: %s", filename)
        # Open once to get metadata, then close
        ds = gdal.Open(filename, gdal.GA_ReadOnly)
        if ds is None:
            raise ValueError(f"Could not open {filename}")
        band = ds.GetRasterBand(band_index)
        
        self._shape = (ds.RasterYSize, ds.RasterXSize)
        self._dtype = self._gdal_to_numpy_dtype(band.DataType)
        self._block_size = band.GetBlockSize()
        ds = None  # close
    
    def _get_band(self):
        if not hasattr(self._local, 'ds'):
            self._local.ds = gdal.Open(self.filename, gdal.GA_ReadOnly)
            self._local.band = self._local.ds.GetRasterBand(self.band_index)
        return self._local.band
    
    def __dask_tokenize__(self):
        # Fast unique identifier
        return (type(self).__name__, self.filename, self.band_index)
    
    @staticmethod
    def _gdal_to_numpy_dtype(gdal_dtype):
        """Convert GDAL data type to numpy dtype."""
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
        return np.prod(self._shape)

    def __getitem__(self, key):
        import logging
        logging.getLogger(__name__).debug("key type=%s key=%r", type(key).__name__, key)
        
        # Handle xarray's explicit indexing objects
        from xarray.core import indexing as xr_indexing
        
        if isinstance(key, xr_indexing.BasicIndexer):
            key = key.tuple
        elif isinstance(key, xr_indexing.OuterIndexer):
            key = key.tuple
        elif isinstance(key, xr_indexing.VectorizedIndexer):
            key = key.tuple
        
        # Handle direct array indexing
        if isinstance(key, tuple):
            return self._raw_indexing_method(key)
        else:
            return self._raw_indexing_method((key,))
      
    def _raw_indexing_method(self, key):
      """Read data from GDAL using basic indexing."""
      # Ensure we have a tuple
      if not isinstance(key, tuple):
          key = (key,)
      
      # Pad key with full slices if needed
      if len(key) < 2:
          key = key + (slice(None),) * (2 - len(key))
      
      if len(key) > 2:
          raise IndexError(f"Expected at most 2D index, got {len(key)}D")
      
      y_idx, x_idx = key
      
      # Convert integers and slices to window parameters
      if isinstance(y_idx, int):
          y_start, y_size = y_idx, 1
          squeeze_y = True
      elif isinstance(y_idx, slice):
          y_start = y_idx.start if y_idx.start is not None else 0
          y_stop = y_idx.stop if y_idx.stop is not None else self.shape[0]
          y_size = y_stop - y_start
          squeeze_y = False
      else:
          raise IndexError(f"Unsupported y index type: {type(y_idx)}")
      
      if isinstance(x_idx, int):
          x_start, x_size = x_idx, 1
          squeeze_x = True
      elif isinstance(x_idx, slice):
          x_start = x_idx.start if x_idx.start is not None else 0
          x_stop = x_idx.stop if x_idx.stop is not None else self.shape[1]
          x_size = x_stop - x_start
          squeeze_x = False
      else:
          raise IndexError(f"Unsupported x index type: {type(x_idx)}")
      
      # Handle zero-sized slices (Dask uses these for _meta)
      if y_size == 0 or x_size == 0:
          shape = []
          if not squeeze_y:
              shape.append(y_size)
          if not squeeze_x:
              shape.append(x_size)
          return np.empty(shape, dtype=self._dtype)
      
      band = self._get_band()
      logger.debug("read: yoff=%s, xoff=%s, ysize=%s, xsize=%s", 
         y_start, x_start, y_size, x_size)
      data = band.ReadAsArray(
          xoff=x_start,
          yoff=y_start,
          win_xsize=x_size,
          win_ysize=y_size
      )
      
      # Squeeze dimensions if we indexed with integers
      if squeeze_y and squeeze_x:
          return data[0, 0]
      elif squeeze_y:
          return data[0, :]
      elif squeeze_x:
          return data[:, 0]
      else:
          return data


class GDALMultiBandArray(BackendArray):
    """Wrapper exposing all bands of a GDAL dataset as a single 3D (band, y, x) array.
    
    Used by _open_raster when band_as_dim=True (the default), so that bands behave
    as an xarray dimension rather than separate data variables. Read efficiency is
    typically better than per-band reads because GDAL handles BIP/BIL/BSQ interleaving
    internally — multi-band selections on interleaved imagery only touch the source
    once per (y, x) window.
    """
    
    def __init__(self, filename, band_indices=None):
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
        
        # Assume all bands share dtype and y/x shape (the GDAL convention)
        band1 = ds.GetRasterBand(self.band_indices[0])
        self._shape = (len(self.band_indices), ds.RasterYSize, ds.RasterXSize)
        self._dtype = GDALBackendArray._gdal_to_numpy_dtype(band1.DataType)
        self._block_size = band1.GetBlockSize()  # [x, y]
        ds = None
    
    def _get_dataset(self):
        if not hasattr(self._local, 'ds'):
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
        logger.debug("multiband __getitem__ key type=%s key=%r",
                     type(key).__name__, key)
        
        from xarray.core import indexing as xr_indexing
        if isinstance(key, xr_indexing.BasicIndexer):
            key = key.tuple
        elif isinstance(key, xr_indexing.OuterIndexer):
            key = key.tuple
        elif isinstance(key, xr_indexing.VectorizedIndexer):
            key = key.tuple
        
        if isinstance(key, tuple):
            return self._raw_indexing_method(key)
        else:
            return self._raw_indexing_method((key,))
    
    def _raw_indexing_method(self, key):
        """Read data from GDAL via dataset.ReadAsArray with a band_list."""
        # Pad key with full slices if needed
        if len(key) < 3:
            key = key + (slice(None),) * (3 - len(key))
        if len(key) > 3:
            raise IndexError(f"Expected at most 3D index, got {len(key)}D")
        
        b_idx, y_idx, x_idx = key
        squeeze_b = squeeze_y = squeeze_x = False
        
        # Resolve band indexer to a list of 1-based GDAL band numbers.
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
            band_list = [self.band_indices[i]
                         for i in range(b_start, b_stop, b_step)]
        elif isinstance(b_idx, (list, np.ndarray)):
            band_list = [self.band_indices[int(i)] for i in b_idx]
        else:
            raise IndexError(f"Unsupported band index type: {type(b_idx)}")
        
        # Resolve y indexer
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
        
        # Resolve x indexer
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
        
        # Handle zero-sized slices (Dask uses these for _meta)
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
        logger.debug("multiband read: bands=%s yoff=%s xoff=%s ysize=%s xsize=%s",
                     band_list, y_start, x_start, y_size, x_size)
        
        data = ds.ReadAsArray(
            xoff=x_start,
            yoff=y_start,
            xsize=x_size,
            ysize=y_size,
            band_list=band_list,
        )
        
        # ReadAsArray returns 2D for a single band, 3D for multiple - normalise.
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        
        # Squeeze integer-indexed dims; do it in one pass to keep axis numbering sane.
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


class GDALMultiDimArray(BackendArray):
    """Wrapper around GDAL multidimensional array."""
    
    def __init__(self, mdarray):
        self.mdarray = mdarray
        
        # Get shape and dtype from multidim array
        dims = mdarray.GetDimensions()
        self._shape = tuple(dim.GetSize() for dim in dims)
        
        # Get numpy dtype
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
     import logging
     logging.getLogger(__name__).debug("__getitem__ key type=%s key=%r", type(key).__name__, key)
     
     # Handle xarray's explicit indexing objects
     from xarray.core import indexing as xr_indexing
    
     if isinstance(key, xr_indexing.BasicIndexer):
         key = key.tuple
     elif isinstance(key, xr_indexing.OuterIndexer):
         key = key.tuple
     elif isinstance(key, xr_indexing.VectorizedIndexer):
         key = key.tuple
    
     # Handle direct array indexing
     if not isinstance(key, tuple):
         key = (key,)
     return self._raw_indexing_method(key)
   
    def _raw_indexing_method(self, key):
      """Read data from GDAL multidim array."""
      # Convert key to array of slices
      if not isinstance(key, tuple):
          key = (key,)
      
      # Build start, count, and step arrays for GDAL
      ndim = len(self.shape)
      starts = []
      counts = []
      steps = []
      squeeze_dims = []  # Track which dimensions to squeeze
      
      for i, k in enumerate(key):
          if isinstance(k, slice):
            start = k.start if k.start is not None else 0
            stop = k.stop if k.stop is not None else self.shape[i]
            step = k.step if k.step is not None else 1
            if step > 0 and stop < start:
              start, stop = stop, start + 1
            elif step < 0:
              # explicit reverse: also canonicalise
              start, stop, step = stop + 1, start + 1, -step
              # (worry about this branch later if it ever fires)
            count = (stop - start + step - 1) // step
          elif isinstance(k, int) | isinstance(k, float):
              start = k
              count = 1
              step = 1
              squeeze_dims.append(i)  # Mark this dimension for squeezing
          else:
              raise IndexError(f"Unsupported index type: {type(k)}")
          
          starts.append(start)
          counts.append(count)
          steps.append(step)
      # Handle zero-sized slices (Dask uses these for _meta)
      if any(c == 0 for c in counts):
         shape = [c for i, c in enumerate(counts) if i not in squeeze_dims]
         return np.empty(shape, dtype=self._dtype)
      # Read from GDAL multidim array
      # scale = self.mdarray.GetScale() 
      # offset = self.mdarray.GetOffset() 
      block = np.array(self.mdarray.GetBlockSize())
      ## avoid div by 0
      ## see issue https://github.com/OSGeo/gdal/issues/13324 
      #block = [1 if x == 0 else x for x in block]
      for i in range(len(block)): 
        if block[i] == 0: 
          block[i] = self.shape[i]
      
      num_elem =  int(np.ceil(np.prod(np.ceil((np.array(counts) * np.array(steps))  / block) * block)))
      num_bytes = int(self._dtype().itemsize * num_elem * 1.2)
      if num_bytes < 16777216:
        num_bytes = 16777216
      self.mdarray.AdviseRead(
            array_start_idx=starts,
            count=counts,
            options=[f"CACHE_SIZE={str(num_bytes)}"]
      )
      logger.debug("starts=%s, counts=%s, steps=%s, shape=%s, chunks=%s",
             starts, counts, steps, self.shape, self._chunks)
        
      data = self.mdarray.ReadAsArray(
          array_start_idx=starts,
          count=counts,
          array_step=steps
      )
      # if scale is not None:
      #   data = data * scale
      # if offset is not None: 
      #   data = data + offset
      #   
      
      # Squeeze out dimensions that were indexed with integers
      for dim_idx in reversed(squeeze_dims):
          data = np.squeeze(data, axis=dim_idx)
      
      return data


class GDALBackendEntrypoint(BackendEntrypoint):
    """Xarray backend for reading geospatial files with GDAL."""
    
    available = True
    
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
        """
        Open a dataset using GDAL.
        
        Parameters
        ----------
        filename_or_obj : str
            Path to the file to open
        drop_variables : list, optional
            Variables to drop from the dataset
        chunks : dict, optional
            Chunk sizes for Dask arrays. None (default) returns a lazy
            non-Dask Dataset; ``{}`` uses GDAL's native block sizes;
            an explicit mapping (e.g. ``{"y": 256, "x": 256}``) is honoured.
        multidim : bool, default True
            If True, use GDAL's multidimensional API (OpenEx with OF_MULTIDIM_RASTER).
            If False, use the classic raster API.
        group : str, optional
            Group path for multidimensional datasets (e.g., "/group/subgroup").
        band_as_dim : bool, default True
            Classic-raster only. If True, bands become an xarray "band" dimension
            on a single ``band_data`` DataArray (the rioxarray-compatible idiom).
            If False, each band becomes a separate data variable named after its
            description (or ``band_N``). The True default suits multispectral
            imagery; False is preferable when bands carry semantically distinct
            quantities (e.g. NetCDF-translated multivariable rasters).
        """
        
        if multidim:
            return self._open_multidim(filename_or_obj, chunks, group, drop_variables)
        else:
            return self._open_raster(
                filename_or_obj, chunks, drop_variables, band_as_dim=band_as_dim
            )
    
    def _open_raster(self, filename_or_obj, chunks, drop_variables, band_as_dim=True):
        """Open using GDAL's classic raster API."""
        logger.debug("filename_or_obj: %s", filename_or_obj)
        dataset = gdal.Open(filename_or_obj, gdal.GA_ReadOnly)
        if dataset is None:
            raise ValueError(f"Could not open {filename_or_obj} with GDAL")
        
        # Shared setup: geotransform → RasterIndex, projection, band count.
        geotransform = dataset.GetGeoTransform()
        index = RasterIndex.from_transform(
            Affine.from_gdal(*geotransform),
            width=dataset.RasterXSize,
            height=dataset.RasterYSize,
        )
        projection = dataset.GetProjection()
        num_bands = dataset.RasterCount
        
        if band_as_dim:
            ds = self._raster_as_band_dim(
                filename_or_obj, dataset, chunks, drop_variables, num_bands
            )
        else:
            ds = self._raster_as_vars(
                filename_or_obj, dataset, chunks, drop_variables, num_bands
            )
        
        # Apply spatial coordinates and CRS uniformly to both layouts.
        ds = ds.assign_coords(xr.Coordinates.from_xindex(index))
        if len(projection) > 0:
            ds = ds.proj.assign_crs(crs=projection)
        return ds
    
    def _raster_as_band_dim(
        self, filename_or_obj, dataset, chunks, drop_variables, num_bands
    ):
        """Build a Dataset with bands collapsed into a 'band' dimension.
        
        Returns a single ``band_data`` DataArray with dims (band, y, x). Per-band
        metadata (description, nodata, scale, offset) is attached as coordinate
        variables along the band axis when it varies, or as scalar attrs when
        it's uniform across bands.
        """
        # Gather per-band metadata up-front.
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
        
        # Decide whether per-band metadata is uniform (→ scalar attr) or varies
        # (→ coordinate along the band axis). Descriptions are always per-band
        # since they're meant to label individual bands.
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
            pass  # default, omit
        elif len(set(scales)) == 1:
            attrs["scale"] = scales[0]
        else:
            coords["scale"] = ("band", np.array(scales))
        
        if all(o == 0.0 for o in offsets):
            pass  # default, omit
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
    
    def _raster_as_vars(
        self, filename_or_obj, dataset, chunks, drop_variables, num_bands
    ):
        """Build a Dataset with each band as a separate (y, x) data variable.
        
        Use this when bands carry semantically distinct quantities (e.g. a
        NetCDF translated to multiband GeoTIFF where bands are different
        physical variables).
        """
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
            
            data_vars[band_name] = xr.DataArray(
                data, dims=["y", "x"], attrs=band_attrs
            )
        
        return xr.Dataset(data_vars)
    
    def _open_multidim(self, filename_or_obj, chunks, group, drop_variables):
        """Open using GDAL multidimensional API."""
        
        # Open with multidimensional API
        dataset = gdal.OpenEx(filename_or_obj, gdal.OF_MULTIDIM_RASTER | gdal.GA_ReadOnly)
        if dataset is None:
            raise ValueError(f"Could not open {filename_or_obj} with GDAL multidim API")
        
        # Get root group
        root_group = dataset.GetRootGroup()
        if root_group is None:
            raise ValueError(f"No root group found in {filename_or_obj}")
        
        # Navigate to the requested group, handling None, "", "/", "a/b/c", "/a/b/c"
        parts = [p for p in (group or "").strip("/").split("/") if p]
        target_group = root_group
        for part in parts:
          target_group = target_group.OpenGroup(part)
          if target_group is None:
            raise ValueError(
                f"Group component {part!r} not found in path {group!r}"
            )
    
    
        # Get arrays from the group
        array_names = target_group.GetMDArrayNames()
        
        data_vars = {}
        coords = {}
        
        for array_name in array_names:
            if drop_variables and array_name in drop_variables:
                continue
            
            mdarray = target_group.OpenMDArray(array_name)
            if mdarray is None:
                continue
            
            # Get dimensions
            dims = mdarray.GetDimensions()
            dim_names = [dim.GetName() or f"dim_{i}" for i, dim in enumerate(dims)]
            
            # Create backend array
            backend_array = GDALMultiDimArray(mdarray)
            # Wrap with Dask if chunks specified
            if chunks is not None:
                if chunks == {}:
                    # Use native block sizes from GDAL
                    block_size = mdarray.GetBlockSize()
                    dim_sizes = [dim.GetSize() for dim in dims]
                    chunk_tuple = tuple(
                        b if b > 0 else dim_sizes[i]
                        for i, b in enumerate(block_size)
                    )
                else:
                    chunk_tuple = tuple(
                        chunks.get(dim_name, -1) for dim_name in dim_names
                    )
                dask_array = da.from_array(
                    backend_array,
                    chunks=chunk_tuple,
                    name=f"gdal-multidim-{filename_or_obj}-{array_name}",
                    asarray=False
                )
                data = dask_array
            else:
                from xarray.core import indexing
                data = indexing.LazilyIndexedArray(backend_array)


            # Get attributes
            attrs = {}
            md = mdarray.GetAttributes()
            for attr in md:
                attr_name = attr.GetName()
                attr_value = attr.Read()
                if attr_value is not None:
                    attrs[attr_name] = attr_value
            
            # Check if this is a coordinate variable
            is_coord = any(dim.GetName() == array_name for dim in dims)
            
            if is_coord and len(dim_names) == 1:
                # Add as coordinate - load eagerly for index variables
                coord_data = backend_array[:]  # Load the data
                    # Decode CF time coordinates
                units = mdarray.GetUnit()
                if _is_time_coord(array_name, attrs, units):
                  calendar = attrs.get('calendar', 'standard')
                  if units:
                     coord_data = decode_cf_datetime(coord_data, units, calendar)
                coords[array_name] = xr.DataArray(coord_data, dims=dim_names, attrs=attrs)
            else:
                # Add as data variable
                data_vars[array_name] = xr.DataArray(data, dims=dim_names, attrs=attrs)
                data_vars[array_name].encoding['gdal_backend'] = backend_array 
                # Create coordinate arrays for each dimension if not already present
                for dim, dim_name in zip(dims, dim_names):
                    if dim_name not in coords and dim_name not in data_vars:
                        # Create simple index coordinate
                        coords[dim_name] = np.arange(dim.GetSize())
        
        # Get group attributes
        group_attrs = {}
        group_md = target_group.GetAttributes()
        for attr in group_md:
            attr_name = attr.GetName()
            attr_value = attr.Read()
            if attr_value is not None:
                group_attrs[attr_name] = attr_value
  

        # Create dataset
        ds = xr.Dataset(data_vars, coords=coords, attrs=group_attrs)
        ds.encoding['gdal_dataset'] = dataset
        ds.encoding['gdal_group'] = target_group
    
        ds.encoding['_gdal_arrays'] = {
               name: data_vars[name].encoding.get('gdal_backend') 
              for name in data_vars
                                  }
        return ds  #{"data_vars": data_vars, "coords": coords}
    
    def guess_can_open(self, filename_or_obj):
        """Guess if this backend can open the file."""
        if isinstance(filename_or_obj, str):
            try:
                ds = gdal.Open(filename_or_obj, gdal.GA_ReadOnly)
                if ds is not None:
                    ds = None
                    return True
            except:
                pass
        return False


