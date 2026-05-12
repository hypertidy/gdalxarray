import xarray as xr
from xarray.backends import BackendEntrypoint, BackendArray
from xarray.core import indexing
from xarray.coding.times import decode_cf_datetime

import numpy as np
import dask.array as da
from osgeo import gdal
gdal.UseExceptions()
from typing import Iterable, Optional
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
        # Fast unique identifier - avoids Dask trying to hash the mdarray
        return (type(self).__name__, id(self.dataset), self.band_index)
    
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
            Chunk sizes for Dask arrays
        multidim : bool, default False
            If True, use GDAL multidimensional API (OpenEx with OF_MULTIDIM_RASTER)
            If False, use standard raster API
        group : str, optional
            Group path for multidimensional datasets (e.g., "/group/subgroup")
        **kwargs
            Additional backend-specific options
        """
        
        if multidim:
            return self._open_multidim(filename_or_obj, chunks, group, drop_variables)
        else:
            return self._open_raster(filename_or_obj, chunks, drop_variables)
    
    def _open_raster(self, filename_or_obj, chunks, drop_variables):
        """Open using standard GDAL raster API."""
        logger.debug("filename_or_obj: %s", filename_or_obj)
        # Open with GDAL
        dataset = gdal.Open(filename_or_obj, gdal.GA_ReadOnly)
        if dataset is None:
            raise ValueError(f"Could not open {filename_or_obj} with GDAL")
        
        # Get geotransform for coordinates
        geotransform = dataset.GetGeoTransform()
        
        # Calculate coordinates
        #x_coords = np.arange(dataset.RasterXSize) * geotransform[1] + geotransform[0]
        #y_coords = np.arange(dataset.RasterYSize) * geotransform[5] + geotransform[3]
        index = RasterIndex.from_transform(Affine.from_gdal(geotransform[0], geotransform[1], geotransform[2], 
                                                            geotransform[3], geotransform[4], geotransform[5]), 
                                                            width=dataset.RasterXSize, height=dataset.RasterYSize)

        # Get CRS
        projection = dataset.GetProjection()
        
        # Create data variables for each band
        data_vars = {}
        num_bands = dataset.RasterCount
        
        for band_idx in range(1, num_bands + 1):
            band = dataset.GetRasterBand(band_idx)
            band_name = band.GetDescription() or f"band_{band_idx}"
            
            if drop_variables and band_name in drop_variables:
                continue
            
            # Create backend array
            backend_array = GDALBackendArray(filename_or_obj, band_idx)
            logger.debug("band: %i", band_idx)
            # Wrap with Dask if chunks specified
            if chunks is not None:
                if chunks == {}:
                    # FIXME what if no  bands available
                    # Use native block sizes from GDAL
                    block_size = dataset.GetRasterBand(1).GetBlockSize()  # [x, y]
                    # Flip for (y, x) array order:
                    chunk_tuple = (block_size[1], block_size[0]) 
                    block_size = dataset.GetRasterBand(1).GetBlockSize()
                    dim_sizes = [dataset.RasterXSize, dataset.RasterYSize]
                    logger.debug("shape=%s(x,y), chunks=%s(y,x)", dim_sizes, block_size)
                else:
                    chunk_tuple = tuple(
                        chunks.get(dim_name, -1) for dim_name in dim_names
                    )
                dask_array = da.from_array(
                    backend_array,
                    chunks=chunk_tuple,
                    name=f"gdal-{filename_or_obj}-{band_name}",
                    asarray=False
                )
                data = dask_array
            else:
                data = backend_array

            # Get band metadata
            band_attrs = {
                'nodata': band.GetNoDataValue(),
                'scale': band.GetScale() or 1.0,
                'offset': band.GetOffset() or 0.0,
            }
            band_attrs = {k: v for k, v in band_attrs.items() if v is not None}
            
            # Create DataArray
            data_vars[band_name] = xr.DataArray(
                data,
                dims=["y", "x"],
                attrs=band_attrs
            )
        # Create dataset
        ds = xr.Dataset(data_vars)
        ##https://gist.github.com/mdsumner/911c181467abb2c91d08544a94d8510a
        ds = ds.assign_coords(xr.Coordinates.from_xindex(index))
          
        # https://xarray.dev/blog/flexible-indexing#xprojcrsindex
        if len(projection) > 0: 
          ds = ds.proj.assign_crs(crs = projection)
        # Add global attributes
        #ds.attrs['crs'] = projection
        #ds.attrs['geotransform'] = geotransform
        return ds
    
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
        
        # Navigate to specified group if provided
        if group:
            target_group = root_group.OpenGroup(group)
            if target_group is None:
                raise ValueError(f"Group {group} not found")
        else:
            target_group = root_group
        
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
                data = backend_array


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

                if array_name == 'time':
                  units = mdarray.GetUnit()
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


