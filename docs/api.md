<!-- docs/api.md -->
# API Reference

## Engine entrypoint

The xarray engine itself. Most users invoke it via
`xr.open_dataset(path, engine="gdalxarray", ...)` rather than directly.

::: gdalxarray.GDALBackendEntrypoint
    options:
      show_source: false
    
## Recipe builders

Helper functions that build GDAL configuration recipes - VRT strings,
typically - for downstream consumption by the engine.
  
::: gdalxarray.warp
    options:
      show_source: false