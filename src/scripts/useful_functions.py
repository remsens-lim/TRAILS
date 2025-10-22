from shapely.geometry import Point
import pandas as pd
import numpy as np
import h5py
import glob
import natsort
import datetime
import geopandas as gpd
import xarray as xr


############################################################################################################################################
#         OMPS         UV AEROSOL INDEX                         -                      SWATHS - TO - GRIDS
############################################################################################################################################

def read_suomi(path_suomi):
    with h5py.File(path_suomi, 'r') as f:
        uvai = f['BinScheme1/ScienceData/Pair340_379/UVAerosolIndex'][:]
        lat = f['BinScheme1/GeolocationData/Latitude'][:]
        lon = f['BinScheme1/GeolocationData/Longitude'][:]
    return lon, lat, uvai
def create_global_grid(uvai, lon, lat):
    grid_sum = np.zeros((180, 360))
    counts   = np.zeros((180, 360))
    grid_max = np.full((180, 360), -np.inf)

    uvai = uvai.flatten()
    lon  = lon.flatten()
    lat  = lat.flatten()

    valid = np.isfinite(uvai) & np.isfinite(lat) & np.isfinite(lon)
    uvai  = uvai[valid]
    lon   = lon[valid]
    lat   = lat[valid]

    lat_idx = (lat + 90).astype(int)
    lon_idx = (lon + 180).astype(int)

    mask = (lat_idx >= 0) & (lat_idx < 180) & (lon_idx >= 0) & (lon_idx < 360)
    lat_idx = lat_idx[mask]
    lon_idx = lon_idx[mask]
    uvai    = uvai[mask]

    for la, lo, val in zip(lat_idx, lon_idx, uvai):
        grid_sum[la, lo] += val
        counts[la, lo]   += 1
        grid_max[la, lo] = max(grid_max[la, lo], val)

    return grid_sum, counts, grid_max

def swaths_to_grd(paths_omps):
        # Initialize accumulators
        grid_sum = np.zeros((180, 360))
        count_sum = np.zeros((180, 360))
        grid_max = np.full((180, 360), -np.inf)  # start with -inf for max
    
        for path in paths_omps:
            lon, lat, uvai = read_suomi(path)
            grid, counts, g_max = create_global_grid(uvai, lon, lat)
            
            grid_sum += grid * counts
            count_sum += counts
            grid_max  = np.fmax(grid_max, g_max)
        # Compute mean and mask invalids
        with np.errstate(invalid='ignore', divide='ignore'):
            uvai_mean = np.where(count_sum > 0, grid_sum / count_sum, np.nan)
            uvai_max  = np.where(grid_max == -np.inf, np.nan, grid_max)
        return uvai_mean, uvai_max

def create_nc_dataset(obs_date, uvai_mean, uvai_max): 
    ds = xr.Dataset(
            {
                # 2D Variables 
                'uvai_mean': (('longitude' , 'latitude'), uvai_mean.T,
                        {'long_name': "Aerosol Index" , 'units': "" , 
                         'description': "OMPS Aerosol Index (PyroCumuloNimbus) layer"}),
                 # 2D Variables 
                'uvai_max': (('longitude' , 'latitude'), uvai_max.T,
                        {'long_name': "Aerosol Index" , 'units': "" , 
                         'description': "OMPS Aerosol Index (PyroCumuloNimbus) layer"}),
                
            },
            
            coords={
                'latitude': (('latitude',), np.linspace(-90,90,180),
                          {'long_name': 'Laltitude (°)', 'units': '°'}),
                
                'longitude': (('longitude',), np.linspace(-180,180,360),
                             {'long_name': 'Longitude (°)', 'units': '°'}),
                
            }
        )
    
    # Add global attributes
    ds.attrs = {
        'title': f'Aerosol Index on {obs_date} (date of observation)',
        'description': "OMPS Aerosol Index (PyroCumuloNimbus) layer indicates the presence of ultraviolet (UV)-absorbing particles in the air (aerosols) such as desert dust and soot particles in the atmosphere; it is related to both the thickness of the aerosol layer located in the atmosphere and to the height of the layer. The Aerosol Index (PyroCumuloNimbus) layer is useful for identifying and tracking the extent and spread of pyrocumulonimbus and other high-aerosol events in the atmosphere. The layer has a unitless range from < 0 to >= 50. It differs from the Aerosol Index layer which detects aerosol index values between 0 and 5, because in the case of pyrocumulonimbus events, which are both dense and high in the atmosphere, the aerosol index value can easily become much larger than 5. Larger aerosol index values between 5 and 10 usually indicate dense smoke from intensely burning fires that reach higher in the troposphere, and once the aerosol index value is above 10, the smoke has likely been produced from a pyrocumulonimbus event, with dense smoke lofted into the upper troposphere, and often, into the stratosphere.",
        'processing_date': datetime.datetime.now().strftime("%Y-%m-%d"),
        'data_origin': 'NMMIEAI-L2-NRT; OMPS_NPP_NMMIEAI_L2 doi:10.5067/40L92G8144IV; Earthdata - OMPS Product Provides a Better View of High-Aerosol Events',
        'author': 'Johanna Roschke/ Institue for Meteorology , University of Leipzig',
        'contact': 'johanna.roschke@uni-leipzig.de',
        'project': 'TRAILS - TRAjectory based Airmass Identification of Lofted Smoke',
        'Conventions': 'CF-1.8'
    }
    return ds


############################################################################################################################################
#                    MODIS         GRIDS
############################################################################################################################################







############################################################################################################################################
