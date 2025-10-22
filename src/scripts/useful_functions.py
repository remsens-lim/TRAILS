from shapely.geometry import Point
import pandas as pd
import numpy as np
import h5py
import glob
import natsort
import datetime
import geopandas as gpd
import xarray as xr
from shapely.geometry import Point
import os
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



def active_fires_per_date(df_active, date_min, date_max):
    geometry = [Point(lon, lat) for lon, lat in zip(df_active['lon'], df_active['lat'])]
    gdf_active = gpd.GeoDataFrame(df_active, geometry=geometry)
    
    # Pad the 'Time' column with leading zeros to ensure it has 4 digits
    gdf_active['HHMM'] = gdf_active['HHMM'].astype(str).str.zfill(4)
    
    # Combine the 'Date' and 'Time' columns into a single string
    gdf_active['DateTime'] = gdf_active['YYYYMMDD'].astype(str) + gdf_active['HHMM'].astype(str)
    gdf_active['Date'] = gdf_active['YYYYMMDD'].astype(str)
    
    # Convert the combined string to a pandas datetime column
    gdf_active['DateTime'] = pd.to_datetime(gdf_active['DateTime'], format='%Y%m%d%H%M')
    gdf_active['Date'] = pd.to_datetime(gdf_active['Date'], format='%Y%m%d')

    # CORRECTED: Use AND operator to get dates within the range
    test_active = gdf_active[(gdf_active['DateTime'] >= date_min) & (gdf_active['DateTime'] <= date_max)] 
    return test_active

def create_daily_nc_files(gdf_day, output_dir, day):
    """
    Create NetCDF file for a single day with mean FRP binned to 0.5° grid
    """
    # Create 0.5° grid bins
    gdf_day['lon_bin'] = (gdf_day.geometry.x / 0.5).apply(np.floor) * 0.5 + 0.25
    gdf_day['lat_bin'] = (gdf_day.geometry.y / 0.5).apply(np.floor) * 0.5 + 0.25
    
    # Group by bins and calculate mean FRP
    binned = gdf_day.groupby(['lon_bin', 'lat_bin']).agg(
        FRP_mean=('FRP', 'mean'),
        counts=('FRP', 'count')
    ).reset_index()
    
    # Create coordinate arrays for the grid
    lon_unique = np.sort(binned['lon_bin'].unique())
    lat_unique = np.sort(binned['lat_bin'].unique())
    
    # Create empty arrays for the full grid
    frp_grid = np.full((len(lat_unique), len(lon_unique)), np.nan)
    counts_grid = np.full((len(lat_unique), len(lon_unique)), 0)
    
    # Fill the arrays with data
    for _, row in binned.iterrows():
        lon_idx = np.where(lon_unique == row['lon_bin'])[0][0]
        lat_idx = np.where(lat_unique == row['lat_bin'])[0][0]
        frp_grid[lat_idx, lon_idx] = row['FRP_mean']
        counts_grid[lat_idx, lon_idx] = row['counts']
    
    # Create xarray Dataset
    ds = xr.Dataset({
        'FRP_mean': (['latitude', 'longitude'], frp_grid),
        'counts': (['latitude', 'longitude'], counts_grid)
    }, coords={
        'longitude': lon_unique,
        'latitude': lat_unique,
        'time': pd.to_datetime(day)
    })
    
    # Add attributes
    ds['FRP_mean'].attrs = {
        'long_name': 'Mean Fire Radiative Power',
        'units': 'MW',
        'description': 'Mean FRP value in 0.5° grid cells'
    }
    
    ds['counts'].attrs = {
        'long_name': 'Number of fire detections',
        'units': 'count',
        'description': 'Number of fire detections in 0.5° grid cells'
    }
    
    ds.attrs = {
        'title': f'Daily Fire Radiative Power - {day.strftime("%Y-%m-%d")}',
        'institution': 'Your Institution',
        'source': 'MODIS Active Fires',
        'history': f'Created {pd.Timestamp.now().strftime("%Y-%m-%d")}',
        'grid_resolution': '1 degrees'
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to NetCDF
    output_file = os.path.join(output_dir, f"{day.strftime('%Y%m%d')}_MODIS_FRP_daily.nc")
    ds.to_netcdf(output_file)
    print(f"Created: {output_file}")
    
    return output_file
def create_daily_nc_files_efficient(gdf_day, day):
    """
    More efficient version using np.digitize
    """
    # Create uniform global grid coordinates (1° resolution)
    lon_edges = np.linspace(-180, 180, 361)  # 361 edges for 360 cells
    lat_edges = np.linspace(-90, 90, 181)    # 181 edges for 180 cells
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    
    # Create empty arrays
    frp_sum = np.zeros((180, 360))
    counts  = np.zeros((180, 360), dtype=int)
    
    # Get coordinates
    lons = gdf_day.geometry.x.values
    lats = gdf_day.geometry.y.values
    frp_values = gdf_day['FRP'].values
    
    # Bin the data
    lon_idx = np.digitize(lons, lon_edges) - 1
    lat_idx = np.digitize(lats, lat_edges) - 1
    
    # Filter indices that are within bounds
    valid_mask = (lon_idx >= 0) & (lon_idx < 360) & (lat_idx >= 0) & (lat_idx < 180)
    lon_idx = lon_idx[valid_mask]
    lat_idx = lat_idx[valid_mask]
    frp_values = frp_values[valid_mask]
    
    # Accumulate sums and counts
    for i, (lat_i, lon_i, frp_val) in enumerate(zip(lat_idx, lon_idx, frp_values)):
        frp_sum[lat_i, lon_i] += frp_val
        counts[lat_i, lon_i] += 1
    
    # Calculate mean FRP
    frp_mean = np.divide(frp_sum, counts, where=counts > 0, out=np.full_like(frp_sum, np.nan))
    
    # Create Dataset
    ds = xr.Dataset({
        'FRP_mean': (['latitude', 'longitude'], frp_mean),
        'counts': (['latitude', 'longitude'], counts)
    }, coords={
        'longitude': lon_centers,
        'latitude': lat_centers,
        'time': pd.to_datetime(day)
    })
    ds.attrs = {
        'resolution': f'1 degrees',}
    
    return ds






############################################################################################################################################
