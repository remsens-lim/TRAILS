import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np
import cmaps as c

def plot_uvai_global_grd(ds, obs_date, extent):

    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    ax.set_title(f"OMPS UV Aerosol Index: {obs_date}")
    # Plot data
    
    cp = plt.pcolormesh(ds.longitude.values, ds.latitude.values ,  
                        ds.uvai_mean.values.T, vmin = 0, vmax = 10, cmap=c.uv_map, 
                      transform=ccrs.PlateCarree())
    # Coastlines
    ax.coastlines('50m', linewidth=1, color='black')
    
    # Gridlines with labels
    gd = ax.gridlines(
        crs=ccrs.PlateCarree(), 
        draw_labels=True,
        linewidth=1, 
        linestyle='--', 
        color='k',
        alpha=0.5
    )
    
    # Control label visibility
    gd.xlabels_top = False     # Disable top labels
    gd.ylabels_right = False   # Disable right labels
    gd.xlabel_bottom = True    # Ensure bottom labels are ON
    gd.ylabel_left = True      # Ensure left labels are ON
    gd.xformatter = LONGITUDE_FORMATTER
    gd.yformatter = LATITUDE_FORMATTER
    
    aspect_ratio = (80 / 170) * 0.7  # Adjusted for better fit
    cbar = plt.colorbar(cp, ax=ax, label='UV Aerosol Index', 
                       shrink=aspect_ratio, pad=0.1)
    cbar.ax.set_ylabel( 'UV Aerosol index ', fontsize=12)
        
    # Plot a point (example)
    #ax.plot( 12.435, 51.353, '^', color = "red")
    
    # Set extent (focus on your region of interest)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    plt.tight_layout()
    return fig, ax