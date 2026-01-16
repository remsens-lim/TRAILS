import os
import re
import datetime
import numpy as np
import bcolz
import xarray as xr
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")
from pyproj import Proj
import rasterio
from affine import Affine

# COPIED FROM trace_airmss_source code: https://github.com/martin-rdz/trace_airmass_source
def read_flexpart_traj_meta(fname, ncluster = 5):
    """ """
    
    data = {}
    with open(fname) as f:
        l = f.readline().split()
        data['end_of_sim'] = l[0] + "_" + l[1].zfill(6)
        data['version'] = l[2]
        l = f.readline().split()
        #print("second line? ", l)
        l = f.readline().split()

        data['no_releases'] = int(l[0])
        data['ncluster'] = ncluster
        data['releases_meta'] = {} 
        data['releases_traj'] = defaultdict(lambda: defaultdict(list))
        for i in range(1, data['no_releases']+1):
            l = f.readline().split()
            data['releases_meta'][i] = {}
            data['releases_meta'][i]['start_times'] = list(map(float, l[0:2]))
            data['releases_meta'][i]['lat_lon_bounds'] = list(map(float, l[2:6]))
            data['releases_meta'][i]['heights'] = list(map(float, l[6:8]))
            data['releases_meta'][i]['species'] = float(l[8])
            data['releases_meta'][i]['no_particles'] = float(l[9])
            data['releases_meta'][i]['string'] =  f.readline().strip()
            #print('releases meta', data['releases_meta'][i])
            
        for line in f:
            l = line.split()
            i = int(l.pop(0))
            
            props = ['age', 'lon', 'lat', 'height', 'mean_topo',
                     'mean_mlh', 'mean_tph', 'mean_PV', 'rms_distance',
                     'rms', 'zrms_distance', 'zrms', 'frac_ml', 'frac_lt_2pvu',
                     'frac_tp']
            for p in props:
                match = re.match('([-+]?[0-9]*\.?[0-9]*)(-[0-9]*\.?[0-9]*)', l[0])
                if match and not match.group(1) == '':
                    l[0] = match.group(1)
                    l.insert(1, match.group(2))
                elem = float(l.pop(0))
                #print(p, elem)
                data['releases_traj'][i][p].append(elem)
                
                
            # cluster are not continuous
            # fix ourselfs
            avail_clusters = list(range(ncluster))
            cluster_data = []
            for k in avail_clusters:
                cluster_props = ['lon', 'lat', 'height', 'frac', 'rms']
                cluster_data.append({})
                for cp in cluster_props:
                    match = re.match('([-+]?[0-9]*\.?[0-9]*)(-[0-9]*\.?[0-9]*)', l[0])
                    if match and not match.group(1) == '':
                        l[0] = match.group(1)
                        l.insert(1, match.group(2))
                    elem = float(l.pop(0))
                    #key = 'c{}_{}'.format(k, cp)
                    #print(key, cp, elem)
                    cluster_data[k][cp] = elem
            

                for ci, elem in enumerate(cluster_data):
                    for k, v in elem.items():
                            key = 'c{}_{}'.format(ci, k)
                            data['releases_traj'][i][key].append(v)
                      
                    
            assert len(l) == 0, "line not fully consumend"
        
        return data

def get_quantized_ctable(dtype, cparams, quantize=None, expectedlen=None):
    """Return a ctable with the quantize filter enabled for floating point cols.
    
    License
        This function is taken from the reflexible package (https://github.com/spectraphilic/reflexible/tree/master/reflexible).
        Authored by John F Burkhart <jfburkhart@gmail.com> with contributions Francesc Alted <falted@gmail.com>.
        Licensed under: 'This script follows creative commons usage.'
    """
    columns, names = [], []
    for fname, ftype in dtype.descr:
        names.append(fname)
        if 'f' in ftype:
            cparams2 = bcolz.cparams(clevel=cparams.clevel, cname=cparams.cname, quantize=quantize)
            columns.append(bcolz.zeros(0, dtype=ftype, cparams=cparams2, expectedlen=expectedlen))
        else:
            columns.append(bcolz.zeros(0, dtype=ftype, cparams=cparams, expectedlen=expectedlen))
    return bcolz.ctable(columns=columns, names=names)

def read_partpositions(filename, nspec, ctable=True, clevel=5, cname="lz4", quantize=None):
    """Read the particle positions in `filename`.

    This function strives to use as less memory as possible; for this, a
    bcolz ctable container is used for holding the data.  Besides to be compressed
    in-memory, its chunked nature makes a natural fit for data that needs to
    be appended because it does not need expensive memory resize operations.

    NOTE: This code reads directly from un UNFORMATTED SEQUENTIAL data Fortran
    file so care has been taken to skip the record length at the beginning and
    the end of every record.  See:
    http://stackoverflow.com/questions/8751185/fortran-unformatted-file-format

    Parameters
    ----------
    filename : string
        The file name of the particle raw data
    nspec : int
        number of species in particle raw data
    ctable : bool
        Return a bcolz ctable container.  If not, a numpy structured array is returned instead.
    clevel : int
        Compression level for the ctable container
    cname : string
        Codec name for the ctable container.  Can be 'blosclz', 'lz4', 'zlib' or 'zstd'.
    quantize : int
        Quantize data to improve (lossy) compression.  Data is quantized using
        np.around(scale*data)/scale, where scale is 2**bits, and bits is
        determined from the quantize value.  For example, if quantize=1, bits
        will be 4.  0 means that the quantization is disabled.

    Returns
    -------
    ctable object OR structured_numpy_array

    Returning a ctable is preferred because it is used internally so it does not require to be
    converted to other formats, so it is faster and uses less memory.

    Note: Passing a `quantize` param > 0 can increase the compression ratio of the ctable
    container, but it may also slow down the reading speed significantly.

    License
        This function is taken from the reflexible package (https://github.com/spectraphilic/reflexible/tree/master/reflexible).
        Authored by John F Burkhart <jfburkhart@gmail.com> with contributions Francesc Alted <falted@gmail.com>.
        Licensed under: 'This script follows creative commons usage.'


    """

    CHUNKSIZE = 10 * 1000
    xmass_dtype = [('xmass_%d' % (i + 1), 'f4') for i in range(nspec)]
    # note age is calculated from itramem by adding itimein
    out_fields = [
                     ('npoint', 'i4'), ('xtra1', 'f4'), ('ytra1', 'f4'), ('ztra1', 'f4'),
                     ('itramem', 'i4'), ('topo', 'f4'), ('pvi', 'f4'), ('qvi', 'f4'),
                     ('rhoi', 'f4'), ('hmixi', 'f4'), ('tri', 'f4'), ('tti', 'f4')] + xmass_dtype
    raw_fields = [('begin_recsize', 'i4')] + out_fields + [('end_recsize', 'i4')]
    raw_rectype = np.dtype(raw_fields)
    recsize = raw_rectype.itemsize

    cparams = bcolz.cparams(clevel=clevel, cname=cname)
    if quantize is not None and quantize > 0:
        out = get_quantized_ctable(raw_rectype, cparams=cparams, quantize=quantize, expectedlen=int(1e6))
    else:
        out = bcolz.zeros(0, dtype=raw_rectype, cparams=cparams, expectedlen=int(1e6))

    with open(filename, "rb", buffering=1) as f:
        # The timein value is at the beginning of the file
        reclen = np.ndarray(shape=(1,), buffer=f.read(4), dtype="i4")[0]
        assert reclen == 4
        itimein = np.ndarray(shape=(1,), buffer=f.read(4), dtype="i4")
        reclen = np.ndarray(shape=(1,), buffer=f.read(4), dtype="i4")[0]
        assert reclen == 4
        nrec = 0
        while True:
            # Try to read a complete chunk
            data = f.read(CHUNKSIZE * recsize)
            read_records = int(len(data) / recsize)  # the actual number of records read
            chunk = np.ndarray(shape=(read_records,), buffer=data, dtype=raw_rectype)
            # Add the chunk to the out array
            out.append(chunk[:read_records])
            nrec += read_records
            if read_records < CHUNKSIZE:
                # We reached the end of the file
                break

    # Truncate at the max length (last row is always a sentinel, so remove it)
    out.trim(1)
    # Remove the first and last columns
    out.delcol("begin_recsize")
    out.delcol("end_recsize")

    if ctable:
        return out
    else:
        return out[:]




# 1° grids MODIS
LON_GRID_MODIS = np.linspace(-180, 180, 360)  
LAT_GRID_MODIS = np.linspace(-90, 90, 180)    

# 1° grids UVAI
LAT_GRID = np.linspace(-90, 90, 180)
LON_GRID = np.linspace(-180, 180, 360)

# Helper functions
def time_list(begin, end, delta):
    return [begin + datetime.timedelta(hours=i*delta) 
            for i in range(int((end - begin).total_seconds() // 3600 // delta) + 1)]

def get_uvai_for_coords(uvai_map, lats, lons):
    lat_idx = np.argmin(np.abs(LAT_GRID[:, None] - lats.ravel()), axis=0)
    lon_idx = np.argmin(np.abs(LON_GRID[:, None] - lons.ravel()), axis=0)
    return uvai_map[lat_idx, lon_idx].reshape(lats.shape)
def get_frp_for_coords(frp_map, lats, lons):
    lat_idx = np.argmin(np.abs(LAT_GRID_MODIS[:, None] - lats.ravel()), axis=0)
    lon_idx = np.argmin(np.abs(LON_GRID_MODIS[:, None] - lons.ravel()), axis=0)
    return frp_map[lat_idx, lon_idx].reshape(lats.shape)

def get_LCcat_for_coords(lc_map, lc_lon, lc_lat, lats, lons):
    lat_idx = np.argmin(np.abs(lc_lat[:, None] - lats.ravel()), axis=0)
    lon_idx = np.argmin(np.abs(lc_lon[:, None] - lons.ravel()), axis=0)
    return lc_map[lat_idx, lon_idx].reshape(lats.shape)


def load_uvai_map(config, dt):
    """Load daily UVAI map for given datetime (ignores time of day)"""
    date_str = dt.strftime("%Y%m%d")
    path =  config["omps_dir"]
    
    file = path + f"{date_str}_OMPS-NPP_NMMIEAI-L2_UVAI.nc"
    
    #print(file)
    if not os.path.exists(file):
        print(f"⚠️ Missing UVAI file: {file}")
        return np.zeros((180, 360))  # Fallback
    
    dataset = xr.open_dataset(file)
    uvai_max = dataset.uvai_max.values.T
    uvai  = np.ma.masked_where(uvai_max   <= 0.7, uvai_max  )

    return uvai


def load_frp_map(config, dt):
    """Load daily FRP map for given datetime (ignores time of day)"""
    date_str = dt.strftime("%Y%m%d")
    file = config["frp_dir"] + f"/{date_str}_MODIS_FRP_daily.nc"
    
    #print(file)
    if not os.path.exists(file):
        print(f"⚠️ Missing FRP file: {file}")
        return np.zeros((180, 360))  # Fallback
    
    dataset = xr.open_dataset(file)
    frp = dataset.FRP_mean.values
   
    return np.ma.masked_where(frp <=0, frp)

def load_LC_map(config):

    # Code runs very slow on 500m LCType grid, resampled ones to 1degree much faster
    #filename = config["LCType_dir"] + f"lc_mcd12q1v061.t1_c_500m_s_20210101_20211231_go_epsg.4326_v20230818.tif"
    filename =  config["LCType_dir"] + f"resampledLCType.tif"
    #with rasterio.divers():
    with rasterio.open(filename, 'r') as src:
        meta = src.meta
        width = meta['width']
        height = meta['height']
        count = meta['count']
        dtype = meta['dtype']
        shape = src.shape
        transform = src.transform
        T0 = src.transform
        # Resolution is the pixel size: transform.a and transform.e
        pixel_width = transform.a  # Size of a pixel in the x-direction
        pixel_height = abs(transform.e)  # Size of a pixel in the y-direction (positive value)
    
        print(f"Pixel width (resolution in x): {pixel_width}")
        print(f"Pixel height (resolution in y): {pixel_height}")
        p1 = Proj(src.crs)
        print('T0 aka affine transformation ', T0)
        print('src.crs', src.crs)
        print('p1', p1)
        # allocate memory for image
        im = np.empty([height, width], dtype)
        # read image into memory
        print(meta)
        im[:, :] = src.read(1)
    ls_data = im
    
    T1 = T0 * Affine.translation(0.5, 0.5)
    # Function to convert pixel row/column index (from 0) to easting/northing at centre
    
    etemp, northings = T1 * (np.meshgrid(np.arange(1), np.arange(im.shape[0])))
    eastings, ntemp = T1 * (np.meshgrid(np.arange(im.shape[1]), np.arange(1)))
    
    lon = eastings[0, :]
    
    lat = northings[:, 0]
    return lon, lat, ls_data


# FLEXPART caclulates heights above ground level, UVAI relates to above mean sea level -- this needs to be corrected using elevation 

def load_elevation_map():
    """Load daily elevation map """
    
    file = f"/projekt1/remsens/work/jroschke/repositories/SMOKE/trace_smoke/elevation_global_1d.nc"
    
    
    dataset = xr.open_dataset(file)
    elevation = dataset.elevation.values
    
    return dataset.longitude.values, dataset.latitude.values, np.ma.masked_invalid(elevation).filled(0)

