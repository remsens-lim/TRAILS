import sys
import os
import re
import datetime
import time
import numpy as np
import toml
import matplotlib.dates as mdates
import xarray as xr
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import functions as funct
from pathlib import Path


import sys
import os
import re
import gc
import datetime
import time
import numpy as np
import toml
import bcolz
import matplotlib
import matplotlib.dates as mdates
import xarray as xr
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm
import numpy.ma as ma
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")
from pyproj import Proj, transform
import rasterio
from affine import Affine

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


# Configuration
# CORRECT PATH: Go up 2 levels to project root, then into config folder
script_dir = Path(__file__).parent  # This is /.../TRAILS/src/trails/
project_root = script_dir.parent.parent  # Go up 2 levels to /.../TRAILS/
config_file = project_root / "config" / "config_leipzig.toml"

with open(config_file) as f:
    config = toml.load(f)


# Constants
AFRICA_LAT_MIN, AFRICA_LAT_MAX = -10, 37
AFRICA_LON_MIN, AFRICA_LON_MAX = -20, 55

DUST_LAT_MIN, DUST_LAT_MAX = -30, 30
DUST_LON_MIN, DUST_LON_MAX = -180, 180


N_DATA      = int(abs(config["time"]["tr_duration"]/config["time"]["step"]))
N_LEVELS    = int(config["height"]["top"]/config["flexpart"]["no_particles"])
N_PARTICLES = int(config["flexpart"]["no_particles"])



# ADDED OR SUBSTRACTED TO Smoke height regression 
H_UVAI_TROPO_ADD = 2
H_UVAI_TROPO_SUB = 1
H_UVAI_STRATO_ADD = 2
# mask clouds in uvai grid
UVAI_THRESHOLD = 1

# Stratospheric smoke uvai thresholds
UVAI_STRAT_THRESHOLD = 10
UVAI_STRAT_POST_THRESHOLD = 2

# dust uvai maximum threshold and maximum plume height
UVAI_DUST = 5
H_DUST    = 6000  # Height threshold for dust (in meters)

# MODIS FRP thresholds
MODIS_DEGREE = 1
FRP_THRESHOLD = 30
FRP_MAJOR_THRESHOLD = 100

if MODIS_DEGREE == 1:
   lon_size = 360
   lat_size = 180
elif MODIS_DEGREE == 0.5:
    lon_size = 720
    lat_size = 360
elif MODIS_DEGREE == 2:
    lon_size = 180
    lat_size = 90
    
# 0.5° grid with bin centers (like your floor approach)
LON_GRID_MODIS = np.linspace(-180, 180, lon_size)  # 360*2 = 720 points
LAT_GRID_MODIS = np.linspace(-90, 90, lat_size)    # 180*2 = 360 points


LAT_GRID = np.linspace(-90, 90, 180)
LON_GRID = np.linspace(-180, 180, 360)


# Progress tracker class
class ProgressTracker:
    def __init__(self, total_tasks):
        self.start_time = time.time()
        self.completed = 0
        self.total = total_tasks
        self.times = []
    
    def update(self):
        self.completed += 1
        current_time = time.time()
        elapsed = current_time - self.start_time
        self.times.append(elapsed)
        
        if self.completed > 0:
            avg_time = sum(self.times) / len(self.times)
            remaining = (self.total - self.completed) * avg_time
            percent = (self.completed / self.total) * 100
            
            # Convert to human-readable time
            elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
            remaining_str = str(datetime.timedelta(seconds=int(remaining)))
            
            print(f"\nPROGRESS: {self.completed}/{self.total} ({percent:.1f}%) "
                  f"Elapsed: {elapsed_str} | Remaining: {remaining_str}")
            sys.stdout.flush()



def process_arrival_time(args):
    dt, config, it, tracker = args
    start_time = time.time()
    #print(f"⏳ Starting arrival time: {dt.strftime('%Y-%m-%d %H:%M')} (Index {it})")
    
    folder = config['partposit_dir'] + dt.strftime("%Y%m%d_%H") + "/"
    
    # Load trajectory metadata
    traj_meta = read_flexpart_traj_meta(folder + "trajectories.txt")
    
    # Precompute particle times
    part_times = [dt - datetime.timedelta(hours=3*i) for i in reversed(range(N_DATA))]
    #print(part_times)
    # Create UVAI map cache
    uvai_cache = {}
    uvai_max_cache = {}
    frp_cache = {}
    elevation_cache = {}
    
    
    # Load particle position files and sort by TIMESTAMP, not number
    files = [f for f in os.listdir(folder) if "partposit_" in f]
    
    # Extract timestamp from filename and sort by it
    def extract_timestamp(filename):
        # Extract the timestamp part: partposit_20230421090000 -> 20230421090000
        match = re.search(r"partposit_(\d+)", filename)
        return match.group(1) if match else "0"

    # Sort files by timestamp (ascending order = chronological)
    files = sorted(files, key=extract_timestamp)
    
    
    
    # Preallocate data arrays
    height_arr = np.empty((N_DATA, N_LEVELS, N_PARTICLES), dtype=np.float32)
    pbl_arr    = np.empty((N_DATA, N_LEVELS, N_PARTICLES), dtype=np.float32)
    tropo_arr  = np.empty((N_DATA, N_LEVELS, N_PARTICLES), dtype=np.float32)
    temp_arr   = np.empty((N_DATA, N_LEVELS, N_PARTICLES), dtype=np.float32)
    part_num   = np.empty((N_DATA, N_LEVELS, N_PARTICLES), dtype=np.float32)
    part_it   = np.empty((N_DATA, N_LEVELS, N_PARTICLES), dtype=np.float32)
    lon_arr    = np.empty((N_DATA, N_LEVELS, N_PARTICLES), dtype=np.float32)
    lat_arr    = np.empty((N_DATA, N_LEVELS, N_PARTICLES), dtype=np.float32)
    class_arr  = np.zeros((N_DATA, N_LEVELS, N_PARTICLES), dtype=np.int8)
    
    # Create PER-PARTICLE history trackers (shape: N_LEVELS x N_PARTICLES)
    particle_passed_africa      = np.zeros((N_LEVELS, N_PARTICLES), dtype=bool)
    particle_passed_dustbelt        = np.zeros((N_LEVELS, N_PARTICLES), dtype=bool)
    particle_smoke_was_in_strat = np.zeros((N_LEVELS, N_PARTICLES), dtype=bool)

    # Create PER-PARTICLE statistics trackers
    particle_uvai_gt1       = np.zeros((N_LEVELS, N_PARTICLES), dtype=bool)
    particle_uvai_gt10      = np.zeros((N_LEVELS, N_PARTICLES), dtype=bool)
    particle_in_africa      = np.zeros((N_LEVELS, N_PARTICLES), dtype=bool)
    particle_in_dustbelt    = np.zeros((N_LEVELS, N_PARTICLES), dtype=bool)
    particle_in_strat       = np.zeros((N_LEVELS, N_PARTICLES), dtype=bool)
    particle_smoke_in_strat = np.zeros((N_LEVELS, N_PARTICLES), dtype=bool)
    particle_in_duststorm  = np.zeros((N_LEVELS, N_PARTICLES), dtype=bool)
    
    particle_major_fire = np.zeros((N_LEVELS, N_PARTICLES), dtype=bool)
    particle_wildfire   = np.zeros((N_LEVELS, N_PARTICLES), dtype=bool)

    ### LAND SURFACE DATA AND ELEVATION ###
    lc_lon, lc_lat, lc_map       = funct.load_LC_map()
    el_lon, el_lat, elevation_map = funct.load_elevation_map()   
    
    # Load and process all files
    for i, f in enumerate(files):
        
        # Get ACTUAL particle time (not arrival time)
        particle_time = part_times[i]
        date_key = particle_time.strftime("%Y%m%d")
        
        uvai_cache[date_key] = funct.load_uvai_map(particle_time)   
        uvai_map = uvai_cache[date_key]
        
        uvai_max_cache[date_key] = funct.load_uvai_map(particle_time)   
        uvai_max_map = uvai_max_cache[date_key]

        frp_cache[date_key] = funct.load_frp_map(particle_time)
        frp_map = frp_cache[date_key]

        # Assuming part_pos is a bcolz ctable returned from read_partpositions
        part_pos = read_partpositions(folder + f, 1, ctable=True)
        
        # Convert to a 2D numeric array: (n_records, N_DATA)
        fields = part_pos.dtype.names           # all column names
        part_pos_numeric = np.vstack([part_pos[field] for field in fields]).T
        
        # Now reshape to (N_LEVELS, N_PARTICLES, N_DATA)
        part_pos_reshaped = part_pos_numeric.reshape(N_LEVELS, N_PARTICLES, -1)
        #print("Original shape:", part_pos_numeric.shape)
        #print("Reshaped shape:", part_pos_reshaped.shape)
        # Store basic data
        part_num[i]   = part_pos_reshaped[:, :, 0] # height idx
        lon_arr[i]    = part_pos_reshaped[:, :, 1]
        lat_arr[i]    = part_pos_reshaped[:, :, 2]
        height_arr[i] = part_pos_reshaped[:, :, 3]
        part_it[i]    = part_pos_reshaped[:, :, 4] # time idx
        pbl_arr[i]    = part_pos_reshaped[:, :, 9]
        tropo_arr[i]  = part_pos_reshaped[:, :, 10]
        temp_arr[i]   = part_pos_reshaped[:, :, 11]
        
        # Compute CURRENT timestep flags
        in_africa_now = (lat_arr[i] >= AFRICA_LAT_MIN) & \
                        (lat_arr[i] <= AFRICA_LAT_MAX) & \
                        (lon_arr[i] >= AFRICA_LON_MIN) & \
                        (lon_arr[i] <= AFRICA_LON_MAX)
        # Compute CURRENT timestep flags
        in_dustbelt_now  = (lat_arr[i] >= DUST_LAT_MIN) & \
                        (lat_arr[i] <= DUST_LAT_MAX) & \
                        (lon_arr[i] >= DUST_LON_MIN) & \
                        (lon_arr[i] <= DUST_LON_MAX)
        
        in_strat_now = height_arr[i] > tropo_arr[i]

        # Get UVAI values for THIS DAY
        uvai_vals     = funct.get_uvai_for_coords(uvai_map, lat_arr[i], lon_arr[i])
        uvai_max_vals = funct.get_uvai_for_coords(uvai_max_map, lat_arr[i], lon_arr[i])
        frp_vals      = funct.get_frp_for_coords(frp_map, lat_arr[i], lon_arr[i])
        lc_vals       = funct.get_LCcat_for_coords(lc_map, lc_lon, lc_lat, lat_arr[i], lon_arr[i])
        el_vals       = funct.get_elevation_for_coords(elevation_map, el_lon, el_lat, lat_arr[i], lon_arr[i])
        
        major_fire         = frp_vals > FRP_MAJOR_THRESHOLD
        wildfire           = frp_vals > FRP_THRESHOLD
        large_uvai         = uvai_max_vals > UVAI_STRAT_THRESHOLD
        smoke_in_strat_now = in_strat_now & large_uvai
        desert             = (lc_vals == 16)
        dust_storm         = desert & (uvai_vals < 3)
        
        # UPDATE PER-PARTICLE HISTORY (using bitwise OR)
        particle_passed_africa |= in_africa_now
        particle_smoke_was_in_strat |= smoke_in_strat_now
        particle_major_fire |= major_fire
        particle_wildfire |= wildfire
        
        # UPDATE PER-PARTICLE STATISTICS
        particle_uvai_gt1 |= (uvai_vals > UVAI_THRESHOLD)
        particle_uvai_gt10 |= large_uvai
        particle_in_africa |= in_africa_now
        particle_in_dustbelt |= in_dustbelt_now
        particle_in_strat |= in_strat_now
        particle_smoke_in_strat |= smoke_in_strat_now
        particle_in_duststorm |= dust_storm

    # After processing all timesteps, compute FINAL statistics
    particle_stats = {
        'major_fire': np.sum(particle_major_fire),
        'wildfire':   np.sum(particle_wildfire),
        'uvai_gt1':   np.sum(particle_uvai_gt1),
        'uvai_gt10': np.sum(particle_uvai_gt10),
        'in_africa': np.sum(particle_in_africa),
        'in_strat': np.sum(particle_in_strat),
        'smoke_in_strat': np.sum(particle_smoke_in_strat),
        'passed_africa': np.sum(particle_passed_africa),
        'in_dustbelt': np.sum(particle_in_dustbelt),
        'smoke_was_in_strat': np.sum(particle_smoke_was_in_strat),
        'in_duststorm': np.sum(particle_in_duststorm)
    }
    
    # Print particle statistics
    print(f"  📊 Particle statistics for {dt.strftime('%Y%m%d_%H')}:")
    for stat, value in particle_stats.items():
        print(f"    {stat}: {value}")
    
    # Now use the accumulated history for classification
    passed_africa        = particle_passed_africa
    passed_dustbelt      = particle_passed_dustbelt
   # in_dustbelt         = particle_in_dustbelt
    smoke_was_in_strat   = particle_smoke_was_in_strat
    passed_duststorm     = particle_in_duststorm
    
    # Classification loop with progress tracking
    class_counts    = {'dust': 0, 'smoke': 0, 'strat_smoke': 0, 'strat_post_smoke': 0 , 'strat_post_smoke': 0 , 'smoke_post_utls':0, 'contaminated_dust': 0,'duststorm': 0, 'other': 0}
    total_particles = N_DATA * N_LEVELS * N_PARTICLES
    processed       = 0
    last_reported   = 0
    report_interval = max(1, total_particles // 20)  # Report every 5%
    
    print(f"  🚀 Starting classification of {total_particles:,} particles...")
    classification_start = time.time()
    # DON'T copy from first loop - start fresh
    smoke_was_in_strat = np.zeros((N_LEVELS, N_PARTICLES), dtype=bool)
    smoke_was_in_tropo = np.zeros((N_LEVELS, N_PARTICLES), dtype=bool)
    for idata in range(N_DATA):
        particle_time = part_times[idata]
        date_key = particle_time.strftime("%Y%m%d")
        
        uvai_map     = uvai_cache[date_key]
        uvai_max_map = uvai_max_cache[date_key]
        frp_map      = frp_cache[date_key]
    
        # Get UVAI and FRP for all particles at this timestep
        uvai_vals     = funct.get_uvai_for_coords(uvai_map, lat_arr[idata], lon_arr[idata])
        uvai_max_vals = funct.get_uvai_for_coords(uvai_max_map, lat_arr[idata], lon_arr[idata])
        #print("Fraction of particles with UVAI>1:", np.sum(uvai_vals>1)/(N_LEVELS*N_PARTICLES),  np.sum(uvai_max_vals>1)/(N_LEVELS*N_PARTICLES))

        frp_vals  = funct.get_frp_for_coords(frp_map, lat_arr[idata], lon_arr[idata])
        el_vals   = funct.get_elevation_for_coords(elevation_map, el_lon, el_lat, lat_arr[idata], lon_arr[idata])
        lc_vals   = funct.get_LCcat_for_coords(lc_map, lc_lon, lc_lat, lat_arr[idata], lon_arr[idata])
        # Mask invalid UVAI (<1 or NaN)
        uvai_valid = (~np.isnan(uvai_vals)) & (uvai_vals > UVAI_THRESHOLD)
        
        height_particle_msl = height_arr[idata] + el_vals

        # Tropopause heights are always in MSL 
        
        # Compute boolean masks for regions and stratosphere
        in_africa_now   = (lat_arr[idata] >= AFRICA_LAT_MIN) & (lat_arr[idata] <= AFRICA_LAT_MAX) & \
                          (lon_arr[idata] >= AFRICA_LON_MIN) & (lon_arr[idata] <= AFRICA_LON_MAX) & (uvai_vals < UVAI_DUST) & (uvai_vals > UVAI_THRESHOLD) & (height_particle_msl < H_DUST)  
        in_dustbelt_now = (~in_africa_now) & (lat_arr[idata] >= DUST_LAT_MIN) & (lat_arr[idata] <= DUST_LAT_MAX) & \
                          (lon_arr[idata] >= DUST_LON_MIN) & (lon_arr[idata] <= DUST_LON_MAX)  & (uvai_vals < UVAI_DUST) & (uvai_vals > UVAI_THRESHOLD) & (height_particle_msl< H_DUST)  
        in_strat_now     = height_particle_msl > tropo_arr[idata]
        in_utls_now      = (height_particle_msl > (tropo_arr[idata] - 2) ) & (height_particle_msl < (tropo_arr[idata] + 2) )
        #print('TROPOPAUSE', tropo_arr[idata])
        in_duststorm_now = (~in_africa_now) & (~in_dustbelt_now) & (lc_vals == 16) & (uvai_vals < UVAI_DUST) & (uvai_vals > UVAI_THRESHOLD) & (height_particle_msl < H_DUST)  # Desert & low UVAI & low altitude
        

        # Compute smoke height thresholds only for valid UVAI
        uv_max_hs = np.where(uvai_valid, (0.51 * uvai_vals + 2.2 + H_UVAI_TROPO_ADD) * 1000, np.nan)
        uv_min_hs = np.where(uvai_valid, (0.51 * uvai_vals + 2.2 - H_UVAI_TROPO_SUB) * 1000, np.nan)
        uv_str_hs = np.where(uvai_valid, (0.51 * uvai_vals + 2.2 + H_UVAI_STRATO_ADD) * 1000, np.nan)
        # Update memory flags
        passed_africa    |= in_africa_now
        passed_dustbelt  |= in_dustbelt_now
        passed_duststorm |= in_duststorm_now
        dust_influenced   = passed_africa | passed_dustbelt | passed_duststorm
    
        
        # Initialize classification array
        class_val = np.zeros((N_LEVELS, N_PARTICLES), dtype=np.int8)
    
        # 1️⃣ Stratospheric smoke
        # Fresh detection: strict threshold
        smoke_was_in_strat |= in_strat_now &  (uvai_max_vals > UVAI_STRAT_THRESHOLD) & (height_particle_msl <=  uv_str_hs)
       
        mask_strat         = in_strat_now  &  (uvai_max_vals > UVAI_STRAT_THRESHOLD) & (height_particle_msl <=  uv_str_hs)
        mask_post_strat    = (~mask_strat) & smoke_was_in_strat &  (uvai_max_vals > UVAI_STRAT_POST_THRESHOLD) #& (height_particle_msl <=  uv_max_hs + 12) #& (height_particle_msl<= 15) 
        
        class_val[mask_strat] = 2
        class_val[mask_post_strat] = 3
        class_counts['strat_smoke'] += np.sum(mask_strat)
        class_counts['strat_post_smoke'] += np.sum(mask_post_strat)

        
    
        # 2️⃣ Dust/Africa influence
        # Classification (mutually exclusive)
        mask_dust_africa = passed_africa & in_africa_now & (class_val == 0)
        mask_dustbelt    = passed_dustbelt & in_dustbelt_now & (class_val == 0)
        mask_duststorm   = passed_duststorm & in_duststorm_now & (class_val == 0)
        
        class_val[mask_dust_africa]  = 4
        class_val[mask_dustbelt]     = 5
        class_val[mask_duststorm]    = 6
        class_counts['dust'] += np.sum(mask_dust_africa)
        class_counts['duststorm'] += np.sum(mask_duststorm)
        class_counts['contaminated_dust'] += np.sum(mask_dustbelt)
    
        # 3️⃣ Tropospheric smoke (never passed dust/africa)
        # have to exclude aged plumes from tropospheric smoke detection !!
        # FRESH tropospheric 
        mask_tropo     = (~dust_influenced) & (~in_strat_now) & (class_val == 0)
        mask_smoke_frp = mask_tropo & (frp_vals >= FRP_THRESHOLD) & (height_particle_msl <=  uv_max_hs)
        mask_smoke_uv  = mask_tropo & (height_particle_msl >= uv_min_hs) & (height_particle_msl<= uv_max_hs) #
        mask_smoke     = mask_smoke_frp | mask_smoke_uv
        
        
        # AGED tropospheric ONLY for UTLS smoke with fresh tropo classified previously
        smoke_was_in_tropo |= mask_smoke
        mask_post_utls      = (~dust_influenced) & (in_utls_now) & smoke_was_in_tropo &  (uvai_max_vals > UVAI_STRAT_POST_THRESHOLD)
        
        
        class_val[mask_smoke] = 1
        class_val[mask_post_utls] = 3
        class_counts['smoke'] += np.sum(mask_smoke)
        class_counts['smoke_post_utls'] += np.sum(mask_post_utls)
    
        # Remaining particles = other
        mask_other = class_val == 0
        class_counts['other'] += np.sum(mask_other)
    
        # Store results
        class_arr[idata] = class_val




    # Print classification summary
    print(f"  🔍 Classification summary for {dt.strftime('%Y%m%d_%H')}:")
    for cls, count in class_counts.items():
        print(f"    {cls}: {count}")
    
    # Update progress
    duration = time.time() - start_time
    print(f"✅ Completed arrival time: {dt.strftime('%Y-%m-%d %H:%M')} in {duration:.1f} seconds")
    tracker.update()
    
    return {
        'class': class_arr,
        'height': height_arr,
        'pbl': pbl_arr,
        'tropo': tropo_arr,
        'temp': temp_arr,
        'lon': lon_arr,
        'lat': lat_arr,
        'times': part_times,
        'arrival_time': dt,
        'index': it
    }

def process_single_day(current_date, config):
    # Process a single day and return the dataset
    print(f"📅 Processing date: {current_date.strftime('%Y-%m-%d')}")
    
    # Generate time list for this day only
    dt_range = (current_date, current_date + datetime.timedelta(hours=23))#, minutes=59
    dt_list = funct.time_list(dt_range[0], dt_range[1], config['time']['step'])
    n_end_time = len(dt_list)
    
    print(f"   📊 Processing {n_end_time} arrival times")
    
    # Preallocate global arrays for this day only
    global_arrays = {
        'class':  np.empty((N_DATA, N_LEVELS, N_PARTICLES, n_end_time), dtype=np.int8),
        'height': np.empty((N_DATA, N_LEVELS, N_PARTICLES, n_end_time), dtype=np.float32),
        'pbl':    np.empty((N_DATA, N_LEVELS, N_PARTICLES, n_end_time), dtype=np.float32),
        'tropo':  np.empty((N_DATA, N_LEVELS, N_PARTICLES, n_end_time), dtype=np.float32),
        'temp':   np.empty((N_DATA, N_LEVELS, N_PARTICLES, n_end_time), dtype=np.float32),
        'lon':    np.empty((N_DATA, N_LEVELS, N_PARTICLES, n_end_time), dtype=np.float32),
        'lat':    np.empty((N_DATA, N_LEVELS, N_PARTICLES, n_end_time), dtype=np.float32),
    }
    
    time_data = []
    traj_end = []
    
    # Setup progress tracking for this day
    tracker = ProgressTracker(n_end_time)
    
    # Parallel processing for this day
    args = [(dt, config, i, tracker) for i, dt in enumerate(dt_list)]
    
    with Pool(processes=cpu_count()) as pool:
        results = []
        for res in tqdm(pool.imap(process_arrival_time, args), total=len(args), desc=f"Processing {current_date.strftime('%Y-%m-%d')}"):
            results.append(res)
    
    # Aggregate results for this day
    print("🔁 Aggregating results...")
    for res in results:
        idx = res['index']
        for key in global_arrays:
            global_arrays[key][..., idx] = res[key]
        time_data.append(res['times'])
        traj_end.append(res['arrival_time'])
    
    # Create netCDF dataset for this day
    print("💾 Creating dataset...")
    ds = xr.Dataset(
        {
            'type': (('position', 'level', 'particle', 'traj_end_time'), global_arrays['class'],
                {'long_name': "Particle Type", 'units': "", 
                 'description': "0=non-absorbing, 1=smoke, 2=stratospheric smoke, 4=dust, 5=smoke-contaminated dust, 6= dust storm"}),
            'altitude': (('position', 'level', 'particle', 'traj_end_time'), global_arrays['height'],
                {'long_name': "Altitude", 'units': "m"}),
            'longitude': (('position', 'level', 'particle', 'traj_end_time'), global_arrays['lon'],
                {'long_name': "Longitude", 'units': "°"}),
            'latitude': (('position', 'level', 'particle', 'traj_end_time'), global_arrays['lat'],
                {'long_name': "Latitude", 'units': "°"}),
            'pbl': (('position', 'level', 'particle', 'traj_end_time'), global_arrays['pbl'],
                {'long_name': "PBL Height", 'units': "m"}),
            'tropo': (('position', 'level', 'particle', 'traj_end_time'), global_arrays['tropo'],
                {'long_name': "Tropopause Height", 'units': "m"}),
            'temp': (('position', 'level', 'particle', 'traj_end_time'), global_arrays['temp'],
                {'long_name': "Temperature", 'units': "K"}),
            'time': (('traj_end_time', 'position'), 
                     np.array([[mdates.date2num(t) for t in times] for times in time_data], dtype=np.float32),
                     {'long_name': "Particle position times", 'units': "days since 0001-01-01"})
        },
        coords={
            'position': np.arange(N_DATA),
            'level': np.arange(N_LEVELS),
            'particle': np.arange(N_PARTICLES),
            'traj_end_time': ('traj_end_time', [mdates.date2num(t) for t in traj_end],
                             {'long_name': 'Arrival time'})
        }
    )
    
    # Global attributes
    ds.attrs = {
        'title': f'FLEXPART Particle Classification - {current_date.strftime("%Y-%m-%d")}',
        'institution': 'University of Leipzig',
        'source': 'FLEXPART v10.4',
        'history': f'Created {datetime.datetime.now()}',
        'author': 'Johanna Roschke',
        'contact': 'johanna.roschke@uni-leipzig.de',
        'processing_time': f"{time.time() - tracker.start_time:.1f} seconds",
        'version': f"Tropospheric smoke height: H = 0.51 UVAI+ 2.2 (+{H_UVAI_TROPO_ADD}km/-{H_UVAI_TROPO_SUB}km), Stratospheric smoke height: H = 0.51 UVAI+ 2.2 (+{H_UVAI_STRATO_ADD}km/,  MODIS {MODIS_DEGREE} degree grid, MODIS FRP THRESH = {FRP_THRESHOLD} , MODIS MAJOR FIRE THRESHOLD {FRP_MAJOR_THRESHOLD}, UVAI CLOUDS < {UVAI_THRESHOLD}, UVAI stratospheric smoke > {UVAI_STRAT_THRESHOLD}, UVAI aged stratospheric smoke > {UVAI_STRAT_POST_THRESHOLD}",
        
    }
    
    return ds

def main():
    # Load configuration
    with open(config_file) as f:
        config = toml.load(f)
    
    # Date range setup - now for multiple days
    start_date = datetime.datetime(2023, 5, 14)
    end_date   = datetime.datetime(2023, 7, 20)  # Example: 7 days
    current_date = start_date
    # We need the memory efect also for tropospheric smoke logic, otherwise we 
    # have tropospheric smoke identification over the atlantic and too low altitudes.. which eventually increase SOF within the PBL
    # Process each day individually
    while current_date <= end_date:
        day_start_time = time.time()
        
        try:
            # Process single day
            ds = process_single_day(current_date, config)
            
            # Save to netCDF for this day
            date_str = current_date.strftime("%Y%m%d")
            station = config["station"]["short_name"]
            output_path = config["output_dir"] +f"/{date_str}_trails_{station}.nc"
            
            print(f"Saving results to {output_path}")
            ds.to_netcdf(output_path, format='NETCDF4')
            
            day_processing_time = time.time() - day_start_time
            print(f"✅ Finished {current_date.strftime('%Y-%m-%d')} in {datetime.timedelta(seconds=int(day_processing_time))}")
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ Error processing {current_date.strftime('%Y-%m-%d')}: {e}")
            # Continue with next day even if this one fails
            continue
        
        finally:
            # Move to next day
            current_date += datetime.timedelta(days=1)
    
    print("🎉 All days processed successfully!")

if __name__ == '__main__':
    main()