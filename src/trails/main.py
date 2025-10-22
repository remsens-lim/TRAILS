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
# Configuration
# CORRECT PATH: Go up 2 levels to project root, then into config folder
script_dir = Path(__file__).parent  # This is /.../TRAILS/src/trails/
project_root = script_dir.parent.parent  # Go up 2 levels to /.../TRAILS/
config_file = project_root / "config" / "config_leipzig.toml"

with open(config_file) as f:
    config = toml.load(f)


# Constants
AFRICA_LAT_MIN, AFRICA_LAT_MAX = -35, 37
AFRICA_LON_MIN, AFRICA_LON_MAX = -20, 55

DUST_LAT_MIN, DUST_LAT_MAX = -30, 30
DUST_LON_MIN, DUST_LON_MAX = -180, 180


N_DATA      = config["time"]["tr_duration"]/config["time"]["step"]
N_LEVELS    = config["height"]["top"]/config["flexpart"]["no_particles"]
N_PARTICLES = config["flexpart"]["no_particles"]

# ADDED OR SUBSTRACTED TO Smoke height regression 
H_UVAI_TROPO_ADD = 2
H_UVAI_TROPO_SUB = 1

# mask clouds in uvai grid
UVAI_THRESHOLD = 0.7

# Stratospheric smoke uvai thresholds
UVAI_STRAT_THRESHOLD = 10
UVAI_STRAT_POST_THRESHOLD = 3

# dust uvai maximum threshold and maximum plume height
UVAI_DUST = 5
H_DUST = 6000  # Height threshold for dust (in meters)

# MODIS FRP thresholds
MODIS_DEGREE = 1
FRP_THRESHOLD = 50
FRP_MAJOR_THRESHOLD = 100


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
    traj_meta = funct.read_flexpart_traj_meta(folder + "trajectories.txt")
    
    # Precompute particle times
    part_times = [dt - datetime.timedelta(hours=3*i) for i in reversed(range(N_DATA))]
    #print(part_times)
    # Create UVAI map cache
    uvai_cache = {}
    uvai_max_cache = {}
    frp_cache = {}
    
    
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
    #print(folder, files[0], files[-1])
    #print(part_times[0], part_times[-1])
    
    ### LAND SURFACE DATA ###
    lc_lon, lc_lat, lc_map = funct.load_LC_map(config)
    print("Loaded Land Cover data", lc_map.shape)
    
    # Load and process all files
    for i, f in enumerate(files):
        
        # Get ACTUAL particle time (not arrival time)
        particle_time = part_times[i]
        date_key = particle_time.strftime("%Y%m%d")
        
        # Load UVAI for partposition 
        #uvai_map =  load_uvai_map(particle_time)
        #frp_map  =  load_frp_map(particle_time)
        
        
        uvai_cache[date_key] = funct.load_uvai_map(config, particle_time, which="Mean")   
        uvai_map = uvai_cache[date_key]
        
        uvai_max_cache[date_key] = funct.load_uvai_map(config,particle_time, which = "Max")   
        uvai_max_map = uvai_max_cache[date_key]

        frp_cache[date_key] = funct.load_frp_map(config, particle_time)
        frp_map = frp_cache[date_key]
        
        
        
        #if i % 10 == 0:
        #print(f"  📁 Processing file {i+1}/{len(files)} for {particle_time.strftime('%Y%m%d')} at {particle_time.strftime('%H')} UTC")
        print("Folder: " , folder ,"position", i,  "Load particle data" , f )
        print(f"  🌍 Loaded UVAI for {date_key}")
        print(f"  🔥 Loaded FRP for {date_key}, {frp_map.shape}")
        print("___________________________________")
        #part_pos = read_partpositions(folder + f, 1, ctable=True)
        #part_pos = np.array(part_pos)
        #part_pos_reshaped = part_pos.reshape((N_LEVELS, N_PARTICLES, -1))
        #print("from: " , part_pos.shape, "shaped to:" , part_pos_reshaped.shape)
        
        # Assuming part_pos is a bcolz ctable returned from read_partpositions
        part_pos = funct.read_partpositions(folder + f, 1, ctable=True)
        
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
    class_counts    = {'dust': 0, 'smoke': 0, 'strat_smoke': 0, 'contaminated_dust': 0,'duststorm': 0, 'other': 0}
    total_particles = N_DATA * N_LEVELS * N_PARTICLES

    print(f"Starting classification of {total_particles:,} particles...")

    for idata in range(N_DATA):
        particle_time = part_times[idata]
        date_key = particle_time.strftime("%Y%m%d")
        
        uvai_map     = uvai_cache[date_key]
        uvai_max_map = uvai_max_cache[date_key]
        frp_map  = frp_cache[date_key]
    
        # Get UVAI and FRP for all particles at this timestep
        uvai_vals     = funct.get_uvai_for_coords(uvai_map, lat_arr[idata], lon_arr[idata])
        uvai_max_vals = funct.get_uvai_for_coords(uvai_max_map, lat_arr[idata], lon_arr[idata])
        #print("Fraction of particles with UVAI>1:", np.sum(uvai_vals>1)/(N_LEVELS*N_PARTICLES),  np.sum(uvai_max_vals>1)/(N_LEVELS*N_PARTICLES))

        frp_vals  = funct.get_frp_for_coords(frp_map, lat_arr[idata], lon_arr[idata])
        lc_vals   = funct.get_LCcat_for_coords(lc_map, lc_lon, lc_lat, lat_arr[idata], lon_arr[idata])
        # Mask invalid UVAI (<1 or NaN)
        uvai_valid = (~np.isnan(uvai_vals)) & (uvai_vals > UVAI_THRESHOLD)
        
        # Compute boolean masks for regions and stratosphere
        in_africa_now   = (lat_arr[idata] >= AFRICA_LAT_MIN) & (lat_arr[idata] <= AFRICA_LAT_MAX) & \
                          (lon_arr[idata] >= AFRICA_LON_MIN) & (lon_arr[idata] <= AFRICA_LON_MAX) & (uvai_vals < UVAI_DUST) & (uvai_vals > UVAI_THRESHOLD) & (height_arr[idata] < H_DUST)  
        in_dustbelt_now = (~in_africa_now) & (lat_arr[idata] >= DUST_LAT_MIN) & (lat_arr[idata] <= DUST_LAT_MAX) & \
                          (lon_arr[idata] >= DUST_LON_MIN) & (lon_arr[idata] <= DUST_LON_MAX)  & (uvai_vals < UVAI_DUST) & (uvai_vals > UVAI_THRESHOLD) & (height_arr[idata] < H_DUST)  
        in_strat_now     = height_arr[idata] > tropo_arr[idata]
        in_duststorm_now = (~in_africa_now) & (~in_dustbelt_now) & (lc_vals == 16) & (uvai_vals < UVAI_DUST) & (uvai_vals > UVAI_THRESHOLD) & (height_arr[idata] < H_DUST)  # Desert & low UVAI & low altitude
        
        # Compute smoke height thresholds
        #uv_max_hs = (0.51 * uvai_vals + 2.2 + 2) * 1000
        #uv_min_hs = (0.51 * uvai_vals + 2.2 - 2) * 1000
        # Compute smoke height thresholds only for valid UVAI
        uv_max_hs = np.where(uvai_valid, (0.51 * uvai_vals + 2.2 + H_UVAI_TROPO_ADD) * 1000, np.nan)
        uv_min_hs = np.where(uvai_valid, (0.51 * uvai_vals + 2.2 - H_UVAI_TROPO_SUB) * 1000, np.nan)

        # Update memory flags
        passed_africa    |= in_africa_now
        passed_dustbelt  |= in_dustbelt_now
        passed_duststorm |= in_duststorm_now
        dust_influenced   = passed_africa | passed_dustbelt | passed_duststorm
    
        
        # Initialize classification array
        class_val = np.zeros((N_LEVELS, N_PARTICLES), dtype=np.int8)
    
        # Stratospheric smoke
        # Fresh detection: strict threshold
        smoke_was_in_strat |= in_strat_now & (uvai_max_vals > UVAI_STRAT_THRESHOLD)
        
        mask_strat = in_strat_now & (uvai_max_vals > UVAI_STRAT_THRESHOLD)
        mask_post_strat = (~mask_strat) & smoke_was_in_strat &  (uvai_max_vals > UVAI_STRAT_POST_THRESHOLD) 
        class_val[mask_strat] = 2
        class_val[mask_post_strat] = 3
        class_counts['strat_smoke'] += np.sum(mask_strat)

        
    
        # Dust/Africa influence
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
    
        # Tropospheric smoke (never passed dust/africa)
        mask_tropo = (~dust_influenced) & (~in_strat_now) & (class_val == 0)
        mask_smoke_frp = mask_tropo & (frp_vals >= FRP_THRESHOLD) & (height_arr[idata] <=  uv_max_hs)
        mask_smoke_uv  = mask_tropo & (height_arr[idata] >= uv_min_hs) & (height_arr[idata] <= uv_max_hs) #
        mask_smoke = mask_smoke_frp | mask_smoke_uv
        class_val[mask_smoke] = 1
        class_counts['smoke'] += np.sum(mask_smoke)
    
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
    print(f"Completed arrival time: {dt.strftime('%Y-%m-%d %H:%M')} in {duration:.1f} seconds")
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
    print(f"Processing date: {current_date.strftime('%Y-%m-%d')}")
    
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
        
    }
    
    return ds

def main():
    # Load configuration
    with open(config_file) as f:
        config = toml.load(f)
    
     # Get dates from config
    start_date = datetime.datetime.fromisoformat(config["time"]["begin"])
    end_date = datetime.datetime.fromisoformat(config["time"]["end"])
    
    current_date = start_date
    
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
            print(f"Finished {current_date.strftime('%Y-%m-%d')} in {datetime.timedelta(seconds=int(day_processing_time))}")

            
        except Exception as e:
            print(f"Error processing {current_date.strftime('%Y-%m-%d')}: {e}")
            # Continue with next day even if this one fails
            continue
        
        finally:
            # Move to next day
            current_date += datetime.timedelta(days=1)
    
    print("All days processed successfully!")

if __name__ == '__main__':
    main()