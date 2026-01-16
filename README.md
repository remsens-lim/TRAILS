# TRAILS  
### **TRAjectory-based Identification of Lofted Smoke**

**TRAILS** is a Python-based tool for attributing the influence of **wildfire smoke** on air masses.  
It extends the automated air mass source attribution tool of  [Radenz et al. (2021)](https://acp.copernicus.org/articles/21/3015/2021/) . Have a look at the corresponding github repository to run FLEXPART via docker → [trace_airmass_source](https://github.com/martin-rdz/trace_airmass_source).

TRAILS combines:
- **Air mass transport simulations** from the **FLEXPART** particle dispersion model  
- **Satellite observations** of the **OMPS Ultraviolet Aerosol Index (UVAI)**  
- **Active fire data** from **MODIS/VIIRS**

Full documentation and methodology are detailed in the accompanying publication:  
**_[CITATION TO BE ADDED]_**

---

## Overview

**TRAILS** (TRAjectory-based Identification of Lofted Smoke) processes **FLEXPART particle positions** to classify the influence of smoke using **OMPS UVAI** and **MODIS FRP** satellite data.

---

## Quick Start

### Prerequisites

Before running TRAILS, ensure you have:

- **Python 3.8+**
- **FLEXPART particle position data**  
  (see [trace_airmass_source](https://github.com/martin-rdz/trace_airmass_source) for running FLEXPART via Docker)
- **OMPS UVAI** NetCDF files *(see Jupyter notebooks for preprocessing examples)*
- **MODIS FRP** NetCDF files *(see Jupyter notebooks for preprocessing examples)*

---


### Installation

1. **The package can be installed via pip:**
```bash
python -m pip install git+https://github.com/remsens-lim/TRAILS
```
2. **Install the package :**
```bash
python setup.py install
```
3. **Run the main script:**
```bash
python src/trails/main.py
```

### Config files

TRAILS can be configured using TOML files located in `config/station_name.toml`. Examples are provided for station Leipzig. For each station, the configuration file should be updated with the corresponding directories for the input grids. Jupyter notebooks are provided to create OMPS UVAI global grids from swaths, as well as for MODIS/VIIRS

Please update in `main.py` the name of your `.toml` file:
```python
config_file = project_root / "config" / "station_name.toml"
```

#### Directories
```toml
# configuration file for TRAILS

output_dir    = "/path_to_/output/"
plot_dir      = "/path_to_/figures/"
traj_dir      = "/path_to_/flexpart_particle_positions/"
partposit_dir = "/path_to_/flexpart_particle_positions/" 

frp_dir       = "/path_to/MODIS_1Degree_daily_grids/"
omps_dir      = "/path_to/OMPS_UVAI_1Degree_daily_grids/"
LCType_dir    = "/path_to/land_surface_type_global_yearly_grids/"
```
#### Time settings
Define dates to process and time resolution (step). Trajectories are calculated every 3 hours over 10 days (-240 hours backward).
```toml

[time]
    # dates to be processed
    begin = '2023-05-01'
    end   = '2023-07-31'
    # time step for which trajectories are calculated
    step = 3
    # duration of each trajectory
    tr_duration = -240
```
#### FLEXPART settings
500 particles are traced back at each height level, from 500 m up to 12 km, every 500 m.
```toml
[flexpart]
    no_particles = 500
    rel_before_minutes = 5
    rel_after_minutes = 0
    rel_pm_height = 200
    species = 24
    outstep = 3


[height]
    top =  12000
    base = 500
    interval = 500
```

#### Station info 
```toml

[station]
   name = 'Leipzig, Germany'
   short_name = 'leipzig'
   lat = 51.3397
   lon = 12.3731
   altitude = 113 # in meters
```
In ```python main.py``` u can also change thresholds as explained in the manuscript:

```python
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
```
### References 
Radenz, M., Seifert, P., Baars, H., Floutsi, A. A., Yin, Z., and Bühl, J.: Automated time–height-resolved air mass source attribution for profiling remote sensing applications, Atmos. Chem. Phys., 21, 3015–3033, https://doi.org/10.5194/acp-21-3015-2021, 2021.

### Licence
See the [LICENSE file](https://github.com/remsens-lim/TRAILS/blob/main/LICENCE) for more information Copyright 2025, Johanan Roschke [MIT License](https://opensource.org/license/mit) 