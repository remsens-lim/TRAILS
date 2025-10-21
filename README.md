# TRAILS - TRAjectory based Identification of Lofted Smoke

A Python-based tool for attributing the influence of wildfire smoke using airparcel tracer from FLEXPART backward trajectories constrained by OMPS UVAI data.

## 📋 Overview

TRAILS (TRAjectory based Identification of Lofted Smoke) processes FLEXPART particle positions to classify the potential influence of particles by type (smoke, dust, stratospheric smoke, etc.) using satellite observations from OMPS UVAI and MODIS FRP data.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- FLEXPART output data
- OMPS UVAI NetCDF files
- MODIS FRP data

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/TRAILS.git
cd TRAILS

pip install -r requirements.txt

python src/trails/main.py

```

### Configuration

The tool is configured cai config/config_leipzig.toml:
```toml
# Main directories
output_dir = "/path/to/output/"
traj_dir = "/path/to/flexpart/trajectories/"
partposit_dir = "/path/to/particle/positions/"

# Time settings
[time]
begin = '2023-05-01'  # Start date
end = '2023-07-31'    # End date  
step = 3              # Time step in hours
tr_duration = -240    # Trajectory duration in hours

# FLEXPART settings
[flexpart]
no_particles = 500    # Number of particles per release
species = 24          # FLEXPART species ID

# Height settings  
[height]
top = 12000          # Top height in meters
base = 500           # Base height in meters
interval = 500       # Height interval in meters
```

