# mmWave ISAC point cloud data collection System

This repository contains codes for mmwave point clouds, lidar point clouds and depth camera data collection. For each data modality, there's a script providing specific fucntion/class.

## Project Structure

```
├── collect_cam_radar.py             # collect depth camera & mmwave data
├── depth_camera_rec.py              # mini test for depth camera recording
├── depth_cam_postprocessor.py       # post processing for depth camera data
├── dualcam.py                       # functions related to depth camera
├── lidar.py                         # functions related to lidar
├── main_collect_P_radar_lidar.py    # collect lidar & mmwave data
├── data                             # store collected data (timestamp)
│   ├── 1761012485744354782
│   ├── 1761274904121066263
│   └── .........
├── DEFINED                          # predefined matrix for isac data
│   ├── E.npy
│   ├── OUSTER_BEAM.json
│   └── RANDOM_BITS.npy
├── isac_host                        # isac processing folder
│   ├── ant_calib.bin
│   ├── cpp
│   ├── E_ant_pattern0.npy
│   ├── E_ant_pattern1.npy
│   ├── E_ant_pattern_all.npy
│   ├── E.npy
│   ├── IPAddressFinder.py
│   ├── main_isac_app.py            # mmwave related functions
│   ├── ofdm.py
│   ├── plot_2d.py
│   ├── plot_3d.py
│   ├── __pycache__
│   ├── SpatialProjection.py
│   └── test.py
├── librealsense                    # depth camera(realsense) library
└── README.md
```

## Usage

### Prepare

1. **Connect Devices**
  - Ensure the mmWave device, LiDAR device and depth camera are all physically connected and powered on.

2. **Start mmWave Service**
  - On your host computer, open a browser and access: [http://192.168.2.99:9090/tree?](http://192.168.2.99:9090/tree?)
  - Open a terminal in the web interface and execute:
    ```bash
    cd /home/xilinx/isac_rfsoc/
    ./run.sh
    ```

3. **Check LiDAR Connection**
  - On your host computer, access: [http://os-122210002400.local/](http://os-122210002400.local/)
  - Confirm that the LiDAR device is reachable and operational.

4. **(Optinal) Check RealSense Connection**
  - If the librealsense has compiled following the instructions on the [official website](https://dev.realsenseai.com/docs/compiling-librealsense-for-linux-ubuntu-guide), you can just run **realsense-viewer** to check.
  - If the code is running on Ubuntu(<=24.04) or Windows, the prebuild packages can easily to be installed following the [instructions for Ubuntu](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md) or [for Windows](https://github.com/IntelRealSense/librealsense/releases).


### Basic Usage
```bash
cd data_collection
python main_collect_P_radar_lidar.py $RECORD_DURATION # for lidar and radar
python collect_cam_radar.py $RECORD_DURATION # for depth camera and radar
# if collected depth camera data, then run
python depth_cam_postprocessor.py
```

Where `RECORD_DURATION` is the recording duration in seconds.


### Output Data Structure
```
data/
└── <timestamp>/
    ├── realsense/      # Processed depth camera data
    ├── radar_points/       # processed radar point clouds
    ├── lidar_points/   # processed lidar point clouds
    └── pose.jsonl   # Recording pose trajectory(if using lidar data)
```

## Dependencies

### Installation
Install the required packages using pip:

```bash
pip install jsonlines numpy more-itertools ouster-sdk matplotlib realsense2 open3d pyvista
```

## System Components

### Predefined Data (`DEFINED/`)
- **`E.npy`**: Antenna pattern projection matrices for spatial beamforming
- **`OUSTER_BEAM.json`**: LiDAR beam angle configurations (azimuth/altitude angles)
- **`RANDOM_BITS.npy`**: Predefined random bits for consistent QPSK symbol generation

### Adjustable Parameters
- **`main_collect_P_radar_lidar.py/collect_cam_radar.py`**: `mask = strength > 0.5` can determine the minimum strength of output radar point clouds, which can reduce noise. 
- **`lidar.py`**: `ouster_processor` can adjust the output angle of point clouds, which can restrict saved region.


