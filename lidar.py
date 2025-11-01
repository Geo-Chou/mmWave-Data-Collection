"""
Ouster LiDAR sensor integration module for ISAC project.

This module provides functionality to:
- Configure and connect to Ouster LiDAR sensors
- Receive and process LiDAR scan data with SLAM
- Real-time visualization of range maps
"""

import os
import time
import json
import jsonlines
import numpy as np
import matplotlib.pyplot as plt
from contextlib import closing
from multiprocessing import Queue
from more_itertools import time_limited
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

import open3d as o3d

from ouster.sdk import core, sensor
from ouster.sdk.mapping import SlamConfig, SlamEngine
from ouster.sdk.client import ChanField
from ouster.sdk.client import XYZLut


class OusterConfig:
    """
    Configuration class for the Ouster LiDAR sensor.
    """
    # Network configuration
    HOST_NAME = "os-122210002400.local"  # Ouster sensor hostname
    LIDAR_PORT = 7502                     # UDP port for LiDAR data
    IMU_PORT = 7503                       # UDP port for IMU data
    
    # Sensor operation settings
    OPERATING_MODE = core.OperatingMode.OPERATING_NORMAL  # Standard operating mode
    LIDAR_MODE = core.LidarMode.MODE_1024x10             # 1024 columns, 10Hz rotation

    # Load predefined beam angle configurations from JSON file
    with open(os.path.join("./DEFINED", "OUSTER_BEAM.json"), "r", encoding="utf-8") as f:
        OUSTER_BEAM = json.load(f)

    # Extract beam angle arrays for azimuth (horizontal) and altitude (vertical) directions
    BEAM_AZIMUTH_ANGLES = np.array(OUSTER_BEAM["BEAM_AZIMUTH_ANGLES"])
    BEAM_ALTITUDE_ANGLES = np.array(OUSTER_BEAM["BEAM_ALTITUDE_ANGLES"])

    # Range processing parameters
    RANGE_SCALE = 1000  # Scale factor to convert range to meters (from mm)
    
    # Define field of view limits for data filtering
    # Azimuth: 120° to 240° (rear-facing 120° sector)
    AZIMUTH_ANGLE_START = np.where(BEAM_AZIMUTH_ANGLES >= 120)[0][0]
    AZIMUTH_ANGLE_END = np.where(BEAM_AZIMUTH_ANGLES <= 240)[0][-1]
    
    # Altitude: -30° to +30° (±30° vertical field of view)
    ALTITUDE_ANGLE_START = np.where(BEAM_ALTITUDE_ANGLES <= 30)[0][0]
    ALTITUDE_ANGLE_END = np.where(BEAM_ALTITUDE_ANGLES >= -30)[0][-1]

    # SLAM (Simultaneous Localization and Mapping) configuration
    SLAM_RANGE_MIN = 0.5    # Minimum detection range in meters
    SLAM_RANGE_MAX = 20.0   # Maximum detection range in meters  
    SLAM_VOXEL_SIZE = 0.2   # Voxel size for point cloud processing
    SLAM_BACKEND = "kiss"   # SLAM algorithm backend
    

    @classmethod
    def set_sensor_config(cls) -> None:
        """
        Configure the Ouster sensor with predefined settings.
        """
        config = core.SensorConfig()
        config.operating_mode = cls.OPERATING_MODE
        config.lidar_mode = cls.LIDAR_MODE
        config.udp_port_lidar = cls.LIDAR_PORT
        config.udp_port_imu = cls.IMU_PORT

        sensor.set_config(cls.HOST_NAME, config, persist=True, udp_dest_auto=True)

    @classmethod
    def connect_sensor(cls) -> sensor.SensorScanSource:
        """
        Establish connection to the Ouster sensor and return a scan source.
        """
        return sensor.SensorScanSource(
            cls.HOST_NAME, 
            lidar_port=cls.LIDAR_PORT, 
            imu_port=cls.IMU_PORT
        )
    
    @classmethod
    def get_slam_config(cls) -> SlamConfig:
        """
        Create and return SLAM configuration with predefined parameters.
        """
        config = SlamConfig()
        config.min_range = cls.SLAM_RANGE_MIN
        config.max_range = cls.SLAM_RANGE_MAX
        config.voxel_size = cls.SLAM_VOXEL_SIZE
        config.backend = cls.SLAM_BACKEND
        return config


def ouster_receiver(shared_ouster_queue: Queue, RECORD_DURATION: int, stop_evt) -> None:
    """
    Receive LiDAR scan data from Ouster sensor and process with SLAM.
    
    This function runs as a separate process/thread to:
    1. Configure and connect to the Ouster sensor
    2. Initialize SLAM engine for pose estimation
    3. Continuously receive scan data for the specified duration
    4. Process each scan to extract range maps and poses
    5. Put processed data into shared queue for further processing
    """
    try:
        # Configure sensor with predefined settings
        OusterConfig.set_sensor_config()
        frames = 0
        
        # Connect to sensor and record lidar/imu packets
        with closing(OusterConfig.connect_sensor()) as stream:
            print("Ouster Receiver: Connected to Ouster sensor.")
            
            # Get sensor metadata and initialize SLAM engine
            metadata = stream.sensor_info[0]
            xyz_lut = XYZLut(metadata)
            slam = SlamEngine(stream.sensor_info, OusterConfig.get_slam_config())
            
            # Process scans for the specified duration
            for scan, *_ in time_limited(RECORD_DURATION, stream):
                if scan is None:
                    continue
                
                # Generate timestamp for data synchronization (centiseconds precision)
                timestamp = time.time_ns()
                
                # Update SLAM with current scan to get pose estimate
                scans_w_poses = slam.update([scan])
                if scans_w_poses is None:
                    continue

                # Extract pose information from the first valid column
                col = scans_w_poses[0].get_first_valid_column()
                scan_pose = scans_w_poses[0].pose[col].astype(np.float64)
                ranges = scan.field(core.ChanField.RANGE)
                intensity = scan.field(core.ChanField.SIGNAL)
                intensity = intensity.reshape(-1, 1).astype(np.float32)
                # Process range data: destagger and extract region of interest
                # range_destaggered = core.destagger(metadata, scan.field(core.ChanField.RANGE))
                # range_map = range_destaggered[
                #     OusterConfig.ALTITUDE_ANGLE_START:OusterConfig.ALTITUDE_ANGLE_END + 1,
                #     OusterConfig.AZIMUTH_ANGLE_START:OusterConfig.AZIMUTH_ANGLE_END + 1
                # ] / OusterConfig.RANGE_SCALE  # Convert to meters
                xyz = xyz_lut(ranges)   # 单位：米
                xyz = xyz.astype(np.float32)

                points = xyz.reshape(-1, 3)

                # Send processed data to queue for saving/visualization
                # shared_ouster_queue.put((timestamp, scan_pose, points))
                shared_ouster_queue.put((timestamp, scan_pose, points, intensity))
                frames += 1
                

                # Progress reporting every 100 frames
                if frames % 100 == 0:
                    print(f"Ouster Receiver: Captured {frames} scans")

            print(f"Ouster Receiver: Finished. {frames} scans received.")
            stop_evt.set()
    
    finally:
        # Signal termination to consumer processes
        shared_ouster_queue.put(None)
        print("Ouster Receiver: Connection closed")


def ouster_processor(shared_ouster_queue: Queue, plot_ouster_queue: Queue, OUTPUT_PATH: str, stop_evt) -> None:
    """
    Process and save LiDAR data received from the ouster_receiver.
    
    This function runs as a separate process to:
    1. Consume processed scan data from the shared queue
    2. Save range map data as individual .npy files
    3. Forward range maps to visualization queue
    4. Generate pose trajectory data as a .jsonl file
    5. Use multithreading for efficient file I/O operations
    """
    # Create output directory for scan files
    SCAN_PATH = os.path.join(OUTPUT_PATH, "lidar_points")
    os.makedirs(SCAN_PATH, exist_ok=True)

    pose_trajectory = []
    threads = []
    
    # Use thread pool for efficient parallel file I/O
    with ThreadPoolExecutor(max_workers=4) as executor:

        while True:
            # Block until an item is available in the queue
            ouster_package = shared_ouster_queue.get()
            
            # Check for termination signal
            if stop_evt.is_set():
                print("Ouster Processor: Termination signal detected.")
                break




            timestamp, scan_pose, points, intensity = ouster_package
            # pose_trajectory.append((timestamp, pose_matrix))

            ranges = np.linalg.norm(points, axis=1)
            mask = (ranges>=0.5) & (ranges <= 10.0)


            azimuth = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
            mask &= (azimuth >= -60) & (azimuth <= 60)


            altitude = np.degrees(np.arcsin(points[:, 2] / (ranges + 1e-6)))
            mask &= (altitude >= -45) & (altitude <= 45)

            
            # range_map = range_map.astype(np.float32)
            points = points[mask]
            intensity = intensity[mask] 
            
            # Forward range map to visualization queue
            plot_ouster_queue.put((points, intensity))
            pose_trajectory.append((timestamp, scan_pose))
            # Prepare file path and submit save task to thread pool
            SAVE_PATH = os.path.join(SCAN_PATH, f"{timestamp}.npy")
            thread = executor.submit(_save_processed_data, np.hstack((points, intensity)), SAVE_PATH)
            threads.append(thread)

        # Wait for all file saving operations to complete
        wait(threads, return_when=ALL_COMPLETED)
    
    # Save pose trajectory data to JSONL format
    with jsonlines.open(os.path.join(OUTPUT_PATH, "pose.jsonl"), mode="w") as f:
        for timestamp, pose_matrix in pose_trajectory:
            pose_list = pose_matrix.flatten().tolist()
            f.write({
                "timestamp": timestamp,
                "pose": pose_list
            })

    print(f"Ouster Processor: {len(threads)} frames saved.")


def _save_processed_data(range_map: np.ndarray, SAVE_PATH: str) -> None:
    """
    Save processed LiDAR range map data to a numpy file.
    """
    np.save(SAVE_PATH, range_map, allow_pickle=True)


def ouster_visualizer(plot_ouster_queue: Queue) -> None:
    """
    Real-time visualization of LiDAR range maps using matplotlib.
    
    This function creates and maintains a live heatmap display showing:
    - Distance measurements as color-coded pixels
    - Azimuth angles (horizontal axis) vs Altitude angles (vertical axis)
    - Real-time updates as new scan data arrives
    """
    # # Visualization parameters
    # VISUALIZE_DISTANCE_MIN = 0.5  # Minimum distance for color mapping (meters)
    # VISUALIZE_DISTANCE_MAX = 6.0  # Maximum distance for color mapping (meters)
    
    # # Set up interactive matplotlib plotting
    # plt.ion()  # Enable interactive mode for real-time updates
    # fig, ax = plt.subplots(figsize=(8, 4))
    
    # # Configure colormap with special handling for out-of-range values
    # cmap = plt.get_cmap('viridis_r').copy()  # Reverse viridis (dark=far, bright=close)
    # cmap.set_under('black')  # Values below min_range appear black
    # cmap.set_over('black')   # Values above max_range appear black
    
    # # Calculate expected image dimensions from configuration
    # h = OusterConfig.AZIMUTH_ANGLE_END - OusterConfig.AZIMUTH_ANGLE_START + 1
    # v = OusterConfig.ALTITUDE_ANGLE_END - OusterConfig.ALTITUDE_ANGLE_START + 1
    # placeholder_data = np.full((v, h), -1.0)  # Initialize with invalid values
    
    # # Extract angle extents for proper axis scaling
    # h_angle_start = float(OusterConfig.BEAM_AZIMUTH_ANGLES[OusterConfig.AZIMUTH_ANGLE_START])
    # h_angle_end = float(OusterConfig.BEAM_AZIMUTH_ANGLES[OusterConfig.AZIMUTH_ANGLE_END])
    # v_angle_start = float(OusterConfig.BEAM_ALTITUDE_ANGLES[OusterConfig.ALTITUDE_ANGLE_START])
    # v_angle_end = float(OusterConfig.BEAM_ALTITUDE_ANGLES[OusterConfig.ALTITUDE_ANGLE_END])

    # # Create initial image with proper scaling and labels
    # im = ax.imshow(
    #     placeholder_data, 
    #     cmap=cmap, 
    #     aspect='auto',           # Auto-adjust aspect ratio
    #     origin='upper',          # Origin at top-left
    #     extent=(h_angle_start, h_angle_end, v_angle_start, v_angle_end),  # Real angle coordinates
    #     vmin=VISUALIZE_DISTANCE_MIN, 
    #     vmax=VISUALIZE_DISTANCE_MAX
    # )
    
    # # Add color bar and labels
    # cbar = fig.colorbar(im, ax=ax)
    # cbar.set_label('Distance (meters)')
    # ax.set_title('Real-time Ouster LiDAR Range Map')
    # ax.set_xlabel('Azimuth Angle (Degrees)')
    # ax.set_ylabel('Altitude Angle (Degrees)')
    
    # fig.tight_layout()
    # fig.show()

    # Main visualization loop
    # pcd = o3d.geometry.PointCloud()
    vis = o3d.visualization.Visualizer()
    vis.create_window("Live 3D Point Cloud Update", width=2048, height=960)  # Set window size
    pcd = o3d.geometry.PointCloud()
    plot_index = 0
    while True:
        if plot_index == 0:
            # Get next range map from queue (blocks until available)
            points = plot_ouster_queue.get()
            pcd.points = o3d.utility.Vector3dVector(points)
            vis.add_geometry(pcd)
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])    
            vis.add_geometry(axis)
            view_control = vis.get_view_control()
            view_control.set_lookat([0, 0, 0])  # Adjust center
            view_control.set_front([-1, 0, 0])  # Adjust camera direction
            view_control.set_up([0, 0, 1])  # Adjust upward direction
            view_control.set_zoom(0.1)  # Adjust zoom to fit the scene properly
            # o3d.visualization.draw_geometries([pcd])
        else: 
            points = plot_ouster_queue.get()
            if points is None:
                print("Ouster Visualizer: Termination signal detected.")
                break
            pcd.points = o3d.utility.Vector3dVector(points)
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
        plot_index = plot_index + 1
        if plot_index%10 == 0:
            print(f"PLOT: plot_index: {plot_index}")
            # # Check for termination signal
            # if range_map is None:
            #     print("Ouster Visualizer: Termination signal detected.")
            #     break
                
            # # Update image data and refresh display
            # im.set_data(range_map)
            # fig.canvas.draw()           # Redraw the figure
            # fig.canvas.flush_events()   # Process GUI events
            # plt.pause(0.01)             # Small pause for smooth animation                

    # Cleanup: disable interactive mode and close window
    # plt.ioff()
    # plt.close(fig)
    # print("Ouster Visualizer: Plot window closed.")



# if __name__ == "__main__":
#     from multiprocessing import Process, Queue, set_start_method

#     try:
#         set_start_method('spawn')  # Use 'spawn' for compatibility across platforms
#     except RuntimeError:
#         pass  # Start method has already been set

#     RECORD_DURATION = 3000  # Duration to record data in seconds
#     OUTPUT_PATH = "./data"  # Directory to save output data
#     os.makedirs(OUTPUT_PATH, exist_ok=True)

#     # Create shared queues for inter-process communication
#     shared_ouster_queue = Queue(maxsize=10)  # Queue for raw data from receiver to processor
#     plot_ouster_queue = Queue(maxsize=10)    # Queue for processed data to visualizer

#     # Initialize and start processes
#     receiver_process = Process(target=ouster_receiver, args=(shared_ouster_queue, RECORD_DURATION))
#     processor_process = Process(target=ouster_processor, args=(shared_ouster_queue, plot_ouster_queue, OUTPUT_PATH))
#     visualizer_process = Process(target=ouster_visualizer, args=(plot_ouster_queue,))

#     receiver_process.start()
#     processor_process.start()
#     visualizer_process.start()

#     # Wait for all processes to complete
#     receiver_process.join()
#     processor_process.join()
#     visualizer_process.join()

#     print("Main: All processes have completed.")