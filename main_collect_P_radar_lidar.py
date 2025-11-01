'''
This script collects and saves radar and lidar cloud point data for a specified time of frames.
'''
import itertools
import os
import time
import json
import jsonlines
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
from contextlib import closing
from multiprocessing import Queue, Array, Process, Event, set_start_method
from more_itertools import time_limited
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import argparse

import open3d as o3d

from ouster.sdk import core, sensor
from ouster.sdk.mapping import SlamConfig, SlamEngine
from ouster.sdk.client import ChanField
from ouster.sdk.client import XYZLut

from lidar import OusterConfig,ouster_receiver, ouster_processor

from isac_host.main_isac_app import udp_data_receive, plot_point_cloud
from isac_host.SpatialProjection import SpatialProjection


import open3d as o3d
from open3d.visualization import gui, rendering

SAVE_DIR = './data/'


def process_save_radar_data(shared_udp_queue, shared_radar_queue, stop_evt) -> None:
    num_cir_points = 64  # number of CIR points
    SP0 = SpatialProjection(Ant_Pattern_Scheme=0, R = num_cir_points)
    SP1 = SpatialProjection(Ant_Pattern_Scheme=1, R = num_cir_points)
    SP = SP0
    
    # vis = o3d.visualization.Visualizer()
    # vis.create_window("Live 3D Point Cloud Update", width=2048, height=960)  # Set window size
    # pcd = o3d.geometry.PointCloud()
    # frame_index = 0
    
    # arr = np.frombuffer(shared_esti_cir.get_obj(), dtype=np.float64)
    # shared_complex_arr = arr.view(np.complex128)  # Interpret as complex numbers
    ##############################################
    # process the 3D point cloud
    ##############################################
    W = np.linspace(1, 3, num_cir_points)
    frame_index = 0
    while True:
        if stop_evt.is_set():
            print("Radar termination signal received by Processing. exiting.")
            break
        if shared_udp_queue.empty():
            continue
        timestamp, remote_frame_index, remote_pattern_index, cir_data = shared_udp_queue.get()
        # print(f"Received frame index: {remote_frame_index}, pattern index: {remote_pattern_index}")
        S = cir_data.reshape(64, 128)
        S = S[:, 0:num_cir_points]
        # remote_frame_index = int(shared_complex_arr[0].real)
        # remote_pattern_index = int(shared_complex_arr[0].imag)
        if remote_frame_index == 0:
            time.sleep(0.01)
            # continue
            
        S[:, 0] = 0
        S[:, 1] = 0
        S[:, 2] = 0
        S[:, 3] = 0
        S = S*W

        Ant_Pattern_Scheme = int(remote_pattern_index/10.0)
        pattern_index = int(remote_pattern_index - Ant_Pattern_Scheme*10)

        if Ant_Pattern_Scheme == 0:
            if pattern_index == 0:
                B = SP0.E0 @ S
            elif pattern_index == 1:
                B = SP0.E1 @ S
            elif pattern_index == 2:
                B = SP0.E2 @ S
            else:
                B = SP0.E3 @ S
        else:
            if pattern_index == 0:
                B = SP1.E0 @ S
            elif pattern_index == 1:
                B = SP1.E1 @ S
            elif pattern_index == 2:
                B = SP1.E2 @ S
            else:
                B = SP1.E3 @ S

        # Iterate over theta and phi values
        # Assuming B is already calculated as per your previous steps
        strength = np.abs(B)  
        strength = strength.flatten()  
        strength_normalized = strength/20

        mask = strength > 1 
        x_masked = SP.X[mask] 
        y_masked = SP.Y[mask]
        z_masked = SP.Z[mask]
        strength_normalized_masked = strength_normalized[mask]

        np.savez(os.path.join(SAVE_DIR, 'radar_points', f"{timestamp:06d}.npz"),
             points=np.column_stack((x_masked, y_masked, z_masked)), intensity=strength_normalized_masked,
             frame_idx=remote_frame_index,
             pattern_idx=remote_pattern_index)
        
        if frame_index == 0:
            radar_points = np.column_stack((SP.X, SP.Y, SP.Z))
            colors = None
            shared_radar_queue.put((radar_points, colors))

        else:
            radar_points = np.column_stack((x_masked, y_masked, z_masked))
            colormap = cm.viridis  # You can choose any colormap like 'viridis', 'plasma', etc.
            colors = colormap(strength_normalized_masked)[:, :3]  # Extract RGB values (without alpha channel)
            shared_radar_queue.put((radar_points, colors))
        frame_index = frame_index+1
        if frame_index % 100 == 0:
            print(f"Processed and saved {frame_index} radar frames.")
    

def mat_unlit(size=2.0):
    m = rendering.MaterialRecord()
    m.shader = "defaultUnlit"
    m.point_size = size
    return m


def visualize_radar_data(radar_queue: Queue, stop_evt) -> None:
    '''
    Visualizes  point cloud data.
    '''
    frame_index = 0
    vis = o3d.visualization.Visualizer()
    vis.create_window("Live Radar Point Cloud Update", width=1280, height=960)  # Set window size
    pcd = o3d.geometry.PointCloud()
    while True:
        if stop_evt.is_set():
            print("Radar termination signal received by Visualizer. exiting.")
            break
        if radar_queue.empty():
            continue
        radar_points, colors = radar_queue.get()
        # print(f"Radar Visualizer: Received {radar_points.shape}.")
        pcd.points = o3d.utility.Vector3dVector(radar_points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            pcd.paint_uniform_color([0.5, 0.5, 0.5])
        if frame_index == 0:
            vis.add_geometry(pcd)
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])    
            vis.add_geometry(axis)
            view_control = vis.get_view_control()
            view_control.set_lookat([0, 2, 0])  # Adjust center
            # view_control.set_front([0, 0, -1])  # Adjust camera direction
            view_control.set_up([0, 1, 0])  # Adjust upward direction
            view_control.set_zoom(0.2)  # Adjust zoom to fit the scene properly
            theta = np.radians(30)   # 60
            front = np.array([0, 0, -1])  
            R = o3d.geometry.get_rotation_matrix_from_axis_angle([theta, 0, 0])  # 绕 y 轴
            front_rot = R @ front
            view_control.set_front(front_rot.tolist())
        else: 
            # Create Open3D PointCloud object
            pcd.points = o3d.utility.Vector3dVector(radar_points)
            if not np.array_equal(pcd.colors, colors):
                pcd.colors = o3d.utility.Vector3dVector(colors)

            # Update visualization
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

        frame_index = frame_index + 1

def visualize_lidar_data(lidar_queue: Queue, stop_evt) -> None:
    '''
    Visualizes  point cloud data.
    '''
    vis = o3d.visualization.Visualizer()
    vis.create_window("Live Lidar 3D Point Cloud Update", width=1280, height=960)  # Set window size
    pcd = o3d.geometry.PointCloud()
    plot_index = 0
    while True:
        if stop_evt.is_set():
            print("Lidar termination signal received by Visualizer. exiting.")
            break
        if lidar_queue.empty():
            continue
        points, intensity = lidar_queue.get()
        
        # if intensity is not None:
        #     intensity = np.asarray(intensity, dtype=np.float64).reshape(-1, 1)
        #     intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-8)

        #     intensity = np.repeat(intensity, 3, axis=1)
        #     pcd.colors = o3d.utility.Vector3dVector(intensity)
        # else:
        #     print("No intensity data, using uniform color.")
        #     pcd.paint_uniform_color([0.5, 0.5, 0.5])
        if plot_index == 0:
            # Get next range map from queue (blocks until available)
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
            if points is None:
                print("Ouster Visualizer: Termination signal detected.")
                break
            pcd.points = o3d.utility.Vector3dVector(points)
            # if not np.array_equal(pcd.colors, intensity):
            #     pcd.colors = o3d.utility.Vector3dVector(intensity)
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
        plot_index = plot_index + 1

    


if __name__ == "__main__":
    set_start_method('spawn')
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
    )
    os.makedirs(os.path.join(SAVE_DIR, 'radar_points'),exist_ok=True)
    parser.add_argument("RECORD_DURATION", type=int, help="Duration of the recording in seconds")
    args = parser.parse_args()
    stop_evt_radar = Event()
    stop_evt_lidar = Event()
    

    # shared_esti_cir = Array('d', 64*128*2)  # 'd' is for float64
    shared_udp_queue = Queue(maxsize=10)
    shared_radar_queue = Queue(maxsize=10)

    shared_ouster_queue = Queue(maxsize=10)
    plot_ouster_queue = Queue(maxsize=10)

    
    receiver_process = Process(target=ouster_receiver, args=(shared_ouster_queue, args.RECORD_DURATION, stop_evt_lidar))
    processor_process = Process(target=ouster_processor, args=(shared_ouster_queue, plot_ouster_queue, SAVE_DIR, stop_evt_lidar))
    

    udp_process = Process(target=udp_data_receive, args=(shared_udp_queue,args.RECORD_DURATION, stop_evt_radar)) 
    radar_process = Process(target=process_save_radar_data, args=(shared_udp_queue,shared_radar_queue, stop_evt_radar))

    radar_visualizer_process = Process(target=visualize_radar_data, args=(shared_radar_queue, stop_evt_radar))
    lidar_visualizer_process = Process(target=visualize_lidar_data, args=(plot_ouster_queue, stop_evt_lidar))

    udp_process.start()
    radar_process.start()
    receiver_process.start()
    processor_process.start()
    radar_visualizer_process.start()
    lidar_visualizer_process.start()

    udp_process.join()
    radar_process.join()
    receiver_process.join()
    processor_process.join()
    radar_visualizer_process.join()
    lidar_visualizer_process.join()