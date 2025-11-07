'''
This script collects and saves radar and lidar cloud point data for a specified time of frames.
'''
import itertools
import os
import time
import json
import jsonlines
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
from contextlib import closing
from multiprocessing import Queue, Array, Process, Event, set_start_method
from more_itertools import time_limited
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import argparse

import open3d as o3d
import pyvista as pv

from lidar import OusterConfig,ouster_receiver, ouster_processor

from isac_host.main_isac_app import udp_data_receive, plot_point_cloud
from isac_host.SpatialProjection import SpatialProjection


import open3d as o3d
# from open3d.visualization import gui, rendering

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

        #mask = strength > 1.15 
        mask = strength > 2.5
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
            colors = strength_normalized_masked  # Extract RGB values (without alpha channel)
            shared_radar_queue.put((radar_points, colors))
        frame_index = frame_index+1
        if frame_index % 100 == 0:
            print(f"Processed and saved {frame_index} radar frames.")
    

# def mat_unlit(size=2.0):
#     m = rendering.MaterialRecord()
#     m.shader = "defaultUnlit"
#     m.point_size = size
#     return m


def visualize_radar_data(radar_queue: Queue, stop_evt) -> None:
    '''
    Visualizes  point cloud data.
    '''
    frame_index = 0
    plotter = pv.Plotter(window_size=(1280, 960))
    while True:
        if stop_evt.is_set():
            print("Radar termination signal received by Visualizer. exiting.")
            break
        if radar_queue.empty():
            continue
        radar_points, colors = radar_queue.get()
        
        # print(f"Radar Visualizer: Received {radar_points.shape}.")
        if colors is not None and len(colors) != 0:
            # print(colors.shape)
            norm = mcolors.Normalize(vmin=np.percentile(colors, 2), vmax=np.percentile(colors, 98))
            intensity_norm = norm(colors)
            cmap = cm.plasma
            mapped_colors = cmap(intensity_norm)
            # alpha = np.zeros((mapped_colors.shape[0], 0))
            # mapped_colors = np.concatenate([mapped_colors,alpha], axis = 1)
            mapped_colors[:, 3] = np.clip(intensity_norm, 0.1, 1.0)  # alpha
        else:
            continue
        # Create Open3D PointCloud object
        if frame_index == 0:
            cloud = pv.PolyData(radar_points)
            cloud.points = radar_points
            cloud["rgba"] = (mapped_colors * 255).astype(np.uint8)
            actor = plotter.add_mesh(cloud, scalars="rgba", rgba=True, point_size=5)
            plotter.add_axes(line_width=2, labels_off=False)
            lookat = [0, 1, 0]
            front = np.array([0, 0, -1])
            up = [0, 0, 0]
            zoom = 1.0

            theta = np.radians(15)
            R = o3d.geometry.get_rotation_matrix_from_axis_angle([theta, 0, 0])  # 注意旋转轴顺序
            front_rot = R @ front

            # 相机位置 = 观察点 + front * 距离
            distance = 5.0 / zoom  # 距离可以理解为反比于zoom
            camera_position = lookat + front_rot * distance 

            # 设置 PyVista 相机参数
            plotter.camera.position = camera_position
            plotter.camera.focal_point = lookat
            plotter.camera.up = up
            plotter.camera.zoom(zoom)
            plotter.show(auto_close=False, interactive_update=True)  # Keep the plotter open for updates
        else:
            if len(radar_points) > len(cloud.points):
                # plotter.remove_actor(actor)
                # cloud = pv.PolyData(radar_points)
                # cloud["rgba"] = mapped_colors
                # actor = plotter.add_mesh(cloud, scalars="rgba", rgba=True, point_size=5)
                radar_points = radar_points[:len(cloud.points)]
                mapped_colors = mapped_colors[:len(cloud.points)]
            elif len(radar_points) < len(cloud.points):
                pad_n = len(cloud.points)-len(radar_points) 
                pad_points = np.repeat(radar_points[-1][None, :], pad_n, axis=0)
                pad_colors = np.repeat(mapped_colors[-1][None, :], pad_n, axis=0)
                radar_points = np.vstack([radar_points, pad_points])
                mapped_colors = np.vstack([mapped_colors, pad_colors])
                # cloud.points = radar_points
                # cloud["rgba"] = mapped_colors
                # plotter.update_coordinates(cloud.points, render=False)
                # plotter.update_scalars(cloud["rgba"], render=True)

            cloud.points = radar_points
            cloud["rgba"] = (mapped_colors * 255).astype(np.uint8)

            # Update visualization
            plotter.update()
            plotter.render()
            # plotter.update_coordinates(radar_points, render=False)
            # plotter.update_scalars(cloud["rgba"], render=True)
        frame_index = frame_index + 1
    plotter.clear()
    plotter.close()
    del plotter

def visualize_lidar_data(lidar_queue: Queue, stop_evt) -> None:
    '''
    Visualizes  point cloud data.
    '''
    vis = o3d.visualization.Visualizer()
    vis.create_window("Live Lidar 3D Point Cloud Update", width=1280, height=960)  # Set window size
    render_option = vis.get_render_option()
    render_option.point_size = 5.0
    render_option.background_color = np.array([0, 0, 0])
    pcd = o3d.geometry.PointCloud()
    plot_index = 0
    while True:
        if stop_evt.is_set():
            print("Lidar termination signal received by Visualizer. exiting.")
            break
        if lidar_queue.empty():
            continue
        points, intensity = lidar_queue.get()
        
        if intensity is None or len(intensity) == 0:
            colors = np.ones((points.shape[0], 3)) * 0.5  
        else:
            intensity = np.asarray(intensity).reshape(-1)

            norm = mcolors.Normalize(
                vmin=np.percentile(intensity, 2),
                vmax=np.percentile(intensity, 98)
            )
            intensity_norm = np.power(norm(intensity), 0.8 ) 
            mapped = cm.inferno(intensity_norm)[:, :3]        
            colors = mapped.astype(np.float32)
        if plot_index == 0:
            # Get next range map from queue (blocks until available)
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            pcd.voxel_down_sample(voxel_size=0.2)
            vis.add_geometry(pcd)
            # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])    
            # vis.add_geometry(axis)
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
            pcd.colors = o3d.utility.Vector3dVector(colors)
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.voxel_down_sample(voxel_size=0.2)
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
