'''
This script collects and saves radar and lidar cloud point data for a specified time of frames.
'''
import itertools
import os
import time
import json
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from multiprocessing import Queue, Array, Process, Event, set_start_method, shared_memory
import argparse

import open3d as o3d
import pyvista as pv

from isac_host.main_isac_app import udp_data_receive, plot_point_cloud
from isac_host.SpatialProjection import SpatialProjection

from dualcam import collect_realsense_data, visualize_realsense_data

import open3d as o3d




def process_save_radar_data(shared_udp_queue, shared_radar_queue, save_dir, stop_evt) -> None:
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

        mask = strength > 0.5 
        x_masked = SP.X[mask] 
        y_masked = SP.Y[mask]
        z_masked = SP.Z[mask]
        strength_normalized_masked = strength_normalized[mask]
        B_masked = B.flatten()[mask]

        np.savez(os.path.join(save_dir, 'radar_points', f"{timestamp:06d}.npz"),
             points=np.column_stack((x_masked, y_masked, z_masked)), intensity=strength_normalized_masked, phase=B_masked)
        
        if frame_index == 0:
            radar_points = np.column_stack((SP.X, SP.Y, SP.Z))
            colors = None
            shared_radar_queue.put((radar_points, colors))

        else:
            radar_points = np.column_stack((x_masked, y_masked, z_masked))
            colors =  strength_normalized_masked# Extract RGB values (without alpha channel)
            shared_radar_queue.put((radar_points, colors))
        frame_index = frame_index+1
        if frame_index % 100 == 0:
            print(f"Processed and saved {frame_index} radar frames.")
    


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
            norm = mcolors.Normalize(vmin=np.percentile(colors, 2), vmax=np.percentile(colors, 98))
            intensity_norm = norm(colors)
            cmap = cm.Purples
            mapped_colors = cmap(intensity_norm)
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
            lookat = [0, 2, 0]
            front = np.array([0, 0, -1])
            up = [0, 0, 0]
            zoom = 1.0

            theta = np.radians(30)
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

if __name__ == "__main__":
    set_start_method('spawn')
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
    )
    start_time = time.time_ns()
    SAVE_DIR = './data/'+str(start_time)+'/'
    # os.makedirs(SAVE_DIR,exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, 'radar_points'),exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, 'realsense'), exist_ok=True)
    parser.add_argument("RECORD_DURATION", type=int, help="Duration of the recording in seconds")
    args = parser.parse_args()
    stop_evt_radar = Event()
    stop_evt_cam = Event()
    

    # shared_esti_cir = Array('d', 64*128*2)  # 'd' is for float64
    shared_udp_queue = Queue(maxsize=20)
    shared_radar_queue = Queue(maxsize=20)

    shared_cam_queue = Queue(maxsize=50)
    shape_vtx = (1280*720, 3)
    shape_img = (720, 1280, 3)
    shm_vtx = shared_memory.SharedMemory(create=True, size=np.prod(shape_vtx) * np.dtype(np.float32).itemsize)
    shm_img = shared_memory.SharedMemory(create=True, size=np.prod(shape_img) * np.dtype(np.uint8).itemsize)

    collect_realsense_process = Process(target=collect_realsense_data, 
                                        args=(shared_cam_queue, shm_vtx.name, shm_img.name, shape_vtx, shape_img,
                                               args.RECORD_DURATION, SAVE_DIR, start_time, stop_evt_cam))
    visualize_realsense_process = Process(target=visualize_realsense_data, 
                                          args=(shared_cam_queue, shm_vtx.name, shm_img.name,
                                                shape_vtx, shape_img, stop_evt_cam))


    udp_process = Process(target=udp_data_receive, args=(shared_udp_queue,args.RECORD_DURATION, stop_evt_radar)) 
    radar_process = Process(target=process_save_radar_data, args=(shared_udp_queue,shared_radar_queue, SAVE_DIR, stop_evt_radar))

    radar_visualizer_process = Process(target=visualize_radar_data, args=(shared_radar_queue, stop_evt_radar))


    udp_process.start()
    radar_process.start()
    radar_visualizer_process.start()
    collect_realsense_process.start()
    visualize_realsense_process.start()

    udp_process.join()
    radar_process.join()

    radar_visualizer_process.join()


   
    
    collect_realsense_process.join()
    visualize_realsense_process.join()
    shm_vtx.close(); shm_vtx.unlink()
    shm_img.close(); shm_img.unlink()
