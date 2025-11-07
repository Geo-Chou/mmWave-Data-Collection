import pyrealsense2 as rs
import numpy as np
# import cv2
import open3d as o3d
# import time
from more_itertools import time_limited
import itertools
from multiprocessing import shared_memory, Event, set_start_method, shared_memory, Queue, Array, Process
import os


def collect_realsense_data(shared_id_queue, shm_name_vtx, shm_name_img,
                            shape_vtx, shape_img, RECORD_DURATION, save_dir, start_time, stop_evt):

    shm_vtx = shared_memory.SharedMemory(name=shm_name_vtx)
    shm_img = shared_memory.SharedMemory(name=shm_name_img)

    buf_vtx = np.ndarray(shape_vtx, dtype=np.float32, buffer=shm_vtx.buf)
    buf_img = np.ndarray(shape_img, dtype=np.uint8, buffer=shm_img.buf)


    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_record_to_file(os.path.join(save_dir, f"../{start_time}.bag"))
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    pc = rs.pointcloud()

    for i in range(30): 
        pipeline.wait_for_frames()
    print("Starting frame capture...")

    for i in time_limited(RECORD_DURATION, itertools.count()):
        frames = pipeline.wait_for_frames()
        
        if frames.get_depth_frame() and frames.get_color_frame():
            aligned_frames = align.process(frames)
        else:
            continue

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)
        vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        # tex = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
        img = np.asanyarray(color_frame.get_data())
        
        if not depth_frame or not color_frame:
            raise RuntimeError("No frames received from the camera.")
        if i==0:
            intr = color_frame.profile.as_video_stream_profile().intrinsics
            np.save(os.path.join(save_dir, "intrinsics.npy"), {
                "width": intr.width,
                "height": intr.height,
                "ppx": intr.ppx,
                "ppy": intr.ppy,
                "fx": intr.fx,
                "fy": intr.fy,
                "model": intr.model,
                "coeffs": intr.coeffs
            })
        # print(vtx.dtype, img.dtype, depth_image.dtype, color_image.dtype)
        np.copyto(buf_vtx, vtx)
        np.copyto(buf_img, img)
        try:
            shared_id_queue.put(i, block=False)
        except:
            pass
    pipeline.stop()
    stop_evt.set()
    print(" Termination signal sent to Realsense frame collector.")




    
def visualize_realsense_data(vis_cam_queue, shm_name_vtx, shm_name_img,
                             vtx_shape, img_shape, stop_evt):

    frame_id = 0
    vis = o3d.visualization.Visualizer()
    vis.create_window("Live Realsense Point Cloud Update", width=1280, height=960)  # Set window size
    pcd = o3d.geometry.PointCloud()
    while True:
        if stop_evt.is_set():
            print("Realsense visualizer received termination signal. Exiting.")
            break
        if vis_cam_queue.empty():
            continue
        frame_id = vis_cam_queue.get()
        shm_vtx = shared_memory.SharedMemory(name=shm_name_vtx)
        shm_img = shared_memory.SharedMemory(name=shm_name_img)
        buf_vtx = np.ndarray(vtx_shape, dtype=np.float32, buffer=shm_vtx.buf)
        buf_img = np.ndarray(img_shape, dtype=np.uint8, buffer=shm_img.buf)
        vtx = buf_vtx.copy()
        img = buf_img.copy()
        pcd.points = o3d.utility.Vector3dVector(vtx)
        pcd.colors = o3d.utility.Vector3dVector(img.reshape(-1, 3) / 255.0)

        if frame_id == 0:
            vis.add_geometry(pcd)
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])    
            vis.add_geometry(axis)
            view_control = vis.get_view_control()
            view_control.set_lookat([0, 0, 1])  
            view_control.set_front([0, 0, -1])  
            view_control.set_up([0, -1, 0])    
            view_control.set_zoom(0.05)
            frame_id += 1
        else:
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            frame_id += 1
    vis.destroy_window()

# SAVE_DIR = "./data/"

# if __name__ == "__main__":
#     os.makedirs(os.path.join(SAVE_DIR, 'realsense'),exist_ok=True)
#     stop_evt_cam = Event()
#     shape_depth = (720, 1280)
#     shape_color = (720, 1280, 3)
#     shape_vtx = (1280*720, 3)
#     shape_img = (720, 1280, 3)
#     shm_depth = shared_memory.SharedMemory(create=True, size=np.prod(shape_depth) * np.dtype(np.uint16).itemsize)
#     shm_color = shared_memory.SharedMemory(create=True, size=np.prod(shape_color) * np.dtype(np.uint8).itemsize)
#     shm_vtx = shared_memory.SharedMemory(create=True, size=np.prod(shape_vtx) * np.dtype(np.float32).itemsize)
#     shm_img = shared_memory.SharedMemory(create=True, size=np.prod(shape_img) * np.dtype(np.uint8).itemsize)
#     shared_id_queue = Queue(maxsize=50)
#     vis_cam_queue = Queue(maxsize=50)
#     collect_realsense_process = Process(target=collect_realsense_data, 
#                                         args=(shared_id_queue, shm_depth.name, shm_color.name, shm_vtx.name, shm_img.name,
#                                               shape_depth, shape_color, shape_vtx, shape_img,
#                                                120, SAVE_DIR, stop_evt_cam))
#     save_realsense_process = Process(target=save_realsense_data, 
#                                      args=(shared_id_queue,vis_cam_queue, shm_depth.name, shm_color.name,
#                                            shape_depth, shape_color,
#                                             SAVE_DIR, stop_evt_cam))
#     visualize_realsense_process = Process(target=visualize_realsense_data, 
#                                           args=(shared_id_queue, shm_vtx.name, shm_img.name,
#                                                 shape_vtx, shape_img, stop_evt_cam))
#     collect_realsense_process.start()
#     save_realsense_process.start()
#     visualize_realsense_process.start()
#     collect_realsense_process.join()
#     save_realsense_process.join()
#     visualize_realsense_process.join()
#     shm_depth.close(); shm_depth.unlink()
#     shm_color.close(); shm_color.unlink()
#     shm_vtx.close(); shm_vtx.unlink()
#     shm_img.close(); shm_img.unlink()
# if __name__ == "__main__":
#     set_start_method('spawn')
#     start_time = time.time_ns()
#     SAVE_DIR = './data/'
#     RECORD_DURATION = 120  # seconds
#     # os.makedirs(SAVE_DIR,exist_ok=True)
#     os.makedirs(os.path.join(SAVE_DIR, 'realsense'), exist_ok=True)
#     stop_evt_radar = Event()
#     stop_evt_cam = Event()
    

#     # shared_esti_cir = Array('d', 64*128*2)  # 'd' is for float64
#     shared_udp_queue = Queue(maxsize=20)
#     shared_radar_queue = Queue(maxsize=20)

#     shared_cam_queue = Queue(maxsize=50)

#     shape_depth = (720, 1280)
#     shape_color = (720, 1280, 3)
#     shape_vtx = (1280*720, 3)
#     shape_img = (720, 1280, 3)
#     shm_depth = shared_memory.SharedMemory(create=True, size=np.prod(shape_depth) * np.dtype(np.uint16).itemsize)
#     shm_color = shared_memory.SharedMemory(create=True, size=np.prod(shape_color) * np.dtype(np.uint8).itemsize)
#     shm_vtx = shared_memory.SharedMemory(create=True, size=np.prod(shape_vtx) * np.dtype(np.float32).itemsize)
#     shm_img = shared_memory.SharedMemory(create=True, size=np.prod(shape_img) * np.dtype(np.uint8).itemsize)
#     collect_realsense_process = Process(target=collect_realsense_data, 
#                                         args=(shared_cam_queue, shm_vtx.name, shm_img.name, shape_vtx, shape_img,
#                                                RECORD_DURATION, SAVE_DIR, stop_evt_cam))
#     visualize_realsense_process = Process(target=visualize_realsense_data, 
#                                           args=(shared_cam_queue, shm_vtx.name, shm_img.name,
#                                                 shape_vtx, shape_img, stop_evt_cam))


#     collect_realsense_process.start()
#     visualize_realsense_process.start()
    
#     collect_realsense_process.join()
#     visualize_realsense_process.join()
#     shm_depth.close(); shm_depth.unlink()
#     shm_color.close(); shm_color.unlink()
#     shm_vtx.close(); shm_vtx.unlink()
#     shm_img.close(); shm_img.unlink()
