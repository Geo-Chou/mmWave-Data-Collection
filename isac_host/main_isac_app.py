import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from more_itertools import time_limited
import itertools

import os
import pdb
import time
from multiprocessing import Process, Array, Lock, set_start_method
import socket
import open3d as o3d

from .ofdm import ofdm_generator 
from .SpatialProjection import SpatialProjection

save_dir = "/Users/geozhou/OneDrive/Phd/code/mmwave/data"

##############################################
# data date from UDP
##############################################
def udp_data_receive(shared_udp_queue, RECORD_DURATION, stop_evt):
    OG = ofdm_generator()
    # Define the IP and port to listen on
    UDP_IP = "0.0.0.0"  # Listen on all available interfaces
    UDP_PORT = 12000  # Choose an appropriate port
    BUFFER_SIZE = 40000  # Set the buffer size to 2000 bytes

    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    print(f"Listening for UDP packets on {UDP_IP}:{UDP_PORT}...")
    # arr = np.frombuffer(shared_esti_cir.get_obj(), dtype=np.float64)
    # shared_complex_arr = arr.view(np.complex128)  # Interpret as complex numbers

    local_frame_index = 0
    segment_index = 0
    big_buffer = []  # Initialize a buffer to store received data
    for i in time_limited(RECORD_DURATION, itertools.count()):
        data, addr = sock.recvfrom(BUFFER_SIZE)  # Receive data from sender
        if len(data) != 8000*4 and len(data) != 5500*4: 
            print("udp_data error!")
            continue
        data_uint32 = np.frombuffer(data, dtype=np.uint32)     
        rx_signal_one = OG.convert_int32_to_complex(data_uint32)
        big_buffer.append(rx_signal_one)
        if segment_index == 0:
            remote_frame_index = (data_uint32[0] & 0xFFFF0000) >> 16
            remote_pattern_index = (data_uint32[0] & 0x0000FFFF)  
        segment_index = segment_index + 1
        if len(data_uint32) == 5500:
            # print("Warning: incomplete packet!")
            rx_time_signal = np.concatenate(big_buffer)
            if len(rx_time_signal) == 77500:
                timestamp = time.time_ns()
                samp_shift = 224 + 130 + 128
                esti_cir = OG.channel_impulse_response_estimate(rx_time_signal[samp_shift:samp_shift+1200*64])
                g_esti_cir = esti_cir[:, 0:128] #copy.deepcopy(esti_cir)
                shared_udp_queue.put((timestamp, remote_frame_index, remote_pattern_index, g_esti_cir.flatten()))
                # shared_complex_arr[:] = g_esti_cir.flatten()
                # shared_complex_arr[0] = complex(remote_frame_index, remote_pattern_index)
                local_frame_index = local_frame_index + 1 

            big_buffer = []
            segment_index = 0

            if local_frame_index%100 == 0:
                print(f"Local/remote packet count: ({local_frame_index}, {remote_frame_index}); pattern_index: {remote_pattern_index}")

    print(f"UDP Receiver: Finished. {local_frame_index} scans received.")
    stop_evt.set()
    sock.close()



def plot_point_cloud(shared_esti_cir):
    num_cir_points = 64  # number of CIR points
    SP0 = SpatialProjection(Ant_Pattern_Scheme=0, R = num_cir_points)
    SP1 = SpatialProjection(Ant_Pattern_Scheme=1, R = num_cir_points)
    SP = SP0
    
    vis = o3d.visualization.Visualizer()
    vis.create_window("Live 3D Point Cloud Update", width=2048, height=960)  # Set window size
    pcd = o3d.geometry.PointCloud()
    plot_index = 0
    
    arr = np.frombuffer(shared_esti_cir.get_obj(), dtype=np.float64)
    shared_complex_arr = arr.view(np.complex128)  # Interpret as complex numbers
    ##############################################
    # plot the 3D point cloud
    ##############################################
    W = np.linspace(1, 3, num_cir_points)
    frame_index = 0
    while True:
        frame_index = frame_index+1
        S = shared_complex_arr.reshape(64, 128)
        S = S[:, 0:num_cir_points]
        remote_frame_index = int(shared_complex_arr[0].real)
        remote_pattern_index = int(shared_complex_arr[0].imag)
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

        mask = strength > 3 
        x_masked = SP.X[mask] 
        y_masked = SP.Y[mask]
        z_masked = SP.Z[mask]
        strength_normalized_masked = strength_normalized[mask]
        
        timestamp = time.time_ns()
        # np.savez(os.path.join(save_dir, f"{timestamp:06d}.npz"),
        #      points=np.column_stack((x_masked, y_masked, z_masked)), intensity=strength_normalized_masked,
        #      frame_idx=remote_frame_index,
        #      pattern_idx=remote_pattern_index)

        # Remove the old point cloud and add the new one
        if plot_index == 0:  # Skip the first iteration since there's nothing to remove
            pcd.points = o3d.utility.Vector3dVector(np.column_stack((SP.X, SP.Y, SP.Z)))
            vis.add_geometry(pcd)

            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])    
            vis.add_geometry(axis)
            
            view_control = vis.get_view_control()
            view_control.set_lookat([0, 0, 0])  # Adjust center
            view_control.set_front([0, 0, -1])  # Adjust camera direction
            view_control.set_up([0, 1, 0])  # Adjust upward direction
            view_control.set_zoom(0.1)  # Adjust zoom to fit the scene properly
                        
        else: 
            colormap = cm.viridis  # You can choose any colormap like 'viridis', 'plasma', etc.
            colors = colormap(strength_normalized_masked)[:, :3]  # Extract RGB values (without alpha channel)

            # Create Open3D PointCloud object
            pcd.points = o3d.utility.Vector3dVector(np.column_stack((x_masked, y_masked, z_masked)))
            if not np.array_equal(pcd.colors, colors):
                pcd.colors = o3d.utility.Vector3dVector(colors)

            # Update visualization
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
        time.sleep(0.1)  # Small delay for visualization updates
        plot_index = plot_index + 1
        if plot_index%10 == 0:
            print(f"PLOT: plot_index: {plot_index}; strength max: {int(strength.max())}")
    vis.destroy_window()


def antenna_calib(shared_esti_cir):
    SP = SpatialProjection()
    
    arr = np.frombuffer(shared_esti_cir.get_obj(), dtype=np.float64)
    shared_complex_arr = arr.view(np.complex128)  # Interpret as complex numbers    
    calib_list = []
    calib_count = 0
    
    print("***********************************")
    print("Perform antenna calibration")
    print("Please place corner reflector at 1.2m (BIN 10)")
    print("***********************************")
    
    while True:
        remote_frame_index = int(shared_complex_arr[0].real)
        remote_pattern_index = int(shared_complex_arr[0].imag)
        S = shared_complex_arr.reshape(64, 128)
        #-----------------------------------------------
        # calculate antenna phase difference; use FFT BIN 10 as the reference
        #-----------------------------------------------
        ant_calib = S[:, 10]
        ant_calib_coeff = np.conj(ant_calib)/np.square(np.abs(ant_calib)+0.00001)
        SP.save_ant_calib_coeff(remote_pattern_index, ant_calib_coeff)
        #-----------------------------------------------
        # calculate antenna phase difference; use FFT BIN 10 as the reference
        #-----------------------------------------------
        if remote_pattern_index not in calib_list:
            calib_list.append(remote_pattern_index)        
        if len(calib_list) == 4:        
            SP.calc_and_save_E()
            calib_list = []
            calib_count = calib_count + 1
            print(f"******************** calib_count = {calib_count}")
            time.sleep(0.001)

    
    
def cir_display(shared_esti_cir):
    arr = np.frombuffer(shared_esti_cir.get_obj(), dtype=np.float64)
    shared_complex_arr = arr.view(np.complex128)  # Interpret as complex numbers    


    num_samples = 128
    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Initialize empty lines for animation
    line1, = ax1.plot([], [], 'r-', label='abs')
    line2, = ax2.plot([], [], 'b-', label='angle')

    # Set axis labels
    ax2.set_xlabel('Sample Index')
    ax1.set_ylabel('Amplitude')
    ax2.set_ylabel('Phase')

    # Set axis limits
    ax1.set_xlim(0, num_samples/2)
    ax2.set_xlim(0, num_samples/2)
    ax1.set_ylim(0, 5)  # Adjust amplitude range
    ax2.set_ylim(-np.pi, np.pi)  # Phase range (-π to π)
    ax1.grid(True)  # Enable grid for amplitude plot
    ax2.grid(True)  # Enable grid for phase plot

    # Animation update function
    def update(frame):
        S = shared_complex_arr.reshape(64, 128)
        S_new = S[0, :]
        #S_new = np.exp(1j * np.linspace(frame / 10, 2 * np.pi + frame / 10, num_samples))

        line1.set_data(np.arange(num_samples), np.abs(S_new))  # Magnitude
        line2.set_data(np.arange(num_samples), np.angle(S_new))  # Phase
        
        return line1, line2

    # Run animation (update every 100ms)
    ani = animation.FuncAnimation(fig, update, frames=63, interval=10, blit=True)

    plt.show()
        
        
# if __name__ == "__main__": 
#     # set_start_method("spawn", force=True)
#     shared_esti_cir = Array('d', 64*128*2)  # 'd' is for float64

#     flag_antenna_calib = False
#     flag_signal_display = False
#     flag_plot_point_cloud = True
    
#     if flag_antenna_calib == True:        
#         udp_process = Process(target=udp_data_receive, args=(shared_esti_cir,))
#         calib_process = Process(target=antenna_calib, args=(shared_esti_cir,))
#         display_process = Process(target=cir_display, args=(shared_esti_cir,))

#         udp_process.start()
#         calib_process.start()
#         display_process.start()
        
#         udp_process.join()
#         calib_process.join()
#         display_process.join()

#     elif flag_signal_display == True:
    
#         udp_process = Process(target=udp_data_receive, args=(shared_esti_cir,))
#         display_process = Process(target=cir_display, args=(shared_esti_cir,))

#         udp_process.start()
#         display_process.start()
        
#         udp_process.join()
#         display_process.join()
#     else:
        
#         udp_process = Process(target=udp_data_receive, args=(shared_esti_cir,)) 
#         plot_process = Process(target=plot_point_cloud, args=(shared_esti_cir,))
        
#         udp_process.start()
#         plot_process.start()
#         udp_process.join()
#         plot_process.join()
        
