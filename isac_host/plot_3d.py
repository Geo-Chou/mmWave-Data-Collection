import numpy as np
# from numpy import imag, real
import matplotlib.pyplot as plt
import pdb
import time
import open3d as o3d
from ofdm import ofdm_generator 



##############################################
# data date from UDP
##############################################
def udp_data_receive():
    import socket
    # Define the IP and port to listen on
    UDP_IP = "0.0.0.0"  # Listen on all available interfaces
    UDP_IP = "127.0.0.1"  # Listen on all available interfaces
    UDP_PORT = 12000  # Choose an appropriate port
    BUFFER_SIZE = 40000  # Set the buffer size to 2000 bytes

    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    print(f"Listening for UDP packets on {UDP_IP}:{UDP_PORT}...")

    packet_index = 0
    big_buffer = []  # Initialize a buffer to store received data
    while True:
        data, addr = sock.recvfrom(BUFFER_SIZE)  # Receive data from sender
        print(f"Received {len(data)} bytes from {addr}")
        if len(data) == 0: 
            continue
        rx_signal_one = og.convert_int32_to_complex(data)
        big_buffer.append(rx_signal_one)
        packet_index = packet_index + 1
        if np.real(rx_signal_one[-1]) == 200 and imag(rx_signal_one[-1]) == 200:
            print(f"The number of udp packets is: {packet_index}")   
            #wait_chan_esti_idle()
            rx_time_signal = np.concatenate(big_buffer)
            big_buffer = []
            return 0
    
    
def wait_chan_esti_idle():
    while True:
        if chan_esti_busy == True:
            time.sleep(0.001)
        else:
            break
 
 
udp_data_receive()



###########################################
# calabiration  of channel 
###########################################
with open("ant_calib.bin", "rb") as f:
    data = np.fromfile(f, dtype=np.float32)

# Step 2: Convert to complex vector
# Assuming the file contains interleaved real and imaginary parts
if len(data) != 2*256:
    print("The data len is incorrect!")
    exit(0);
ant_calib = data[0::2] + 1j * data[1::2]  # Real + Imaginary * i
ant_calib = np.array(ant_calib) + 0.000001
ant_calib_coeff = np.conj(ant_calib)/np.square(np.abs(ant_calib))
 
 
#plt.figure(figsize=(10, 4))
#plt.subplot(1, 2, 1)
#plt.stem(np.abs(ant_calib), basefmt=" ", linefmt="b-", markerfmt="bo")
#plt.subplot(1, 2, 2)
#plt.stem(np.angle(ant_calib), basefmt=" ", linefmt="r-", markerfmt="ro")
#plt.tight_layout()
#plt.show()
#exit(1)


###########################################
# calabiration  of channel 
###########################################
N = 16
M = 16
R = 128
T = N*M*R
lambda_val = 3e11 / 60.48e9  # unit: mm
pix2_div_lambda_val = 2*np.pi/lambda_val

###################################
# antenna arrangement
###################################
# ANTENNA POSITION
# a = 2.25    # unit: mm
# b = 3.00    # unit: mm
# c = 16 / 7  # unit: mm
# d = 25.5    # unit: mm

a = 2.3    # unit: mm
b = 2.9    # unit: mm
c = 2.2    # unit: mm
d = 25.7   # unit: mm ??


# Define TX_X and TX_Z arrays
TX_X = np.array([
    0 * c, 0 * c, 0 * c, 0 * c,
    1 * c, 1 * c, 1 * c, 1 * c,
    2 * c, 2 * c, 2 * c, 2 * c,
    3 * c, 3 * c, 3 * c, 3 * c,
    4 * c, 4 * c, 4 * c, 4 * c,
    5 * c, 5 * c, 5 * c, 5 * c,
    6 * c, 6 * c, 6 * c, 6 * c,
    7 * c, 7 * c, 7 * c, 7 * c
])

TX_Z = np.array([
    2 * a + b, a + b, a, 0,
    2 * a + b, a + b, a, 0,
    2 * a + b, a + b, a, 0,
    2 * a + b, a + b, a, 0,
    2 * a + b, a + b, a, 0,
    2 * a + b, a + b, a, 0,
    2 * a + b, a + b, a, 0,
    2 * a + b, a + b, a, 0
])


# Define RX_X and RX_Z
RX_X = TX_X
RX_Z = TX_Z + 2 * a + b + d

# Define P_TX and P_RX
P_TX = np.vstack([TX_X, np.zeros_like(TX_X), TX_Z])
P_RX = np.vstack([RX_X, np.zeros_like(RX_X), RX_Z])



##############################
# construct the plot 3D 
##############################
# Constants
distance = np.arange(R) * 0.12  # Distance array (1:fft_bin_len) * 0.12
azi_angle_arr = np.arange(30, 150, 1)  # azi_rad array from 30 to 150 degrees
ele_angle_arr = np.arange(-30, 30, 1)  # ele_rad array with only one value 0 (could be expanded)

Az, El, Dist = np.meshgrid(ele_angle_arr, azi_angle_arr, distance)

# Convert spherical coordinates to Cartesian coordinates
xx = Dist * np.cos(np.radians(El)) * np.cos(np.radians(Az))  # X-coordinates
yy = Dist * np.cos(np.radians(El)) * np.sin(np.radians(Az))  # Y-coordinates
zz = Dist * np.sin(np.radians(El))  # Z-coordinates


# Reshape the data for plotting (flatten the 3D grid into 1D vectors)
x = xx.flatten()
y = yy.flatten()
z = zz.flatten()


rx_time_signal = np.array([])
og = ofdm_generator()


##############################################
# beamforming for query function B(theta, phi)
##############################################
if False:
    E = np.zeros((len(azi_angle_arr), len(ele_angle_arr), N*M), dtype=complex)
    for ll in range(len(azi_angle_arr)):
        for kk in range(len(ele_angle_arr)):
            theta = azi_angle_arr[ll] * np.pi / 180  # Convert to radians
            phi = ele_angle_arr[kk] * np.pi / 180  # Convert to radians
            d = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])
            steering_direction = np.zeros(N*M, dtype=complex)  # Initialize steering direction

            # Nested loop over TX and RX antenna indices
            for m in range(M):  # TX antenna index (1 to 16)
                Pm1 = P_TX[:, 2*m]
                Pm2 = P_TX[:, 2*m+1]
                for n in range(N):  # RX antenna index (1 to 16)
                    Pn1 = P_RX[:, 2*n]
                    Pn2 = P_RX[:, 2*n+1]

                    # Calculate phase for the current TX-RX pair
                    phase_mn = np.exp(-1j * pix2_div_lambda_val * np.dot(d, (Pm1 + Pn1))) + \
                               np.exp(-1j * pix2_div_lambda_val * np.dot(d, (Pm1 + Pn2))) + \
                               np.exp(-1j * pix2_div_lambda_val * np.dot(d, (Pm2 + Pn1))) + \
                               np.exp(-1j * pix2_div_lambda_val * np.dot(d, (Pm2 + Pn2)))

                    # Assign to the steering direction matrix
                    steering_direction[m*16+n] = phase_mn

            # Store the result in the B matrix
            E[ll, kk, :] = steering_direction * ant_calib_coeff

    np.save("E.npy", E)
else:
    E = np.load("E.npy")

pattern_index0 = [x+16*0 for x in range(16)] + [x+16*4 for x in range(16)] + [x+16*8  for x in range(16)] + [x+16*12 for x in range(16)]
pattern_index1 = [x+16*1 for x in range(16)] + [x+16*5 for x in range(16)] + [x+16*9  for x in range(16)] + [x+16*13 for x in range(16)]
pattern_index2 = [x+16*2 for x in range(16)] + [x+16*6 for x in range(16)] + [x+16*10 for x in range(16)] + [x+16*14 for x in range(16)]
pattern_index3 = [x+16*3 for x in range(16)] + [x+16*7 for x in range(16)] + [x+16*11 for x in range(16)] + [x+16*15 for x in range(16)]
E0 = E[:, :, pattern_index0]
E1 = E[:, :, pattern_index1]
E2 = E[:, :, pattern_index2]
E3 = E[:, :, pattern_index3]


vis = o3d.visualization.Visualizer()
vis.create_window("Live 3D Point Cloud Update")
pcd = o3d.geometry.PointCloud()

 
##############################
# load data from file 
##############################
B = np.zeros((len(azi_angle_arr), len(ele_angle_arr), R), dtype=complex)
filename = "cir_frame.bin"
loop_index = 0
for ii in range(10000):
    print(f"Iteration {ii}")
    loop_index = loop_index + 1
    
    data_source = 0
    if data_source == 0:
        with open(filename, "rb") as f:
            data = np.fromfile(f, dtype=np.float32)
            f.close()
        if len(data) != 2*T:
            #print("The data len is incorrect!")
            continue
        complex_vector = data[0::2] + 1j * data[1::2]  # Real + Imaginary * i
        S = complex_vector.reshape(N*M, R)    
    else:
        udp_data_receive()
        esti_chan = channel_estimate(rx_time_signal)
        S = esti_chan[:, 0:128]
        
        
    # remove tx/rx ant index 
    S[:, 0] = 0
    S[:, 1] = 0

    # weight the waveform
    W = np.linspace(1, 5, 128)
    S = S*W
    #S = S.T
    #S[:, 64:128] = 0
    
    B = E0 @ S

    ##############################################
    # beamforming for query function B(theta, phi)
    ##############################################
    # Iterate over theta and phi values
    #for ll in range(len(azi_angle_arr)):
    #    for kk in range(len(ele_angle_arr)):
    #        B[ll, kk, :] = S @ E[ll, kk, :]



    ##############################################
    # plot the 3D point cloud
    ##############################################
    # Iterate over theta and phi values
    # Assuming B is already calculated as per your previous steps
    strength = np.abs(B)  # Compute the strength (absolute value of B)
    strength = strength.flatten()  # Flatten for consistency
    strength_min = np.min(strength)
    strength_max = np.max(strength)
    
    strength_normalized = (strength - strength.min()) / (strength.max() - strength.min())
    
    #threshold = float(input("Enter a decimal number: "))
    threshold = 0.1
    mask = strength_normalized > threshold
    x_masked = x[mask] 
    y_masked = y[mask]
    z_masked = z[mask]
    # x_masked = x_masked + 0.02*np.random.uniform(-1, 1, len(x_masked))
    strength_normalized_masked = strength_normalized[mask]


    colormap = plt.get_cmap("viridis")
    colors = colormap(strength_normalized_masked)[:, :3]  # Extract RGB values
    
    # Create Open3D PointCloud object
    pcd.points = o3d.utility.Vector3dVector(np.column_stack((x_masked, y_masked, z_masked)))
    pcd.colors = o3d.utility.Vector3dVector(colors)  # Assign colors



    
    # Remove the old point cloud and add the new one
    if ii == 0:  # Skip the first iteration since there's nothing to remove
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])    
        vis.add_geometry(pcd)
        vis.add_geometry(axis)
        
        view_ctrl = vis.get_view_control()

        # Set the camera look-at center (modify as needed)
        view_ctrl.set_lookat([0, 0, 0])  # Adjust center
        view_ctrl.set_front([0, 0, -1])  # Adjust camera direction
        view_ctrl.set_up([0, 1, 0])  # Adjust upward direction
        
    else: 
        # Update visualization
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

    time.sleep(0.25)  # Small delay for visualization updates
    
vis.destroy_window()
    
