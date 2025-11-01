import numpy as np
import matplotlib.pyplot as plt
import pdb
import time





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
ant_calib = np.array(ant_calib)
ant_calib_coeff = np.conj(ant_calib)/np.square(np.abs(ant_calib))

'''
plt.figure()
plt.plot(np.abs(ant_calib), marker='o', label='amp')
plt.plot(np.abs(ant_calib1), marker='*', label='amp1')
plt.plot(np.abs(ant_calib2), marker='+', label='amp2')

plt.figure()
plt.plot(np.angle(ant_calib), marker='o', label='pha')
plt.plot(np.angle(ant_calib1), marker='*', label='pha1')
plt.plot(np.angle(ant_calib2), marker='+', label='pha2')
plt.legend()
plt.show()    


exit()
'''



###########################################
# calabiration  of channel 
###########################################
N = 16
M = 16
R = 128
T = N*M*R
lambda_val = 3e11 / 60.48e9  # unit: mm



# Constants
distance = np.arange(R) * 0.12  # Distance array (1:fft_bin_len) * 0.12
azi_angle_arr = np.arange(30, 150, 1)  # azi_rad array from 30 to 150 degrees
ele_angle_arr = np.arange(-40, 40, 1)  # ele_rad array with only one value 0 (could be expanded)

# Create meshgrid for Azimuth, Elevation, and Distance
Az, El, Dist = np.meshgrid(azi_angle_arr, np.array([0]), distance)
# Convert spherical coordinates to Cartesian coordinates
x = Dist * np.cos(np.radians(El)) * np.cos(np.radians(Az))  # X-coordinates
y = Dist * np.cos(np.radians(El)) * np.sin(np.radians(Az))  # Y-coordinates
# Reshape the data for plotting (flatten the 3D grid into 1D vectors)
x_azi = x.flatten()
y_azi = y.flatten()


Az, El, Dist = np.meshgrid(np.array([90]), ele_angle_arr, distance)
# Convert spherical coordinates to Cartesian coordinates
y = Dist * np.cos(np.radians(El)) * np.sin(np.radians(Az))  # Y-coordinates
z = Dist * np.sin(np.radians(El))  # Z-coordinates
# Reshape the data for plotting (flatten the 3D grid into 1D vectors)
y_ele = y.flatten()
z_ele = z.flatten()





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




loop_index = 0

plt.ion()  # Turn on interactive mode
#fig1, ax1 = plt.subplots(figsize=(18, 6))
#fig2, ax2 = plt.subplots(figsize=(6, 12))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 18))  # One row, two columns of subplots
# Create scatter plots outside the loop
scatter_azi = ax1.scatter(x_azi, y_azi, c=np.linspace(0, 20, len(x_azi)), cmap='jet', s=50)
scatter_ele = ax2.scatter(z_ele, y_ele, c=np.linspace(0, 15, len(z_ele)), cmap='jet', s=50)
b_azi_old = 0
b_ele_old = 0

# Create colorbars once
cbar1 = fig.colorbar(scatter_azi, ax=ax1)
cbar2 = fig.colorbar(scatter_ele, ax=ax2)
#cbar1.set_label('Value')
ax1.set_xlim([-5, 5])
ax1.set_ylim([0, 5])
ax1.set_title('Azimuth map: '+str(loop_index))
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.grid(True)

# Add colorbar
#cbar2.set_label('Value')
# Set plot limits, titles, and labels
ax2.set_ylim([0, 5])
ax2.set_xlim([-4, 4])
ax2.set_title('Elevation map: '+str(loop_index))
ax2.set_ylabel('Y')
ax2.set_xlabel('Z')
ax2.grid(True)    
    
    

##############################################
# beamforming for query function B(azi_rad, ele_rad)
##############################################
E = np.zeros((len(azi_angle_arr), N*M), dtype=complex)
for ll in range(len(azi_angle_arr)):
    azi_rad = azi_angle_arr[ll] * np.pi / 180  # Convert to radians
    ele_rad = 0 * np.pi / 180  # Convert to radians
    d = np.array([np.cos(azi_rad) * np.cos(ele_rad), np.sin(azi_rad) * np.cos(ele_rad), np.sin(ele_rad)])
    steer_dir = np.zeros(N*M, dtype=complex)  # Initialize steering direction

    # Nested loop over TX and RX antenna indices
    for m in range(M):  # TX antenna index (1 to 16)
        for n in range(N):  # RX antenna index (1 to 16)
            Pm1 = P_TX[:, 2*m]
            Pm2 = P_TX[:, 2*m+1]
            Pn1 = P_RX[:, 2*n]
            Pn2 = P_RX[:, 2*n+1]

            # Calculate phase for the current TX-RX pair
            phase_mn = np.exp(1j * 2 * np.pi / lambda_val * np.dot(d, (Pm1 + Pn1))) + \
                       np.exp(1j * 2 * np.pi / lambda_val * np.dot(d, (Pm1 + Pn2))) + \
                       np.exp(1j * 2 * np.pi / lambda_val * np.dot(d, (Pm2 + Pn1))) + \
                       np.exp(1j * 2 * np.pi / lambda_val * np.dot(d, (Pm2 + Pn2)))

            # Assign to the steering direction matrix
            steer_dir[m*16+n] = phase_mn
    E[ll, :] = steer_dir * ant_calib_coeff


##############################################
# beamforming for query function B(azi_rad, ele_rad)
##############################################
H = np.zeros((len(ele_angle_arr), N*M), dtype=complex)
for kk in range(len(ele_angle_arr)):
    azi_rad = 90 * np.pi / 180  # Convert to radians
    ele_rad = ele_angle_arr[kk] * np.pi / 180  # Convert to radians
    d = np.array([np.cos(azi_rad) * np.cos(ele_rad), np.sin(azi_rad) * np.cos(ele_rad), np.sin(ele_rad)])
    steer_dir = np.zeros(N*M, dtype=complex)  # Initialize steering direction

    # Nested loop over TX and RX antenna indices
    for m in range(M):  # TX antenna index (1 to 16)
        for n in range(N):  # RX antenna index (1 to 16)
            Pm1 = P_TX[:, 2*m]
            Pm2 = P_TX[:, 2*m+1]
            Pn1 = P_RX[:, 2*n]
            Pn2 = P_RX[:, 2*n+1]

            # Calculate phase for the current TX-RX pair
            phase_mn = np.exp(1j * 2 * np.pi / lambda_val * np.dot(d, (Pm1 + Pn1))) + \
                       np.exp(1j * 2 * np.pi / lambda_val * np.dot(d, (Pm1 + Pn2))) + \
                       np.exp(1j * 2 * np.pi / lambda_val * np.dot(d, (Pm2 + Pn1))) + \
                       np.exp(1j * 2 * np.pi / lambda_val * np.dot(d, (Pm2 + Pn2)))

            # Assign to the steering direction matrix
            steer_dir[m*16+n] = phase_mn

    # Store the result in the B matrix
    H[kk, :] = steer_dir * ant_calib_coeff



    
##############################
# load data from file 
##############################
filename = "cir_frame.bin"
B_azi = np.zeros((len(azi_angle_arr), R), dtype=complex)
B_ele = np.zeros((len(ele_angle_arr), R), dtype=complex)

while(True):
    loop_index = loop_index + 1
    with open(filename, "rb") as f:
        data = np.fromfile(f, dtype=np.float32)
        f.close()

    # Step 2: Convert to complex vector
    # Assuming the file contains interleaved real and imaginary parts
    if len(data) != 2*T:
        #print("The data len is incorrect!")
        continue
    
    complex_vector = data[0::2] + 1j * data[1::2]  # Real + Imaginary * i
    S = complex_vector.reshape(N*M, R)
    #S = complex_matrix * ant_calib_coeff[:, np.newaxis]



    # remove tx/rx ant index 
    S[:, 0] = 0
    S[:, 1] = 0

    # weight the waveform
    W = np.linspace(1, 5, 128)
    S = S*W
    #S = S.T
    
    
    ##############################################
    # beamforming for query function B(azi_rad, ele_rad)
    ##############################################
    B_azi = E @ S
    
    ##############################################
    # beamforming for query function B(azi_rad, ele_rad)
    ##############################################
    # Iterate over azi_rad and ele_rad values
    B_ele = H @ S

    ###################################
    # plot azimuth map 
    ###################################
    #pdb.set_trace()
    COLOR_BD = 150
    
    # Assuming B is already calculated as per your previous steps
    B_azi_value = np.abs(B_azi)  # Compute the B_azi_value (absolute value of B)
    B_azi_value = B_azi_value.flatten()  # Flatten for consistency
    #b_azi_alpha = (B_azi_value - B_azi_value.min()) / (B_azi_value.max() - B_azi_value.min())
    #b_azi_alpha = np.abs(b_azi_alpha - b_azi_old)
    
    #B_azi_value[0] = COLOR_BD
    #B_azi_value[B_azi_value > COLOR_BD] = COLOR_BD
    
    #ax1.clear()
    #sc = ax1.scatter(x_azi, y_azi, c=B_azi_value, cmap='jet', s=50)
    scatter_azi.set_array(np.sqrt(B_azi_value))
    #scatter_azi.set_offsets(np.c_[x_azi, y_azi]) 
    ax1.set_title('Azimuth map: '+str(loop_index))


    ###################################
    # plot elevation map 
    ###################################
    #pdb.set_trace()
    
    
    # Assuming B is already calculated as per your previous steps
    B_ele_value = np.abs(B_ele)  # Compute the B_azi_value (absolute value of B)
    B_ele_value = B_ele_value.flatten()  # Flatten for consistency
    #b_ele_alpha = (B_ele_value - B_ele_value.min()) / (B_ele_value.max() - B_ele_value.min())
    #b_ele_alpha = np.abs(b_ele_alpha - b_ele_old)
    
    #B_ele_value[0] = COLOR_BD
    #B_ele_value[B_ele_value > COLOR_BD] = COLOR_BD
    
    #ax2.clear()
    # Scatter plot
    #sc = ax2.scatter(z_ele, y_ele, c=B_ele_value, cmap='jet', s=50)
    scatter_ele.set_array(np.sqrt(B_ele_value))
    #scatter_ele.set_offsets(np.c_[z_ele, y_ele])
 
    
    b_azi_old = B_azi_value
    b_ele_old = B_ele_value
    
    #plt.draw()
    plt.pause(0.001)  # Pause to visualize both figures
 
plt.ioff()  # Turn off interactive mode
plt.show()