import numpy as np


class SpatialProjection():
    def __init__(self, Ant_Pattern_Scheme=1, R=64):   

        self.N = 16
        self.M = 16
        self.R = R
        self.lambda_val = 3e11 / 60.48e9  # unit: mm
        self.pix2_div_lambda_val = 2*np.pi/self.lambda_val
        self.ant_calib_matrix = np.ones((4, 64), dtype=np.complex128)
        
        self.E  = 0    
        self.E0 = 0     
        self.E1 = 0     
        self.E2 = 0     
        self.E3 = 0     
        self.X  = 0
        self.Y  = 0
        self.Z  = 0
        self.P_TX = 0
        self.P_RX = 0

        if Ant_Pattern_Scheme == 0:
            tx_ant_pattern0 = [0, 4, 8,  12]    #evk.wr('trx_tx_on', 0x1F1111)
            tx_ant_pattern1 = [1, 5, 9,  13]    #evk.wr('trx_tx_on', 0x1F2222)
            tx_ant_pattern2 = [2, 6, 10, 14]    #evk.wr('trx_tx_on', 0x1F4444)
            tx_ant_pattern3 = [3, 7, 11, 15]    #evk.wr('trx_tx_on', 0x1F8888)
        else:
            tx_ant_pattern0 = [0, 5, 8,  13]    #evk.wr('trx_tx_on', 0x1F2121)
            tx_ant_pattern1 = [1, 4, 9,  12]    #evk.wr('trx_tx_on', 0x1F1212)
            tx_ant_pattern2 = [2, 7, 10, 15]    #evk.wr('trx_tx_on', 0x1F8484)
            tx_ant_pattern3 = [3, 6, 11, 14]    #evk.wr('trx_tx_on', 0x1F4848)

        self.ant_calib_coeff = np.zeros(16*16, dtype=np.complex128)
        ant_index_matrix = np.arange(16*16).reshape(16, 16)
        self.pattern_index0 = ant_index_matrix[tx_ant_pattern0, :].flatten()
        self.pattern_index1 = ant_index_matrix[tx_ant_pattern1, :].flatten()
        self.pattern_index2 = ant_index_matrix[tx_ant_pattern2, :].flatten()
        self.pattern_index3 = ant_index_matrix[tx_ant_pattern3, :].flatten()

        # Constants
        self.distance = np.arange(self.R) * 0.12  # Distance array (1:fft_bin_len) * 0.12
        self.azi_angle_arr = np.arange(30, 150, 1)  # azi_rad array from 30 to 150 degrees
        self.ele_angle_arr = np.arange(-30, 30, 1)  # ele_rad array with only one value 0 (could be expanded)

        self.antenna_config()
        self.calc_XYZ()
        if True:
            self.load_E_from_file()
        else:
            self.load_ant_calib_coeff()
            self.calc_and_save_E()



    ###################################
    # antenna arrangement
    ###################################
    def antenna_config(self):
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
        self.P_TX = np.vstack([TX_X, np.zeros_like(TX_X), TX_Z])
        self.P_RX = np.vstack([RX_X, np.zeros_like(RX_X), RX_Z])

    ##############################
    # construct the plot 3D 
    ##############################
    def calc_XYZ(self):
        Az, El, Dist = np.meshgrid(self.ele_angle_arr, self.azi_angle_arr, self.distance)

        # Convert spherical coordinates to Cartesian coordinates
        xx = Dist * np.cos(np.radians(El)) * np.cos(np.radians(Az))  # X-coordinates
        yy = Dist * np.cos(np.radians(El)) * np.sin(np.radians(Az))  # Y-coordinates
        zz = Dist * np.sin(np.radians(El))  # Z-coordinates

        # Reshape the data for plotting (flatten the 3D grid into 1D vectors)
        self.X = xx.flatten()
        self.Y = yy.flatten()
        self.Z = zz.flatten()
        
        
    ###########################################
    # calc_and_save_E 
    ###########################################
    def calc_and_save_E(self):
        self.E = np.zeros((len(self.azi_angle_arr), len(self.ele_angle_arr), self.N*self.M), dtype=complex)
        for ll in range(len(self.azi_angle_arr)):
            for kk in range(len(self.ele_angle_arr)):
                theta = self.azi_angle_arr[ll] * np.pi / 180  # Convert to radians
                phi = self.ele_angle_arr[kk] * np.pi / 180  # Convert to radians
                d = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])
                steering_direction = np.zeros(self.N*self.M, dtype=complex)  # Initialize steering direction

                # Nested loop over TX and RX antenna indices
                for m in range(self.M):  # TX antenna index (1 to 16)
                    Pm1 = self.P_TX[:, 2*m]
                    Pm2 = self.P_TX[:, 2*m+1]
                    for n in range(self.N):  # RX antenna index (1 to 16)
                        Pn1 = self.P_RX[:, 2*n]
                        Pn2 = self.P_RX[:, 2*n+1]

                        # Calculate phase for the current TX-RX pair
                        phase_mn = np.exp(-1j * self.pix2_div_lambda_val * np.dot(d, (Pm1 + Pn1))) + \
                                   np.exp(-1j * self.pix2_div_lambda_val * np.dot(d, (Pm1 + Pn2))) + \
                                   np.exp(-1j * self.pix2_div_lambda_val * np.dot(d, (Pm2 + Pn1))) + \
                                   np.exp(-1j * self.pix2_div_lambda_val * np.dot(d, (Pm2 + Pn2)))

                        # Assign to the steering direction matrix
                        steering_direction[m*16+n] = phase_mn

                # Store the result in the B matrix
                self.E[ll, kk, :] = steering_direction * self.ant_calib_coeff

        E_avg = np.mean(np.abs(self.E))
        self.E = self.E / E_avg
        np.save("E.npy", self.E)


    def load_E_from_file(self):
        self.E = np.load("./isac_host/E.npy")
        self.E0 = self.E[:, :, self.pattern_index0]
        self.E1 = self.E[:, :, self.pattern_index1]
        self.E2 = self.E[:, :, self.pattern_index2]
        self.E3 = self.E[:, :, self.pattern_index3]


    ###########################################
    # load_ant_calib_coeff 
    ###########################################
    def load_ant_calib_coeff(self):
        with open("ant_calib.bin", "rb") as f:
            data = np.fromfile(f, dtype=np.float32)

        # Step 2: Convert to complex vector
        # Assuming the file contains interleaved real and imaginary parts
        if len(data) != 2*256:
            print("The data len is incorrect!")
            exit(0);
        ant_calib = data[0::2] + 1j * data[1::2]  # Real + Imaginary * i
        ant_calib = np.array(ant_calib) + 0.000001
        self.ant_calib_coeff = np.conj(ant_calib)/np.square(np.abs(ant_calib))
     
         
         
    ###########################################
    # record_calibration_coeff 
    ###########################################
    def save_ant_calib_coeff(self, pattern_index, calib_coeff):
        if pattern_index == 0:
            pattern_indices = self.pattern_index0
        elif pattern_index == 1:
            pattern_indices = self.pattern_index1
        elif pattern_index == 2:
            pattern_indices = self.pattern_index2
        else:
            pattern_indices = self.pattern_index3
        
        self.ant_calib_coeff[pattern_indices] = calib_coeff

