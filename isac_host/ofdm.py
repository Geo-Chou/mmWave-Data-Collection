import numpy as np
   
defined_random_bits = [1,1,1,1,1,0,0,1,1,0,1,0,1,1,0,0,1,0,1,1,1,0,0,1,1,0,0,1,1,0,1,1,1,1,1,0,1,0,0,1,0,1,1,0,1,0,0,0,0,1,1,0,0,1,0,0,0,0,1,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,0,0,1,1,1,1,0,0,0,0,0,0,0,1,0,0,1,0,1,1,1,0,0,0,0,1,1,1,0,1,1,0,0,1,1,1,0,1,1,1,1,0,0,0,1,0,1,0,1,1,1,0,1,0,1,0,1,1,1,1,1,1,0,1,0,0,0,1,1,1,1,0,1,1,1,1,0,1,1,0,0,0,0,1,0,1,1,1,0,0,1,0,1,1,1,1,1,0,0,0,0,1,0,1,0,1,0,1,0,1,1,0,1,1,0,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,1,0,1,0,1,1,1,0,0,1,1,0,0,1,0,0,1,1,0,0,0,0,0,0,1,0,1,1,1,1,1,0,0,0,0,1,1,0,0,0,1,0,1,1,0,0,1,1,1,1,0,1,0,1,0,1,0,0,0,1,1,0,1,0,0,0,1,1,0,0,0,0,1,0,0,1,1,1,0,0,1,0,1,1,1,0,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,0,1,0,1,1,1,0,0,1,0,0,0,0,1,1,0,1,0,0,1,0,0,1,0,1,0,1,1,1,1,1,0,0,0,1,0,0,1,0,1,1,0,1,0,0,0,0,1,1,1,1,0,0,1,0,1,0,1,1,0,0,0,1,1,1,0,0,0,1,0,1,1,0,1,0,1,0,1,0,0,1,0,0,1,1,0,0,1,1,0,0,1,0,1,1,0,0,1,1,0,1,0,1,0,0,0,1,1,1,0,1,1,1,0,1,0,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,1,1,0,0,0,1,0,1,1,1,1,0,0,1,1,0,0,0,0,1,1,1,1,0,1,0,1,0,0,1,0,0,1,0,1,0,1,1,0,1,1,0,0,0,1,0,0,0,0,0,0,1,0,1,1,1,0,0,0,1,1,0,1,1,1,1,0,0,1,1,1,1,1,0,1,1,1,0,1,1,0,1,0,1,1,0,1,1,1,1,0,1,0,1,1,1,0,1,1,1,0,0,0,1,1,0,1,1,1,1,1,0,0,0,1,0,1,0,1,0,1,0,0,1,1,0,0,0,1,0,0,1,0,1,1,0,0,1,1,0,0,1,0,1,1,1,0,0,1,1,1,1,1,1,1,1,0,1,1,0,0,1,1,0,0,1,0,1,0,0,1,1,1,0,0,0,1,0,1,1,1,1,1,1,1,0,0,0,0,1,0,0,0,1,1,0,0,0,0,1,1,0,1,0,1,1,0,0,0,1,0,0,1,0,1,1,0,1,1,0,1,1,0,1,0,1,0,0,1,0,0,1,1,1,0,0,0,1,1,1,1,1,1,1,1,0,0,1,1,0,1,0,1,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,1,0,1,1,1,1,0,1,1,0,1,1,1,0,0,1,1,1,0,0,1,0,1,0,0,0,1,0,1,1,1,0,1,1,0,0,1,0,1,1,1,1,0,0,1,0,1,1,0,0,0,0,1,0,0,1,1,0,0,1,1,0,1,1,0,1,0,1,1,0,0,0,1,0,0,0,0,0,1,1,1,1,0,0,0,1,1,1,0,0,1,1,0,0,0,1,1,1,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,1,1,0,1,0,1,0,0,1,1,1,1,1,1,1,0,0,0,0,1,0,0,1,0,1,0,1,1,1,1,0,1,1,1,0,0,0,0,1,1,0,1,1,0,1,1,1,1,0,0,1,1,0,1,1,1,0,1,0,1,0,1,1,1,0,0,1,0,0,0,1,0,1,1,1,0,0,0,1,0,1,1,0,1,0,1,1,1,1,0,0,0,0,0,1,1,1,1,0,1,0,0,1,1,0,0,1,0,0,1,0,1,1,1,1,0,1,0,1,1,0,0,1,0,1,1,1,0,1,0,1,0,0,0,1,1,1,0,0,0,1,0,1,1,1,1,0,0,0,1,1,0,0,0,0,1,1,1,1,1,1,0,0,1,0,1,0,0,1,0,1,1,1,0,0,1,1,0,1,1,1,0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,0,1,1,0,1,0,1,1,0,1,0,0,0,1,0,0,1,0,1,1,1,1,0,1,1,1,1,0,1,0,1,0,1,1,0,0,1,0,1,1,0,0,0,1,0,1,1,0,0,0,1,0,1,1,1,0,0,0,1,1,0,0,0,1,0,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,1,1,1,0,1,1,0,1,0,0,1,0,0,1,1,1,1,0,1,0,0,0,0,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,0,1,1,0,0,1,0,1,0,0,1,1,1,0,1,0,0,0,1,0,0,1,1,0,0,0,0,0,0,1,0,0,0,1,1,0,1,0,1,1,0,0,1,0,0,1,0,1,0,0,1,0,0,1,0,0,1,0,1,1,1,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,1,1,0,1,0,1,0,1,1,0,1,1,0,1,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,1,0,0,1,0,1,1,1,1,0,1,1,0,0,1,1,0,0,0,1,0,1,1,1,0,1,1,1,0,1,0,0,0,0,0,0,0,1,1,1,0,1,1,0,1,0,0,0,0,0,1,1,0,0,1,1,0,0,1,0,1,0,1,1,1,0,1,0,0,0,0,1,0,0,0,0,1,1,0,1,0,1,1,0,0,0,1,0,0,0,0,1,0,1,1,1,1,1,1,0,1,1,0,0,1,0,1,0,0,0,1,0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,1,1,0,0,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,1,1,0,1,1,0,1,0,1,1,0,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,1,1,0,1,0,0,1,1,1,1,0,1,0,1,0,0,1,0,1,0,1,1,1,0,1,1,0,1,0,1,1,1,0,1,0,1,0,1,1,1,0,0,1,0,1,1,0,0,0,1,0,1,1,0,0,1,1,1,0,1,0,1,0,1,1,0,0,0,1,1,0,1,1,1,0,0,1,0,0,0,1,1,0,0,1,0,1,0,1,0,1,1,1,0,0,1,0,0,1,0,1,0,1,1,1,1,0,1,0,1,0,0,0,0,0,0,0,1,0,0,1,1,0,1,0,0,0,1,1,0,0,1,0,0,1,1,1,1,1,1,0,0,0,1,1,0,1,1,0,0,1,0,1,1,0,1,0,1,1,1,1,0,1,1,0,0,1,0,0,0,1,0,0,1,1,1,0,1,1,1,1,0,1,0,0,0,1,1,0,1,0,1,1,1,1,1,0,0,1,1,0,0,1,0,1,1,1,0,0,1,0,1,1,0,0,0,0,0,1,1,1,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,0,0,1,0,0,1,0,0,1,1,0,0,1,1,0,0,0,1,0,1,0,0,0,1,1,0,1,0,0,1,0,1,0,0,0,0,0,0,0,1,1,0,1,1,1,0,1,1,0,1,0,1,1,0,1,0,0,0,0,1,1,0,1,0,1,1,1,0,1,1,1,1,0,1,1,0,1,0,1,1,0,1,1,0,1,1,1,1,0,1,0,1,1,1,0,1,1,1,0,1,0,0,1,1,0,1,1,1,0,0,1,0,0,1,1,1,1,0,1,1,0,1,1,0,1,0,1,1,1,1,1,1,1,1,1,0,0,1,1,1,0,1,1,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,1,1,1,0,0,1,1,0,1,0,1,0,1,1,1,0,1,0,0,1,0,1,1,0,1,0,0,0,1,0,1,0,0,1,0,1,0,1,0,0,1,1,1,1,0,0,0,1,1,1,1,1,0,0,0,0,1,1,0,1,0,0,0,0,1,0,1,0,0,0,1,1,1,1,0,0,1,0,1,0,0,0,1,0,1,0,0,0,1,1,1,0,1,0,1,0,1,0,1,0,0,0,0,1,0,0,0,0,0,1,1,0,0,1,0,0,1,1,1,1,0,1,1,1,1,0,0,1];
defined_random_bits = np.array(defined_random_bits)
defined_random_bits = 2*defined_random_bits - 1

class ofdm_generator():
    def __init__(self):   
        self.num_symbols = 64
        self.num_sc_total = 1024
        self.num_sc_valid = 900
        self.cp_len = 176
        self.modu_order = 2
        self.gap_len = 10
        self.frame_len = self.num_symbols*(self.num_sc_total + self.cp_len)
        self.SC_INDEX_up = np.arange(self.num_sc_valid//2) + self.gap_len
        self.SC_INDEX_lw = (-1)*self.SC_INDEX_up[::-1] + self.num_sc_total
        self.SC_INDEX = np.concatenate((self.SC_INDEX_lw, self.SC_INDEX_up)).astype(np.int32)
        self.MODU_DATA_MATRIX = self.generate_qpsk_signal()
        
        
    def generate_qpsk_signal(self):
        #data_bits = np.random.randint(0, 2, self.modu_order*self.num_sc_valid*self.num_symbols)
        num_bits_required = self.modu_order*self.num_sc_valid*self.num_symbols;
        ref_ofdm_symb = defined_random_bits[0::2] + 1j*defined_random_bits[1::2]
        ref_ofdm_symb = ref_ofdm_symb[0:self.num_sc_valid]
        modulated_data_matrix = np.zeros((self.num_symbols, self.num_sc_valid), dtype=complex)
        for l in range(self.num_symbols):
            for k in range(self.num_sc_valid):
                modulated_data_matrix[l, k] = ref_ofdm_symb[(k+16*l)%self.num_sc_valid]
        return modulated_data_matrix

    def modulate_ofdm_signal(self):
        freq_signal = np.zeros((self.num_symbols, self.num_sc_total), dtype=complex)
        freq_signal[:, self.SC_INDEX] = self.MODU_DATA_MATRIX
        time_signal = np.fft.ifft(freq_signal, axis=1)*np.sqrt(self.num_sc_total)
        ampl_max = np.max([np.max(time_signal.real), np.max(time_signal.imag)])
        time_signal = time_signal/ampl_max
        time_signal_w_cp = np.concatenate((time_signal[:,-self.cp_len:], time_signal), axis=1)
        #print(time_signal_w_cp.shape)
        time_signal_frame = time_signal_w_cp.flatten()
        return time_signal_frame


    def channel_estimate(self, rx_time_signal_frame):
        if len(rx_time_signal_frame) < self.num_symbols*(self.num_sc_total+self.cp_len):
            print("Error in length.")
        rx_time_signal_frame = rx_time_signal_frame[0:self.num_symbols*(self.num_sc_total+self.cp_len)]
        rx_time_signal_w_cp = rx_time_signal_frame.reshape(self.num_symbols, -1)
        rx_time_signal_wo_cp = rx_time_signal_w_cp[:, self.cp_len:]
        #print(rx_time_signal_wo_cp.shape)
        rx_freq_signal = np.fft.fft(rx_time_signal_wo_cp)/np.sqrt(self.num_sc_total)
        #print(rx_freq_signal.shape)
        esti_chan = rx_freq_signal[:,self.SC_INDEX] / self.MODU_DATA_MATRIX
        return esti_chan

    def channel_impulse_response_estimate(self, rx_time_signal_frame):
        #print(rx_time_signal_frame.shape)
    
        if len(rx_time_signal_frame) < self.num_symbols*(self.num_sc_total+self.cp_len):
            print("Error in length.")
        rx_time_signal_frame = rx_time_signal_frame[0:self.num_symbols*(self.num_sc_total+self.cp_len)]
        #print(rx_time_signal_frame.shape)
        
        rx_time_signal_w_cp = rx_time_signal_frame.reshape(self.num_symbols, -1)
        rx_time_signal_wo_cp = rx_time_signal_w_cp[:, self.cp_len:]
        #print(rx_time_signal_wo_cp.shape)
        rx_freq_signal = np.fft.fft(rx_time_signal_wo_cp)/np.sqrt(self.num_sc_total)
        #print(rx_freq_signal.shape)
        freq_chan = np.zeros((self.num_symbols, self.num_sc_total), dtype=complex)
        freq_chan[:, self.SC_INDEX] = rx_freq_signal[:,self.SC_INDEX] / self.MODU_DATA_MATRIX
        esti_cir = np.fft.ifft(freq_chan, axis=1)*np.sqrt(1024)
        return esti_cir




    #float point to fixed point  
    def convert_complex_to_int32(self, tx_signal):  
        tx_signal_real = tx_signal.real*0x7FFF
        tx_signal_imag = tx_signal.imag*0x7FFF
        tx_signal_real[tx_signal_real < 0] += 0x10000
        tx_signal_imag[tx_signal_imag < 0] += 0x10000
        tx_signal_real_uint32 = np.uint32(tx_signal_real)
        tx_signal_imag_uint32 = np.uint32(tx_signal_imag)
        #tx_signal_imag_uint32 = np.uint32(tx_signal_real)
        tx_signal_uint32 = ((tx_signal_imag_uint32 & 0xFFFF) << 16) | (tx_signal_real_uint32 & 0xFFFF)
        return tx_signal_uint32
        #print(f"tx_signal_uint32.shape = {tx_signal_uint32.shape}")


    def convert_int32_to_complex(self, rx_buf):  
        rx_buf_int16 = rx_buf.view(np.int16)
        rx_real = rx_buf_int16[1::2]  # Upper 16 bits
        rx_imag = rx_buf_int16[0::2]  # Lower 16 bits
        rx_signal = (np.int32(rx_real) + 1j*np.int32(rx_imag))/float(0x7FFF)
        return rx_signal

    def generate_golay_waveform(self):
        golay_sequ = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1];
        golay_wave = [0.514, 0.706, 0.820, 0.852, 0.835, 0.808, 0.790, 0.785, 0.789, 0.795, 0.800, 0.798, 0.787, 0.779, 0.801, 0.863, 0.919, 0.868, 0.614, 0.158, -0.356, -0.686, -0.654, -0.267, 0.268, 0.655, 0.689, 0.363, -0.148, -0.613, -0.874, -0.917, -0.842, -0.776, -0.788, -0.861, -0.911, -0.845, -0.610, -0.224, 0.223, 0.613, 0.849, 0.908, 0.849, 0.780, 0.782, 0.855, 0.916, 0.853, 0.610, 0.220, -0.215, -0.593, -0.852, -0.977, -0.977, -0.852, -0.593, -0.215, 0.220, 0.612, 0.857, 0.918, 0.854, 0.775, 0.773, 0.851, 0.920, 0.860, 0.611, 0.212, -0.224, -0.592, -0.840, -0.975, -1, -0.886, -0.591, -0.135, 0.349, 0.664, 0.660, 0.340, -0.142, -0.587, -0.876, -1, -0.996, -0.869, -0.586, -0.150, 0.331, 0.660, 0.672, 0.349, -0.156, -0.620, -0.881, -0.925, -0.857, -0.789, -0.776, -0.793, -0.796, -0.787, -0.798, -0.854, -0.913, -0.869, -0.620, -0.165, 0.348, 0.688, 0.664, 0.270, -0.284, -0.683, -0.686, -0.288, 0.276, 0.675, 0.684, 0.313, -0.220, -0.640, -0.775, -0.632];
        golay_wave_complex = np.array(golay_wave) + 1j*np.array(golay_wave)
        return golay_wave_complex
    
        

