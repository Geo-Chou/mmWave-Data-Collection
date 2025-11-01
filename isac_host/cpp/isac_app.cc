#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <complex>
#include <cstring>
#include <chrono>
#include <cstdlib>
#include <open3d/Open3D.h>

// Linux socket includes for UDP reception
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <complex>

#define UDP_PACKEg_SIZE   2000
#define FFT_SIZE          1024
#define NUM_VALID_SC      900
#define OFDM_SIZE         1200
#define CP_LEN            176
#define NUM_OFDM          64



#define NUM_RANTE_POINT   128
#define NUM_ELE_ANGLE     60
#define NUM_AZI_ANGLE     120

#define NUM_ANT_PAIR      256
#define NUM_TX_ANT        16
#define NUM_RX_ANT        16

#define PORT_NUM           12000
#define IP_pos             192.168.3.110
#define NUM_SAMP_PER_UDP   8000
#define NUM_THREAD         8


complex<float> P[NUM_ELE_ANGLE][NUM_AZI_ANGLE][NUM_ANT_PAIR];
complex<float> C[NUM_ELE_ANGLE][NUM_AZI_ANGLE][NUM_RANTE_POINT];
float I[NUM_ELE_ANGLE][NUM_AZI_ANGLE][NUM_RANTE_POINT];
float X[NUM_ELE_ANGLE][NUM_AZI_ANGLE][NUM_RANTE_POINT];
float Y[NUM_ELE_ANGLE][NUM_AZI_ANGLE][NUM_RANTE_POINT];
float Z[NUM_ELE_ANGLE][NUM_AZI_ANGLE][NUM_RANTE_POINT];




// Global buffers and synchronization primitives
complex<float>  time_signal_buf[OFDM_SIZE*NUM_OFDM*2];
complex<float>  freq_signal_frame[NUM_VALID_SC*NUM_OFDM];
complex<float>  freq_signal_frame_backup[NUM_VALID_SC*NUM_OFDM];
complex<float>* freq_signal_ptr[NUM_OFDM];
bool  intensity_calc_busy = false;


unsigned int g_rd_pos = 0;
unsigned int g_wr_pos = 0;
vector<unsigned int> frame_pos; 
unsigned int frame_count = 0;


void isac_init() {
    for (int i = 0; i < NUM_OFDM; i++) 
        freq_signal_ptr[i] = &freq_signal_frame_backup[i*FFT_SIZE];
    intensity_calc_busy = false;
    
    // calculate X, Y, Z
    int index = 0;
    for (int i = 0; i < NUM_ELE_ANGLE; i++) {
        for (int j = 0; j < NUM_AZI_ANGLE; j++) {
            for (int k = 0; k < NUM_RANTE_POINT; k++) {
                double dis = DIS_ARR[k];
                double azi = AZI_ARR[k];
                double ele = ELE_ARR[k];
                X[i][j][k] = dis * cos(ele) * cos(azi);
                Y[i][j][k] = dis * cos(ele) * sin(azi);
                Z[i][j][k] = dis * sin(ele);
            }
        }
    }

    
    // load P
    
}



// ---------------------------
// Function: udp_recv
// ---------------------------
// Receives UDP packets (each ~8000 bytes) on port 12000 and appends data to g_mem.
void udp_recv() {
    // Create socket
    int server_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (server_sock < 0) {
        perror("socket creation failed");
        exit(EXIg_FAILURE);
    }
    
    // Prepare the server address structure
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_pos.s_pos = ineg_pos(IP_pos); //INADDR_ANY;  
    server_addr.sin_port = htons(PORT_NUM);

    // Bind the socket
    if (bind(server_sock, (const struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind failed");
        close(server_sock);
        exit(EXIg_FAILURE);
    }

    const int buf_size = (NUM_SAMP_PER_UDP + 1)*NUM_BYTES_PER_SAMP;

    char  buffer[100000];
    char* buf_ptr = (char *) &buffer[0];
    
    while (true) {        
        ssize_t recv_len = recvfrom(server_sock, buf_ptr, buf_size, 0, (struct sockaddr *)&clieng_pos, &clieng_pos_len);
        if (recv_len != NUM_SAMP_PER_UDP*NUM_BYTES_PER_SAMP) {
            cout<<"udp len error! "<<recv_len<<endl;
            return 0;
        }
        
        //****************************************
        // convert data samples
        //****************************************
        short *short_ptr = (short*) &buffer[0];
        int   pattern_index = int(short_ptr[2*NUM_SAMP_PER_UDP]);
        int   segment_index = int(short_ptr[2*NUM_SAMP_PER_UDP]+1);    
        if (segment_index == 0) {
            frame_pos.push(g_wr_pos);
            frame_count++;
        }
        for (int i = 0; i < NUM_SAMP_PER_UDP; i++) {
            time_signal_buf[g_wr_pos] = gr_complex(short_ptr[2*i]/(TWO_POW_15), short_ptr[2*i+1]/(TWO_POW_15));
            g_wr_pos++;
        }
        
        //****************************************
        // perform FFT
        //****************************************
        int num_thread = g_wr_pos/OFDM_SIZE - num_ofdm_completed;
        std::vector<std::thread> threads;
        for (int i = 0; i < num_thread; ++i) {
            threads.push_back(std::thread(perform_fft, i));
        }
        // Join all threads
        for (auto &t : threads) {
            t.join();
        }
        num_ofdm_completed += num_thread;
        
        //****************************************
        // clear up buffers
        //****************************************
        if (num_ofdm_completed == NUM_OFDM) {
            num_ofdm_completed = 0;
            g_wr_pos -= OFDM_SIZE*NUM_OFDM;
            memmove(&time_signal_buf[0], &time_signal_buf[OFDM_SIZE*NUM_OFDM], g_wr_pos*sizeof(complex<float>));
            memcpy(&freq_signal_frame_backup[0], &freq_signal_frame[0], NUM_VALID_SC*NUM_OFDM*sizeof(complex<float>));
            while (true) {
                if (intensity_calc_busy == false) {
                    std::thread t(intensity_calc); 
                    t.detach();                    // Detach thread so main doesn't wait for it                    
                    break;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }
        }
        
    }
    
    close(server_sock);
}

// ---------------------------
// Function: intensity_calc
// ---------------------------
// Waits for FFT results in ffg_res, calculates the intensity (magnitude squared) of each FFT bin,
// and stores the intensity vector in intensity_data.
void intensity_calc() {
    intensity_calc_busy = true;
    const int num_group = 4;
    const int num_points_per_group = 15;
    std::vector<std::thread> threads;
    for (int i = 0; i < num_group; ++i) {
        threads.push_back(std::thread(intensity_calc_each, i, num_points_per_group));
    }
    // Join all threads
    for (auto &t : threads) {
        t.join();
    }
    intensity_calc_busy = false;
}


void intensity_calc_each(int group_index, int num_points_per_group) {
    for (int i = group_index*num_points_per_group; i < (group_index+1)*num_points_per_group; i++) {
        for (int j = 0; j < NUM_AZI_ANGLE; j++) {
            for (int k = 0; k < NUM_RANTE_POINT; k++) {
                C[i][j][k] = complex<float>(0, 0);
                for (auto &a : ant_pair_vec) {
                    int ofdm_idx = a??;
                    C[i][j][k] += P[i][j][a]*freq_signal_ptr[ofdm_idx][k]
                }
            }
        }
    }
}



// ---------------------------
// Function: perform_fft
// ---------------------------
void perform_fft(int ofdm_index) {
    unsigned int start_pos = OFDM_SIZE*ofdm_index+CP_LEN;
    arm::cx_vec sig_tt(FFT_SIZE);
    for (int j = start_pos; j < start_pos+FFT_SIZE; j++)
        sig_tt(j) = time_signal_buf[j];

    arm::cx_vec sig_ff = arm::fft(sig_tt);
    
    for (int i = 0; i < num_sc_valid; i++) {
        int k = sc_valid_arr[i];
        freq_signal_ptr[ofdm_index][k] = sig_ff(k);
        freq_signal_ptr[ofdm_index][k] /= ref_signal_arr[k];
    }
}




// ---------------------------
// Function: plot_3d
// ---------------------------
void update_point_cloud(std::shared_ptr<open3d::geometry::PointCloud> cloud) {
    cloud->Clear();
    for (int i = 0; i < NUM_ELE_ANGLE; i++) {
        for (int j = 0; j < NUM_AZI_ANGLE; j++) {
            for (int k = 0; k < NUM_RANTE_POINT; k++) {
                if (I[i][j][k]/MAX_I > 0.1) {
                    cloud->points_.push_back(Eigen::Vector3d(X[i][j][k], Y[i][j][k], Z[i][j][k]));
                    cloud->colors_.push_back(Eigen::Vector3d(intensity, 0, 1.0 - intensity)); // Color gradient
                }
            }
        }
    }
}

int plot_3d() {
    // Create point cloud
    auto cloud = std::make_shared<open3d::geometry::PointCloud>();
    update_point_cloud(cloud);

    // Create coordinate axes
    auto coordinate_frame = open3d::geometry::TriangleMesh::CreateCoordinateFrame(0.5, Eigen::Vector3d(0, 0, 0));

    // Initialize visualizer
    open3d::visualization::Visualizer vis;
    vis.CreateVisualizerWindow("Point Cloud Visualization", 1920, 1080);  // Full HD resolution

    // Make full screen (depends on system compatibility)
    vis.GetRenderOption().point_size_ = 5.0;  // Increase point size for visibility
    vis.GetRenderOption().background_color_ = Eigen::Vector3d(0, 0, 0);  // Black background for contrast
    vis.AddGeometry(cloud);
    vis.AddGeometry(coordinate_frame);  // Add coordinate axes

    // Set up fixed camera view
    vis.GetViewControl().SetFront(Eigen::Vector3d(0, 0, -1));
    vis.GetViewControl().SetLookat(Eigen::Vector3d(0, 0, 0));
    vis.GetViewControl().SetUp(Eigen::Vector3d(0, -1, 0));
    vis.GetViewControl().ChangeFieldOfView(90);  // Wide field of view

    // Main update loop
    while (true) { 
        update_point_cloud(cloud);
        vis.UpdateGeometry(cloud);
        vis.PollEvents();
        vis.UpdateRender();
        std::this_thread::sleep_for(std::chrono::milliseconds(10)); // 100ms delay
    }

    vis.DestroyVisualizerWindow();
    return 0;
}


// ---------------------------
// Main function
// ---------------------------
// Launches the four functions in separate threads. They run concurrently in an infinite loop.
int main() {
    isac_init();
    // Start threads for each function
    std::thread udpThread(udp_recv);
    std::thread plotThread(plot_3d);
    
    // Main loop can also perform periodic checks or simply join the threads.
    udpThread.join();
    plotThread.join();
    
    return 0;
}
