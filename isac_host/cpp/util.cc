#include "util.h"
#include <iostream>
using namespace std;

 
void convert__int__to_qpsk(int &num, vec_1d_c &constel_arr) {
	
	constel_arr.clear();
	
	// convert channel to modulation
	vec_1d_c constellation_list(4);
	constellation_list[0] = complex<float>(-0.7074, -0.7074);
	constellation_list[1] = complex<float>(-0.7074,  0.7074);
	constellation_list[2] = complex<float>( 0.7074, -0.7074);
	constellation_list[3] = complex<float>( 0.7074,  0.7074);
	
	int char_len = sizeof(int);
	unsigned char *char_ptr = (unsigned char*) (&num);
	for (int i = 0; i < char_len; i++) {
		unsigned char a = char_ptr[i];
		for (int j = 0; j < 4; j++) {
			constel_arr.push_back(constellation_list[a & 0x03]);
			a = a >> 2;
		}
	}
}



void convert_qpsk_to___int(vec_1d_c &constel_arr, int &num) {

	vec_1d_i const_pow(8);
	const_pow[0] = 1; 
	const_pow[1] = 2; 
	const_pow[2] = 4; 
	const_pow[3] = 8; 
	const_pow[4] = 16; 
	const_pow[5] = 32; 
	const_pow[6] = 64; 
	const_pow[7] = 128; 
	
	int char_len = sizeof(int);
	unsigned char *char_ptr = (unsigned char*) (&num);
	for (int i = 0; i < char_len; i++) {
		unsigned char a = 0;
		for (int j = 0; j < 4; j++) {
			a += constel_arr[4*i+j].real() > 0 ? const_pow[2*j+1] : 0;
			a += constel_arr[4*i+j].imag() > 0 ? const_pow[2*j] : 0;
		}
		char_ptr[i] = a;
	}
}



void convert_short_to_qpsk(short &num, vec_1d_c &constel_arr) {
	
	constel_arr.clear();
	
	// convert channel to modulation
	vec_1d_c constellation_list(4);
	constellation_list[0] = complex<float>(-0.7074, -0.7074);
	constellation_list[1] = complex<float>(-0.7074,  0.7074);
	constellation_list[2] = complex<float>( 0.7074, -0.7074);
	constellation_list[3] = complex<float>( 0.7074,  0.7074);
	
	int char_len = sizeof(short);
	unsigned char *char_ptr = (unsigned char*) (&num);
	for (int i = 0; i < char_len; i++) {
		unsigned char a = char_ptr[i];
		for (int j = 0; j < 4; j++) {
			constel_arr.push_back(constellation_list[a & 0x03]);
			a = a >> 2;
		}
	}
}



void convert_qpsk_to_short(vec_1d_c &constel_arr, short &num) {

	vec_1d_i const_pow(8);
	const_pow[0] = 1; 
	const_pow[1] = 2; 
	const_pow[2] = 4; 
	const_pow[3] = 8; 
	const_pow[4] = 16; 
	const_pow[5] = 32; 
	const_pow[6] = 64; 
	const_pow[7] = 128; 
	
	int char_len = sizeof(short);
	unsigned char *char_ptr = (unsigned char*) (&num);
	for (int i = 0; i < char_len; i++) {
		unsigned char a = 0;
		for (int j = 0; j < 4; j++) {
			a += constel_arr[4*i+j].real() > 0 ? const_pow[2*j+1] : 0;
			a += constel_arr[4*i+j].imag() > 0 ? const_pow[2*j] : 0;
		}
		char_ptr[i] = a;
	}
}




void convert_complex_to_qpsk(vec_1d_c &complex_arr, vec_1d_c &constel_arr) {
	
	constel_arr.clear();
	
	// convert channel to modulation
	vec_1d_c constellation_list(4);
	constellation_list[0] = complex<float>(-0.7074, -0.7074);
	constellation_list[1] = complex<float>(-0.7074,  0.7074);
	constellation_list[2] = complex<float>( 0.7074, -0.7074);
	constellation_list[3] = complex<float>( 0.7074,  0.7074);
	
	int char_len = complex_arr.size()*sizeof(complex<float>);
	unsigned char *char_ptr = (unsigned char*) (&complex_arr[0]);
	for (int i = 0; i < char_len; i++) {
		unsigned char a = char_ptr[i];
		for (int j = 0; j < 4; j++) {
			constel_arr.push_back(constellation_list[a & 0x03]);
			a = a >> 2;
		}
	}
	
}
	
	
	
void convert_qpsk_to_complex(vec_1d_c &constel_arr, vec_1d_c &complex_arr) {

	vec_1d_i const_pow(8);
	const_pow[0] = 1; 
	const_pow[1] = 2; 
	const_pow[2] = 4; 
	const_pow[3] = 8; 
	const_pow[4] = 16; 
	const_pow[5] = 32; 
	const_pow[6] = 64; 
	const_pow[7] = 128; 
	
	if (constel_arr.size()%32 != 0) cout<<"The number of constellation is not correct!"<<endl;
	int comp_len = constel_arr.size()/32;
	complex_arr.clear();
	for (int i = 0; i < comp_len; i++) complex_arr.push_back(complex<float>(0.0, 0.0));
	
	int char_len = comp_len*sizeof(complex<float>);
	unsigned char *char_ptr = (unsigned char*) (&complex_arr[0]);
	
	for (int i = 0; i < char_len; i++) {
		unsigned char a = 0;
		for (int j = 0; j < 4; j++) {
			a += constel_arr[4*i+j].real() > 0 ? const_pow[2*j+1] : 0;
			a += constel_arr[4*i+j].imag() > 0 ? const_pow[2*j] : 0;
		}
		char_ptr[i] = a;
	}
}

