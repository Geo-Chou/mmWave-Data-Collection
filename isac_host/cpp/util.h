

#ifndef INCLUDED_UTIL_H
#define INCLUDED_UTIL_H
#include <iostream>
#include <iterator>
#include <algorithm>
#include <vector>
#include <complex>
using namespace std;

#define cf_type       complex<float> 

#define div_cf_type(x, y)  complex<float>(x.real()/y, x.imag()/y)
#define mul_cf_type(x, y)  complex<float>(x.real()*y, x.imag()*y)

// 4-dimentional vector
#define vec_4d_c  vector<vector<vector<vector<complex<float>>>>>
#define vec_4d_f  vector<vector<vector<vector<float>>>>
#define vec_4d_i  vector<vector<vector<vector<int>>>>

// 3-dimentional vector
#define vec_3d_c  vector<vector<vector<complex<float>>>>
#define vec_3d_f  vector<vector<vector<float>>>
#define vec_3d_i  vector<vector<vector<int>>>

// 2-dimentional vector
#define vec_2d_c  vector<vector<complex<float>>>
#define vec_2d_f  vector<vector<float>>
#define vec_2d_i  vector<vector<int>>

// 1-dimentional vector
#define vec_1d_c  vector<complex<float> >
#define vec_1d_f  vector<float>
#define vec_1d_i  vector<int>


// 4-dimentional vector
#define new_vec_4d_c(p1, p2, p3, p4)  vec_4d_c(p1, vec_3d_c(p2, vec_2d_c(p3, vec_1d_c(p4))))
#define new_vec_4d_f(p1, p2, p3, p4)  vec_4d_f(p1, vec_3d_f(p2, vec_2d_f(p3, vec_1d_f(p4))))
#define new_vec_4d_i(p1, p2, p3, p4)  vec_4d_i(p1, vec_3d_i(p2, vec_2d_i(p3, vec_1d_i(p4))))
 
// 3-dimentional vector
#define new_vec_3d_c(p1, p2, p3)  vec_3d_c(p1, vec_2d_c(p2, vec_1d_c(p3)))
#define new_vec_3d_f(p1, p2, p3)  vec_3d_f(p1, vec_2d_f(p2, vec_1d_f(p3)))
#define new_vec_3d_i(p1, p2, p3)  vec_3d_i(p1, vec_2d_i(p2, vec_1d_i(p3)))

// 2-dimentional vector
#define new_vec_2d_c(p1, p2)  vec_2d_c(p1, vec_1d_c(p2))
#define new_vec_2d_f(p1, p2)  vec_2d_f(p1, vec_1d_f(p2))
#define new_vec_2d_i(p1, p2)  vec_2d_i(p1, vec_1d_i(p2))

// 1-dimentional vector
#define new_vec_1d_c(p1)  vec_1d_c(p1)
#define new_vec_1d_f(p1)  vec_1d_f(p1)
#define new_vec_1d_i(p1)  vec_1d_i(p1)



void convert_complex_to_qpsk(vec_1d_c &complex_arr, vec_1d_c &constel_arr);
void convert_qpsk_to_complex(vec_1d_c &constel_arr, vec_1d_c &complex_arr);
void convert__int__to_qpsk(int &num, vec_1d_c &constel_arr);
void convert_qpsk_to___int(vec_1d_c &constel_arr, int &num);
void convert_short_to_qpsk(short &num, vec_1d_c &constel_arr);
void convert_qpsk_to_short(vec_1d_c &constel_arr, short &num);
#endif






