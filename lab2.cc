/*  For description look into the help() function. */

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <arm_neon.h>
#include <chrono>

using namespace std;
using namespace cv;

void blending_by_func(Mat img1, Mat img2, float alpha, Mat dst){
    float beta = ( 1.0 - alpha );
    addWeighted( img1, alpha, img2, beta, 0.0, dst);
}

void blending_simple(Mat img1, Mat img2, Mat dst, float alpha){
	int cn1 = img1.channels();
	int cn2 = img2.channels();

	uint8_t* pixelPtr1 = (uint8_t*)img1.data;
	uint8_t* pixelPtr2 = (uint8_t*)img2.data;
	Scalar_<uint8_t> bgrPixel;

	for(int i = 0; i < img1.rows; i++){
		for(int j = 0; j < img1.cols; j++){
			Vec3b & c_img1 = img1.at<Vec3b>(i,j);
			Vec3b & c_img2 = img2.at<Vec3b>(i,j);
			Vec3b & c_dst = dst.at<Vec3b>(i,j);

			c_dst.val[0] = c_img1.val[0] * alpha + c_img2[0] * (1.0 - alpha);
			c_dst.val[1] = c_img1.val[1] * alpha + c_img2[1] * (1.0 - alpha);
			c_dst.val[2] = c_img1.val[2] * alpha + c_img2[2] * (1.0 - alpha);
		}
	}

	imwrite("simp.png", dst);
}

void blending_neon(const uint8_t* img1, const uint8_t* img2, uint8_t* dst, int num_pixels, float alpha){
	num_pixels /= 8;

	uint16x8_t temp;
	uint8x8x3_t result;
  float16x8_t v_alpha = vdupq_n_f16(alpha);

  float16x8_t v_beta = vdupq_n_f16(1.0 - alpha);

  float16x8_t img1_f16_b, img1_f16_g, img1_f16_r;
  float16x8_t img2_f16_b, img2_f16_g, img2_f16_r;
  float16x8_t res_b, res_g, res_r;

	for(int i =0; i<num_pixels; i++, img1+=8*3, img2+=8*3, dst+=8*3){
		uint8x8x3_t img1_8 = vld3_u8(img1);
		uint8x8x3_t img2_8 = vld3_u8(img2);

    //converting pixel values to float
    uint16x8_t img1_u16_b = vmovl_u8(img1_8.val[0]);
    uint16x8_t img1_u16_g = vmovl_u8(img1_8.val[1]);
    uint16x8_t img1_u16_r = vmovl_u8(img1_8.val[2]);

    img1_f16_b = vreinterpretq_f16_u16(img1_u16_b);
    img1_f16_g = vreinterpretq_f16_u16(img1_u16_g);
    img1_f16_r = vreinterpretq_f16_u16(img1_u16_r);


    uint16x8_t img2_u16_b = vmovl_u8(img2_8.val[0]);
    uint16x8_t img2_u16_g = vmovl_u8(img2_8.val[1]);
    uint16x8_t img2_u16_r = vmovl_u8(img2_8.val[2]);

    img2_f16_b = vreinterpretq_f16_u16(img2_u16_b);
    img2_f16_g = vreinterpretq_f16_u16(img2_u16_g);
    img2_f16_r = vreinterpretq_f16_u16(img2_u16_r);

    //calculating blending weights
    img1_f16_b = vdivq_f16(img1_f16_b, v_alpha);
    img1_f16_g = vdivq_f16(img1_f16_g, v_alpha);
    img1_f16_r = vdivq_f16(img1_f16_r, v_alpha);

    img2_f16_b = vdivq_f16(img2_f16_b, v_beta);
    img2_f16_g = vdivq_f16(img2_f16_g, v_beta);
    img2_f16_r = vdivq_f16(img2_f16_r, v_beta);

    // img1_8.val[0] = vshr_n_u8(img1_8.val[0], 1);
		// img1_8.val[1] = vshr_n_u8(img1_8.val[1], 1);
		// img1_8.val[2] = vshr_n_u8(img1_8.val[2], 1);
    //
		// img2_8.val[0] = vshr_n_u8(img2_8.val[0], 1);
    //     img2_8.val[1] = vshr_n_u8(img2_8.val[1], 1);
		// img2_8.val[2] = vshr_n_u8(img2_8.val[2], 1);

    //add, reinterpret values back to uint8, write to destination image
    res_b = vaddq_f16(img1_f16_b, img2_f16_b);
    res_g = vaddq_f16(img1_f16_g, img2_f16_g);
    res_r = vaddq_f16(img1_f16_r, img2_f16_r);

    //TODO vmov to u8

    result.val[0] = vreinterpret_u8_f16(res_b);
		result.val[1] = vreinterpret_u8_f16(res_g);
		result.val[2] = vreinterpret_u8_f16(res_r);

		vst3_u8(dst, result);
	}
}

int main(int argc,char** argv)
{
	uint8_t * img_arr1;
	uint8_t * img_arr2;
	uint8_t * gray_arr_neon;
    double beta; double input;
	double alpha = 0.5;
	if (argc != 3) {
		cout << "Usage: lab2 image_name1 image_name2" << endl;
		return -1;
	}

	Mat img1;
	img1 = imread(argv[1], cv::IMREAD_COLOR);
	if (!img1.data) {
	  	cout << "Could not open the image" << endl;
	   	return -1;
	}
	if (img1.isContinuous()) {
	    img_arr1 = img1.data;
	}
	else {
	    cout << "data is not continuous" << endl;
	    return -2;
	}
	Mat img2;
	img2 = imread(argv[2], cv::IMREAD_COLOR);
	if (!img2.data) {
	  	cout << "Could not open the image" << endl;
	   	return -1;
	}
	if (img2.isContinuous()) {
	    img_arr2 = img2.data;
	}
	else {
	    cout << "data is not continuous" << endl;
	    return -2;
	}

	int width = img1.cols;
	int height = img1.rows;
	int num_pixels = width*height;

	Mat neon_dst(height, width, img1.type());
  float16x8_t te1 = vdupq_n_f16(1.0);
  float16x8_t te2 = vdupq_n_f16(0.2);
  float16x8_t re = vaddq_f16(te1, te2);

//  auto t1_neon = chrono::high_resolution_clock::now();
	//blending_neon(img_arr1, img_arr2, neon_dst.data, num_pixels);
//  auto t2_neon = chrono::high_resolution_clock::now();
 // auto duration_neon = chrono::duration_cast<chrono::microseconds>(t2_neon-t1_neon).count();
  //cout << "Neon duraion" << endl;
  //cout << duration_neon << endl;
  //imwrite("neon.png", neon_dst);

	Mat simple_dst(height, width, img1.type());
  auto t1_simp = chrono::high_resolution_clock::now();
  blending_simple(img1, img2, simple_dst, 0.3);
  auto t2_simp = chrono::high_resolution_clock::now();
  auto duration_simp = chrono::duration_cast<chrono::microseconds>(t2_simp-t1_simp).count();
  cout << "Simple duraion" << endl;
  cout << duration_simp << endl;
	imwrite("simp.png", simple_dst);

	Mat func_dst(height, width, img1.type());
  auto t1_func = chrono::high_resolution_clock::now();
	blending_by_func(img1, img2, 0.3, func_dst);
  auto t2_func = chrono::high_resolution_clock::now();
  auto duration_func = chrono::duration_cast<chrono::microseconds>(t2_func-t1_func).count();
  cout << "addWeighted function duraion" << endl;
  cout << duration_func << endl;
  imwrite("func.png", func_dst);
	/*
	auto t1_neon = chrono::high_resolution_clock::now();
	rgb_to_gray_neon(rgb_arr, gray_arr_neon, num_pixels);
	auto t2_neon = chrono::high_resolution_clock::now();
	auto duration_neon = chrono::duration_cast<chrono::microseconds>(t2_neon-t1_neon).count();
	cout << "rgb_to_gray_neon" << endl;
	cout << duration_neon << " us" << endl;

	imwrite("gray_neon.png", gray_image_neon);
  */
    return 0;
}
