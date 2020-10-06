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

void blending_neon(const float32_t* img1, const float32_t* img2, float32_t* dst, int num_pixels, float alpha){
	num_pixels /= 8;

	float32x2x3_t result;
  float32x2_t v_alpha = vdup_n_f32(alpha);

  float32x2_t v_beta = vdup_n_f32(1.0 - alpha);

  float32x2_t img1_f32_b, img1_f32_g, img1_f32_r;
  float32x2_t img2_f32_b, img2_f32_g, img2_f32_r;
  float32x2_t res_b, res_g, res_r;

	for(int i =0; i<num_pixels; i++, img1+=2*3, img2+=2*3, dst+=2*3){
		float32x2x3_t img1_8 = vld3_f32(img1);
		float32x2x3_t img2_8 = vld3_f32(img2);
		
    //calculating blending weights
    img1_f32_b = vmul_n_f32(img1_8.val[0], alpha);
    img1_f32_g = vmul_n_f32(img1_8.val[1], alpha);
    img1_f32_r = vmul_n_f32(img1_8.val[2], alpha);

    img2_f32_b = vmul_n_f32(img2_8.val[0], 1.0 - alpha);
    img2_f32_g = vmul_n_f32(img2_8.val[1], 1.0 - alpha);
    img2_f32_r = vmul_n_f32(img2_8.val[2], 1.0 - alpha);

    // img1_8.val[0] = vshr_n_u8(img1_8.val[0], 1);
		// img1_8.val[1] = vshr_n_u8(img1_8.val[1], 1);
		// img1_8.val[2] = vshr_n_u8(img1_8.val[2], 1);
    //
		// img2_8.val[0] = vshr_n_u8(img2_8.val[0], 1);
    //     img2_8.val[1] = vshr_n_u8(img2_8.val[1], 1);
		// img2_8.val[2] = vshr_n_u8(img2_8.val[2], 1);

    result.val[0] = vadd_f32(img1_f32_b, img2_f32_b);
    result.val[1] = vadd_f32(img1_f32_g, img2_f32_g);
    result.val[2] = vadd_f32(img1_f32_r, img2_f32_r);
		
		vst3_f32(dst, result);
	}
}

int main(int argc,char** argv)
{
	uint8_t * img_arr1;
	uint8_t * img_arr2;
	uint8_t * gray_arr_neon;
    double beta; double input;
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

  auto t1_neon = chrono::high_resolution_clock::now();
  blending_neon((float32_t*)img_arr1, (float32_t*)img_arr2, (float32_t*)neon_dst.data, num_pixels, 0.5);
  auto t2_neon = chrono::high_resolution_clock::now();
 auto duration_neon = chrono::duration_cast<chrono::microseconds>(t2_neon-t1_neon).count();
 cout << "Neon duraion" << endl;
 cout << duration_neon << endl;
  imwrite("neon_1.png", neon_dst);

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
