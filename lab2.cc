/*  For description look into the help() function. */

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <arm_neon.h>
#include <chrono>

using namespace std;
using namespace cv;

void rgb_to_gray(const uint8_t* rgb, uint8_t* gray, int num_pixels)
{
	cout << "inside function rgb_to_gray" << endl;
	auto t1 = chrono::high_resolution_clock::now();
	for(int i=0; i<num_pixels; ++i, rgb+=3) {
		int v = (77*rgb[0] + 150*rgb[1] + 29*rgb[2]);
		gray[i] = v>>8;
	}
	auto t2 = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(t2-t1).count();
	cout << duration << " us" << endl;
}
void blending_by_func(Mat img1, Mat img2, double alpha){
    Mat dst;
    double beta = ( 1.0 - alpha );

    addWeighted( img1, alpha, img2, beta, 0.0, dst);

    imwrite( "Linear.png", dst );
}

void blending_simple(Mat img1, Mat img2, Mat dst){
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

			c_dst.val[0] = c_img1.val[0] * 0.5 + c_img2[0] * 0.5;
			c_dst.val[1] = c_img1.val[1] * 0.5 + c_img2[1] * 0.5;
			c_dst.val[2] = c_img1.val[2] * 0.5 + c_img2[2] * 0.5;
		}
	}

	imwrite("simp.png", dst);
}

void blending_neon(const uint8_t* img1, const uint8_t* img2, uint8_t* dst, int num_pixels){
	num_pixels /= 8;

	uint16x8_t temp;
	uint8x8x3_t result;

	for(int i =0; i<num_pixels; i++, img1+=8*3, img2+=8*3, dst+=8*3){
		uint8x8x3_t img1_8 = vld3_u8(img1);
		uint8x8x3_t img2_8 = vld3_u8(img2);

		img1_8.val[0] = vshr_n_u8(img1_8.val[0], 1);
		img1_8.val[1] = vshr_n_u8(img1_8.val[1], 1);
		img1_8.val[2] = vshr_n_u8(img1_8.val[2], 1);

		img2_8.val[0] = vshr_n_u8(img2_8.val[0], 1);
    img2_8.val[1] = vshr_n_u8(img2_8.val[1], 1);
		img2_8.val[2] = vshr_n_u8(img2_8.val[2], 1);

		result.val[0] = vadd_u8(img1_8.val[0], img2_8.val[0]);
		result.val[1] = vadd_u8(img1_8.val[1], img2_8.val[1]);
		result.val[2] = vadd_u8(img1_8.val[2], img2_8.val[2]);
		//result = vshrn_n_u16(temp, 1);

		vst3_u8(dst, result);
	}
}
void rgb_to_gray_neon(const uint8_t* rgb, uint8_t* gray, int num_pixels) {
	// We'll use 64-bit NEON registers to process 8 pixels in parallel.
	num_pixels /= 8;
	// Duplicate the weight 8 times.
	uint8x8_t w_r = vdup_n_u8(77);
	uint8x8_t w_g = vdup_n_u8(150);
	uint8x8_t w_b = vdup_n_u8(29);
	// For intermediate results. 16-bit/pixel to avoid overflow.
	uint16x8_t temp;
	// For the converted grayscale values.
	uint8x8_t result;
	auto t1_neon = chrono::high_resolution_clock::now();
	for(int i=0; i<num_pixels; ++i, rgb+=8*3, gray+=8) {
	    // Load 8 pixels into 3 64-bit registers, split by channel.
	    uint8x8x3_t src = vld3_u8(rgb);
	    // Multiply all eight red pixels by the corresponding weights.
	    temp = vmull_u8(src.val[0], w_r);
	    // Combined multiply and addition.
	    temp = vmlal_u8(temp, src.val[1], w_g);
	    temp = vmlal_u8(temp, src.val[2], w_b);
	    // Shift right by 8, "narrow" to 8-bits (recall temp is 16-bit).
	    result = vshrn_n_u16(temp, 8);
	    // Store converted pixels in the output grayscale image.
	    vst1_u8(gray, result);
	}

	auto t2_neon = chrono::high_resolution_clock::now();
	auto duration_neon = chrono::duration_cast<chrono::microseconds>(t2_neon-t1_neon).count();
	cout << "inside function rgb_to_gray_neon" << endl;
	cout << duration_neon << " us" << endl;
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

	//blending(img1, img2, alpha);

	Mat dst(height, width, 16);
	//blending_neon(img_arr1, img_arr2, dst.data, num_pixels);
	blending_simple(img1, img2, dst);
	//imshow("Neon blending", dst);
  //waitKey(0);
  imwrite("test.png", dst);

	/*int width = rgb_image.cols;
	int height = rgb_image.rows;
	int num_pixels = width*height;
	Mat gray_image_neon(height, width, CV_8UC1, Scalar(0));
	gray_arr_neon = gray_image_neon.data;


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
