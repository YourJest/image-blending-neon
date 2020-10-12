/*  For description look into the help() function. */

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <arm_neon.h>
#include <chrono>

using namespace std;
using namespace cv;

void blending_by_func(Mat img1, Mat img2, float alpha, Mat dst){
    //https://docs.opencv.org/3.4/d5/dc4/tutorial_adding_images.html

    addWeighted( img1, alpha, img2, 1.0 - alpha, 0.0, dst);
}

void blending_simple(Mat img1, Mat img2, Mat dst, float alpha){

  Scalar_<uint8_t> bgrPixel;

  //Just taking every value and calculating it;
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
}

void blending_neon(const uint8_t* img1, const uint8_t* img2, uint8_t* dst, int num_pixels, float alpha){
  num_pixels /= 8; //Format of the images should be 8-bits per pixel;

  uint8x8x3_t result;
  bool first = true;
  //We're loading triplets of data, so we're working with 24(8*3) values per iteration
  for(int i =0; i<num_pixels; i++, img1+=8*3, img2+=8*3, dst+=8*3){
    uint8x8x3_t img1_u8 = vld3_u8(img1);
    uint8x8x3_t img2_u8 = vld3_u8(img2);
    /*Debug print of loaded value
    if(first){
      cout << "orig_image first values: "<< endl;
      uint8_t r_d[8];
      for(int k = 0; k<3;k++){
        vst1_u8(r_d, img2_u8.val[k]);
        for(int l= 0; l<8;l++){
          cout << to_string(r_d[l]) << ' ';
        }
        cout << endl;
      }
      cout << endl;
    }*/
    for(int j = 0; j < 3; j++){
      //Because ARM does not have integer division(or multiplying int by float), we should convert it to float
      uint16x8_t img1_u16 = vmovl_u8(img1_u8.val[j]);
      uint16x8_t img2_u16 = vmovl_u8(img2_u8.val[j]);

      uint32x4_t img1_u32l, img1_u32h, img2_u32l, img2_u32h;
      float32x4_t img1_f32l, img1_f32h, img2_f32l, img2_f32h;
      //We're loading 8 values per iteration, so we need to divide int by 2 parts
      img1_u32l = vmovl_u16(vget_low_u16(img1_u16));
      img1_u32h = vmovl_u16(vget_high_u16(img1_u16));
      img1_f32l = vcvtq_f32_u32(img1_u32l);
      img1_f32h = vcvtq_f32_u32(img1_u32h);

      img2_u32l = vmovl_u16(vget_low_u16(img2_u16));
      img2_u32h = vmovl_u16(vget_high_u16(img2_u16));
      img2_f32l = vcvtq_f32_u32(img2_u32l);
      img2_f32h = vcvtq_f32_u32(img2_u32h);

      //Image blending formula is alpha*src1 + beta*src2; beta = 1.0 - alpha
      img1_f32h = vmulq_n_f32(img1_f32h, alpha);
      img1_f32l = vmulq_n_f32(img1_f32l, alpha);
      img2_f32h = vmulq_n_f32(img2_f32h, 1.0 - alpha);
      img2_f32l = vmulq_n_f32(img2_f32l, 1.0 - alpha);

      float32x4_t res_f32h = vaddq_f32(img1_f32h, img2_f32h);
      float32x4_t res_f32l = vaddq_f32(img1_f32l, img2_f32l);
      /* Convertion debugging
      if(first){
        cout << "float vector convert: " << endl;
        float32_t r_dl[4];
        float32_t r_dh[4];
        vst1q_f32(r_dl, res_f32l);
        vst1q_f32(r_dh, res_f32h);
        for(int k = 0; k<4;k++){
          cout << to_string(r_dl[k]) << ' ';
        }
        for(int k = 0; k<4;k++){
          cout << to_string(r_dh[k]) << ' ';
        }
        cout << endl;
      }*/

      //Converting result back to uint8x8
      uint32x4_t res_u32h = vcvtq_u32_f32(res_f32h);
      uint32x4_t res_u32l = vcvtq_u32_f32(res_f32l);

      //Lower values must go first! I don't know why, but I'm sure this how vmovn_high works
      uint16x8_t res = vmovn_high_u32(vget_low_u16(res), res_u32l);
      res = vmovn_high_u32(vget_high_u16(res), res_u32h);
      result.val[j] = vmovn_u16(res);
    }
    /*Result debugging
    if(first){
      first = false;
      uint8_t r_d[8];
      std::cout << "result verctor after blending filter" << '\n';
      for(int k = 0; k<3;k++){
        vst1_u8(r_d, result.val[k]);
        for(int l= 0; l<8;l++){
          cout << to_string(r_d[l]) << ' ';
        }
        cout << endl;
      }
      cout << endl;
    }
    */

    //Writing values to destination image
    vst3_u8(dst, result);
  }
}

int main(int argc,char** argv)
{
  if (argc != 4) {
    cout << "Usage: lab2 image_name1 image_name2 alpha" << endl;
    return -1;
  }
  uint8_t * img1_arr;
  uint8_t * img2_arr;

  Mat img1;
  img1 = imread(argv[1], cv::IMREAD_COLOR);
  if (!img1.data) {
      cout << "Could not open the image" << endl;
       return -1;
  }
  if (img1.isContinuous()) {
      img1_arr = img1.data;
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
      img2_arr = img2.data;
  }
  else {
      cout << "data is not continuous" << endl;
      return -2;
  }
  float alpha = stof(argv[3]);
  if(alpha < 0.0 || alpha > 1.0){
    std::cout << "Alpha should be from 0.0 to 1.0" << '\n';
    return -3;
  }
  int width = img1.cols;
  int height = img1.rows;
  int num_pixels = width*height;

  Mat neon_dst(height, width, img1.type());
  auto t1_neon = chrono::high_resolution_clock::now();
  blending_neon(img1_arr, img2_arr, neon_dst.data, num_pixels, alpha);
  auto t2_neon = chrono::high_resolution_clock::now();
  auto duration_neon = chrono::duration_cast<chrono::microseconds>(t2_neon-t1_neon).count();
  cout << "Neon duraion" << endl;
  cout << duration_neon << endl;
  imwrite("neon_1.png", neon_dst);

  /* Testing Uint8 to float convertion
  uint8x8_t test_u8 = vld1_u8(img1_arr);
  uint8_t data_u8[8];
  vst1_u8(data_u8, test_u8);
  for(int i = 0; i < 8; i++){
    cout << to_string(data_u8[i]) << " ";
  }
  cout << endl;
  uint8x8_t test2_u8 = vld1_u8(img2_arr);
  uint8_t data2_u8[8];
  vst1_u8(data2_u8, test2_u8);
  for(int i = 0; i < 8; i++){
    cout << to_string(data2_u8[i]) << " ";
  }
  cout << endl;
  uint16x8_t test_u16 = vmovl_u8(test_u8);
  uint32x4_t i32vl, i32vh;
  float32x4_t f32vl, f32vh;
  i32vl = vmovl_u16(vget_low_u16(test_u16));
  i32vh = vmovl_u16(vget_high_u16(test_u16));
  f32vl = vcvtq_f32_u32(i32vl);
  f32vh = vcvtq_f32_u32(i32vh);
  f32vl = vmulq_n_f32(f32vl, 0.5);
  f32vh = vmulq_n_f32(f32vh, 0.5);

  i32vl = vcvtq_u32_f32(f32vl);
  i32vh = vcvtq_u32_f32(f32vh);

  uint16x8_t u16v = vmovn_high_u32(vget_low_u16(u16v), i32vl);
  u16v = vmovn_high_u32(vget_high_u16(u16v), i32vh);
  uint8x8_t u8v = vmovn_u16(u16v);
  uint8_t data[8];
  vst1_u8(data, u8v);
  for(int i = 0; i < 8; i++){
    cout << to_string(data[i]) << " ";
  }
  cout << endl; */
  Mat simple_dst(height, width, img1.type());
  auto t1_simp = chrono::high_resolution_clock::now();
  blending_simple(img1, img2, simple_dst, alpha);
  auto t2_simp = chrono::high_resolution_clock::now();
  auto duration_simp = chrono::duration_cast<chrono::microseconds>(t2_simp-t1_simp).count();
  cout << "Simple duraion" << endl;
  cout << duration_simp << endl;
  imwrite("simp.png", simple_dst);

  Mat func_dst(height, width, img1.type());
  auto t1_func = chrono::high_resolution_clock::now();
  blending_by_func(img1, img2, alpha, func_dst);
  auto t2_func = chrono::high_resolution_clock::now();
  auto duration_func = chrono::duration_cast<chrono::microseconds>(t2_func-t1_func).count();
  cout << "addWeighted function duraion" << endl;
  cout << duration_func << endl;
  imwrite("func.png", func_dst);

  return 0;
}
