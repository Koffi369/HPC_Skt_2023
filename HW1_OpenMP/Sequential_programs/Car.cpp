#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <thread>

// include omp header file here
#include <omp.h>

#define RGB_COMPONENT_COLOR 255



struct PPMPixel {
    int red;
    int green;
    int blue;
};

typedef struct{
    int x, y, all;
    PPMPixel * data;
} PPMImage;

static const auto THREADS = std::thread::hardware_concurrency();

void readPPM(const char *filename, PPMImage& img){
    std::ifstream file (filename);
    if (file){
        std::string s;
        int rgb_comp_color;
        file >> s;
        if (s!="P3") {std::cout<< "error in format"<<std::endl; exit(9);}
        file >> img.x >>img.y;
        file >>rgb_comp_color;
        img.all = img.x*img.y;
        std::cout << s << std::endl;
        std::cout << "x=" << img.x << " y=" << img.y << " all=" <<img.all << std::endl;
        img.data = new PPMPixel[img.all];
        for (int i=0; i<img.all; i++){
            file >> img.data[i].red >>img.data[i].green >> img.data[i].blue;
        }

    }else{
        std::cout << "the file:" << filename << "was not found" << std::endl;
    }
    file.close();
}

void writePPM(const char *filename, PPMImage & img){
    std::ofstream file (filename, std::ofstream::out);
    file << "P3"<<std::endl;
    file << img.x << " " << img.y << " "<< std::endl;
    file << RGB_COMPONENT_COLOR << std::endl;

    for(int i=0; i<img.all; i++){
        file << img.data[i].red << " " << img.data[i].green << " " << img.data[i].blue << (((i+1)%img.x ==0)? "\n" : " ");
    }
    file.close();
}

//
//write the function for shifting
//

void shift_ppm_omp(PPMImage &image, int shift_amount) 
{
  // create a new image to hold the shifted pixels
  PPMImage new_image;
  new_image.data = new PPMPixel[image.all];

  // loop through the shift amount
  for (int k = 0; k < shift_amount; ++k)
  {

    // parallelize pixel manipulation using OpenMP
    #pragma omp parallel for num_threads(THREADS)
    for (int i = 0; i < image.y; i++) 
    {
      for (int j = 0; j < image.x; j++) 
      {
        // calculate the shifted pixel's index in the new image
        auto &shifted_pixel = new_image.data[(i * image.x + j + 1) % image.all];
        auto &old_pixel = image.data[i * image.x + j];
        // copy the RGB values of the old pixel to the shifted pixel
        shifted_pixel.red = old_pixel.red;
        shifted_pixel.green = old_pixel.green;
        shifted_pixel.blue = old_pixel.blue;
      }
    }

    // parallelize pixel copying using OpenMP
    #pragma omp parallel for num_threads(THREADS)
    for (int i = 0; i < image.all; i++) 
    {
      auto &old_pixel = image.data[i];
      auto &new_pixel = new_image.data[i];
      // copy the RGB values of the new pixel to the old pixel
      old_pixel.red = new_pixel.red;
      old_pixel.green = new_pixel.green;
      old_pixel.blue = new_pixel.blue;
    }
  }
}

void shift_ppm_seq(PPMImage &image, int shift_amount) 
{
  // create a new image to hold the shifted pixels
  PPMImage new_image;
  new_image.data = new PPMPixel[image.all];

  // loop through the shift amount
  for (int k = 0; k < shift_amount; ++k) 
  {
    for (int i = 0; i < image.y; i++) 
    {
      for (int j = 0; j < image.x; j++) 
      {
        // calculate the shifted pixel's index in the new image
        auto &shifted_pixel = new_image.data[(i * image.x + j + 1) % image.all];
        auto &old_pixel = image.data[i * image.x + j];
        // copy the RGB values of the old pixel to the shifted pixel
        shifted_pixel.red = old_pixel.red;
        shifted_pixel.green = old_pixel.green;
        shifted_pixel.blue = old_pixel.blue;
      }
    }

    for (int i = 0; i < image.all; i++) 
    {
      auto &old_pixel = image.data[i];
      auto &new_pixel = new_image.data[i];
      // copy the RGB values of the new pixel to the old pixel
      old_pixel.red = new_pixel.red;
      old_pixel.green = new_pixel.green;
      old_pixel.blue = new_pixel.blue;
    }
  }
}


//Here's the modified code with updated variable names and structure:

int main(int argc, char *argv[]) {
  // read in the PPM image
  PPMImage image;
  readPPM("car.ppm", image);

  // set the amount of pixels to shift
  int shift_amount = 200;

  // measure time taken to perform shiftPPM function
  double start_time, end_time;

  start_time = omp_get_wtime();
  shift_ppm_seq(image, shift_amount);
  end_time = omp_get_wtime();
  printf("Time elapsed with %d pixel shifts (serial): %f seconds.\n", shift_amount, end_time - start_time);

  // write the shifted PPM image to a new file
  writePPM("new_car_1.ppm", image);

  // read in the PPM image again
  readPPM("car.ppm", image);

  // measure time taken to perform shiftPPM_omp function
  start_time = omp_get_wtime();
  shift_ppm_omp(image, shift_amount);
  end_time = omp_get_wtime();
  printf("Time elapsed with %d pixel shifts and %d threads (parallel): %f seconds.\n", shift_amount, THREADS, end_time - start_time);

  // write the shifted PPM image to a new file
  writePPM("new_car_2.ppm", image);

  return 0;
}
