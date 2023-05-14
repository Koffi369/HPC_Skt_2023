
//%%writefile histogram.cu

#include <stdio.h>
#include <stdint.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// kernel function to calculate grayscale image and histogram
__global__ void grayScaleAndHistogram(uint8_t *inputImage, unsigned int *histogram, int numBins, int imageWidth, int imageHeight) {
    int threadId = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (threadId < imageWidth * imageHeight) {
        // calculate grayscale value of pixel
        int grayValue = 0;
        for (int channel = 0; channel < 3; channel++) {
            grayValue += inputImage[threadId * 3 + channel];
        }
        grayValue /= 3;

        // increment histogram bin corresponding to pixel intensity
        atomicAdd(&histogram[grayValue % numBins], 1);
        
        // set grayscale value of pixel to output image
        inputImage[threadId * 3] = grayValue;
        inputImage[threadId * 3 + 1] = grayValue;
        inputImage[threadId * 3 + 2] = grayValue;
    }
}

// function to read in input image
uint8_t* readImage(char* filename, int *imageWidth, int *imageHeight) {
    int channels = 3;
    uint8_t* image = stbi_load(filename, imageWidth, imageHeight, &channels, STBI_rgb);
    return image;
}

// function to write histogram data to file
void writeHistogram(char* filename, unsigned int *histogram, int numBins) {
    FILE *file = fopen(filename, "wb");
    for (int i = 0; i < numBins; i++) {
        fprintf(file, "%d\n", histogram[i]);
    }
    fclose(file);
}

int main(void) {
    int numBins = 256;
    int imageWidth, imageHeight;

    // read in input image
    uint8_t* inputImage = readImage("simsps.jpg", &imageWidth, &imageHeight);

    // allocate memory on GPU
    uint8_t* deviceInputImage;
    unsigned int* deviceHistogram;
    cudaMalloc(&deviceInputImage, sizeof(uint8_t) * imageWidth * imageHeight * 3); // allocate space for three channels (RGB)
    cudaMalloc(&deviceHistogram, sizeof(unsigned int) * numBins);

    // copy input image from CPU to GPU
    cudaMemcpy(deviceInputImage, inputImage, sizeof(uint8_t) * imageWidth * imageHeight * 3, cudaMemcpyHostToDevice);
    
    // set block size and grid size for GPU kernel
    int blockSize = 256;
    int gridSize = (imageWidth * imageHeight + blockSize - 1) / blockSize;
    dim3 block(blockSize, 1, 1);
    dim3 grid(gridSize, 1, 1);

    // call GPU kernel to calculate grayscale image and histogram
    grayScaleAndHistogram<<<grid, block>>>(deviceInputImage, deviceHistogram, numBins, imageWidth, imageHeight);
    cudaDeviceSynchronize();
    
    // copy histogram data back from GPU to CPU
    unsigned int* histogram = (unsigned int*)malloc(sizeof(unsigned int) * numBins);
    cudaMemcpy(histogram, deviceHistogram, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToHost);

    // write histogram data to file
    writeHistogram("histogram.txt", histogram, numBins);

    // save grayscale image to disk
    stbi_write_png("output.png", imageWidth, imageHeight, 3, inputImage, 100);

    // free memory on GPU and CPU
    cudaFree(deviceInputImage);
    cudaFree(deviceHistogram);
    free(inputImage);
    free(histogram);

    return 0;
}

