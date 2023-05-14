

//%%writefile filter.cu


#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


// Copy a pixel from the input to the output image
__device__ void copyPixel(double *inputImage, double *outputImage, int j, int i, int channel, int width)
{
    outputImage[j * width * 3 + i * 3 + channel] = inputImage[j * width * 3 + i * 3 + channel];
}

// Get the colors of the 121 pixels in the 11x11 neighborhood centered at the given pixel
__device__ void getPixelColors(double *inputImage, int j, int i, int channel, int width, double *pixelColors)
{
    int count = 0;
    for (int k = -5; k < 6; k++)
    {
        for (int l = -5; l < 6; l++)
        {
            pixelColors[count] = inputImage[(j + l) * width * 3 + (i + k) * 3 + channel];
            count++;  
        }
    }
}

// Sort an array of colors in descending order
__device__ void sortColors(double *pixelColors)
{
    double x = 0;
    for (int k = 0; k < 120; k++)
    {
        for (int l = k + 1; l < 121; l++)
        {
            if (pixelColors[k] < pixelColors[l])
            {
                x = pixelColors[k];
                pixelColors[k] = pixelColors[l];
                pixelColors[l] = x;
            }
        }
    }
}

// Set the value of the output pixel to the 61st color value
__device__ void setOutputPixel(double *outputImage, int j, int i, int channel, int width, double *pixelColors)
{
    outputImage[j * width * 3 + i * 3 + channel] = pixelColors[60];
}


// Define the NonLinearFilter kernel function
__global__ void NonLinearFilter(int height, int width, double *inputImage, double *outputImage)
{
    // Calculate the global index of the current thread
    int globalIndex = threadIdx.x + blockDim.x * blockIdx.x;

    // Calculate the j and i indices and the channel number of the corresponding pixel
    int j = globalIndex / width / 3;
    int i = globalIndex / 3 - j * width;
    int channel = globalIndex - i * 3 - j * width * 3;
  
    // Calculate the total number of elements in the input and output arrays
    long int size = height * width * 3;

    if (globalIndex >= size)
    {
        // This thread index is outside of the image boundaries
        return;
    }
    
    if (i < 4 || j < 4 || i > width - 5 || j > height - 5)
    {
        // This pixel is at the edge of the image
        copyPixel(inputImage, outputImage, j, i, channel, width);
    }
    else
    {
        // This pixel is not at the edge of the image
        double pixelColors[121];
        getPixelColors(inputImage, j, i, channel, width, pixelColors);
        sortColors(pixelColors);
        setOutputPixel(outputImage, j, i, channel, width, pixelColors);
    }
}

 

// Copy a pixel from the input to the output image
__device__ void copyPixel2(double *inputImage, double *outputImage, int j, int i, int channel, int width)
{
    outputImage[j * width * 3 + i * 3 + channel] = inputImage[j * width * 3 + i * 3 + channel];
}

// Apply the filter kernel to a pixel
__device__ void applyFilter(double *inputImage, double *outputImage, int j, int i, int channel, int width, double *kernel)
{
    outputImage[j*width*3 + i*3 + channel] =  (inputImage[j*width*3 + i*3 + channel]*kernel[4] + \
                             inputImage[(j + 1) *width * 3 + (i - 1) * 3 + channel]*kernel[0] + \
                             inputImage[(j + 1) *width * 3 + (i + 1) * 3 + channel]*kernel[8] + \
                             inputImage[(j - 1) *width * 3 + (i - 1) * 3 + channel]*kernel[6] + \
                             inputImage[(j - 1) *width * 3 + (i + 1) * 3 + channel]*kernel[2] + \
                             inputImage[(j + 1) *width * 3 + i * 3 + channel]*kernel[3] + \
                             inputImage[j *width * 3 + (i - 1) * 3 + channel]*kernel[1] + \
                             inputImage[(j - 1) *width * 3 + i * 3 + channel]*kernel[5] + \
                             inputImage[j * width * 3 + (i + 1)*3 + channel]*kernel[7]); 
}


// Define the Filter kernel function
__global__ void Filter(int height, int width, double *kernel, double *inputImage, double *outputImage)
{
    // Calculate the global index of the current thread
    int globalIndex = threadIdx.x + blockDim.x * blockIdx.x;

    // Calculate the j and i indices and the channel number of the corresponding pixel
    int j = globalIndex / width / 3;
    int i = globalIndex / 3 - j * width;
    int channel = globalIndex - i * 3 - j * width * 3; 
  
    // Calculate the total number of elements in the input and output arrays
    long int size = height * width * 3;
  
    // Check if the current thread index is within the bounds of the input and output arrays
    if (globalIndex < size)
    {
        if (i == 0 || j == 0 || i == width - 1 || j == height - 1)
        {
            // If the pixel is at the edge of the image, copy it directly to the output
            copyPixel2(inputImage, outputImage, j, i, channel, width);
        }
        else
        {
            // If the pixel is not at the edge of the image, apply the filter kernel to it
            applyFilter(inputImage, outputImage, j, i, channel, width, kernel);
        }

        // Clamp the output pixel value to be non-negative
        if (outputImage[j * width * 3 + i * 3 + channel] < 0)
        {
            outputImage[j * width * 3 + i * 3 + channel] = 0;
        }
    }
}



//////////////////



int main(int argc, char **argv)
{
  int image_width, image_height, image_bpp, image_size;

  // Allocate memory for the kernel
  double *kernel = (double *) calloc(sizeof(double), 9);

  // Allocate memory for CUDA kernel
  double *cuda_kernel;

  // Allocate memory for filter name
  char *filter_name;

  // Get filter name from command line arguments
  filter_name = (char *) malloc(sizeof(char) * (strlen(argv[1] + 1)));
  filter_name = argv[1];

  // Set kernel values based on filter name
  if (strcmp(filter_name, "edge") == 0)
  {
      kernel[0] = kernel[6] = kernel[2] = kernel[8] = -1;
      kernel[1] = kernel[3] = kernel[7] = kernel[5] = -1;
      kernel[4] = 8;
  }
  
  if (strcmp(filter_name, "gaussian") == 0)
  {
      kernel[0] = kernel[6] = kernel[2] = kernel[8] = 1 / 16.;
      kernel[1] = kernel[3] = kernel[7] = kernel[5] = 2 / 16.;
      kernel[4] = 3 / 50.;
  }

  // Allocate memory for CUDA kernel
  if (strcmp(filter_name, "median") != 0)
  {
      cudaMalloc(&cuda_kernel, sizeof(double)*9);
      cudaMemcpy(cuda_kernel, kernel, sizeof(double) * 9, cudaMemcpyHostToDevice);  
  }

  // Load image
  uint8_t* image = stbi_load("simps512.jpg", &image_width, &image_height, &image_bpp, 3);  
  image_size = image_height * image_width * 3;

  // Allocate memory for image buffer
  double *image_buffer = (double *) malloc(sizeof(double) * image_size);

  // Allocate memory for CUDA image and result buffers
  double *cuda_image;
  double *cuda_result;

  cudaMalloc(&cuda_image, sizeof(double) * image_size);
  cudaMalloc(&cuda_result, sizeof(double) * image_size);

  // Convert image buffer to double
  for (int i = 0; i < image_size; i++) image_buffer[i] = (double) image[i];
  
  // Copy image buffer to CUDA image buffer
  cudaMemcpy(cuda_image, image_buffer, sizeof(double) * image_size, cudaMemcpyHostToDevice);

  // Set block size and grid size for CUDA kernels

  dim3 dimBlock(image_height);
  dim3 dimGrid(image_width * 3);

  // Apply filter based on filter name
  if (strcmp(filter_name, "median") != 0) Filter<<<dimGrid,dimBlock>>>(image_height, image_width, cuda_kernel, cuda_image, cuda_result);
  else NonLinearFilter<<<dimGrid,dimBlock>>>(image_height, image_width, cuda_image, cuda_result);

  // Synchronize CUDA device
  cudaDeviceSynchronize();  
  
  // Copy CUDA result buffer to image buffer
  double *result_buffer = (double *)malloc(sizeof(double) * image_size);
  cudaMemcpy(result_buffer, cuda_result, sizeof(double) * image_size, cudaMemcpyDeviceToHost);
  
  // Convert result buffer to uint8_t and store as image
  for (int i = 0; i < image_size; i++) image[i] = uint8_t (result_buffer[i]);

  // Save filtered image
  stbi_write_png("simps512_fil.png", image_width, image_height, 3, image, image_width * 3);

  // Free memory
  free(image);
  free(image_buffer);
  cudaFree(cuda_image);
  cudaFree(cuda_result);
}




