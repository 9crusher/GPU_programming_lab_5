#include <stdio.h>
#include <math.h>
#include "ppm_io.h"
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__));

// Define a function used to wrap CUDA API calls and print error
static void HandleError(cudaError_t err, const char*file,  int line ){
	if (err != cudaSuccess){
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line );
	}
}

// Define function to calculate elapsed time in ms
float cpu_time(timespec* start, timespec* end){
	return ((1e9*end->tv_sec + end->tv_nsec) -(1e9*start->tv_sec + start->tv_nsec))/1e6;
}

// Side length of FILTER
static const int FILTER_SIZE = 3;
// Kernel implementation total GPU block size
static const int BLOCK_SIZE = 16;
// Kernel implementation output size for a block
static const int BLOCK_O_SIZE = BLOCK_SIZE - FILTER_SIZE + 1;
// Define filter to be applied by CPU and GPU
//static const float FILTER[] = { 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111};
static const float FILTER[] = { -1, -1, -1, -1, 8, -1, -1, -1, -1};

// Define a function to get the flattened index in a 3-channel image (used in device and host)
__host__ __device__ size_t getPixelIndex(int row, int col, int channel, int imageWidth){
    return (((row * imageWidth) + col) * 3) + channel;
}

// Time the CPU implementation of convolution
int CPUBenchmark(unsigned char* inputImage, int inputHeight, int inputWidth, char* outputFile){

    // Declare start and end times and record start time
    timespec ts, te;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);

    // Allocate an output image
    unsigned char* outputImage = (unsigned char*)malloc(inputWidth * inputHeight * 3);

    // Calculate the radius of the filter
    int filterRadius = FILTER_SIZE / 2;

    // Iterate over each pixel index and each of R, G, B channels in those pixels
    for(int row = 0; row < inputHeight; row++){
        for(int col = 0; col < inputWidth; col++){
            for(int channel = 0; channel < 3; channel++){
                double accum = 0;

                // Iterate over the entire filter, indexed from the center
                for(int filterRow = -1 * filterRadius; filterRow <= filterRadius; filterRow++){
                    for(int filterCol = -1 * filterRadius; filterCol <= filterRadius; filterCol++){

                        // Convert from kernel index to image index
                        int rowOffset = row + filterRow;
                        int colOffset = col + filterCol;

                        // Ensure that we are within the bounds of the image
                        if((rowOffset >= 0) && (rowOffset < inputHeight) && (colOffset >= 0) && (colOffset < inputWidth)){

                            // Get the value of the pixel in the original image and the corresponding mask value
                            unsigned char pixelVal = inputImage[getPixelIndex(rowOffset, colOffset, channel, inputWidth)];
                            double maskVal = FILTER[((filterRadius + filterRow) * FILTER_SIZE) + filterCol + filterRadius];

                            // Add the mask application at the index to the accumulated total
                            accum += (pixelVal * maskVal);
                        }
                    }
                }
                // Assign the accumulated value to the center pixel in the filter
                outputImage[getPixelIndex(row, col, channel, inputWidth)] = (unsigned char)max(min(255.0, accum), 0.0);
            }
        }
    }

    // Report CPU runtime(ms)
    clock_gettime(CLOCK_MONOTONIC_RAW, &te);
    printf("CPU elapsed time: %f\n", cpu_time(&ts, &te));

    // Save the output image
    if(writeImage(outputFile, outputImage, inputHeight, inputWidth) == -1){
        printf("Failed to write image from CPU\n");
        return -1;
    }

    // Free memory
    free(outputImage);
    return 0;
}

__global__ void convolutionalKernel(unsigned char* inputImage, unsigned char* outputImage, int height, int width, const float* __restrict__ M) {

    // Define share memory tile; this needs to accomodate padding
    __shared__ unsigned char tile[BLOCK_SIZE][BLOCK_SIZE];

    // Calculate the radius of the mask (distance from center to edge)
    int maskRadius = FILTER_SIZE / 2;

    // Calculate output element coordinates
    int i_out = (blockIdx.y * BLOCK_O_SIZE) + threadIdx.y;
    int j_out = (blockIdx.x * BLOCK_O_SIZE) + threadIdx.x;

    // Calculate the indices of the values to load
    int i_in = i_out - maskRadius;
    int j_in = j_out - maskRadius;

    // Iterate over each channel in this image
    for(int channel = 0; channel < 3; channel++){

        // Load elements from the image into the tile if we are within the bounds of the image
        if((i_in > -1) && (i_in < height) && (j_in > -1) && (j_in < width)){
            tile[threadIdx.y][threadIdx.x] = inputImage[getPixelIndex(i_in, j_in, channel, width)];
        } else {
            // If we are not within the bounds of the image, load 0
            tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Wait for entire tile to be loaded into shared memory before proceeding
        __syncthreads();

        // Define variable to hold running sum of filtered values at pixel & channel
        float accum = 0.0f;

        // Ensure that we are within the bounds of the output tile
        if(threadIdx.y < BLOCK_O_SIZE && threadIdx.x < BLOCK_O_SIZE){

            // Iterate over all elements in the filter; no boundary check here because we have 0.0 padding
            for(int mask_i = 0; mask_i < FILTER_SIZE; mask_i++){
                for(int mask_j = 0; mask_j < FILTER_SIZE; mask_j++){

                    // Add the pixel value * mask value to the accumulated value for the pixel
                    accum += (M[mask_i * FILTER_SIZE + mask_j] * tile[mask_i + threadIdx.y][mask_j + threadIdx.x]);
                }
            }

            // If our output index overlaps with the image, save the accum result in the output image
            if(i_out < height && j_out < width){
                // Calculate corresponding 1D output index and squeez accum between 0 and 1
                outputImage[getPixelIndex(i_out, j_out, channel, width)] = (unsigned char)max(min(255.0, accum), 0.0);
            }
        }

        // Wait for all calculations to be done on this channel before going to next
        __syncthreads();
    }
}

int GPUBenchmark(unsigned char* h_inputImage, int height, int width, char* outputFile){

    // Get the CUDA device count to ensure we can run on GPU
    cudaError_t error;
    int count;	//stores the number of CUDA compatible devices
    error = cudaGetDeviceCount(&count);	//get the number of devices with compute capability >= 2.0

    if(error != cudaSuccess){	//if there is an error getting the device count
        printf("\nERROR calling cudaGetDeviceCount()\n");	//display an error message
        return -1;// exit the function
    }

    // Define start and end time variables; store start to inculde memory moves
    timespec ts, te;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);

    // Move input image to device
    unsigned char* d_inputImage;
    HANDLE_ERROR(cudaMalloc(&d_inputImage, width * height * 3));
    HANDLE_ERROR(cudaMemcpy(d_inputImage, h_inputImage, width * height * 3, cudaMemcpyHostToDevice));

    // Allocate the output image
    unsigned char* d_outputImage;
    HANDLE_ERROR(cudaMalloc(&d_outputImage, width * height * 3));

    // Move filter to device memory
    float* d_filter;
    HANDLE_ERROR(cudaMalloc(&d_filter, FILTER_SIZE * FILTER_SIZE * sizeof(float)));
    HANDLE_ERROR(cudaMemcpy(d_filter, FILTER, FILTER_SIZE * FILTER_SIZE * sizeof(float), cudaMemcpyHostToDevice));


    // Calculate grid layout
    dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 DimGrid(ceil((float)width / BLOCK_O_SIZE), ceil((float)height / BLOCK_O_SIZE));

    //Run the convolutional kernel
    convolutionalKernel<<<DimGrid, DimBlock>>>(d_inputImage, d_outputImage, height, width, d_filter);

    // Copy output back to host
    unsigned char* h_outputImage = (unsigned char*)malloc(width * height * 3);
    HANDLE_ERROR(cudaMemcpy(h_outputImage, d_outputImage, width * height * 3, cudaMemcpyDeviceToHost));

    // Report GPU runtime (ms) with memory moves
    clock_gettime(CLOCK_MONOTONIC_RAW, &te);
    printf("GPU elapsed time: %f\n", cpu_time(&ts, &te));

    // Write image
    if(writeImage(outputFile, h_outputImage, height, width) == -1){
        printf("Failed to write image from GPU\n");
        return -1;
    }

    // Free all allocated memory
    HANDLE_ERROR(cudaFree(d_inputImage));
    HANDLE_ERROR(cudaFree(d_outputImage));
    HANDLE_ERROR(cudaFree(d_filter));
    free(h_outputImage);

    return 0;
}

int main(int argc, char* argv[]){

    if(argc < 4){
        printf("Please include input image, cpu-output-image, and gpu-output-image\n");
        return 0;
    } else if (argc > 4){
        printf("Too many arguments passed\n");
    }
    // Get vector length from the command line
    char* inputImagePath = argv[1];
    char* outputCPUPath = argv[2];
    char* outputGPUPath = argv[3];

    // Print validation of what file is being used
    printf("Running on file: %s\n", inputImagePath);

    // Declare variables for file reading
    unsigned char* inputImage = NULL;
    int width = 0;
    int height = 0;

    // Intake the image and save it in inputImage
    if(readImage(inputImagePath, &height, &width, &inputImage) == -1){
        printf("Exiting program due to image read error \n");
        return -1;
    }

    // Run the GPU benchamark on the image
    if(CPUBenchmark(inputImage, height, width, outputCPUPath) == -1){
        printf("Exiting program due to CPU benchmark error");
        free(inputImage);
        return -1;
    }

    if(GPUBenchmark(inputImage, height, width, outputGPUPath) == -1){
        printf("Exiting program due to GPU benchmark error");
        free(inputImage);
        return -1;
    }

    free(inputImage);
    return 0;
}
