#include <stdio.h>
#include <math.h>
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__));

static void HandleError(cudaError_t err, const char*file,  int line ){
	if (err != cudaSuccess){
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line );
	}
}

static const int BLOCK_SIZE = 16;
static const int FILTER_SIZE = 3;
static const float FILTER[] = { 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111};

static size_t getPixelIndex(int row, int col, int channel, int imageWidth){
    return (((row * imageWidth) + col) * 3) + channel;
}

int readImage(char* filepath, int* height, int* width, unsigned char** imagePtr){
    FILE* fp = fopen(filepath, "rb");

    // Ensure that the file opening succeeded
    if(fp == NULL){
        printf("Failed to open file: %s\n", filepath);
        return -1;
    }

    // Define variable to store each metadata line as they are read
    char* currentLine = NULL;
    size_t lineLength = 0;

    // Parse first metadata line; getline gets the newline as part of the string
    if(getline(&currentLine, &lineLength, fp) == -1 || !strcmp(currentLine, "P6")){
        printf("Unexpected value in first metadata line from file: %s\n", filepath);
        printf("%s\n", currentLine);
        fclose(fp);
        return -1;
    }
    
    // Read all comment lines; end with the size line in storage
    do {
        if(getline(&currentLine, &lineLength, fp) == -1){
            printf("Unexpected value in metadata line from file: %s\n", filepath);
            fclose(fp);
            return -1;
        }

    } while(currentLine[0] == '#');


    // Read the width and height metadata
    if (sscanf(currentLine, "%d %d", width, height) != 2) {
        printf("Failed to load valid dimensions from file: %s \n", filepath);
        fclose(fp);
        return -1;
   }


    // Parse max value metadata
    if(getline(&currentLine, &lineLength, fp) == -1 || !strcmp(currentLine, "255")){
        printf("Unexpected maximum value in metadata of file: %s\n", filepath);
        fclose(fp);
        return -1;
    }


    // Set the pointer to the image array to the allocated space; 3 channels per pixel
    size_t imageSizeInBytes = (*height) * (*width) * 3;
    (*imagePtr) = (unsigned char*)malloc(imageSizeInBytes);


    // Read pixel values into array (1 byte per pixel)
    if(fread((void*)(*imagePtr), 1, imageSizeInBytes, fp) != imageSizeInBytes){
        printf("Failed to load pixel data from image: %s\n", filepath);
        fclose(fp);
        return -1;
    }
    
    
    // Close the file
    fclose(fp);
    return 0;
}

int writeImage(char* filepath, unsigned char* imagePtr, int height, int width){
    // Open write file
    FILE* fp = fopen(filepath, "wb");

    // Ensure the write file opened
    if (fp == NULL) {
        printf("Could not write file: %s\n", filepath);
        return -1;
   }

   // Format metadata
   fprintf(fp, "P6\n");
   // Comment metadata
   fprintf(fp, "#Placeholder comment\n");
   // Shape metadata
   fprintf(fp, "%d %d\n", width, height);
   // Max value metadata
   fprintf(fp, "%d\n", 255);

    // Write actual pixel data, 3 color values per index
    fwrite(imagePtr, 1,  3 * width * height, fp);
    fclose(fp);
    return 0;
}

int CPUBenchmark(unsigned char* inputImage, int inputHeight, int inputWidth, char* outputFile){
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
                            accum += ((int)pixelVal * maskVal);
                        }
                    }
                }
                // Assign the accumulated value to the center pixel in the filter
                outputImage[getPixelIndex(row, col, channel, inputWidth)] = (unsigned char)max(min(255.0, accum), 0.0);
            }
        }
    }
    // Save the output image
    writeImage(outputFile, outputImage, inputHeight, inputWidth);

    // Free memory
    free(outputImage);
    return 0;
}

__global__ void convolutionalKernel(unsigned char* inputImage, unsigned char *outputImage, int height, int width, const float* __restrict__ M) {

    // Define share memory tile; this needs to accomodate padding
    __shared__ unsigned char tile[BLOCK_SIZE + FILTER_SIZE - 1][BLOCK_SIZE + FILTER_SIZE - 1];

    // Calculate the radius of the mask (distance from center to edge)
    int maskRadius = FILTER_SIZE / 2;

    // Calculate output element coordinates
    int i = (blockIdx.y * blockDim.y) + threadIdx.y;
    int j = (blockIdx.x * blockDim.x) + threadIdx.x;

    int loadIndexI = i - maskRadius;
    int loadIndexJ = j - maskRadius;

    for(int channel = 0; channel < 3; channel++){

        // Load elements from the image into the tile if we are within the bounds of the image
        if(loadIndexI > -1 && loadIndexI < height && loadIndexJ > -1 && loadIndexJ < width){
            tile[threadIdx.y][threadIdx.x] = inputImage[(((loadIndexI * width) + loadIndexJ) * 3) + channel];
        } else {
            // If we are not within the bounds of the image, load 0
            tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        // Wait for entire tile to be loaded into shared memory before proceeding
        __syncthreads();

        float accum = 0.0f;
        // Ensure that we are within the bounds of the output tile
        if(threadIdx.y < BLOCK_SIZE && threadIdx.y < BLOCK_SIZE){
            // Iterate over all elements in the filter
            // We do not need a boundary check here because we have 0.0 padding
            for(int maskI = 0; maskI < FILTER_SIZE; maskI++){
                for(int maskJ = 0; maskJ < FILTER_SIZE; maskJ++){
                    // Add the pixel value * mask value to the accumulated value for the pixel
                    accum += (M[maskI * FILTER_SIZE + maskJ] * tile[maskI + threadIdx.y][maskJ + threadIdx.x]);
                }
            }
            
            // If our output index overlaps with the image, save the accum result in the output image
            if(i < height && j < width){
                // Calculate corresponding 1D output index and squeez accum between 0 and 1
                outputImage[(((i * width) + j) * 3) + channel] = (unsigned char)max(min(255.0, accum), 0.0);
            }
        }
        // Wait for all calculations to be done on this channel before going to next
        __syncthreads();
    }
}

void GPUBenchmark(unsigned char* h_inputImage, int height, int width, char* outputFile){

    // Get the CUDA device count to ensure we can run on GPU
    cudaError_t error;
    int count;	//stores the number of CUDA compatible devices
    error = cudaGetDeviceCount(&count);	//get the number of devices with compute capability >= 2.0

    if(error != cudaSuccess){	//if there is an error getting the device count
        printf("\nERROR calling cudaGetDeviceCount()\n");	//display an error message
        return;	// exit the function
    }

    // Move input image to device
    unsigned char* d_inputImage;
    HANDLE_ERROR(cudaMalloc(&d_inputImage, width * height * 3));
    HANDLE_ERROR(cudaMemcpy(d_inputImage, h_inputImage, width * height * 3, cudaMemcpyHostToDevice));

    // Allocate the output image
    unsigned char* d_outputImage;
    HANDLE_ERROR(cudaMalloc(&d_outputImage, width * height * 3));


    // Calculate grid layout
    dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 DimGrid(ceil((float)width / BLOCK_SIZE), ceil((float)height / BLOCK_SIZE));

    //Run the convolutional kernel
    convolutionalKernel<<<DimGrid, DimBlock>>>(d_inputImage, d_outputImage, height, width, FILTER);
    cudaDeviceSynchronize();

    // Copy output back to host
    unsigned char* h_outputImage = (unsigned char*)malloc(width * height * 3);
    HANDLE_ERROR(cudaMemcpy(h_outputImage, d_outputImage, width * height * 3, cudaMemcpyDeviceToHost));

    // Write image
    writeImage(outputFile, h_outputImage, height, width);

    // Free all allocated memory
    HANDLE_ERROR(cudaFree(d_inputImage));
    HANDLE_ERROR(cudaFree(d_outputImage));
    free(h_outputImage);
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
    readImage(inputImagePath, &height, &width, &inputImage);

    // Run the GPU benchamark on the image
    CPUBenchmark(inputImage, height, width, outputCPUPath);
    GPUBenchmark(inputImage, height, width, outputGPUPath);
    free(inputImage);
    return 0;
}
