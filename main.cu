#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__));

static const int FILTER_SIZE = 3;
static const float FILTER[] = { 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111};

static size_t getPixelIndex(int row, int col, int channel, int imageWidth){
    return (((row * imageWidth) + col) * 3) + channel;
}

int convolutionCPU(char** inputImagePtr, int inputHeight, int inputWidth, char** outputImagePtr){
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
                            char pixelVal = (*inputImagePtr)[getPixelIndex(rowOffset, colOffset, channel, inputWidth)];
                            double maskVal = FILTER[(filterRadius + filterRow) * FILTER_SIZE + filterCol + FILTER_SIZE];

                            // Add the mask application at the index to the accumulated total
                            accum += pixelVal * 0.111111111;
                        }
                    }
                }
                //printf("%f\n", accum);
                //(*outputImagePtr)[getPixelIndex(row, col, channel, inputWidth)] = (char)max(min(255.0, accum), 0.0);
            }
        }
    }
    return 0;
}

int readImage(char* filepath, int* height, int* width, char** imagePtr){
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
    (*imagePtr) = (char*)malloc(imageSizeInBytes);


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

int writeImage(char *filepath, char** imagePtr, int height, int width){
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
    fwrite(*imagePtr, 1,  3 * width * height, fp);
    fclose(fp);
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

    char* inputImage = NULL;
    char* outputImage = NULL;
    int width = 0;
    int height = 0;


    readImage("hereford_512.ppm", &height, &width, &inputImage);
    //readImage("hereford_512.ppm", &height, &width, &outputImage);
    //convolutionCPU(&inputImage, height, width, &outputImage);
    //writeImage("test.ppm", &outputImage, height, width);
    for(int i = 0; i < 10; i++){
        printf("%c\n", inputImage[i]);
    }
    return 0;
}
