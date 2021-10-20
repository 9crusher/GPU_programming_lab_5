#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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