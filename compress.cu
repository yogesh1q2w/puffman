#define COMPRESS_CU
#include <iostream>
#include <cuda.h>

#include "compressKernel.h"
#include "huffman.h"

#define MAX_THREADS 1024
#define BLOCK_SIZE 256
#define PER_THREAD_PROC 8
#define SEGMENT_SIZE 256

using namespace std;

unsigned int numBlocks, numThreads, totalThreads, readSize, fileSize = 0, blockSize = BLOCK_SIZE;
char *fileContent,*dfileContent;
codedict dictionary;
    
void printerr() {
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
        cout << "Error encountered: " << cudaGetErrorString(error) << endl;
}

void numThreadsAndBlocks(unsigned int totalIndices) {
    totalThreads = ceil(totalIndices/(float)(SEGMENT_SIZE*PER_THREAD_PROC)) * SEGMENT_SIZE;
    numThreads = min(MAX_THREADS,totalThreads);
    numBlocks = ceil(totalThreads/(float)numThreads);
}

void getFrequencies(unsigned long long int* frequency, FILE* inputFile) {
    
    unsigned long long int* dfrequency;
    unsigned int fileSizeRead;

    cudaMalloc(&dfrequency,256*sizeof(unsigned long long int));
    printerr();
    cudaMemset(dfrequency,0,256*sizeof(unsigned long long int));
    printerr();

    while(!feof(inputFile)) {
        fileSizeRead = fread(fileContent,sizeof(char),readSize,inputFile);
        
        if(fileSizeRead == 0)
        break;
        
        fileSize += fileSizeRead;

        cudaMemcpy(dfileContent,fileContent,fileSizeRead,cudaMemcpyHostToDevice);
        printerr();

        numThreadsAndBlocks(fileSizeRead);
        updatefrequency<<<numBlocks,numThreads>>>(fileSizeRead,dfileContent,dfrequency);
        printerr();
    }

    cudaMemcpy(frequency,dfrequency,sizeof(unsigned long long int)*256,cudaMemcpyDeviceToHost);
    printerr();
    cudaFree(dfrequency);
    printerr();
    rewind(inputFile);
}

void getOffsetArray(unsigned int &lastBlockIndex, unsigned int* bitOffsets, unsigned int fileSizeRead, unsigned int &encodedFileSize) {
    lastBlockIndex = 0;
    bitOffsets[0] = 0;
    unsigned int searchValue = BLOCK_SIZE;
    for (unsigned int i = 1; i < fileSizeRead; i++) {
        bitOffsets[i] = bitOffsets[i-1] + dictionary.codeSize[fileContent[i-1]];
        if (bitOffsets[i] > searchValue) {
            bitOffsets[i-1] = searchValue;
            searchValue += BLOCK_SIZE;
            i--;
            lastBlockIndex = i;
        }
        else if (bitOffsets[i] == searchValue) {
            searchValue += BLOCK_SIZE;
            lastBlockIndex = i;
        }
    }
    encodedFileSize = searchValue - BLOCK_SIZE;
}

void writeFileContents(FILE* inputFile, FILE* outputFile, char* fileContent, codedict &dictionary) {

    unsigned int lastBlockSize = 0;
    unsigned int lastBlockIndex = 0;
    unsigned int encodedFileSize;
    
    unsigned char *compressedFile, *d_compressedFile;
    unsigned char *dbitCompressedFile;
    unsigned int  *bitOffsets, *dbitOffsets;
    bitOffsets = (unsigned int*)malloc(readSize*sizeof(unsigned int));
    cudaMalloc(&dbitOffsets, readSize*sizeof(unsigned int));

    while(!feof(inputFile)) {
        unsigned int fileSizeRead = fread(fileContent + lastBlockSize,sizeof(char),readSize - lastBlockSize,inputFile);

        if(fileSizeRead == 0) {
            break;
        }

        fileSizeRead += lastBlockSize;

        getOffsetArray(lastBlockIndex,bitOffsets,fileSizeRead,encodedFileSize);

        lastBlockSize = fileSizeRead - lastBlockIndex;

        if(encodedFileSize == 0) {
            continue;
        }

        cudaMemcpy(dfileContent,fileContent,fileSizeRead*sizeof(char),cudaMemcpyHostToDevice);
        cudaMemcpy(dbitOffsets, bitOffsets, fileSizeRead*sizeof(unsigned int), cudaMemcpyHostToDevice);
        
        unsigned int writeSize = encodedFileSize/8;

        compressedFile = (unsigned char*) malloc(writeSize * sizeof(unsigned char));

        cudaMalloc(&dbitCompressedFile,encodedFileSize * sizeof(unsigned char));
        cudaMalloc(&d_compressedFile,writeSize * sizeof(unsigned char));
        
        numThreadsAndBlocks(lastBlockIndex);

        genBitCompressed<<<numBlocks,numThreads>>>(lastBlockIndex,dfileContent,dbitOffsets,dbitCompressedFile);

        numThreadsAndBlocks(writeSize);
        encode<<<numBlocks,numThreads>>>(writeSize,dbitCompressedFile,d_compressedFile);
        
        cudaMemcpy(compressedFile,d_compressedFile,writeSize*sizeof(unsigned char), cudaMemcpyDeviceToHost);
        fwrite(compressedFile,sizeof(unsigned char),writeSize,outputFile);

        cudaFree(dbitCompressedFile);
        cudaFree(d_compressedFile);

        free(compressedFile);

        memcpy(fileContent,fileContent + lastBlockIndex,lastBlockSize*sizeof(char));

    }

    if(lastBlockSize > 0) {
        unsigned char finalBlock[BLOCK_SIZE];
        unsigned int k = 0;
        unsigned int sizeToWrite = 0;
        for(unsigned int i = 0; i < lastBlockSize; i++) {
            sizeToWrite += dictionary.codeSize[fileContent[i]];
            for(int j = 0; j < dictionary.codeSize[fileContent[i]]; j++)
                finalBlock[k++] = dictionary.code[fileContent[i]][j];
        }
        unsigned char writeFinalBlock[BLOCK_SIZE/8];
        for(unsigned int i = 0; i < BLOCK_SIZE; i++) {
            if(finalBlock[i])
                writeFinalBlock[i/8]  = (writeFinalBlock[i/8] << 1) | 1;
            else
                writeFinalBlock[i/8] = writeFinalBlock[i/8] << 1;
        }
        fwrite(&writeFinalBlock,sizeof(unsigned char),ceil(sizeToWrite/8.),outputFile);
    }

    cudaFree(dbitOffsets);
    free(bitOffsets);
}

int main(int argc, char **argv) {
    FILE *inputFile, *outputFile;

    if(argc != 2) {
        cout << "Running format is ./compress <file name>" << endl;
        return 0;
    }

    inputFile = fopen(argv[1],"rb");

    if(!inputFile) {
        cout << "Please give a valid file to open." << endl;
        return 0;
    }
    
    size_t memoryFree,memoryTotal;
    cudaMemGetInfo(&memoryFree,&memoryTotal);
    printerr();
    
    readSize = 0.01*memoryFree;
    
    unsigned long long int frequency[256];

    cudaMalloc(&dfileContent,readSize*sizeof(char));
    fileContent = (char*)malloc(readSize);

    getFrequencies(frequency, inputFile);

    HuffmanTree tree;
    tree.HuffmanCodes(frequency,&dictionary);
    cudaMemcpyToSymbol(d_dictionary,&dictionary,sizeof(codedict));
    printerr();

    outputFile = fopen("compressed_output.bin","wb");

    fwrite(&fileSize, sizeof(unsigned long long int),1,outputFile);
    fwrite(&blockSize, sizeof(unsigned int),1,outputFile);
    fwrite(&tree.noOfLeaves,sizeof(unsigned char),1,outputFile);
    tree.writeTree(outputFile);
    writeFileContents(inputFile,outputFile,fileContent,dictionary);
    
    cudaFree(dfileContent);
    printerr();
    fclose(outputFile);
    fclose(inputFile);
    return 0;
}