#define COMPRESS_CU
#include <bits/stdc++.h>
#include <cuda.h>
#include <fstream>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include "compressKernel.h"
#include "huffman.h"
#include <assert.h>

#define MAX_THREADS 1024
#define BLOCK_SIZE 256
#define PER_THREAD_PROC 8
#define SEGMENT_SIZE 256

using namespace std;

unsigned int numBlocks, numThreads, totalThreads, readSize,
    blockSize = BLOCK_SIZE;
unsigned char *fileContent, *dfileContent;
codedict *dictionary;
unsigned int dictionarySize;
unsigned char useSharedMemory;
unsigned long long int fileSize = 0;

cudaEvent_t start, stop;
float milliseconds = 0;

void startClock() {
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  milliseconds = 0;
  cudaEventRecord(start, 0);
}

void stopClock(const string &message) {
  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  if (message[0] != '0')
    cout << "Time taken for " << message << " = " << milliseconds << endl;
}

void printDictionary(unsigned long long int *frequency) {
  for (unsigned short i = 0; i < 256; i++) {
    if (frequency[i]) {
      cout << char(i) << "\t|\t" << frequency[i] << "\t|\t";
      for (unsigned char j = 0; j < dictionary->codeSize[i]; j++)
        cout << int(dictionary->code[i * dictionary->maxCodeSize + j]) << ",";
      cout << endl;
    }
  }
}

void printerr() {
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess)
    cout << "Error encountered: " << cudaGetErrorString(error) << endl;
}

void numThreadsAndBlocks(unsigned int totalIndices) {
  totalThreads = ceil(totalIndices / (float)(SEGMENT_SIZE * PER_THREAD_PROC)) *
                 SEGMENT_SIZE;
  numThreads = min(MAX_THREADS, totalThreads);
  numBlocks = ceil(totalThreads / (float)numThreads);
}

void getFrequencies(unsigned long long int *frequency, FILE *inputFile) {

  unsigned long long int *dfrequency;
  unsigned int fileSizeRead;

  cudaMalloc(&dfrequency, 256 * sizeof(unsigned long long int));
  printerr();
  cudaMemset(dfrequency, 0, 256 * sizeof(unsigned long long int));
  printerr();

  while (true) {
    fileSizeRead =
        fread(fileContent, sizeof(unsigned char), readSize, inputFile);

    if (fileSizeRead == 0)
      break;

    fileSize += fileSizeRead;

    cudaMemcpy(dfileContent, fileContent, fileSizeRead, cudaMemcpyHostToDevice);
    printerr();

    numThreadsAndBlocks(fileSizeRead);
    updatefrequency<<<numBlocks, numThreads>>>(fileSizeRead, dfileContent,
                                               dfrequency);
    printerr();
  }

  cudaMemcpy(frequency, dfrequency, sizeof(unsigned long long int) * 256,
             cudaMemcpyDeviceToHost);
  printerr();
  cudaFree(dfrequency);
  printerr();
  rewind(inputFile);
}

void deepCopyHostToConstant() {
  cudaMemcpyToSymbol(const_code, dictionary->code,
                     dictionary->maxCodeSize * 256);
  printerr();

  cudaMemcpyToSymbol(const_codeSize, dictionary->codeSize, 256);
  printerr();
}

void getOffsetArray(unsigned int &lastBlockIndex, unsigned int *bitOffsets,
                    unsigned int fileSizeRead, unsigned int &encodedFileSize) {
  lastBlockIndex = 0;
  bitOffsets[0] = 0;
  unsigned int searchValue = BLOCK_SIZE;
  unsigned int i;
  for (i = 1; i < fileSizeRead; i++) {
    bitOffsets[i] =
        bitOffsets[i - 1] + dictionary->codeSize[fileContent[i - 1]];

    if (bitOffsets[i] > searchValue) {
      bitOffsets[i - 1] = searchValue;
      searchValue += BLOCK_SIZE;
      i--;
      lastBlockIndex = i;
    } else if (bitOffsets[i] == searchValue) {
      searchValue += BLOCK_SIZE;
      lastBlockIndex = i;
    }
  }

  if (bitOffsets[i - 1] + dictionary->codeSize[fileContent[i - 1]] >
      searchValue) {
    bitOffsets[i - 1] = searchValue;
    searchValue += BLOCK_SIZE;
    lastBlockIndex = i - 1;
  }

  encodedFileSize = searchValue - BLOCK_SIZE;
}

void writeFileContents(FILE *inputFile, FILE *outputFile,
                       unsigned char *fileContent) {

  unsigned int lastBlockSize = 0;
  unsigned int lastBlockIndex = 0;
  unsigned int encodedFileSize;

  unsigned int *compressedFile, *d_compressedFile;
  unsigned int *bitOffsets, *dbitOffsets;
  bitOffsets = (unsigned int *)malloc(readSize * sizeof(unsigned int));
  cudaMalloc(&dbitOffsets, readSize * sizeof(unsigned int));
  deepCopyHostToConstant();

  float offsetTime = 0;
  float kernelTime = 0;
  while (true) {
    unsigned int fileSizeRead =
        fread(fileContent + lastBlockSize, sizeof(unsigned char),
              readSize - lastBlockSize, inputFile);

    if (fileSizeRead == 0) {
      break;
    }

    fileSizeRead += lastBlockSize;

    startClock();
    getOffsetArray(lastBlockIndex, bitOffsets, fileSizeRead, encodedFileSize);
    stopClock("0");
    offsetTime += milliseconds;

    lastBlockSize = fileSizeRead - lastBlockIndex;

    if (encodedFileSize == 0)
      continue;

    cudaMemcpy(dfileContent, fileContent, fileSizeRead * sizeof(unsigned char),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dbitOffsets, bitOffsets, fileSizeRead * sizeof(unsigned int),
               cudaMemcpyHostToDevice);

    unsigned int writeSize = encodedFileSize / (8 * sizeof(unsigned int));

    compressedFile = (unsigned int *)malloc(writeSize * sizeof(unsigned int));

    cudaMalloc(&d_compressedFile, writeSize * sizeof(unsigned int));

    startClock();
    numThreadsAndBlocks(lastBlockIndex);
    if (useSharedMemory) {
      skss_compress_with_shared<<<numBlocks, numThreads,
                                  (dictionary->maxCodeSize + 1) * 256>>>(
          lastBlockIndex, dfileContent, dbitOffsets, d_compressedFile,
          dictionary->maxCodeSize);
    } else {
      skss_compress<<<numBlocks, numThreads>>>(lastBlockIndex, dfileContent,
                                               dbitOffsets, d_compressedFile,
                                               dictionary->maxCodeSize);
    }
    stopClock("0");
    kernelTime += milliseconds;

    cudaMemcpy(compressedFile, d_compressedFile,
               writeSize * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    fwrite(compressedFile, sizeof(unsigned int), writeSize, outputFile);

    cudaFree(d_compressedFile);
    free(compressedFile);

    memcpy(fileContent, fileContent + lastBlockIndex,
           lastBlockSize * sizeof(unsigned char));
  }

  if (lastBlockSize > 0) {
    unsigned int
        finalBlock[(uint)ceil(BLOCK_SIZE / (8. * sizeof(unsigned int)))];
    unsigned int k = 0;
    for (unsigned int i = 0; i < lastBlockSize; i++) {
      for (unsigned int j = 0; j < dictionary->codeSize[fileContent[i]]; j++) {
        unsigned int finalPos = k / (8 * sizeof(unsigned int));
        unsigned int modifyIndex = k % (8 * sizeof(unsigned int));
        modifyIndex = 8 * (modifyIndex / 8) + 7 - (modifyIndex % 8);
        unsigned int mask = 1 << modifyIndex;
        finalBlock[finalPos] =
            (finalBlock[finalPos] & ~mask) |
            ((dictionary->code[fileContent[i] * dictionary->maxCodeSize + j]
              << modifyIndex) &
             mask);
        k++;
      }
    }
    fwrite(&finalBlock, sizeof(unsigned int), ceil(k / 8.), outputFile);
  }
  cout << "Total offset calculation time = " << offsetTime << endl;
  cout << "Total kernel time = " << kernelTime << endl;

  cudaFree(dbitOffsets);
  free(bitOffsets);
}

int main(int argc, char **argv) {
  FILE *inputFile, *outputFile;
  if (argc != 2) {
    cout << "Running format is ./compress <file name>" << endl;
    return 0;
  }

  inputFile = fopen(argv[1], "rb");

  if (!inputFile) {
    cout << "Please give a valid file to open." << endl;
    return 0;
  }

  size_t memoryFree, memoryTotal;
  cudaMemGetInfo(&memoryFree, &memoryTotal);
  printerr();

  readSize = 0.01 * memoryFree;
  // readSize = 98304;       //No. of MPs * Threads per MP * PER_THREAD_PROC * 2
  cout << readSize << endl;
  unsigned long long int frequency[256];

  cudaMalloc(&dfileContent, readSize * sizeof(unsigned char));
  fileContent = (unsigned char *)malloc(readSize);

  startClock();
  getFrequencies(frequency, inputFile); // build histogram in GPU
  stopClock("Histogramming");

  HuffmanTree tree;
  startClock();
  tree.HuffmanCodes(frequency, dictionary); // build Huffman tree in Host
  stopClock("Codebook generation");
  int sharedMemoryPerBlock;
  cudaDeviceGetAttribute(&sharedMemoryPerBlock,
                         cudaDevAttrMaxSharedMemoryPerBlock, 0);
  printerr();
  dictionarySize = dictionary->getSize();
  cout << "Size of dict. = " << dictionarySize
       << ", Shared memory per block = " << sharedMemoryPerBlock << endl;
  if (sharedMemoryPerBlock > dictionarySize) {
    useSharedMemory = 1;
  } else {
    useSharedMemory = 0;
  }
  cout << "Shared memory using bit is " << int(useSharedMemory) << endl;

  outputFile = fopen("compressed_output.bin", "wb");

  startClock();
  fwrite(&fileSize, sizeof(unsigned long long int), 1, outputFile);
  fwrite(&blockSize, sizeof(unsigned int), 1, outputFile);
  fwrite(&tree.noOfLeaves, sizeof(unsigned int), 1, outputFile);

  tree.writeTree(outputFile);
  stopClock("Write tree and Metadata");
  writeFileContents(inputFile, outputFile, fileContent);
  // printDictionary(frequency);
  cudaFree(dfileContent);
  fclose(outputFile);
  fclose(inputFile);
  return 0;
}