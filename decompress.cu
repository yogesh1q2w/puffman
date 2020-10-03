#define MAX_THREADS 1024
#define MAX_FILE_NAME_SIZE 100
#define MAX_THREADS_TO_USE 65536

#include <chrono>
#include <fstream>
#include <iostream>
#include <math.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "decompressKernel.h"
#include "huffman.h"

using namespace std;

typedef unsigned long long int ull;

inline unsigned findNoOfThreadBlocks(unsigned totalNoOfThreads) {
  unsigned noOfThreadBlocks = ceil((double)totalNoOfThreads / MAX_THREADS);
  return noOfThreadBlocks;
}

unsigned char *calculateOffsetAndWriteOutput(unsigned char *input,
                                             unsigned size, unsigned blockSize,
                                             unsigned &outputSize) {
  unsigned *offsets;
  unsigned char *dOutput, *output, *dInput, *dInputInBytes;
  cudaMalloc(&dInput, size * sizeof(unsigned char));
  cudaMemcpy(dInput, input, size * sizeof(unsigned char),
             cudaMemcpyHostToDevice);

  cudaMalloc(&dInputInBytes, size * 8 * sizeof(unsigned char));
  unsigned noOfThreads = ceil(((double)size) / blockSize);
  unsigned noOfThreadBlocks = findNoOfThreadBlocks(noOfThreads);

  convertBitsToBytes<<<noOfThreadBlocks, MAX_THREADS>>>(dInput, dInputInBytes,
                                                        size, blockSize);
  cudaDeviceSynchronize();
  cudaFree(dInput);
  cudaMalloc(&offsets, (noOfThreads + 1) * sizeof(unsigned));

  calculateNoOfTokensInBlock<<<noOfThreadBlocks, MAX_THREADS>>>(
      dInputInBytes, size * 8, blockSize * 8, offsets);
  cudaDeviceSynchronize();

  thrust::exclusive_scan(thrust::device, offsets, offsets + noOfThreads + 1,
                         offsets);
  cudaDeviceSynchronize();

  cudaMemcpy(&outputSize, offsets + noOfThreads, sizeof(unsigned),
             cudaMemcpyDeviceToHost);
  cudaMalloc(&dOutput, outputSize * sizeof(unsigned char));
  writeOutput<<<noOfThreadBlocks, MAX_THREADS>>>(
      dInputInBytes, dOutput, size * 8, blockSize * 8, offsets);
  cudaDeviceSynchronize();

  cudaFree(offsets);
  cudaFree(dInputInBytes);

  output = new unsigned char[outputSize * sizeof(unsigned char)];
  cudaMemcpy(output, dOutput, outputSize * sizeof(unsigned char),
             cudaMemcpyDeviceToHost);
  // printf("--->%s\n",output);
  cudaFree(dOutput);
  return output;
}

void readContentFromFile(ifstream &inputFile, ofstream &outputFile,
                         const HuffmanTree &tree, unsigned blockSize,
                         ull sizeOfInputFile) {
  size_t memoryFree, memoryTotal;
  cudaError_t error;

  error = cudaMemGetInfo(&memoryFree, &memoryTotal);
  if (error != cudaSuccess) {
    printf("Error encountered: %s\n", cudaGetErrorString(error));
    return;
  }

  const unsigned chunkSize = MAX_THREADS_TO_USE * blockSize;
  unsigned char input[chunkSize], *output;
  unsigned outputSize;
  ull sizeWrittenToFile = 0;

  unsigned size = tree.treeInArray.size();
  cudaMemcpyToSymbol(deviceTree, tree.treeInArray.data(),
                     size * sizeof(TreeArrayNode));
  size--;
  cudaMemcpyToSymbol(rootIndex, &size, sizeof(int));

  while (inputFile) {
    inputFile.read((char *)input, chunkSize);
    unsigned noOfBytesRead = inputFile.gcount();

    output = calculateOffsetAndWriteOutput(input, noOfBytesRead, blockSize,
                                           outputSize);

    ull sizeToWrite = min((ull)outputSize, sizeOfInputFile - sizeWrittenToFile);
    outputFile.write((char *)output, sizeToWrite);
    sizeWrittenToFile += sizeToWrite;
    delete[] output;
  }
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    cout << "The usage is ./a.out <fileToDecompress>" << endl;
    return 0;
  }
  char *filename = argv[1];
  ifstream file(filename, ios::in | ios::binary);
  if (!file) {
    cout << "The file could not be opened" << endl;
    return 0;
  }
  ull sizeOfFile;
  unsigned int blockSize;
  file.read((char *)&sizeOfFile, sizeof(ull));
  file.read((char *)&blockSize, sizeof(unsigned int));
  cout << "Filesize= " << sizeOfFile << ", Blocksize = " << blockSize << endl;
  blockSize /= 8;

  HuffmanTree tree;
  tree.readFromFile(file);
  cout << "Tree read from file" << endl;
  ofstream outputFile("decompressed_output", ios::out | ios::binary);

  readContentFromFile(file, outputFile, tree, blockSize, sizeOfFile);
  file.close();
  outputFile.close();

  return 0;
}