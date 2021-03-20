#include <chrono>
#include <fstream>
#include <iostream>
#include <math.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "decompressKernel.h"
#include "huffman.h"

using namespace std;


inline unsigned findNoOfThreadBlocks(unsigned totalNoOfThreads) {
  unsigned noOfThreadBlocks = ceil((double)totalNoOfThreads / MAX_THREADS);
  return noOfThreadBlocks;
}

unsigned char *calculateOffsetAndWriteOutput(unsigned char *input,
                                             ull size, unsigned blockSize) {
  unsigned *offsets;
  unsigned char *dOutput, *dInput, *dInputInBytes;
  cudaMalloc(&dInput, size);
  cudaMemcpy(dInput, input, size, cudaMemcpyHostToDevice);
  cudaFreeHost(input);
  cudaMalloc(&dInputInBytes, size * 8);
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

  uint outputSize;
  cudaMemcpy(&outputSize, offsets + noOfThreads, sizeof(unsigned),
             cudaMemcpyDeviceToHost);
  cudaMalloc(&dOutput, outputSize);
  writeOutput<<<noOfThreadBlocks, MAX_THREADS>>>(
      dInputInBytes, dOutput, size * 8, blockSize * 8, offsets);
  cudaDeviceSynchronize();

  cudaFree(offsets);
  cudaFree(dInputInBytes);

  unsigned char* output;
  cudaMallocHost(&output, outputSize);
  cudaMemcpy(output, dOutput, outputSize,
             cudaMemcpyDeviceToHost);
  cudaFree(dOutput);
  return output;
}

ull findSizeOfInputFile(ifstream& inputFile) {
  streampos currentPositionInFile = inputFile.tellg();
  inputFile.seekg(0, inputFile.end);
  ull maxSizeOfInputFile = inputFile.tellg();
  inputFile.seekg(currentPositionInFile);
  return maxSizeOfInputFile;
}

void readContentFromFile(ifstream &inputFile, ofstream &outputFile,
                         const HuffmanTree &tree, unsigned blockSize,
                         ull sizeOfOriginalFile) {

  unsigned size = tree.treeInArray.size();
  cudaMemcpyToSymbol(deviceTree, tree.treeInArray.data(),
                     size * sizeof(TreeArrayNode));
  size--;
  cudaMemcpyToSymbol(rootIndex, &size, sizeof(int));

  ull maxSizeOfInputFile = ceil(findSizeOfInputFile(inputFile)/4.);
  cout << "MAXSIZEOFINPUTFILE: " << maxSizeOfInputFile << endl;
  unsigned int *input, *output;
  cudaMallocHost(&input, maxSizeOfInputFile);

  inputFile.read((char *)input, maxSizeOfInputFile);

  ull noOfBytesRead = inputFile.gcount();
  for(uint i=0; i<noOfBytesRead; i++) {
		for(uint j=0; j<32; j++)
			cout << ( 1 & (((uint)input[i]) >> (31-j)));
	}
  cout << "noOfBytesRead: " << noOfBytesRead << endl;
  output = calculateOffsetAndWriteOutput(input, noOfBytesRead, blockSize);

  outputFile.write((char *)output, sizeOfOriginalFile);
  cudaFreeHost(output);
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
  blockSize /= 8;
  cout << sizeOfFile << "," << blockSize << endl;

  HuffmanTree tree;
  tree.readFromFile(file);
  ofstream outputFile("decompressed_output", ios::out | ios::binary);

  readContentFromFile(file, outputFile, tree, blockSize, sizeOfFile);
  file.close();
  outputFile.close();

  return 0;
}