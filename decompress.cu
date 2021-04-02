#include <bits/stdc++.h>
#include <cuda.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <unistd.h>

#include "decompressKernel.h"
#include "huffman.h"

using namespace std;

void decode(FILE *inputFile, FILE *outputFile, HuffmanTree tree, uint blockSize,
            uint sizeOfFile, unsigned long long int encodedFileSize, uint numNodes) {
  uint *encodedString, *d_encodedString;
  uint *decodedString, *d_decodedString;
  unsigned long long int *d_charOffset;
  uint *d_taskCounter;
  unsigned char *d_treeToken;
  uint *d_treeLeft, *d_treeRight;

  uint numBlocksInEncodedString = ceil(encodedFileSize / (1. * blockSize));
  cudaMallocHost(&encodedString, sizeof(uint) * ((encodedFileSize + 31) / 32));
  cudaMallocHost(&decodedString, sizeof(uint) * ((sizeOfFile + 3) / 4));
  if (((encodedFileSize + 31) / 32) !=
      fread(encodedString, sizeof(uint), (encodedFileSize + 31) / 32, inputFile))
    fatal("File read error 4");

  cudaMalloc((void **)&d_encodedString,
             sizeof(uint) * ((encodedFileSize + 31) / 32));
  cudaMalloc((void **)&d_decodedString, sizeof(uint) * ((sizeOfFile + 3) / 4));
  cudaMalloc((void **)&d_charOffset,
             sizeof(unsigned long long int) * (numBlocksInEncodedString + 1));
  cudaMalloc((void **)&d_taskCounter, sizeof(uint));
  cudaMalloc((void **)&d_treeToken, sizeof(unsigned char) * numNodes);
  cudaMalloc((void **)&d_treeLeft, sizeof(uint) * numNodes);
  cudaMalloc((void **)&d_treeRight, sizeof(uint) * numNodes);

  cudaMemset(d_taskCounter, 0, sizeof(uint));
  cudaMemset(d_charOffset, 0, sizeof(unsigned long long int) * (numBlocksInEncodedString + 1));
  cudaMemset(d_decodedString, 0, sizeof(uint) * ((sizeOfFile + 3) / 4));

  cudaMemcpy(d_encodedString, encodedString,
             sizeof(uint) * ((encodedFileSize + 31) / 32),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_treeToken, tree.tree.token, numNodes * sizeof(unsigned char),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_treeLeft, tree.tree.left, numNodes * sizeof(uint),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_treeRight, tree.tree.right, numNodes * sizeof(uint),
             cudaMemcpyHostToDevice);
  uint shm_needed = numNodes * 9;
  uint numTasks = ceil(encodedFileSize / (NUM_THREADS * BLOCK_SIZE * 1.0));
  TIMER_START(kernel)
  single_shot_decode<<<BLOCK_NUM, NUM_THREADS, shm_needed>>>(
      d_encodedString, encodedFileSize, d_treeToken, d_treeLeft, d_treeRight,
      d_charOffset, d_decodedString,
      d_taskCounter, numNodes, numTasks);
  TIMER_STOP(kernel)
  cudaMemcpy(decodedString, d_decodedString, sizeof(char) * sizeOfFile,
             cudaMemcpyDeviceToHost);
  fwrite(decodedString, sizeof(char), sizeOfFile, outputFile);
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    cout << "The usage is ./a.out <fileToDecompress>" << endl;
    return 0;
  }
  char *filename = argv[1];
  FILE *inputFile, *outputFile;
  inputFile = fopen(filename, "rb");
  if (!inputFile) {
    cout << "The file could not be opened" << endl;
    return 0;
  }
  outputFile = fopen("decompressed_output", "wb");

  ull sizeOfFile;
  unsigned int blockSize;
  if (1 != fread(&sizeOfFile, sizeof(ull), 1, inputFile))
    fatal("File read error 1");
  if (1 != fread(&blockSize, sizeof(uint), 1, inputFile))
    fatal("File read error 2");
  cout << sizeOfFile << "," << blockSize << endl;

  HuffmanTree tree;
  TIMER_START(tree_generation)
  tree.readFromFile(inputFile);
  TIMER_STOP(tree_generation)
  unsigned long long int encodedFileSize;
  if (1 !=
      fread(&encodedFileSize, sizeof(unsigned long long int), 1, inputFile))
    fatal("File read error 3");
  decode(inputFile, outputFile, tree, blockSize, sizeOfFile, encodedFileSize,
         2 * tree.noOfLeaves - 1);
  fclose(inputFile);
  fclose(outputFile);

  return 0;
}