#include <fstream>
#define COMPRESS_CU
#include <cuda.h>
#include <iostream>

#include "compressKernel.h"
#include "huffman.h"
#include <assert.h>

#define MAX_THREADS 1024
#define BLOCK_SIZE 256
#define PER_THREAD_PROC 8
#define SEGMENT_SIZE 256

using namespace std;

unsigned int numBlocks, numThreads, totalThreads, readSize,
    fileSize = 0, blockSize = BLOCK_SIZE;
unsigned char *fileContent, *dfileContent;
codedict *dictionary;
unsigned int dictionarySize;
unsigned char useSharedMemory;

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

void getFrequencies(unsigned long long int *frequency, ifstream &inputFile) {

  unsigned long long int *dfrequency;
  unsigned int fileSizeRead;

  cudaMalloc(&dfrequency, 256 * sizeof(unsigned long long int));
  printerr();
  cudaMemset(dfrequency, 0, 256 * sizeof(unsigned long long int));
  printerr();

  while (true) {
    inputFile.read((char *)fileContent, readSize);
    fileSizeRead = inputFile.gcount();
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
  inputFile.clear();
  inputFile.seekg(0);
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

void writeFileContents(ifstream &inputFile, ofstream &outputFile,
                       unsigned char *fileContent) {

  unsigned int lastBlockSize = 0;
  unsigned int lastBlockIndex = 0;
  unsigned int encodedFileSize;

  unsigned int *compressedFile, *d_compressedFile;
  unsigned int *bitOffsets, *dbitOffsets;
  bitOffsets = (unsigned int *)malloc(readSize * sizeof(unsigned int));
  cudaMalloc(&dbitOffsets, readSize * sizeof(unsigned int));
  deepCopyHostToConstant();

  while (true) {
    inputFile.read((char *)fileContent + lastBlockSize,
                   readSize - lastBlockSize);
    unsigned int fileSizeRead = inputFile.gcount();
    if ((!inputFile) && (fileSizeRead == 0))
      break;

    fileSizeRead += lastBlockSize;

    getOffsetArray(lastBlockIndex, bitOffsets, fileSizeRead, encodedFileSize);

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

    cudaMemcpy(compressedFile, d_compressedFile,
               writeSize * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    outputFile.write((char *)compressedFile, writeSize * sizeof(unsigned int));
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
    outputFile.write((char *)finalBlock, ceil(k / 8.));
  }

  cudaFree(dbitOffsets);
  free(bitOffsets);
}

int main(int argc, char **argv) {
  if (argc != 2) {
    cout << "Running format is ./compress <file name>" << endl;
    return 0;
  }

  ifstream inputFile(argv[1], ios::in | ios::binary);

  if (!inputFile) {
    cout << "Please give a valid file to open." << endl;
    return 0;
  }

  size_t memoryFree, memoryTotal;
  cudaMemGetInfo(&memoryFree, &memoryTotal);
  printerr();

  readSize = 0.01 * memoryFree;
  unsigned long long int frequency[256];

  cudaMalloc(&dfileContent, readSize * sizeof(unsigned char));
  fileContent = (unsigned char *)malloc(readSize);

  getFrequencies(frequency, inputFile); // build histogram in GPU

  HuffmanTree tree;
  tree.HuffmanCodes(frequency, dictionary); // build Huffman tree in Host
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
  // useSharedMemory = 0;
  cout << "Shared memory using bit is " << int(useSharedMemory) << endl;

  ofstream outputFile("compressed_testFile.bin", ios::out | ios::binary);

  outputFile.write((char *)&fileSize, sizeof(long long int));
  outputFile.write((char *)&blockSize, sizeof(unsigned int));
  outputFile.write((char *)&tree.noOfLeaves, sizeof(unsigned int));
  tree.writeTree(outputFile);
  writeFileContents(inputFile, outputFile, fileContent);
  // printDictionary(frequency);
  cudaFree(dfileContent);
  outputFile.close();
  inputFile.close();
  return 0;
}