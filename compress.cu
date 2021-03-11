#define COMPRESS_CU
#include <bits/stdc++.h>
#include <cuda.h>
#include <unistd.h>

#include "compressKernel.h"
#include "constants.h"
#include "huffman.h"

using namespace std;

uint blockSize = BLOCK_SIZE;
uint *fileContent, *dfileContent;
codedict dictionary;
uint dictionarySize;
unsigned char useSharedMemory;
unsigned long long int fileSize;
uint intFileSize;

void printDictionary(uint *frequency) {
  for (unsigned short i = 0; i < 256; i++) {
    if (frequency[i]) {
      cout << char(i) << "\t|\t" << frequency[i] << "\t|\t";
      for (unsigned char j = 0; j < dictionary.codeSize[i]; j++)
        cout << int(1 & (dictionary.code[i] >> (31 - j)));
      cout << endl;
    }
  }
}

void getFrequencies(uint *frequency) {

  uint *dfrequency;
  cudaMalloc((void **)&dfrequency, 256 * sizeof(uint));
  CUERROR
  cudaMemset(dfrequency, 0, 256 * sizeof(uint));
  CUERROR
  uint *d_PartialHistograms;
  cudaMalloc((void **)&d_PartialHistograms,
             sizeof(uint) * HIST_BLOCK * HIST_SIZE);
  CUERROR
  TIMER_START(hist)
  cu_histgram<<<HIST_BLOCK, HIST_THREADS>>>(d_PartialHistograms, dfileContent,
                                            intFileSize, fileSize);
  mergeHistogram<<<HIST_BLOCK, HIST_SIZE>>>(dfrequency, d_PartialHistograms);
  cudaMemcpy(frequency, dfrequency, 256 * sizeof(uint), cudaMemcpyDeviceToHost);
  TIMER_STOP(hist)
  cudaFree(d_PartialHistograms);
  CUERROR
  cudaFree(dfrequency);
  CUERROR
}

inline unsigned char getcharAt(uint pos) {
  return (fileContent[pos >> 2] >> ((pos & 3U) << 3)) & 0xFFU;
}

void getOffsetArray(uint *bitOffsets, uint &encodedFileSize) {
  bitOffsets[0] = 0;
  uint searchValue = BLOCK_SIZE;
  uint i;
  for (i = 1; i < fileSize; i++) {
    bitOffsets[i] = bitOffsets[i - 1] + dictionary.codeSize[getcharAt(i - 1)];

    if (bitOffsets[i] > searchValue) {
      bitOffsets[i - 1] = searchValue;
      searchValue += BLOCK_SIZE;
      i--;
    } else if (bitOffsets[i] == searchValue) {
      searchValue += BLOCK_SIZE;
    }
  }

  if (bitOffsets[i - 1] + dictionary.codeSize[getcharAt(i - 1)] > searchValue) {
    bitOffsets[i - 1] = searchValue;
  }
  encodedFileSize =
      bitOffsets[fileSize - 1] + dictionary.codeSize[getcharAt(fileSize - 1)];
}

void writeFileContents(FILE *outputFile) {

  uint *compressedFile, *d_compressedFile;
  uint *bitOffsets, *dbitOffsets;
  cudaMallocHost(&bitOffsets, fileSize * sizeof(uint));
  CUERROR
  cudaMalloc((void **)&dbitOffsets, fileSize * sizeof(uint));
  CUERROR

  uint encodedFileSize;
  TIMER_START(offset)
  getOffsetArray(bitOffsets, encodedFileSize);
  TIMER_STOP(offset)

  cudaMemcpy(dbitOffsets, bitOffsets, fileSize * sizeof(uint),
             cudaMemcpyHostToDevice);
  CUERROR

  uint writeSize = (encodedFileSize + 31) >> 5;

  cudaMallocHost(&compressedFile, writeSize * sizeof(uint));
  CUERROR
  cudaMalloc((void **)&d_compressedFile, writeSize * sizeof(uint));
  CUERROR

  TIMER_START(kernel)
  skss_compress_with_shared<<<BLOCK_NUM, 256, 1280>>>(
      fileSize, dfileContent, dbitOffsets, d_compressedFile);
  TIMER_STOP(kernel)

  cudaMemcpy(compressedFile, d_compressedFile, writeSize * sizeof(uint),
             cudaMemcpyDeviceToHost);
  CUERROR
  fwrite(compressedFile, sizeof(uint), writeSize, outputFile);
  fdatasync(outputFile->_fileno);
  cudaFree(d_compressedFile);
  CUERROR
  cudaFreeHost(compressedFile);
  CUERROR
  cudaFree(dbitOffsets);
  CUERROR
  cudaFreeHost(bitOffsets);
  CUERROR
}

void readFile(uint *&fileContent, uint *&dfileContent, FILE *inputFile) {
  fseek(inputFile, 0L, SEEK_END);
  fileSize = ftell(inputFile);
  fseek(inputFile, 0L, SEEK_SET);
  intFileSize = (fileSize + 3) >> 2;
  cudaMallocHost(&fileContent, sizeof(uint) * intFileSize);
  CUERROR
  cudaMalloc((void **)&dfileContent, sizeof(uint) * intFileSize);
  CUERROR
  uint sizeRead =
      fread(fileContent, sizeof(unsigned char), fileSize, inputFile);
  fsync(inputFile->_fileno); // What is this fsync for?
  if (sizeRead != fileSize) {
    cout << "Error in reading the file. Aborting..." << endl;
    exit(0);
  }
  TIMER_START(HtD1)
  cudaMemcpy(dfileContent, fileContent, sizeof(uint) * intFileSize,
             cudaMemcpyHostToDevice);
  CUERROR
  TIMER_STOP(HtD1)
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

  uint frequency[256];

  TIMER_START(readFile)
  readFile(fileContent, dfileContent, inputFile);
  TIMER_STOP(readFile)

  getFrequencies(frequency); // build histogram in GPU

  HuffmanTree tree;
  TIMER_START(tree)
  tree.HuffmanCodes(frequency, dictionary); // build Huffman tree in Host
  TIMER_STOP(tree)
  printDictionary(frequency);

  outputFile = fopen("compressed_output.bin", "wb");

  TIMER_START(meta)
  fwrite(&fileSize, sizeof(unsigned long long int), 1, outputFile);
  fwrite(&blockSize, sizeof(uint), 1, outputFile);
  fwrite(&tree.noOfLeaves, sizeof(uint), 1, outputFile);
  tree.writeTree(outputFile);
  TIMER_STOP(meta)
  writeFileContents(outputFile);
  // cudaFreeHost(fileContent);
  // CUERROR
  // cudaFree(dfileContent);
  // CUERROR
  // fclose(inputFile);
  fclose(outputFile);
  return 0;
}