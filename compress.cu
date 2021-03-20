#define COMPRESS_CU
#include <bits/stdc++.h>
#include <cuda.h>
#include <unistd.h>

#include "compressKernel.h"
#include "constants.h"
#include "huffman.h"

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

using namespace std;

uint blockSize = BLOCK_SIZE;
uint *fileContent, *dfileContent;
codedict dictionary;
uint dictionarySize;
unsigned char useSharedMemory;
unsigned long long int fileSize;
uint intFileSize;
uint frequency[256];
uint *d_dictionary_code;
unsigned char *d_dictionary_codelens;
uint *counter;

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

void getFrequencies() {

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

unsigned long long int getMaxOutputFileSize() {
  unsigned long long int maxOutputFileSize = 0;
  for (uint i = 0; i < 256; i++) {
    maxOutputFileSize += frequency[i] * dictionary.codeSize[i];
  }
  maxOutputFileSize =
      ceil(maxOutputFileSize * (blockSize / (blockSize - 32.0)));
  return maxOutputFileSize;
}

inline unsigned char getcharAt(uint pos) {
  return (fileContent[pos >> 2] >> ((pos & 3U) << 3)) & 0xFFU;
}

void printOut(uint *out, uint size) {
  cout << "The file written was-\n";
  for (uint i = 0; i < size; i++) {
    for (uint j = 0; j < 32; j++)
      cout << int(1 & (out[i] >> (31 - j)));
  }
  cout << "\n-------------------------------------------------------" << endl;
}

void getOffsetArray(uint *dbitOffsets, uint *boundary_index,
                    uint &encodedFileSize) {
  cudaMalloc(&counter, sizeof(uint));
  cudaMemset(counter, 0, sizeof(uint));
  cudaMalloc(&d_dictionary_codelens, 256 * sizeof(unsigned char));
  cudaMemcpy(d_dictionary_codelens, dictionary.codeSize,
             256 * sizeof(unsigned char), cudaMemcpyHostToDevice);
  uint numTasks = ceil(fileSize / (256. * PER_THREAD_PROC));
  cout << "initializing offsets now..." << endl;
  initOffsets<<<BLOCK_NUM, 256>>>(fileSize, dfileContent, dbitOffsets,
                                  d_dictionary_codelens, counter, numTasks);
  cudaDeviceSynchronize();
  cout << "Going for exclusive scan..." << endl;
  thrust::exclusive_scan(thrust::device, dbitOffsets, dbitOffsets + fileSize,
                         dbitOffsets, 0);
  uint modifyIndex = 0;
  float searchValue = BLOCK_SIZE;

  while (1) {
    if (modifyIndex == fileSize - 1)
      break;
    modifyIndex =
        (thrust::lower_bound(thrust::device, dbitOffsets + modifyIndex,
                             dbitOffsets + fileSize, searchValue + 0.1) -
         dbitOffsets) -
        1;
    // cout << "MODIFYINDEX FOR " << searchValue << " IS " << modifyIndex << endl;
    if (modifyIndex == fileSize - 1)
      break;
    thrust::transform(thrust::device, dbitOffsets + modifyIndex,
                      dbitOffsets + fileSize, dbitOffsets + modifyIndex,
                      add_functor((uint)searchValue, modifyIndex, dbitOffsets));
    // cudaMemcpy(bitOffsets, dbitOffsets, fileSize * sizeof(uint),
    //            cudaMemcpyDeviceToHost);
    // for (uint i = 0; i < fileSize; i++)
    //   cout << bitOffsets[i] << ",";
    // cout << endl;
    searchValue += BLOCK_SIZE;
  }
}

void writeFileContents(FILE *outputFile) {

  uint *compressedFile, *d_compressedFile;
  uint *dbitOffsets;
  uint *boundary_index, *d_boundary_index;
  cudaMallocHost(&boundary_index, fileSize * sizeof(uint));
  cudaMemset(boundary_index, 0, fileSize * sizeof(uint));
  CUERROR
  cudaMalloc((void **)&dbitOffsets, fileSize * sizeof(uint));
  cudaMalloc((void **)&d_boundary_index, fileSize * sizeof(uint));
  CUERROR

  uint encodedFileSize;
  TIMER_START(offset)
  getOffsetArray(dbitOffsets, boundary_index, encodedFileSize);
  TIMER_STOP(offset)

  cudaMemcpy(d_boundary_index, boundary_index, fileSize * sizeof(uint),
             cudaMemcpyHostToDevice);
  CUERROR

  // cout << "Prefix array and Boundary" << endl;
  // for (uint i = 0; i < fileSize; i++) {
  //   cout << i << "-->" << bitOffsets[i] << " , " << boundary_index[i] <<
  //   endl;
  // }

  uint writeSize = (encodedFileSize + 31) >> 5;

  cudaMallocHost(&compressedFile, writeSize * sizeof(uint));
  CUERROR
  cudaMalloc((void **)&d_compressedFile, writeSize * sizeof(uint));
  cudaMemset(d_compressedFile, 0, writeSize * sizeof(uint));
  CUERROR
  cudaMalloc(&d_dictionary_code, 256 * sizeof(uint));
  cudaMemcpy(d_dictionary_code, dictionary.code, 256 * sizeof(uint),
             cudaMemcpyHostToDevice);
  // print_dict<<<1, 1>>>(d_dictionary_code, d_dictionary_codelens);

  cudaMemset(counter, 0, sizeof(uint));

  uint numTasks = ceil(fileSize / (256. * PER_THREAD_PROC));

  TIMER_START(kernel)
  encode<<<BLOCK_NUM, 256>>>(
      fileSize, dfileContent, dbitOffsets, d_boundary_index, d_compressedFile,
      d_dictionary_code, d_dictionary_codelens, counter, numTasks);
  TIMER_STOP(kernel)
  CUERROR
  cudaMemcpy(compressedFile, d_compressedFile, writeSize * sizeof(uint),
             cudaMemcpyDeviceToHost);
  CUERROR
  fwrite(compressedFile, sizeof(uint), writeSize, outputFile);
  fdatasync(outputFile->_fileno);
  cudaFree(d_compressedFile);
  CUERROR
  // printOut(compressedFile, writeSize);
  cudaFreeHost(compressedFile);
  CUERROR
  cudaFree(dbitOffsets);
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

  TIMER_START(readFile)
  readFile(fileContent, dfileContent, inputFile);
  TIMER_STOP(readFile)

  getFrequencies(); // build histogram in GPU

  HuffmanTree tree;
  TIMER_START(tree)
  tree.HuffmanCodes(frequency, dictionary); // build Huffman tree in Host
  TIMER_STOP(tree)
  // printDictionary(frequency);

  outputFile = fopen("compressed_output.bin", "wb");

  TIMER_START(meta)
  fwrite(&fileSize, sizeof(unsigned long long int), 1, outputFile);
  fwrite(&blockSize, sizeof(uint), 1, outputFile);
  fwrite(&tree.noOfLeaves, sizeof(uint), 1, outputFile);
  tree.writeTree(outputFile);
  TIMER_STOP(meta)
  writeFileContents(outputFile);
  cudaFreeHost(fileContent);
  CUERROR
  cudaFree(dfileContent);
  CUERROR
  fclose(inputFile);
  fclose(outputFile);
  return 0;
}