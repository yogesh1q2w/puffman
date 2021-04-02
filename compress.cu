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
unsigned long long int fileSize;
uint intFileSize;
uint frequency[256];

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

void getOffsetArray(unsigned long long int *bitOffsets,
                    unsigned long long int *boundary_index,
                    unsigned long long int &encodedFileSize) {
  bitOffsets[0] = 0;
  unsigned long long int searchValue = BLOCK_SIZE;
  unsigned long long int i;
  for (i = 1; i < fileSize; i++) {
    bitOffsets[i] = bitOffsets[i - 1] + dictionary.codeSize[getcharAt(i - 1)];

    if (bitOffsets[i] > searchValue) {
      boundary_index[i - 1] = bitOffsets[i - 1];
      bitOffsets[i - 1] = searchValue;
      searchValue += BLOCK_SIZE;
      i--;
      // cout << "BOUNDARY AT " << i << " = " << bitOffsets[i] << endl;
    } else if (bitOffsets[i] == searchValue) {
      searchValue += BLOCK_SIZE;
    }
  }

  if (bitOffsets[i - 1] + dictionary.codeSize[getcharAt(i - 1)] > searchValue) {
    boundary_index[i - 1] = bitOffsets[i - 1];
    bitOffsets[i - 1] = searchValue;
  }
  encodedFileSize =
      bitOffsets[fileSize - 1] + dictionary.codeSize[getcharAt(fileSize - 1)];
}

void writeFileContents(FILE *outputFile) {

  uint *compressedFile, *d_compressedFile;
  unsigned long long int *bitOffsets, *dbitOffsets;
  unsigned long long int *boundary_index, *d_boundary_index;
  cudaMallocHost(&bitOffsets, fileSize * sizeof(unsigned long long int));
  cudaMallocHost(&boundary_index, fileSize * sizeof(unsigned long long int));
  cudaMemset(boundary_index, 0, fileSize * sizeof(unsigned long long int));
  CUERROR
  cudaMalloc((void **)&dbitOffsets, fileSize * sizeof(unsigned long long int));
  cudaMalloc((void **)&d_boundary_index,
             fileSize * sizeof(unsigned long long int));
  CUERROR

  unsigned long long int encodedFileSize;
  TIMER_START(offset)
  getOffsetArray(bitOffsets, boundary_index, encodedFileSize);
  TIMER_STOP(offset)

  cudaMemcpy(dbitOffsets, bitOffsets, fileSize * sizeof(unsigned long long int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_boundary_index, boundary_index,
             fileSize * sizeof(unsigned long long int), cudaMemcpyHostToDevice);
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

  uint *d_dictionary_code;
  unsigned char *d_dictionary_codelens;
  cudaMalloc(&d_dictionary_code, 256 * sizeof(uint));
  cudaMalloc(&d_dictionary_codelens, 256 * sizeof(unsigned char));
  cudaMemcpy(d_dictionary_code, dictionary.code, 256 * sizeof(uint),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_dictionary_codelens, dictionary.codeSize,
             256 * sizeof(unsigned char), cudaMemcpyHostToDevice);
  // print_dict<<<1, 1>>>(d_dictionary_code, d_dictionary_codelens);

  uint *counter;
  cudaMalloc(&counter, sizeof(uint));
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
  cout << "enc file size = " << encodedFileSize << endl;
  fwrite(&encodedFileSize, sizeof(unsigned long long int), 1, outputFile);
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