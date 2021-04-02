#include "../include/compress_utils.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

int main(int argc, char **argv) {
  FILE *inputFile, *outputFile;
  if (argc != 2) {
    printf("Running format is ./compress <file name>\n");
    return 0;
  }

  inputFile = fopen(argv[1], "rb");

  if (!inputFile) {
    printf("Please give a valid file to open.\n");
    return 0;
  }

  uint blockSize = BLOCK_SIZE;
  uint *fileContent, *dfileContent;
  codedict dictionary;
  unsigned long long int fileSize;
  uint intFileSize;
  uint *frequency;

  readFile(fileContent, dfileContent, inputFile, fileSize, intFileSize);

  getFrequencies(dfileContent, fileSize, frequency,
                 intFileSize); // build histogram in GPU
  HuffmanTree tree;
  CPU_TIMER_START(tree_build)
  tree.HuffmanCodes(frequency, dictionary); // build Huffman tree in Host
  CPU_TIMER_STOP(tree_build)
  cudaFreeHost(frequency);

  outputFile = fopen("temp/compressed_output.bin", "wb");

  CPU_TIMER_START(meta)
  fwrite(&fileSize, sizeof(unsigned long long int), 1, outputFile);
  fwrite(&blockSize, sizeof(uint), 1, outputFile);
  fwrite(&tree.noOfLeaves, sizeof(uint), 1, outputFile);
  tree.writeTree(outputFile);
  CPU_TIMER_STOP(meta)
  writeFileContents(outputFile, fileSize, fileContent, dfileContent,
                    dictionary);
  fclose(inputFile);
  fclose(outputFile);
  return 0;
}