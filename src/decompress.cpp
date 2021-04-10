#include "../include/decompress_utils.cuh"

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("The usage is ./a.out <fileToDecompress>\n");
    return 0;
  }
  char *filename = argv[1];
  FILE *inputFile, *outputFile;
  inputFile = fopen(filename, "rb");
  if (!inputFile) {
    printf("The file could not be opened\n");
    return 0;
  }
  outputFile = fopen("temp/decompressed_output", "wb");

  ull sizeOfFile;
  unsigned int blockSize;
  if (1 != fread(&sizeOfFile, sizeof(ull), 1, inputFile))
    fatal("File read error 1");
  if (1 != fread(&blockSize, sizeof(uint), 1, inputFile))
    fatal("File read error 2");

  HuffmanTree tree;
  CPU_TIMER_START(tree_generation)
  tree.readFromFile(inputFile);
  CPU_TIMER_STOP(tree_generation)
  unsigned long long int encodedFileSize;
  if (1 !=
      fread(&encodedFileSize, sizeof(unsigned long long int), 1, inputFile))
    fatal("File read error 3");
  decode(inputFile, outputFile, tree, blockSize, sizeOfFile, encodedFileSize,
         2 * tree.noOfLeaves - 1, tree.leastSizeCode);
  fclose(inputFile);
  fclose(outputFile);

  return 0;
}