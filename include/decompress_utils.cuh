#ifndef DECOMPRESS_UTILS
#define DECOMPRESS_UTILS
#include "constants.h"
#include "huffman.h"
#include "timer.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void decode(FILE *inputFile, FILE *outputFile, HuffmanTree tree, uint blockSize,
            uint sizeOfFile, unsigned long long int encodedFileSize,
            uint numNodes, unsigned char leastCodeSize);
#endif