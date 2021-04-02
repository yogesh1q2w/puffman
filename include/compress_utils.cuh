#ifndef COMPRESS_UTILS
#define COMPRESS_UTILS

#include "huffman.h"
#include "timer.h"

void readFile(uint *&fileContent, uint *&dfileContent, FILE *inputFile,
              unsigned long long int &fileSize, uint &intFileSize);

void getFrequencies(uint *dfileContent, unsigned long long int &fileSize,
                    uint *&frequency, uint &intFileSize);

void getOffsetArray(unsigned long long int *bitOffsets,
                    unsigned long long int *boundary_index,
                    unsigned long long int &encodedFileSize,
                    unsigned long long int &fileSize, codedict &dictionary,
                    uint *fileContent);

void writeFileContents(FILE *outputFile, unsigned long long int &fileSize,
                       uint *fileContent, uint *dfileContent,
                       codedict &dictionary);

#endif