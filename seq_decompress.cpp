#include <bits/stdc++.h>
#include <cstdio>
using namespace std;



void readAndPrintTree(FILE *fptr, uint noOfLeaves) {
  unsigned sizeOfHuffman = 10 * noOfLeaves - 1;
  unsigned char huffmanTreeInBits[(unsigned)ceil(sizeOfHuffman / 8.)];
  fread((char *)huffmanTreeInBits, 1, (unsigned)ceil(sizeOfHuffman / 8.), fptr);
  for (uint i = 0; i < sizeOfHuffman;) {
    unsigned char readBit = (huffmanTreeInBits[i/8] >> (7 - (i%8))) & 1;
    i++;
    if (readBit) {
      cout << (int)readBit;
      char x =0;
      for (uint j = 0; j < 8; j++) {
        unsigned char tokenBit = (huffmanTreeInBits[i/8] >> (7 - (i%8))) & 1;
        x = x | (tokenBit << (7 - j));
        i++;
      }
      cout << (char)x ;
    } else {
      cout << (int)readBit;
    }
  }
}

void readContent(FILE *fptr) {
  uint fileContent[200];
  uint readSize =
      fread(fileContent, sizeof(unsigned char), 200, fptr);
  cout << readSize <<endl;
  for (uint i = 0; i < readSize; i++) {
    for (uint j = 0; j < 32; j++) {
      cout << (1 & (fileContent[i] >> (31-j)));
    }
  }
  cout << endl;
}

int main() {
  FILE *fptr;
  fptr = fopen("compressed_output.bin", "rb");
  unsigned long long int fileSize;
  fread(&fileSize, sizeof(long long int), 1, fptr);
  cout << "file Size read = " << fileSize << endl;
  uint blockSize;
  fread(&blockSize,sizeof(uint), 1, fptr);
  uint numLeaves;
  fread(&numLeaves, sizeof(uint), 1, fptr);
  cout << "Block size = " << blockSize << "\nNum Leaves = " << numLeaves
       << endl;
  readAndPrintTree(fptr, numLeaves);
  cout << endl;
  readContent(fptr);
}