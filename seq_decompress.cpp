#include <bits/stdc++.h>
#include <cstdio>
using namespace std;

typedef struct node {
  char a;
  struct node *l;
  struct node *r;
  bool isleaf;
  node(char _a, node *_l, node *_r, bool _isleaf) {
    a = _a;
    l = _l;
    r = _r;
    isleaf = _isleaf;
  }
} node;

node *createTree(unsigned char *huffmanTreeInBits, uint &i) {
  unsigned char readBit = (huffmanTreeInBits[i / 8] >> (7 - (i % 8))) & 1;
  i++;
  if (readBit) {
    cout << 1;
    char x = 0;
    for (uint j = 0; j < 8; j++) {
      unsigned char tokenBit = (huffmanTreeInBits[i / 8] >> (7 - (i % 8))) & 1;
      x = x | (tokenBit << (7 - j));
      i++;
    }
    cout << x;
    return new node(x, NULL, NULL, true);
  } else {
    cout << 0;
    node *left = createTree(huffmanTreeInBits, i);
    node *right = createTree(huffmanTreeInBits, i);
    return new node(0, left, right, false);
  }
}

node *readAndPrintTree(FILE *fptr, uint noOfLeaves) {
  unsigned sizeOfHuffman = 10 * noOfLeaves - 1;
  unsigned char huffmanTreeInBits[(unsigned)ceil(sizeOfHuffman / 8.)];
  fread((char *)huffmanTreeInBits, 1, (unsigned)ceil(sizeOfHuffman / 8.), fptr);
  uint i = 0;
  node *root = createTree(huffmanTreeInBits, i);
  return root;
}

void readContent(FILE *fptr, uint fileSize, node *root, uint blockSize) {

  fpos_t ftell_val;
  fgetpos(fptr, &ftell_val);
  uint current = ftell(fptr);
  fseek(fptr, 0L, SEEK_END);
  uint enc_size = ftell(fptr) - current;
  fsetpos(fptr, &ftell_val);
  unsigned int encoding[(uint)ceil(enc_size/4.)];
  uint readSize = fread(encoding, sizeof(unsigned int), ceil(enc_size/4.), fptr);
  unsigned char file[fileSize];
  uint i = 0;
  uint file_loc = 0;
  node *temp = root;
  while (1) {
    if (file_loc == fileSize)
      break;
    uint bit = 1 & (encoding[i / 32] >> (31 - (i % 32)));
    if (temp->isleaf) {
      file[file_loc] = temp->a;
      file_loc++;
      temp = root;
    } else if (i % blockSize == 0) {
      temp = root;
    }
    if (!bit) {
      temp = temp->l;
    } else {
      temp = temp->r;
    }

    i++;
  }
  FILE *optr;
  optr = fopen("decompressed_out_seq", "wb");
  fwrite(file, sizeof(unsigned char), fileSize, optr);

}

int main() {
  FILE *fptr;
  fptr = fopen("compressed_output.bin", "rb");
  unsigned long long int fileSize;
  fread(&fileSize, sizeof(long long int), 1, fptr);
  cout << "file Size read = " << fileSize << endl;
  uint blockSize;
  fread(&blockSize, sizeof(uint), 1, fptr);
  uint numLeaves;
  fread(&numLeaves, sizeof(uint), 1, fptr);
  cout << "Block size = " << blockSize << "\nNum Leaves = " << numLeaves
       << endl;
  node *root = readAndPrintTree(fptr, numLeaves);
  readContent(fptr, fileSize, root, blockSize);
  return 0;
}