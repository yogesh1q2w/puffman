#include "huffman.h"

#include <assert.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <queue>
#include <stdio.h>
#include <string.h>
#include <utility>

using namespace std;

void convertBitsToBytes(unsigned char *arrayInBits, unsigned char *arrayInBytes,
                        unsigned size) {
  unsigned sizeInBytes = ceil(size / 8.);
  for (unsigned i = 0; i < sizeInBytes; i++)
    for (unsigned j = 0; j < 8 && i * 8 + j < size; j++)
      arrayInBytes[i * 8 + j] = (arrayInBits[i] >> (7 - j)) & 1;
}

template <class T>
inline const T minHeapPop(priority_queue<T, vector<T>, greater<T>> &heap) {
  const T top = heap.top();
  heap.pop();
  return top;
}

unsigned char readByte(unsigned char *byte) {
  unsigned char result = 0;
  for (unsigned i = 0; i < 8; i++) {
    assert(byte[i] == 0 || byte[i] == 1);
    result = (result << 1) | byte[i];
  }
  return result;
}

TreeNode::TreeNode(unsigned char token, TreeNode *left = nullptr,
                   TreeNode *right = nullptr) {
  this->token = token;
  this->left = left;
  this->right = right;
}

void TreeArrayNode::assignValues(unsigned char token, int left = -1,
                                 int right = -1) {
  this->token = token;
  this->left = left;
  this->right = right;
}

HuffmanTree::HuffmanTree() {
  root = nullptr;
  noOfLeaves = 0;
}

HuffmanTree::~HuffmanTree() { deleteTree(root); }

void HuffmanTree::deleteTree(TreeNode *node) {
  if (node) {
    deleteTree(node->left);
    deleteTree(node->right);
    delete node;
  }
}

int HuffmanTree::createTreeFromFile(unsigned char *array, unsigned size,
                                    unsigned &offset, int &index) {
  unsigned char token = 0;
  assert(offset < size);
  assert(array[offset] == 0 || array[offset] == 1);
  if (array[offset] == 1) {
    token = readByte(&array[offset + 1]);
    offset += 9;
    treeInArray[index].assignValues(token);
  } else {
    offset++;
    int left = createTreeFromFile(array, size, offset, index);
    int right = createTreeFromFile(array, size, offset, index);
    treeInArray[index].assignValues(0, left, right);
  }
  return index++;
}

void HuffmanTree::readFromFile(ifstream &file) {
  unsigned int noOfLeaves;
  file.read((char *)&noOfLeaves, sizeof(unsigned int));
  unsigned sizeOfHuffman = 10 * noOfLeaves - 1;
  unsigned char huffmanTreeInBits[(unsigned)ceil(sizeOfHuffman / 8.)];
  file.read((char *)huffmanTreeInBits, (unsigned)ceil(sizeOfHuffman / 8.));

  unsigned char huffmanTreeInBytes[sizeOfHuffman];

  convertBitsToBytes(huffmanTreeInBits, huffmanTreeInBytes, sizeOfHuffman);

  treeInArray.resize(2 * noOfLeaves - 1);
  unsigned offset = 0;
  int index = 0;
  createTreeFromFile(huffmanTreeInBytes, sizeOfHuffman, offset, index);
}

void HuffmanTree::buildTreeFromFrequencies(unsigned long long int *frequency) {
  typedef pair<ull, TreeNode *> pullt;
  priority_queue<pullt, vector<pullt>, greater<pullt>> minHeap;
  noOfLeaves = 0;
  for (unsigned int i = 0; i < 256; i++) {
    if (frequency[i] > 0) {
      TreeNode *node = new TreeNode(i);
      minHeap.push(make_pair(frequency[i], node));
      noOfLeaves++;
    }
  }
  while (minHeap.size() > 1) {
    pullt pair1 = minHeapPop(minHeap);
    pullt pair2 = minHeapPop(minHeap);
    TreeNode *node = new TreeNode(0, pair1.second, pair2.second);
    ull newNodeFrequency = pair1.first + pair2.first;
    minHeap.push(make_pair(newNodeFrequency, node));
  }
  root = minHeap.top().second;
}

void HuffmanTree::getCodes(TreeNode *node, unsigned char *code,
                           unsigned char len, codedict *dictionary) {
  if ((node->left == nullptr) && (node->right == nullptr)) {
    dictionary->codeSize[node->token] = len;
    dictionary->addCode(node->token, len, code);
    return;
  }

  if (node->left != nullptr) {
    code[len] = 0;
    getCodes(node->left, code, len + 1, dictionary);
  }

  if (node->right != nullptr) {
    code[len] = 1;
    getCodes(node->right, code, len + 1, dictionary);
  }
}

void HuffmanTree::HuffmanCodes(unsigned long long int *freq,
                               codedict *dictionary) {
  buildTreeFromFrequencies(freq);
  unsigned char code[255];
  getCodes(root, code, 0, dictionary);
}

void HuffmanTree::constructTree(TreeNode *node, unsigned char *bitsRepTree,
                                unsigned int *pos) {
  if ((node->left == nullptr) && (node->right == nullptr)) {
    bitsRepTree[(*pos)++] = 1;
    for (unsigned char i = 0; i < 8; i++)
      bitsRepTree[(*pos)++] = (node->token >> (7 - i)) & 1;
  } else {
    bitsRepTree[(*pos)++] = 0;
    constructTree(node->left, bitsRepTree, pos);
    constructTree(node->right, bitsRepTree, pos);
  }
}

void HuffmanTree::writeTree(ofstream &fptr) {
  unsigned char bitsRepTree[10 * noOfLeaves - 1];
  unsigned int pos = 0;
  constructTree(root, bitsRepTree, &pos);
  unsigned int writeTreeSize = ceil(pos / 8.);
  unsigned char finalTree[writeTreeSize];
  for (unsigned int i = 0; i < writeTreeSize; i++) {
    for (unsigned int j = 0; j < 8; j++) {
      if (bitsRepTree[i * 8 + j])
        finalTree[i] = (finalTree[i] << 1) | 1;
      else
        finalTree[i] = finalTree[i] << 1;
    }
  }
  fptr.write((char *)finalTree, writeTreeSize);
}

codedict::codedict(unsigned char onDeviceBit) {
  onDevice = onDeviceBit;
  if (onDevice) {
    cudaMalloc(&code, 256 * sizeof(unsigned char *));
    cudaError_t error = cudaGetLastError();
    cout << "Error encountered: " << cudaGetErrorString(error) << endl;
    cudaMalloc(&codeSize, 256 * sizeof(unsigned char));
    error = cudaGetLastError();
    cout << "Error encountered: " << cudaGetErrorString(error) << endl;
  } else {
    code = new unsigned char *[256];
    codeSize = new unsigned char[256];
  }
}

void codedict::addCode(const unsigned char &token, const unsigned char &codeLen,
                       const unsigned char *sCode) {
  code[token] = new unsigned char[codeLen];
  memcpy(code[token], sCode, codeLen * sizeof(unsigned char));
}

void codedict::deepCopyHostToDevice(codedict &destination) {
  for (unsigned short i = 0; i < 256; i++) {
    if (codeSize[i] > 0) {
      cout << "HI" << endl;
      cudaMalloc(&destination.code[i], codeSize[i] * sizeof(unsigned char));
      cudaError_t error = cudaGetLastError();
      cout << "Error encountered: " << cudaGetErrorString(error) << endl;
      cudaMemcpy(destination.code[i], code[i], codeSize[i],
                 cudaMemcpyHostToDevice);
      error = cudaGetLastError();
      cout << "Error encountered: " << cudaGetErrorString(error) << endl;
    }
  }
  cudaMemcpy(destination.codeSize, codeSize, 256, cudaMemcpyHostToDevice);
  cudaError_t error = cudaGetLastError();
  cout << "Error encountered: " << cudaGetErrorString(error) << endl;
}

__device__ void codedict::deepCopyDeviceToDevice(codedict *destination) {
  for (unsigned short i = 0; i < 256; i++) {
    if (codeSize[i] > 0)
      memcpy(destination->code[i], code[i], codeSize[i]);
  }
  memcpy(destination->codeSize, codeSize, 256);
}

unsigned short codedict::getSize() {
  unsigned short size = 0;
  for (unsigned short i = 0; i < 256; i++) {
    size += codeSize[i];
  }
  return (size + 256 * (sizeof(unsigned char *) + 1) + 1);
}

codedict::~codedict() {
  if (this->onDevice) {
    for (unsigned short i = 0; i < 256; i++) {
      cudaFree(this->code[i]);
    }
    cudaFree(this->code);
    cudaFree(this->codeSize);
  } else {
    for (unsigned short i = 0; i < 256; i++) {
      delete this->code[i];
    }
    delete this->code;
    delete this->codeSize;
  }
}