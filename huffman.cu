#include "constants.h"
#include "huffman.h"

#include <assert.h>
#include <cstring>
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

TreeNode::TreeNode(unsigned char token, TreeNode *left,
                   TreeNode *right) {
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
    cout << 1 << token;
  } else {
    offset++;
    cout << 0;
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
  cout << endl;
}

void HuffmanTree::buildTreeFromFrequencies(unsigned int *frequency) {
  typedef pair<uint, TreeNode *> pullt;
  priority_queue<pullt, vector<pullt>, greater<pullt> > minHeap;
  noOfLeaves = 0;
  for (unsigned short i = 0; i < 256; i++) {
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
    uint newNodeFrequency = pair1.first + pair2.first;
    minHeap.push(make_pair(newNodeFrequency, node));
  }
  root = minHeap.top().second;
}

void HuffmanTree::getCodes(TreeNode *node, uint &code,
                           unsigned char len, codedict &dictionary) {
  if ((node->left == nullptr) && (node->right == nullptr)) {
    dictionary.codeSize[node->token] = len;
    dictionary.code[node->token] = code;
    return;
  }

  if (node->left != nullptr) {
    code = code & (~(1 << (31-len)));
    getCodes(node->left, code, len + 1, dictionary);
  }

  if (node->right != nullptr) {
    code = code | (1 << (31-len));
    getCodes(node->right, code, len + 1, dictionary);
  }
}

void HuffmanTree::HuffmanCodes(unsigned int *freq, codedict &dictionary) {
  buildTreeFromFrequencies(freq);
  uint code = 0;
  getCodes(root, code, 0, dictionary);
}

void HuffmanTree::constructTree(TreeNode *node, unsigned char *bitsRepTree,
                                unsigned int &pos) {
  unsigned int bitRepPos = pos / 8;
  unsigned int modifyIndex = 7 - pos % 8;
  if ((node->left == nullptr) && (node->right == nullptr)) {
    bitsRepTree[bitRepPos] = bitsRepTree[bitRepPos] | (1 << modifyIndex);
    ++pos;
    for (unsigned char i = 0; i < 8; i++) {
      bitRepPos = pos / 8;
      modifyIndex = 7 - (pos % 8);
      unsigned char mask = (1 << modifyIndex);
      bitsRepTree[bitRepPos] =
          (bitsRepTree[bitRepPos] & ~mask) |
          ((((node->token >> (7 - i)) & 1) << modifyIndex) & mask);
      ++pos;
    }
  } else {
    bitsRepTree[bitRepPos] = bitsRepTree[bitRepPos] & ~(1 << modifyIndex);
    ++pos;
    constructTree(node->left, bitsRepTree, pos);
    constructTree(node->right, bitsRepTree, pos);
  }
}

void HuffmanTree::writeTree(FILE *fptr) {

  unsigned int writeTreeSize = ceil((10 * noOfLeaves - 1) / 8.);
  unsigned char bitsRepTree[writeTreeSize];
  unsigned int pos = 0;
  constructTree(root, bitsRepTree, pos);
  fwrite(bitsRepTree, sizeof(unsigned char), ceil(pos / 8.), fptr);
}
