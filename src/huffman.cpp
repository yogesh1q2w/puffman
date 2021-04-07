#include "../include/huffman.h"
using namespace std;

void fatal(const char *str) {
  fprintf(stderr, "%s! at %s in %d\n", str, __FILE__, __LINE__);
  exit(EXIT_FAILURE);
}

template <class T>
inline const T minHeapPop(priority_queue<T, vector<T>, greater<T>> &heap) {
  const T top = heap.top();
  heap.pop();
  return top;
}

TreeNode::TreeNode(unsigned char token, TreeNode *left, TreeNode *right) {
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

uint HuffmanTree::createTreeFromFile(unsigned char *huffmanTree, uint &offset,
                                     uint &index) {
  unsigned char readBit = (huffmanTree[offset / 8] >> (7 - (offset % 8))) & 1;
  offset++;
  uint index_copy = index;
  index++;
  if (readBit) {
    unsigned char x = 0;
    for (uint j = 0; j < 8; j++) {
      unsigned char tokenBit =
          (huffmanTree[offset / 8] >> (7 - (offset % 8))) & 1;
      x = x | (tokenBit << (7 - j));
      offset++;
    }
    tree.token[index_copy] = x;
    tree.left[index_copy] = -1;
    tree.right[index_copy] = -1;
    return index_copy;
  } else {
    uint leftChildPos = createTreeFromFile(huffmanTree, offset, index);
    uint rightChildPos = createTreeFromFile(huffmanTree, offset, index);
    tree.token[index_copy] = 0;
    tree.left[index_copy] = leftChildPos;
    tree.right[index_copy] = rightChildPos;
    return index_copy;
  }
}

void HuffmanTree::readFromFile(FILE *file) {
  if (1 != fread(&noOfLeaves, sizeof(uint), 1, file))
    fatal("File read error 3");
  uint numNodes = 2 * noOfLeaves - 1;
  uint sizeOfHuffmanTree = 10 * noOfLeaves - 1;
  uint huffmanTreeSize = ceil(sizeOfHuffmanTree / 8.);

  unsigned char huffmanTree[huffmanTreeSize];
  if (huffmanTreeSize !=
      fread(huffmanTree, sizeof(unsigned char), huffmanTreeSize, file))
    fatal("File read error 4");

  tree.token = new unsigned char[numNodes];
  tree.left = new int[numNodes];
  tree.right = new int[numNodes];

  uint offset = 0;
  uint index = 0;
  createTreeFromFile(huffmanTree, offset, index);
}

void HuffmanTree::buildTreeFromFrequencies(unsigned int *frequency) {
  typedef pair<uint, TreeNode *> pullt;
  priority_queue<pullt, vector<pullt>, greater<pullt>> minHeap;
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

void HuffmanTree::getCodes(TreeNode *node, uint &code, unsigned char len,
                           codedict &dictionary) {
  if ((node->left == nullptr) && (node->right == nullptr)) {
    dictionary.codeSize[node->token] = len;
    dictionary.code[node->token] = code >> (32-len);
    return;
  }

  if (node->left != nullptr) {
    code = code & (~(1 << (31 - len)));
    getCodes(node->left, code, len + 1, dictionary);
  }

  if (node->right != nullptr) {
    code = code | (1 << (31 - len));
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