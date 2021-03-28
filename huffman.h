#ifndef HUFFMAN
#define HUFFMAN 1

#include <fstream>
#include <vector>

using namespace std;
void fatal(const char* str);

struct TreeNode {
  unsigned char token;
  TreeNode *left, *right;
  TreeNode(unsigned char token, TreeNode *left = nullptr,
           TreeNode *right = nullptr);
};

class codedict {
public:
  uint code[256];
  unsigned char codeSize[256];
};

struct TreeArray {
  unsigned char *token;
  int *left, *right;
};

class HuffmanTree {
private:
  TreeNode *root;
  uint createTreeFromFile(unsigned char *array, uint &offset, uint &index);
  void deleteTree(TreeNode *node);
  void constructTree(TreeNode *node, unsigned char *bitsRepTree,
                     uint &pos);

  void buildTreeFromFrequencies(uint *frequency);
  void getCodes(TreeNode *node, uint &code, unsigned char len,
                codedict &dictionary);

public:
  TreeArray tree;
  uint noOfLeaves;
  HuffmanTree();
  void readFromFile(FILE *file);
  ~HuffmanTree();
  void writeTree(FILE *fptr);
  void HuffmanCodes(uint *freq, codedict &dictionary);
};

#endif