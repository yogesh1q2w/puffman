#ifndef HUFFMAN
#define HUFFMAN 1

#include <fstream>
#include <vector>

using namespace std;

struct TreeNode {
  unsigned char token;
  TreeNode *left, *right;
  TreeNode(unsigned char token, TreeNode *left=nullptr, TreeNode *right=nullptr);
};

class codedict {
public:
  unsigned int code[256];
  unsigned char codeSize[256];
};

struct TreeArrayNode {
  unsigned char token;
  int left, right;
  void assignValues(unsigned char, int, int);
};

class HuffmanTree {
private:
  TreeNode *root;
  int createTreeFromFile(unsigned char *array, unsigned size, unsigned &offset,
                         int &index);
  void deleteTree(TreeNode *node);
  void constructTree(TreeNode *node, unsigned char *bitsRepTree,
                     unsigned int &pos);
  unsigned char _heightOfTree(TreeNode *node);


  void buildTreeFromFrequencies(unsigned int *frequency);
  void getCodes(TreeNode *node, uint &code, unsigned char len,
                codedict &dictionary);

public:
  vector<TreeArrayNode> treeInArray;
  unsigned int noOfLeaves;
  HuffmanTree();
  void readFromFile(std::ifstream &file);
  ~HuffmanTree();
  unsigned char heightOfTree();
  void writeTree(FILE *fptr);


  void HuffmanCodes(unsigned int *freq, codedict &dictionary);

};

#endif