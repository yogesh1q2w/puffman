#ifndef HUFFMAN
#define HUFFMAN 1

#include <fstream>
#include <vector>

typedef unsigned long long int ull;
using namespace std;

struct TreeNode {
  unsigned char token;
  TreeNode *left, *right;
  TreeNode(unsigned char token, TreeNode *left, TreeNode *right);
};

class codedict {
public:
  unsigned char **code;
  unsigned char *codeSize;
  unsigned char onDevice;
  codedict(unsigned char onDevice = 1);
  void addCode(const unsigned char &index, const unsigned char &codeLen,
               const unsigned char *sCode);
  void deepCopyHostToDevice(codedict &destination);
  __device__ void deepCopyDeviceToDevice(codedict *destination);
  unsigned short getSize();
  ~codedict();
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
  void buildTreeFromFrequencies(unsigned long long int *frequency);
  void getCodes(TreeNode *node, unsigned char *code, unsigned char len,
                codedict *dictionary);
  void constructTree(TreeNode *node, unsigned char *bitsRepTree,
                     unsigned int *pos);

public:
  vector<TreeArrayNode> treeInArray;
  unsigned int noOfLeaves;
  HuffmanTree();
  void readFromFile(std::ifstream &file);
  ~HuffmanTree();
  void HuffmanCodes(unsigned long long int *freq, codedict *dictionary);
  void writeTree(ofstream &fptr);
};

#endif