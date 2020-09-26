#ifndef HUFFMAN
#define HUFFMAN 1

#include <fstream>

typedef unsigned long long int ull;

struct TreeNode {
	unsigned char token;
	TreeNode *left, *right;
	TreeNode(unsigned char token, TreeNode* left, TreeNode* right);
	static TreeNode* createDeviceTreeNode(unsigned char token,  TreeNode* left, TreeNode* right);
};

typedef struct codedict {
    unsigned char code[256][255];
    unsigned char codeSize[256];
} codedict;

class HuffmanTree {
private:
    TreeNode *root, *dRoot;
	TreeNode* createTreeFromFile(unsigned char* array, unsigned size, unsigned& offset);
	void deleteTree(TreeNode* node);
	void buildTreeFromFrequencies(unsigned long long int* frequency);
	void getCodes(TreeNode* node,unsigned char *code, unsigned char len, codedict *dictionary);
	void constructTree(TreeNode* node,unsigned char *bitsRepTree, unsigned int *pos);
	
public:
	unsigned char noOfLeaves;	
	HuffmanTree();
	void readFromFile(std::ifstream& file);
	~HuffmanTree();
	__host__ __device__ unsigned parseTree(unsigned char* byteArray,unsigned size, unsigned char& token) const;
	void HuffmanCodes(unsigned long long int *freq, codedict *dictionary);
	void writeTree(FILE *fptr);
};

#endif