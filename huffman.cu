#include "huffman.h"

#include <assert.h>
#include <math.h>
#include <iostream>
#include <queue>
#include <utility>
#include <cuda.h>
#include <stdio.h>

using namespace std;

void convertBitsToBytes(unsigned char* arrayInBits, unsigned char* arrayInBytes, unsigned size) {
	unsigned sizeInBytes = ceil(size/8.);
	for(unsigned i=0; i<sizeInBytes; i++)
		for(unsigned j=0; j<8 && i*8+j < size; j++)
			arrayInBytes[i*8+j] = (arrayInBits[i] >> (7-j)) & 1;
}

template<class T>
inline const T minHeapPop(priority_queue<T, vector<T>, greater<T>>& heap) {
	const T top = heap.top();
	heap.pop();
	return top;
}

unsigned char readByte(unsigned char* byte) {
	unsigned char result = 0;
	for(unsigned i=0; i<8; i++) {
		assert(byte[i] == 0 || byte[i] == 1);
		result = (result << 1) | byte[i];
	}
	return result;
}

TreeNode::TreeNode(unsigned char token, TreeNode* left=nullptr, TreeNode* right=nullptr) {
	this->token = token;
	this->left = left;
	this->right = right;
}

TreeNode* TreeNode::createDeviceTreeNode(unsigned char token,  TreeNode* left=nullptr, TreeNode* right=nullptr) {
	TreeNode* node = new TreeNode(token, left, right), *dNode;
	cudaMalloc(&dNode, sizeof(TreeNode));
	cudaMemcpy(dNode, node, sizeof(TreeNode), cudaMemcpyHostToDevice);
	delete node;
	return dNode;
}

HuffmanTree::HuffmanTree() {
	root = nullptr;
	dRoot = nullptr;
	noOfLeaves = 0;
}

HuffmanTree::~HuffmanTree() {
	deleteTree(root);
}

void HuffmanTree::deleteTree(TreeNode* node) {
	if(node) {
		deleteTree(node->left);
		deleteTree(node->right);
		delete node;
	}
}

TreeNode* HuffmanTree::createTreeFromFile(unsigned char* array, unsigned size, unsigned &offset) {
	unsigned char token = 0;
	if(offset >= size)
		return nullptr;
	assert(array[offset] == 0 || array[offset] == 1);	
	if(array[offset] == 1) {
		token = readByte(&array[offset+1]);
		offset += 9;
		return TreeNode::createDeviceTreeNode(token);
	}
	else {
		offset++;
		TreeNode *leftChild = createTreeFromFile(array, size, offset);
		TreeNode *rightChild = createTreeFromFile(array, size, offset);
		return TreeNode::createDeviceTreeNode(0, leftChild, rightChild);
	}
}

void HuffmanTree::readFromFile(ifstream& file) {
	unsigned char noOfLeaves;
	file.read((char *)&noOfLeaves, sizeof(unsigned char));
	unsigned sizeOfHuffman = 10*noOfLeaves-1;
	unsigned char huffmanTreeInBits[(unsigned)ceil(sizeOfHuffman/8.)];
	file.read((char *)huffmanTreeInBits, (unsigned)ceil(sizeOfHuffman/8.));	
	
	unsigned char huffmanTreeInBytes[sizeOfHuffman];

	convertBitsToBytes(huffmanTreeInBits, huffmanTreeInBytes, sizeOfHuffman);

	unsigned offset = 0;
	dRoot = createTreeFromFile(huffmanTreeInBytes, sizeOfHuffman, offset);
}

__device__ unsigned HuffmanTree::parseTree(unsigned char* byteArray, unsigned size, unsigned char& token) const {
	TreeNode* node = dRoot;
	unsigned i=0;
	while(node != nullptr && i < size) {
		assert(byteArray[i] == 0 || byteArray[i] == 1);
		if((node->left == nullptr) && (node->right == nullptr)) {
			token = node->token;
			return i;
		}
		if(byteArray[i] == 0)
			node = node->left;
		else
			node = node->right;
		i++;
	}
	if((node->left == nullptr) && (node->right == nullptr)) {
		token = node->token;
		return i;
	}
	token = 0;
	return 0;
}

void HuffmanTree::buildTreeFromFrequencies(ull* frequency) {
	typedef pair<ull, TreeNode*> pullt;
	priority_queue<pullt, vector<pullt>, greater<pullt>> minHeap;
	noOfLeaves = 0;
	for(unsigned int i=0; i<256; i++) {
		if(frequency[i] > 0) {
			TreeNode* node = new TreeNode(i);
			minHeap.push(make_pair(frequency[i], node));
			noOfLeaves++;
		}
	}
	while(minHeap.size() > 1) {
		pullt pair1 = minHeapPop(minHeap);
		pullt pair2 = minHeapPop(minHeap);
		TreeNode* node = new TreeNode(0, pair1.second, pair2.second);
		ull newNodeFrequency = pair1.first+pair2.first;
		minHeap.push(make_pair(newNodeFrequency, node));
	}
	root = minHeap.top().second;
}

void HuffmanTree::getCodes(TreeNode* node, unsigned char *code, unsigned char len, codedict *dictionary) { 
	if ((node->left == nullptr) && (node->right == nullptr)) {
		dictionary->codeSize[node->token] = len;
		memcpy(dictionary->code[node->token], code,len*sizeof(unsigned char));
		return;
    }

    if (node->left != nullptr) { 
		code[len] = 0;
		getCodes(node->left, code, len+1, dictionary); 
	} 

	if (node->right != nullptr) { 
		code[len] = 1;
		getCodes(node->right, code, len+1, dictionary); 
	} 

} 


void HuffmanTree::HuffmanCodes(unsigned long long int *freq, codedict *dictionary) { 
	buildTreeFromFrequencies(freq);
	unsigned char code[255];
	getCodes(root, code, 0, dictionary);
}

void HuffmanTree::constructTree(TreeNode* node, unsigned char *bitsRepTree, unsigned int *pos) {
	if((node->left == nullptr) && (node->right == nullptr)) {
		bitsRepTree[(*pos)++] = 1;
		for(unsigned char i = 0; i < 8; i++)
			bitsRepTree[(*pos)++] = (node->token >> (7 - i)) & 1;
	}
	else {
		bitsRepTree[(*pos)++] = 0;
		constructTree(node->left,bitsRepTree,pos);
		constructTree(node->right,bitsRepTree,pos);
	}
}

void HuffmanTree::writeTree(FILE* fptr) {
	unsigned char bitsRepTree[10*noOfLeaves - 1];
	unsigned int pos = 0;
	constructTree(root,bitsRepTree,&pos);
	unsigned int writeTreeSize = ceil(pos/8.);
	unsigned char finalTree[writeTreeSize];
	for(unsigned int i = 0; i < writeTreeSize; i++) {
		for(unsigned int j = 0; j < 8; j++) {
			if(bitsRepTree[i*8+j])
				finalTree[i] = (finalTree[i] << 1) | 1;
			else
				finalTree[i] = finalTree[i] << 1; 
		}
	}
	fwrite(finalTree,sizeof(unsigned char),writeTreeSize,fptr);
}