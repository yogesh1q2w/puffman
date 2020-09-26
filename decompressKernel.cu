#include "decompressKernel.h"

__global__ void convertBitsToBytes(unsigned char* input, unsigned char* inputInBytes,
	unsigned size, unsigned blockSize)
{
	unsigned id = threadIdx.x + blockIdx.x*blockDim.x;
	if(id*blockSize < size) {
		unsigned sizeInBlock = min(blockSize, size-id*blockSize),
			offset = id*blockSize;
		for(unsigned i=offset; i<offset+sizeInBlock; i++)
			for(unsigned j=0; j<8; j++)
				inputInBytes[i*8+j] = (input[i] >> (7-j)) & 1;
	}
}

__global__ void calculateNoOfTokensInBlock(unsigned char* input, unsigned size, 
 	unsigned blockSize, HuffmanTree* tree, unsigned* outputSizes) 
{
	unsigned id = threadIdx.x + blockIdx.x*blockDim.x;
	if(id*blockSize < size) {
		unsigned sizeInBlock = min(blockSize, size-id*blockSize), 
			index = 0, noOfTokensInBlock = 0, noOfBytesParsed = 1,
			noOfBytesLeftInBlock;
		const unsigned offset = id*blockSize;
        unsigned char token;
		while(noOfBytesParsed) {
			noOfBytesLeftInBlock = sizeInBlock-index;
			noOfBytesParsed = tree->parseTree(&input[offset+index], noOfBytesLeftInBlock, 
				token);
			index += noOfBytesParsed;
			noOfTokensInBlock++;
		}
        outputSizes[id] = noOfTokensInBlock-1;
	}
}

__global__ void writeOutput(unsigned char* input, unsigned char* output, unsigned size,
	unsigned blockSize, HuffmanTree* tree, unsigned* offsets) 
{
	unsigned id = threadIdx.x + blockIdx.x*blockDim.x;
	if(id*blockSize < size) {
		unsigned sizeInBlock = min(blockSize, size-id*blockSize), inputIndex = 0,
			noOfBytesParsed, outputIndex = offsets[id], noOfBytesLeftInBlock;
		unsigned char token;
		const unsigned inputOffset = id*blockSize;
		while(true) {
			noOfBytesLeftInBlock = sizeInBlock-inputIndex;
			noOfBytesParsed = tree->parseTree(&input[inputIndex+inputOffset],
				noOfBytesLeftInBlock, token);
			if(noOfBytesParsed == 0 || outputIndex >= offsets[id+1])
				break;
			
			inputIndex += noOfBytesParsed;
			output[outputIndex] = token;
			outputIndex++;
        }
	}
}