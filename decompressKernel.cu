#include "decompressKernel.h"
#include <assert.h>

// __constant__ TreeArrayNode deviceTree[512];
// __constant__ int rootIndex;

// __device__ unsigned parseTree(unsigned char *byteArray, unsigned size,
//                               unsigned char &token) {
//   int index = rootIndex;
//   unsigned i = 0;
//   while (index != -1 && i < size) {
//     assert(byteArray[i] == 0 || byteArray[i] == 1);
//     if ((deviceTree[index].left == -1) && (deviceTree[index].right == -1)) {
//       token = deviceTree[index].token;
//       return i;
//     }
//     if (byteArray[i] == 0)
//       index = deviceTree[index].left;
//     else
//       index = deviceTree[index].right;
//     i++;
//   }
//   if ((deviceTree[index].left == -1) && (deviceTree[index].right == -1)) {
//     token = deviceTree[index].token;
//     return i;
//   }
//   token = 0;
//   return 0;
// }

// __global__ void convertBitsToBytes(unsigned char *input,
//                                    unsigned char *inputInBytes, ull size,
//                                    unsigned blockSize) {
//   ull id = threadIdx.x + blockIdx.x * blockDim.x;
//   if (id * blockSize < size) {
//     unsigned sizeInBlock = min((ull) blockSize, size - id * blockSize);
//     ull offset = id * blockSize;
//     for (ull i = offset; i < offset + sizeInBlock; i++)
//       for (unsigned j = 0; j < 8; j++)
//         inputInBytes[i * 8 + j] = (input[i] >> (7 - j)) & 1;
//   }
// }

// __global__ void calculateNoOfTokensInBlock(unsigned char *input, ull size,
//                                            unsigned blockSize,
//                                            unsigned *outputSizes) {
//   ull id = threadIdx.x + blockIdx.x * blockDim.x;
//   if (id * blockSize < size) {
//     unsigned sizeInBlock = min((ull) blockSize, size - id * blockSize), index = 0,
//              noOfTokensInBlock = 0, noOfBytesParsed, noOfBytesLeftInBlock;
//     const ull offset = id * blockSize;
//     unsigned char token;
//     while (true) {
//       noOfBytesLeftInBlock = sizeInBlock - index;
//       noOfBytesParsed =
//           parseTree(&input[offset + index], noOfBytesLeftInBlock, token);
//       if (noOfBytesParsed == 0)
//         break;
//       index += noOfBytesParsed;
//       noOfTokensInBlock++;
//     }
//     outputSizes[id] = noOfTokensInBlock;
//   }
// }

// __global__ void writeOutput(unsigned char *input, unsigned char *output,
//                             ull size, unsigned blockSize,
//                             unsigned *offsets) {
//   ull id = threadIdx.x + blockIdx.x * blockDim.x;
//   if (id * blockSize < size) {
//     unsigned sizeInBlock = min((ull)blockSize, size - id * blockSize),
//              inputIndex = 0, noOfBytesParsed, outputIndex = offsets[id],
//              noOfBytesLeftInBlock;
//     unsigned char token;
//     const ull inputOffset = id * blockSize;
//     while (true) {
//       noOfBytesLeftInBlock = sizeInBlock - inputIndex;
//       noOfBytesParsed = parseTree(&input[inputIndex + inputOffset],
//                                   noOfBytesLeftInBlock, token);
//       if (noOfBytesParsed == 0 || outputIndex >= offsets[id + 1])
//         break;

//       inputIndex += noOfBytesParsed;
//       output[outputIndex] = token;
//       outputIndex++;
//     }
//   }
// }