#include "../include/decompress_utils.cuh"

__global__ void single_shot_decode(uint *encodedString,
                                   unsigned long long int encodedFileSize,
                                   unsigned char *treeToken, uint *treeLeft,
                                   uint *treeRight,
                                   volatile unsigned long long int *charOffset,
                                   uint *decodedString, uint *taskCounter,
                                   uint numNodes, uint numTasks) {
  uint *threadInput;
  unsigned long long int threadInput_idx = 0;
  uint task_idx = 0;
  extern __shared__ int tree[];
  int *sh_left = tree;
  int *sh_right = (int *)&tree[numNodes];
  unsigned char *sh_token = (unsigned char *)&tree[2 * numNodes];
  __shared__ unsigned long long int shared_exclusive_sum;
  __shared__ uint shared_task_idx;

  if (threadIdx.x == 0) {
    memcpy(sh_token, treeToken, numNodes * sizeof(unsigned char));
    memcpy(sh_left, treeLeft, numNodes * sizeof(uint));
    memcpy(sh_right, treeRight, numNodes * sizeof(uint));
    shared_task_idx = atomicAdd(taskCounter, 1);
  }
  __syncthreads();

  task_idx = shared_task_idx;
  threadInput_idx = (task_idx * blockDim.x + threadIdx.x) *
                    (unsigned long long int)BLOCK_SIZE;

  while (task_idx < numTasks) {
    threadInput = encodedString + (threadInput_idx / 32);
    uint currWord = threadInput[0];
    uint currentThreadInputPos = 0;
    uint posInTree = 0;
    uint codeCount = 0;
    while (currentThreadInputPos < BLOCK_SIZE &&
           threadInput_idx + currentThreadInputPos < encodedFileSize) {
      char readBit = (currWord >> (31 - (currentThreadInputPos % 32))) & 1;
      posInTree = readBit ? sh_right[posInTree] : sh_left[posInTree];
      if (sh_left[posInTree] == -1 && sh_right[posInTree] == -1) {
        codeCount++;
        posInTree = 0;
      }
      currentThreadInputPos++;
      if ((currentThreadInputPos & 31) == 0)
        currWord = threadInput[currentThreadInputPos / 32];
    }

    int warpLane = threadIdx.x & (WARP_SIZE - 1);
    int warpId = threadIdx.x / WARP_SIZE;
    uint tmp_count = codeCount;
    uint tmp_value = 0;
    for (uint delta = 1; delta < WARP_SIZE; delta <<= 1) {
      tmp_value = __shfl_up_sync(0xFFFFFFFF, tmp_count, delta, WARP_SIZE);
      if (warpLane >= delta)
        tmp_count += tmp_value;
    }

    __shared__ uint blockPrefixSum[THREAD_DIV_WARP];
    if (warpLane == WARP_SIZE - 1) {
      blockPrefixSum[warpId] = tmp_count;
    }
    __syncthreads();

    if (threadIdx.x < THREAD_DIV_WARP) {
      tmp_value = blockPrefixSum[threadIdx.x];
      const uint shfl_mask = ~((~0) << THREAD_DIV_WARP);
      for (uint delta = 1; delta < THREAD_DIV_WARP; delta <<= 1) {
        uint tmp = __shfl_up_sync(shfl_mask, tmp_value, delta, THREAD_DIV_WARP);
        if (threadIdx.x >= delta)
          tmp_value += tmp;
      }
      blockPrefixSum[threadIdx.x] = tmp_value;
    }
    __syncthreads();

    if (warpId == 0) {
      int posIdx = task_idx - warpLane;
      unsigned long long int local_inclusive_sum = 0;
      unsigned long long int exclusive_sum = 0;
      if (warpLane == 0) {
        local_inclusive_sum = blockPrefixSum[THREAD_DIV_WARP - 1];
        if (task_idx == 0) {
          charOffset[task_idx] = local_inclusive_sum | FLAG_P;
          shared_exclusive_sum = 0;
        } else {
          charOffset[task_idx] = local_inclusive_sum | FLAG_A;
        }
      }
      if (task_idx > 0) {
        while (1) {
          while (posIdx > 0 && (exclusive_sum == 0)) {
            // printf("stuck1");
            exclusive_sum = charOffset[posIdx - 1];
          }
          unsigned long long int tmp_sum = 0;
          for (uint delta = 1; delta < WARP_SIZE; delta <<= 1) {
            tmp_sum =
                __shfl_down_sync(0xFFFFFFFF, exclusive_sum, delta, WARP_SIZE);
            if (warpLane < WARP_SIZE - delta && ((exclusive_sum & FLAG_P) == 0))
              exclusive_sum += tmp_sum;
          }
          local_inclusive_sum += (exclusive_sum & (~FLAG_MASK));
          exclusive_sum = __shfl_sync(0xFFFFFFFF, exclusive_sum, 0);
          if (exclusive_sum & FLAG_P) {
            break;
          }
          // printf("stuck2");
          posIdx -= WARP_SIZE;
          exclusive_sum = 0;
        }
        if (warpLane == 0) {
          charOffset[task_idx] =
              ((local_inclusive_sum & (~FLAG_MASK)) | FLAG_P);
          shared_exclusive_sum =
              local_inclusive_sum - blockPrefixSum[THREAD_DIV_WARP - 1];
        }
      }
    }
    __syncthreads();
    unsigned long long int exclusive_sum = 0;
    uint tmp_count_plus = 0;
    if (warpLane == 0) {
      exclusive_sum = shared_exclusive_sum;
    }
    if (warpId > 0 && warpLane == 0) {
      tmp_count_plus = blockPrefixSum[warpId - 1];
    }
    exclusive_sum = __shfl_sync(0xFFFFFFFF, exclusive_sum, 0);
    tmp_count += __shfl_sync(0xFFFFFFFF, tmp_count_plus, 0);

    uint output_ptr = (exclusive_sum + tmp_count - codeCount) / 4;
    uint output_ptr_idx = (exclusive_sum + tmp_count - codeCount) & 3;

    currWord = threadInput[0];
    currentThreadInputPos = 0;
    posInTree = 0;

    uint output_word = 0;
    bool first_flag = 1;
    unsigned char decsymbol;

    while (currentThreadInputPos < BLOCK_SIZE &&
           threadInput_idx + currentThreadInputPos < encodedFileSize) {
      char readBit = (currWord >> (31 - (currentThreadInputPos % 32))) & 1;
      posInTree = readBit ? sh_right[posInTree] : sh_left[posInTree];
      if (sh_left[posInTree] == -1 && sh_right[posInTree] == -1) {
        decsymbol = sh_token[posInTree];
        UINT_OUT(output_word, decsymbol, output_ptr_idx);
        output_ptr_idx++;
        if ((output_ptr_idx & 3) == 0) {
          if (first_flag) {
            first_flag = 0;
            atomicOr(&decodedString[output_ptr++], output_word);
          } else {
            decodedString[output_ptr++] = output_word;
          }
          output_word = 0;
          output_ptr_idx = 0;
        }
        posInTree = 0;
      }
      currentThreadInputPos++;
      if ((currentThreadInputPos & 31) == 0)
        currWord = threadInput[currentThreadInputPos / 32];
    }
    if (output_ptr_idx != 0) {
      atomicOr(&decodedString[output_ptr], output_word);
    }
    if (threadIdx.x == 0) {
      shared_task_idx = atomicAdd(taskCounter, 1);
    }
    __syncthreads();

    task_idx = shared_task_idx;
    threadInput_idx = (task_idx * blockDim.x + threadIdx.x) *
                      (unsigned long long int)BLOCK_SIZE;
  }
}

void decode(FILE *inputFile, FILE *outputFile, HuffmanTree tree, uint blockSize,
            uint sizeOfFile, unsigned long long int encodedFileSize,
            uint numNodes) {
  uint *encodedString, *d_encodedString;
  uint *decodedString, *d_decodedString;
  unsigned long long int *d_charOffset;
  uint *d_taskCounter;
  unsigned char *d_treeToken;
  uint *d_treeLeft, *d_treeRight;

  uint numBlocksInEncodedString = ceil(encodedFileSize / (1. * blockSize));
  cudaMallocHost(&encodedString, sizeof(uint) * ((encodedFileSize + 31) / 32));
  cudaMallocHost(&decodedString, sizeof(uint) * ((sizeOfFile + 3) / 4));
  if (((encodedFileSize + 31) / 32) != fread(encodedString, sizeof(uint),
                                             (encodedFileSize + 31) / 32,
                                             inputFile))
    fatal("File read error 4");

  cudaMalloc((void **)&d_encodedString,
             sizeof(uint) * ((encodedFileSize + 31) / 32));
  cudaMalloc((void **)&d_decodedString, sizeof(uint) * ((sizeOfFile + 3) / 4));
  cudaMalloc((void **)&d_charOffset,
             sizeof(unsigned long long int) * (numBlocksInEncodedString + 1));
  cudaMalloc((void **)&d_taskCounter, sizeof(uint));
  cudaMalloc((void **)&d_treeToken, sizeof(unsigned char) * numNodes);
  cudaMalloc((void **)&d_treeLeft, sizeof(uint) * numNodes);
  cudaMalloc((void **)&d_treeRight, sizeof(uint) * numNodes);

  cudaMemset(d_taskCounter, 0, sizeof(uint));
  cudaMemset(d_charOffset, 0,
             sizeof(unsigned long long int) * (numBlocksInEncodedString + 1));
  cudaMemset(d_decodedString, 0, sizeof(uint) * ((sizeOfFile + 3) / 4));

  cudaMemcpy(d_encodedString, encodedString,
             sizeof(uint) * ((encodedFileSize + 31) / 32),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_treeToken, tree.tree.token, numNodes * sizeof(unsigned char),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_treeLeft, tree.tree.left, numNodes * sizeof(uint),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_treeRight, tree.tree.right, numNodes * sizeof(uint),
             cudaMemcpyHostToDevice);
  uint shm_needed = numNodes * 9;
  uint numTasks = ceil(encodedFileSize / (NUM_THREADS * BLOCK_SIZE * 1.0));
  GPU_TIMER_START(kernel)
  single_shot_decode<<<BLOCK_NUM, NUM_THREADS, shm_needed>>>(
      d_encodedString, encodedFileSize, d_treeToken, d_treeLeft, d_treeRight,
      d_charOffset, d_decodedString, d_taskCounter, numNodes, numTasks);
  GPU_TIMER_STOP(kernel)
  cudaMemcpy(decodedString, d_decodedString, sizeof(char) * sizeOfFile,
             cudaMemcpyDeviceToHost);
  fwrite(decodedString, sizeof(char), sizeOfFile, outputFile);
}