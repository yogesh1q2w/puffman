#include "../include/compress_utils.cuh"

//---------------------------------HISTOGRAM-------------------------------------

inline __device__ void addByte(uint *s_WarpHist, unsigned char data) {
  atomicAdd(s_WarpHist + data, 1);
}

inline __device__ void addWord(uint *s_WarpHist, uint data) {
  addByte(s_WarpHist, (data >> 0) & 0xFFU);
  addByte(s_WarpHist, (data >> 8) & 0xFFU);
  addByte(s_WarpHist, (data >> 16) & 0xFFU);
  addByte(s_WarpHist, (data >> 24) & 0xFFU);
}

__global__ void cu_histgram(uint *d_PartialHistograms, uint *d_Data,
                            uint dataCount, uint byteCount) {

  __shared__ uint s_Hist[S_HIST_SIZE];
  uint *s_WarpHist = s_Hist + (threadIdx.x >> 5) * HIST_SIZE;
  uint warpLane = threadIdx.x & 31;

  for (uint i = warpLane; i < HIST_SIZE; i += WARP_SIZE) {
    s_WarpHist[i] = 0;
  }
  __syncthreads();

  uint pos = 0;
  for (pos = (blockIdx.x * blockDim.x) + threadIdx.x; pos < dataCount - 1;
       pos += (blockDim.x * gridDim.x)) {
    uint data = d_Data[pos];
    addWord(s_WarpHist, data);
  }

  if (pos == dataCount - 1) {
    uint data = d_Data[pos];
    switch (byteCount & 3) {
    case 1:
      addByte(s_WarpHist, (data >> 0) & 0xFFU);
      break;
    case 2:
      addByte(s_WarpHist, (data >> 0) & 0xFFU);
      addByte(s_WarpHist, (data >> 8) & 0xFFU);
      break;
    case 3:
      addByte(s_WarpHist, (data >> 0) & 0xFFU);
      addByte(s_WarpHist, (data >> 8) & 0xFFU);
      addByte(s_WarpHist, (data >> 16) & 0xFFU);
      break;
    default:
      addByte(s_WarpHist, (data >> 0) & 0xFFU);
      addByte(s_WarpHist, (data >> 8) & 0xFFU);
      addByte(s_WarpHist, (data >> 16) & 0xFFU);
      addByte(s_WarpHist, (data >> 24) & 0xFFU);
    }
  }

  __syncthreads();

  //
  for (uint bin = threadIdx.x; bin < HIST_SIZE; bin += HIST_THREADS) {
    uint sum = 0;
    for (uint i = 0; i < WARP_COUNT; i++) {
      sum += s_Hist[bin + i * HIST_SIZE];
    }
    d_PartialHistograms[blockIdx.x * HIST_SIZE + bin] = sum;
  }
}

__global__ void mergeHistogram(uint *d_Histogram, uint *d_PartialHistograms) {

  uint val = d_PartialHistograms[blockIdx.x * HIST_SIZE + threadIdx.x];
  atomicAdd(d_Histogram + threadIdx.x, val);
}

//-----------------------------------------------------------------------------------------------
__host__ __device__ inline unsigned char getcharAt(uint *dfileContent,
                                                   uint pos) {
  return (dfileContent[pos >> 2] >> ((pos & 3U) << 3)) & 0xFFU;
}

__global__ void encode(uint fileSize, uint *dfileContent,
                       unsigned long long int *dbitOffsets,
                       unsigned long long int *d_boundary_index,
                       uint *d_compressedFile, uint *d_dictionary_code,
                       unsigned char *d_dictionary_codelens, uint *counter,
                       uint numTasks) {
  uint task_idx = 0;
  uint threadInput_idx = 0;
  uint *threadInput;
  unsigned long long int *threadBoundaryIndex;
  __shared__ struct codedict sh_dictionary;
  __shared__ unsigned int shared_task_idx;

  sh_dictionary.code[threadIdx.x] = d_dictionary_code[threadIdx.x];
  sh_dictionary.codeSize[threadIdx.x] = d_dictionary_codelens[threadIdx.x];

  if (threadIdx.x == 0) {
    shared_task_idx = atomicAdd(counter, 1);
  }
  __syncthreads();

  task_idx = shared_task_idx;
  threadInput_idx = (task_idx * blockDim.x + threadIdx.x) * PER_THREAD_PROC;

  while (task_idx < numTasks && threadInput_idx < fileSize) {
    threadInput = dfileContent + (threadInput_idx / 4);
    threadBoundaryIndex = d_boundary_index + threadInput_idx;
    uint inputPosInThreadTask = 0;
    uint outputPos = (d_boundary_index[threadInput_idx] == 0)
                         ? dbitOffsets[threadInput_idx] / 32
                         : d_boundary_index[threadInput_idx] / 32;
    uint startPosInOutputWord = (d_boundary_index[threadInput_idx] == 0)
                                    ? dbitOffsets[threadInput_idx] % 32
                                    : d_boundary_index[threadInput_idx] % 32;
    uint outputWord = 0;
    uint input = 0;
    uint pendingBitsFromPreviousCode;
    uint remain_code = 0;
    while (inputPosInThreadTask < PER_THREAD_PROC &&
           threadInput_idx + inputPosInThreadTask < fileSize) {
      if ((inputPosInThreadTask & 3) == 0)
        input = threadInput[inputPosInThreadTask / 4];
      uint code = sh_dictionary.code[GET_CHAR(input, inputPosInThreadTask & 3)];
      unsigned char code_length =
          sh_dictionary.codeSize[GET_CHAR(input, inputPosInThreadTask & 3)];
      code >>= (32 - code_length);
      unsigned long long int boundary_pos =
          threadBoundaryIndex[inputPosInThreadTask];
      if (boundary_pos != 0) {
        code >>=
            (code_length -
             (BLOCK_SIZE * ((uint)ceil(boundary_pos / (1. * BLOCK_SIZE)))) +
             boundary_pos);
        code_length =
            BLOCK_SIZE * ((uint)ceil(boundary_pos / (1. * BLOCK_SIZE))) -
            boundary_pos;
        threadBoundaryIndex[inputPosInThreadTask] = 0;
        inputPosInThreadTask--;
      }
      if (32 - startPosInOutputWord >= code_length) {
        code <<= (32 - startPosInOutputWord - code_length);
        remain_code = 0;
        pendingBitsFromPreviousCode = 0;
        startPosInOutputWord += code_length;
      } else {
        remain_code = code << (32 - code_length + 32 - startPosInOutputWord);
        pendingBitsFromPreviousCode = (code_length - 32 + startPosInOutputWord);
        code >>= pendingBitsFromPreviousCode;
      }
      outputWord |= code;
      if (pendingBitsFromPreviousCode) {
        atomicOr(&d_compressedFile[outputPos++], outputWord);
        outputWord = remain_code;
        startPosInOutputWord = pendingBitsFromPreviousCode;
      }
      inputPosInThreadTask++;
    }
    atomicOr(&d_compressedFile[outputPos++], outputWord);
    if (threadIdx.x == 0) {
      shared_task_idx = atomicAdd(counter, 1);
    }
    __syncthreads();

    task_idx = shared_task_idx;
    threadInput_idx = (task_idx * blockDim.x + threadIdx.x) * PER_THREAD_PROC;
  }
}

void readFile(uint *&fileContent, uint *&dfileContent, FILE *inputFile,
              unsigned long long int &fileSize, uint &intFileSize) {
  fseek(inputFile, 0L, SEEK_END);
  fileSize = ftell(inputFile);
  fseek(inputFile, 0L, SEEK_SET);
  intFileSize = (fileSize + 3) >> 2;
  cudaMallocHost(&fileContent, sizeof(uint) * intFileSize);
  CUERROR
  cudaMalloc((void **)&dfileContent, sizeof(uint) * intFileSize);
  CUERROR
  if (fileSize !=
      fread(fileContent, sizeof(unsigned char), fileSize, inputFile))
    fatal("File read error 1");
  GPU_TIMER_START(HtD1)
  cudaMemcpy(dfileContent, fileContent, sizeof(uint) * intFileSize,
             cudaMemcpyHostToDevice);
  GPU_TIMER_STOP(HtD1)
  CUERROR
}

void getFrequencies(uint *dfileContent, unsigned long long int &fileSize,
                    uint *&frequency, uint &intFileSize) {
  cudaMallocHost(&frequency, 256 * sizeof(uint));
  uint *dfrequency;
  cudaMalloc((void **)&dfrequency, 256 * sizeof(uint));
  cudaMemset(dfrequency, 0, 256 * sizeof(uint));
  uint *d_PartialHistograms;
  cudaMalloc((void **)&d_PartialHistograms,
             sizeof(uint) * HIST_BLOCK * HIST_SIZE);
  GPU_TIMER_START(hist)
  cu_histgram<<<HIST_BLOCK, HIST_THREADS>>>(d_PartialHistograms, dfileContent,
                                            intFileSize, fileSize);
  mergeHistogram<<<HIST_BLOCK, HIST_SIZE>>>(dfrequency, d_PartialHistograms);
  GPU_TIMER_STOP(hist)
  cudaMemcpy(frequency, dfrequency, 256 * sizeof(uint), cudaMemcpyDeviceToHost);
  cudaFree(d_PartialHistograms);
  cudaFree(dfrequency);
  CUERROR
}

void getOffsetArray(unsigned long long int *bitOffsets,
                    unsigned long long int *boundary_index,
                    unsigned long long int &encodedFileSize,
                    unsigned long long int &fileSize, codedict &dictionary,
                    uint *fileContent) {
  bitOffsets[0] = 0;
  unsigned long long int searchValue = BLOCK_SIZE;
  unsigned long long int i;
  for (i = 1; i < fileSize; i++) {
    bitOffsets[i] =
        bitOffsets[i - 1] + dictionary.codeSize[getcharAt(fileContent, i - 1)];

    if (bitOffsets[i] > searchValue) {
      boundary_index[i - 1] = bitOffsets[i - 1];
      bitOffsets[i - 1] = searchValue;
      searchValue += BLOCK_SIZE;
      i--;
    } else if (bitOffsets[i] == searchValue) {
      searchValue += BLOCK_SIZE;
    }
  }

  if (bitOffsets[i - 1] + dictionary.codeSize[getcharAt(fileContent, i - 1)] >
      searchValue) {
    boundary_index[i - 1] = bitOffsets[i - 1];
    bitOffsets[i - 1] = searchValue;
  }
  encodedFileSize = bitOffsets[fileSize - 1] +
                    dictionary.codeSize[getcharAt(fileContent, fileSize - 1)];
}

void writeFileContents(FILE *outputFile, unsigned long long int &fileSize,
                       uint *fileContent, uint *dfileContent,
                       codedict &dictionary) {

  uint *compressedFile, *d_compressedFile;
  unsigned long long int *bitOffsets, *dbitOffsets;
  unsigned long long int *boundary_index, *d_boundary_index;

  cudaMallocHost(&bitOffsets, fileSize * sizeof(unsigned long long int));
  CUERROR
  cudaMallocHost(&boundary_index, fileSize * sizeof(unsigned long long int));
  CUERROR
  cudaMemset(boundary_index, 0, fileSize * sizeof(unsigned long long int));
  CUERROR
  cudaMalloc((void **)&dbitOffsets, fileSize * sizeof(unsigned long long int));
  CUERROR
  cudaMalloc((void **)&d_boundary_index,
             fileSize * sizeof(unsigned long long int));
  CUERROR

  unsigned long long int encodedFileSize;
  CPU_TIMER_START(offset)
  getOffsetArray(bitOffsets, boundary_index, encodedFileSize, fileSize,
                 dictionary, fileContent);
  CPU_TIMER_STOP(offset)

  cudaMemcpy(dbitOffsets, bitOffsets, fileSize * sizeof(unsigned long long int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_boundary_index, boundary_index,
             fileSize * sizeof(unsigned long long int), cudaMemcpyHostToDevice);
  CUERROR

  uint writeSize = (encodedFileSize + 31) >> 5;

  cudaMallocHost(&compressedFile, writeSize * sizeof(uint));
  CUERROR
  cudaMalloc((void **)&d_compressedFile, writeSize * sizeof(uint));
  cudaMemset(d_compressedFile, 0, writeSize * sizeof(uint));
  CUERROR

  uint *d_dictionary_code;
  unsigned char *d_dictionary_codelens;
  cudaMalloc(&d_dictionary_code, 256 * sizeof(uint));
  cudaMalloc(&d_dictionary_codelens, 256 * sizeof(unsigned char));
  cudaMemcpy(d_dictionary_code, dictionary.code, 256 * sizeof(uint),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_dictionary_codelens, dictionary.codeSize,
             256 * sizeof(unsigned char), cudaMemcpyHostToDevice);

  uint *counter;
  cudaMalloc(&counter, sizeof(uint));
  cudaMemset(counter, 0, sizeof(uint));

  uint numTasks = ceil(fileSize / (256. * PER_THREAD_PROC));

  GPU_TIMER_START(kernel)
  encode<<<BLOCK_NUM, 256>>>(
      fileSize, dfileContent, dbitOffsets, d_boundary_index, d_compressedFile,
      d_dictionary_code, d_dictionary_codelens, counter, numTasks);
  GPU_TIMER_STOP(kernel)
  CUERROR
  cudaMemcpy(compressedFile, d_compressedFile, writeSize * sizeof(uint),
             cudaMemcpyDeviceToHost);
  CUERROR
  fwrite(&encodedFileSize, sizeof(unsigned long long int), 1, outputFile);
  fwrite(compressedFile, sizeof(uint), writeSize, outputFile);
  cudaFreeHost(bitOffsets);
  cudaFreeHost(boundary_index);
  cudaFreeHost(compressedFile);
  cudaFreeHost(fileContent);
  cudaFree(dbitOffsets);
  cudaFree(d_boundary_index);
  cudaFree(d_compressedFile);
  cudaFree(dfileContent);
  cudaFree(counter);
  cudaFree(d_dictionary_code);
  cudaFree(d_dictionary_codelens);
  CUERROR
}
