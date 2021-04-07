#include "../include/compress_utils.cuh"
#include <vector>
using namespace std;
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

__global__ void encode(uint fileSize, uint *dfileContent, uint *dblockCharPos,
                       uint *d_compressedFile, uint *d_dictionary_code,
                       unsigned char *d_dictionary_codelens, uint *counter,
                       uint numBlocks, uint numTasks) {
  uint task_idx, block_idx;
  uint inputfile_idx;
  uint *threadInput;
  __shared__ struct codedict sh_dictionary;
  __shared__ uint shared_task_idx;

  sh_dictionary.code[threadIdx.x] = d_dictionary_code[threadIdx.x];
  sh_dictionary.codeSize[threadIdx.x] = d_dictionary_codelens[threadIdx.x];
  __syncthreads();
  if (threadIdx.x == 0) {
    shared_task_idx = atomicAdd(counter, 1);
  }
  

  task_idx = shared_task_idx;
  block_idx = task_idx;

  while (block_idx < numBlocks) {
    inputfile_idx = dblockCharPos[block_idx];
    threadInput = dfileContent + (inputfile_idx / 4);
    uint input = threadInput[0];
    uint bits_written = 0;
    uint window = 0;
    uint window_pos = 0;
    while (bits_written < BLOCK_SIZE && inputfile_idx < fileSize) {
      uint code = sh_dictionary.code[GET_CHAR(input, inputfile_idx & 3)];
      unsigned char code_len =
          sh_dictionary.codeSize[GET_CHAR(input, inputfile_idx & 3)];
      inputfile_idx++;
      if ((inputfile_idx & 3) == 0 && inputfile_idx < fileSize)
        input = threadInput[inputfile_idx / 4];
      while (window_pos + code_len < INT_BITS && inputfile_idx < fileSize) {
        window <<= code_len;
        window |= code;
        window_pos += code_len;

        if (inputfile_idx < fileSize) {
          code = sh_dictionary.code[GET_CHAR(input, inputfile_idx & 3)];
          code_len = sh_dictionary.codeSize[GET_CHAR(input, inputfile_idx & 3)];
          inputfile_idx++;
          if ((inputfile_idx & 3) == 0 && inputfile_idx < fileSize)
            input = threadInput[inputfile_idx / 4];
        }
      }
      const int diff = window_pos + code_len - INT_BITS;
      uint changeIndex = (block_idx * BLOCK_SIZE + bits_written) / 32;
      window <<= (code_len - ((diff >= 0) * diff));
      window |= (code >> ((diff >= 0) * diff));
      window <<= (-diff) * (diff < 0);
      d_compressedFile[changeIndex] |= window;
      if (diff >= 0) {
        window = code & ~(~0 << diff);
        window_pos = diff;
      } else {
        window_pos = 0;
      }
      bits_written += 32;
    }
    if (threadIdx.x == 0) {
      shared_task_idx = atomicAdd(counter, 1);
    }
    __syncthreads();

    task_idx = shared_task_idx;
    block_idx = task_idx;
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

void getOffsetArray(vector<unsigned int> &blockCharPos,
                    unsigned long long int &encodedFileSize,
                    unsigned long long int &fileSize, codedict &dictionary,
                    uint *fileContent) {
  blockCharPos.push_back(0);
  unsigned long long int searchValue = BLOCK_SIZE;
  unsigned long long int i;
  uint offset_sum = 0;
  for (i = 1; i <= fileSize; i++) {
    offset_sum += dictionary.codeSize[getcharAt(fileContent, i - 1)];

    if (offset_sum > searchValue) {
      blockCharPos.push_back(i - 1);
      offset_sum = searchValue;
      searchValue += BLOCK_SIZE;
      i--;
    } else if (offset_sum == searchValue) {
      blockCharPos.push_back(i);
      searchValue += BLOCK_SIZE;
    }
  }
  encodedFileSize = offset_sum;
}

void writeFileContents(FILE *outputFile, unsigned long long int &fileSize,
                       uint *fileContent, uint *dfileContent,
                       codedict &dictionary) {

  uint *compressedFile, *d_compressedFile;
  vector<unsigned int> blockCharPos;
  uint *dblockCharPos;

  unsigned long long int encodedFileSize;
  CPU_TIMER_START(offset)
  getOffsetArray(blockCharPos, encodedFileSize, fileSize, dictionary,
                 fileContent);
  CPU_TIMER_STOP(offset)

  uint numBlocks = blockCharPos.size();
  cudaMalloc((void **)&dblockCharPos, numBlocks * sizeof(uint));
  cudaMemcpy(dblockCharPos, &blockCharPos[0], numBlocks * sizeof(uint),
             cudaMemcpyHostToDevice);
  CUERROR
  // uint block_char_pos[numBlocks];
  // cudaMemcpy(block_char_pos, dblockCharPos, numBlocks*sizeof(uint),
  // cudaMemcpyDeviceToHost); for (int i = 0; i < numBlocks; i++)
  //   printf("%d, ", block_char_pos[i]);
  // printf("\n");

  uint writeSize = (encodedFileSize + 31) >> 5;
  printf("Writesie allowed = %d\n", writeSize);

  cudaMallocHost(&compressedFile, writeSize * sizeof(uint));
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
  uint numTasks = ceil(numBlocks / (NUM_THREADS * 1.));

  GPU_TIMER_START(kernel)
  encode<<<BLOCK_NUM, NUM_THREADS>>>(
      fileSize, dfileContent, dblockCharPos, d_compressedFile,
      d_dictionary_code, d_dictionary_codelens, counter, numBlocks, numTasks);
  GPU_TIMER_STOP(kernel)
  CUERROR
  cudaMemcpy(compressedFile, d_compressedFile, writeSize * sizeof(uint),
             cudaMemcpyDeviceToHost);
  CUERROR
  fwrite(&encodedFileSize, sizeof(unsigned long long int), 1, outputFile);
  fwrite(compressedFile, sizeof(uint), writeSize, outputFile);
  cudaFreeHost(compressedFile);
  cudaFreeHost(fileContent);
  cudaFree(dblockCharPos);
  cudaFree(d_compressedFile);
  cudaFree(dfileContent);
  cudaFree(counter);
  cudaFree(d_dictionary_code);
  cudaFree(d_dictionary_codelens);
  CUERROR
}
