#include <bits/stdc++.h>
#include <cuda.h>
using namespace std;

__global__ void prefix_sum(volatile int *ds, int *da, int sz) {
  if (threadIdx.x >= sz - 1)
    return;
  int temp_val = 0;
  while (temp_val == 0) {
    temp_val = ds[threadIdx.x];
  }
  ds[threadIdx.x + 1] = ds[threadIdx.x] + da[threadIdx.x];
}

int main() {
  int a[] = {1, 2, 3, 4, 5, 6, 7, 8};
  int sz = 8;
  int *d_a;
  cudaMalloc(&d_a, sz * sizeof(int));
  cudaMemcpy(d_a, a, sz * sizeof(int), cudaMemcpyHostToDevice);
  int *d_s;
  cudaMalloc(&d_s, sz * sizeof(int));
  cudaMemset(d_s, 0, sz * sizeof(int));
  cudaMemcpy(d_s, a, sizeof(int), cudaMemcpyHostToDevice);
  int s[sz];
  prefix_sum<<<1, 256>>>(d_s, d_a, sz);
  cudaMemcpy(s, d_s, sz * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < sz; i++)
    cout << s[i] << ",";
  cout << endl;
  return 0;
}