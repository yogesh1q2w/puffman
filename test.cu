#include <bits/stdc++.h>
#include <cuda.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
using namespace std;

int main(int argc, char **argv) {
  uint x[] = {0, 1, 4, 6, 7, 8};
  uint *d_x;
  cudaMalloc(&d_x, 6 * 4);
  cudaMemcpy(d_x, x, 6 * 4, cudaMemcpyHostToDevice);
  uint modifyIndex =
      (thrust::lower_bound(thrust::device, d_x, d_x + 6, 4.1) - d_x)-1;
  cout << modifyIndex << endl;

  return 0;
}