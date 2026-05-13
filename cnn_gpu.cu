// Header inclusions, if any...
#include "lib/cnn.cuh"
#include "cnn_gpu.cuh"

__global__ void cnn_gpu(
    float* input,
    float* weight,
    float* bias,
    float* output)
{

  // Current output pixel
  int p_i = blockIdx.y * blockDim.y + threadIdx.y;
  int p_j = blockIdx.x * blockDim.x + threadIdx.x;
  int p_z = blockIdx.z;

  int base_i = 2 * p_i;
  int base_j = 2 * p_j;

  float res = 0.0f;

  for(int i = 0; i < 2; i++) {
    for(int j = 0; j < 2; j++) {

      int new_i = base_i + i;
      int new_j = base_j + j;

      float sum = 0.0f;
      for(int kk = 0; kk < 256; kk++) {
        for(int ii = 0; ii < 5; ii++) {
          for(int jj = 0; jj < 5; jj++) {
            sum += input(kk, new_i + ii, new_j + jj) * weight(p_z, kk, ii, jj);
          }
        }
      }
      res = fmaxf(res, sum + bias[p_z]);
    }
  }
  output(p_z, p_i, p_j) = res;
}
