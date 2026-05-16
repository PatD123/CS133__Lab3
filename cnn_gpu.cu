// Header inclusions, if any...
#include "lib/cnn.cuh"
#include "cnn_gpu.cuh"

__global__ void cnn_gpu(
    float* input,
    float* weight,
    float* bias,
    float* output)
{

  const int lookOut = 16;

  // Shared memory
  __shared__ float shd[36][36];

  // Current output pixel
  int p_i = blockIdx.y * blockDim.y + threadIdx.y;
  int p_j = blockIdx.x * blockDim.x + threadIdx.x;
  int p_z = blockIdx.z * lookOut;
  int local_i = threadIdx.y * 2;
  int local_j = threadIdx.x * 2;

  int base_i = 2 * p_i;
  int base_j = 2 * p_j;
  int blockBase_i = blockIdx.y * blockDim.y * 2;
  int blockBase_j = blockIdx.x * blockDim.x * 2;

  float pooled[lookOut][4];
  memset(pooled, 0.0f, lookOut * 4 * sizeof(float));

  for(int kk = 0; kk < 256; kk++) {

    // We want to write to the thread block's shared memory
    for (int si = threadIdx.y; si < 36; si += blockDim.y) {
      for (int sj = threadIdx.x; sj < 36; sj += blockDim.x) {
        shd[si][sj] = input(kk, blockBase_i + si, blockBase_j + sj);
      }
    }

    __syncthreads();

    for(int k = 0; k<lookOut; k++)
    for(int i = 0; i < 2; i++) {
      for(int j = 0; j < 2; j++) {

        float sum = 0.0f;
        for(int ii = 0; ii < 5; ii++) {
          for(int jj = 0; jj < 5; jj++) {
            sum += shd[local_i + i + ii][local_j + j + jj] * weight(p_z+k, kk, ii, jj);
          }
        }

        pooled[k][i * 2 + j] += sum;
      }
    }

    __syncthreads();

    // for(int i = 0; i < 2; i++) {
    //   for(int j = 0; j < 2; j++) {

    //     int new_i = base_i + i;
    //     int new_j = base_j + j;

    //     float sum = 0.0f;
    //     for(int ii = 0; ii < 5; ii++) {
    //       for(int jj = 0; jj < 5; jj++) {
    //         sum += input(kk, new_i + ii, new_j + jj) * weight(p_z, kk, ii, jj);
    //       }
    //     }

    //     pooled[i * 2 + j] += sum;
    //   }
    // }
  }

  for(int i = 0; i<lookOut; i++){
    pooled[i][0] = fmaxf(0.0f, pooled[i][0] + bias[p_z+i]);
    pooled[i][1] = fmaxf(0.0f, pooled[i][1] + bias[p_z+i]);
    pooled[i][2] = fmaxf(0.0f, pooled[i][2] + bias[p_z+i]);
    pooled[i][3] = fmaxf(0.0f, pooled[i][3] + bias[p_z+i]);
    output(p_z+i, p_i, p_j) = fmaxf(fmaxf(fmaxf(pooled[i][0], pooled[i][1]), pooled[i][2]), pooled[i][3]);
  }

    
}
