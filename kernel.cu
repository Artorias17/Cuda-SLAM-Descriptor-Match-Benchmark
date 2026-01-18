#include <climits>
#include <cuda_runtime.h>
#include <cstdint>

// Real matching kernel: one block per descriptor in des1
// Each thread in a block scans part of des2 and keeps its local best match.
// Then we do a block-level reduction in shared memory to get the global best.
__global__ void matchKernelHamming(const uint8_t* __restrict__ des1,
                                   const uint8_t* __restrict__ des2,
                                   int N, int M, int dim,
                                   int* bestIdx,
                                   int* bestDist)
{
    int i = blockIdx.x;   // descriptor index in des1
    int tid = threadIdx.x;

    if (i >= N) return;

    // Pointer to descriptor i in des1
    const uint8_t* d1 = des1 + i * dim;

    // Each thread keeps local best (distance, index)
    int localBestDist = INT_MAX;
    int localBestIdx  = -1;

    // Iterate over descriptors in des2 with stride = blockDim.x
    for (int j = tid; j < M; j += blockDim.x) {
        const uint8_t* d2 = des2 + j * dim;

        int dist = 0;
        // Hamming distance: XOR bytes then popcount
        #pragma unroll
        for (int k = 0; k < dim; ++k) {
            unsigned int v = static_cast<unsigned int>(d1[k] ^ d2[k]);
            dist += __popc(v);  // population count of bits set in v
        }

        if (dist < localBestDist) {
            localBestDist = dist;
            localBestIdx  = j;
        }
    }

    // Shared memory for reduction
    extern __shared__ int sdata[];
    int* sDist = sdata;                    // [0 .. blockDim.x-1]
    int* sIdx  = sdata + blockDim.x;       // [blockDim.x .. 2*blockDim.x-1]

    sDist[tid] = localBestDist;
    sIdx[tid]  = localBestIdx;
    __syncthreads();

    // Parallel reduction to find min distance within block
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            int otherDist = sDist[tid + offset];
            int otherIdx  = sIdx[tid + offset];
            if (otherDist < sDist[tid]) {
                sDist[tid] = otherDist;
                sIdx[tid]  = otherIdx;
            }
        }
        __syncthreads();
    }

    // Thread 0 writes the final best match for descriptor i
    if (tid == 0) {
        bestDist[i] = sDist[0];
        bestIdx[i]  = sIdx[0];
    }
}
