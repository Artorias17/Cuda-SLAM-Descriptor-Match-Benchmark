#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <chrono>
#include <cstdlib>
#include <climits>
#include <cuda_runtime.h>

// Simple CUDA error check
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    }

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

// Load binary descriptors (.bin) into a flat vector<uint8_t>
std::vector<uint8_t> loadBin(const std::string& path, size_t expectedBytes)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Failed to open " << path << std::endl;
        std::exit(1);
    }
    std::vector<uint8_t> data(expectedBytes);
    f.read(reinterpret_cast<char*>(data.data()), expectedBytes);
    if (!f) {
        std::cerr << "Failed to read " << path << std::endl;
        std::exit(1);
    }
    return data;
}

int main()
{
    // 1) Read meta.txt to get N1, N2, dim
    int N1, N2, dim;
    {
        std::ifstream meta("descriptors/meta.txt");
        if (!meta) {
            std::cerr << "Cannot open descriptors/meta.txt\n";
            return 1;
        }
        meta >> N1 >> N2 >> dim;
    }

    std::cout << "Meta: N1=" << N1 << " N2=" << N2 << " dim=" << dim << std::endl;

    size_t bytes1 = static_cast<size_t>(N1) * dim;
    size_t bytes2 = static_cast<size_t>(N2) * dim;

    // 2) Load descriptor binaries
    auto h_des1 = loadBin("descriptors/des1.bin", bytes1);
    auto h_des2 = loadBin("descriptors/des2.bin", bytes2);

    // 3) Allocate device memory
    uint8_t *d_des1 = nullptr, *d_des2 = nullptr;
    int *d_bestIdx = nullptr, *d_bestDist = nullptr;

    CUDA_CHECK(cudaMalloc(&d_des1, bytes1));
    CUDA_CHECK(cudaMalloc(&d_des2, bytes2));
    CUDA_CHECK(cudaMalloc(&d_bestIdx,  N1 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bestDist, N1 * sizeof(int)));

    // 4) Copy descriptors to GPU
    CUDA_CHECK(cudaMemcpy(d_des1, h_des1.data(), bytes1, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_des2, h_des2.data(), bytes2, cudaMemcpyHostToDevice));

    // 5) Launch real matching kernel
    const int BLOCK_SIZE = 256;  // threads per block
    dim3 grid(N1);
    dim3 block(BLOCK_SIZE);
    size_t sharedMemBytes = 2 * BLOCK_SIZE * sizeof(int); // sDist + sIdx

    auto t0 = std::chrono::high_resolution_clock::now();
    matchKernelHamming<<<grid, block, sharedMemBytes>>>(
        d_des1, d_des2, N1, N2, dim, d_bestIdx, d_bestDist
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = t1 - t0;
    std::cout << "GPU Hamming matching time: " << elapsed.count() << " ms\n";

    // 6) Copy results back
    std::vector<int> h_bestIdx(N1), h_bestDist(N1);
    CUDA_CHECK(cudaMemcpy(h_bestIdx.data(),  d_bestIdx,  N1 * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bestDist.data(), d_bestDist, N1 * sizeof(int), cudaMemcpyDeviceToHost));

    std::cout << "First 5 matches (GPU):\n";
    for (int i = 0; i < 5 && i < N1; ++i) {
        std::cout << "i=" << i
                  << " bestIdx=" << h_bestIdx[i]
                  << " bestDist=" << h_bestDist[i] << "\n";
    }

    // 7) Cleanup
    cudaFree(d_des1);
    cudaFree(d_des2);
    cudaFree(d_bestIdx);
    cudaFree(d_bestDist);

    return 0;
}
