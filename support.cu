#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <string>
#include <tuple>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <cuda_runtime.h>
#include "support.h"
#include "kernel.cu"

namespace fs = std::filesystem;

std::tuple<int, int, int> loadMetaInfo(const std::string& metaPath) {
    std::ifstream meta(metaPath);
    if (!meta) {
        std::cerr << "Cannot open " << metaPath << std::endl;
        std::exit(1);
    }
    
    int numImages, numFeatures, dim;
    meta >> numImages >> numFeatures >> dim;
    
    return std::make_tuple(numImages, numFeatures, dim);
}

std::vector<uint8_t> loadDescriptorBinary(const std::string& path, size_t expectedBytes) {
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

std::pair<std::vector<std::vector<uint8_t>>, std::vector<int>> loadAllDescriptors(const std::string& descriptorDir) {
    auto [numImages, numFeatures, dim] = loadMetaInfo(descriptorDir + "/meta.txt");
    
    // Find all des*.bin files using filesystem (like Python glob)
    std::vector<fs::path> descriptorFiles;
    for (const auto& entry : fs::directory_iterator(descriptorDir)) {
        if (entry.path().extension() == ".bin" && 
            entry.path().stem().string().find("des") == 0) {
            descriptorFiles.push_back(entry.path());
        }
    }
    
    // Sort by numeric suffix (extract number from "des{N}.bin")
    std::sort(descriptorFiles.begin(), descriptorFiles.end(),
        [](const fs::path& a, const fs::path& b) {
            int numA = std::stoi(a.stem().string().substr(3));
            int numB = std::stoi(b.stem().string().substr(3));
            return numA < numB;
        });
    
    if (descriptorFiles.empty()) {
        std::cerr << "No descriptor files found in " << descriptorDir << std::endl;
        std::exit(1);
    }
    
    std::vector<std::vector<uint8_t>> descriptors;
    std::vector<int> fileNumbers;
    size_t bytesPerDescriptor = static_cast<size_t>(numFeatures) * dim;
    
    std::cout << "Loading " << descriptorFiles.size() << " descriptor sets..." << std::endl;
    
    for (const auto& descPath : descriptorFiles) {
        // Extract file number from "des{N}.bin"
        int fileNum = std::stoi(descPath.stem().string().substr(3));
        fileNumbers.push_back(fileNum);
        
        auto des = loadDescriptorBinary(descPath.string(), bytesPerDescriptor);
        descriptors.push_back(des);
        std::cout << "  Loaded " << descPath.filename().string() << " (frame " << fileNum << "): " 
                  << numFeatures << " x " << dim << " bytes" << std::endl;
    }
    
    return {descriptors, fileNumbers};
}

void matchDescriptorsGPU(const uint8_t* hDes1, const uint8_t* hDes2,
                         int N, int dim,
                         std::vector<int>& hBestIdx, 
                         std::vector<int>& hBestDist,
                         double& elapsedMs) {
    size_t bytes = static_cast<size_t>(N) * dim;
    
    // Allocate device memory
    uint8_t *dDes1 = nullptr, *dDes2 = nullptr;
    int *dBestIdx = nullptr, *dBestDist = nullptr;

    CUDA_CHECK(cudaMalloc(&dDes1, bytes));
    CUDA_CHECK(cudaMalloc(&dDes2, bytes));
    CUDA_CHECK(cudaMalloc(&dBestIdx,  N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dBestDist, N * sizeof(int)));

    // Copy descriptors to GPU (H2D)
    CUDA_CHECK(cudaMemcpy(dDes1, hDes1, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dDes2, hDes2, bytes, cudaMemcpyHostToDevice));

    // Launch matching kernel - KERNEL TIME ONLY using CUDA events
    const int BLOCK_SIZE = 256;
    dim3 grid(N);
    dim3 block(BLOCK_SIZE);
    size_t sharedMemBytes = 2 * BLOCK_SIZE * sizeof(int);

    cudaEvent_t kernelStart, kernelEnd;
    CUDA_CHECK(cudaEventCreate(&kernelStart));
    CUDA_CHECK(cudaEventCreate(&kernelEnd));
    
    CUDA_CHECK(cudaEventRecord(kernelStart));
    matchKernelHamming<<<grid, block, sharedMemBytes>>>(
        dDes1, dDes2, N, N, dim, dBestIdx, dBestDist
    );
    CUDA_CHECK(cudaEventRecord(kernelEnd));
    CUDA_CHECK(cudaEventSynchronize(kernelEnd));

    float kernelTimeMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernelTimeMs, kernelStart, kernelEnd));
    elapsedMs = static_cast<double>(kernelTimeMs);
    
    CUDA_CHECK(cudaEventDestroy(kernelStart));
    CUDA_CHECK(cudaEventDestroy(kernelEnd));

    // Copy results back (D2H)
    hBestIdx.resize(N);
    hBestDist.resize(N);
    CUDA_CHECK(cudaMemcpy(hBestIdx.data(),  dBestIdx,  N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hBestDist.data(), dBestDist, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Cleanup
    cudaFree(dDes1);
    cudaFree(dDes2);
    cudaFree(dBestIdx);
    cudaFree(dBestDist);
}

void matchDescriptorsGPUCrossCheck(const uint8_t* hDes1, const uint8_t* hDes2,
                                   int N, int dim,
                                   std::vector<int>& hBestIdx,
                                   std::vector<int>& hBestDist,
                                   double& elapsedMs) {
    size_t bytes = static_cast<size_t>(N) * dim;
    
    // Allocate device memory
    uint8_t *dDes1 = nullptr, *dDes2 = nullptr;
    int *dBestIdx1 = nullptr, *dBestDist1 = nullptr;
    int *dBestIdx2 = nullptr, *dBestDist2 = nullptr;

    CUDA_CHECK(cudaMalloc(&dDes1, bytes));
    CUDA_CHECK(cudaMalloc(&dDes2, bytes));
    CUDA_CHECK(cudaMalloc(&dBestIdx1, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dBestDist1, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dBestIdx2, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dBestDist2, N * sizeof(int)));

    // Copy descriptors to GPU (H2D)
    CUDA_CHECK(cudaMemcpy(dDes1, hDes1, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dDes2, hDes2, bytes, cudaMemcpyHostToDevice));

    // Launch matching kernels with CUDA events for accurate GPU timing
    const int BLOCK_SIZE = 256;
    dim3 grid(N);
    dim3 block(BLOCK_SIZE);
    size_t sharedMemBytes = 2 * BLOCK_SIZE * sizeof(int);

    cudaEvent_t kernelStart, kernelEnd;
    CUDA_CHECK(cudaEventCreate(&kernelStart));
    CUDA_CHECK(cudaEventCreate(&kernelEnd));
    
    CUDA_CHECK(cudaEventRecord(kernelStart));
    
    // Forward: des1 -> des2
    matchKernelHamming<<<grid, block, sharedMemBytes>>>(
        dDes1, dDes2, N, N, dim, dBestIdx1, dBestDist1
    );
    
    // Backward: des2 -> des1
    matchKernelHamming<<<grid, block, sharedMemBytes>>>(
        dDes2, dDes1, N, N, dim, dBestIdx2, dBestDist2
    );
    
    CUDA_CHECK(cudaEventRecord(kernelEnd));
    CUDA_CHECK(cudaEventSynchronize(kernelEnd));

    float kernelTimeMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernelTimeMs, kernelStart, kernelEnd));
    elapsedMs = static_cast<double>(kernelTimeMs);
    
    CUDA_CHECK(cudaEventDestroy(kernelStart));
    CUDA_CHECK(cudaEventDestroy(kernelEnd));

    // Copy intermediate results back to CPU for cross-check filtering
    std::vector<int> hBestIdx1(N), hBestDist1(N);
    std::vector<int> hBestIdx2(N), hBestDist2(N);
    
    CUDA_CHECK(cudaMemcpy(hBestIdx1.data(), dBestIdx1, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hBestDist1.data(), dBestDist1, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hBestIdx2.data(), dBestIdx2, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hBestDist2.data(), dBestDist2, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Cross-check filtering: keep only mutual matches where i->j and j->i
    hBestIdx.resize(N);
    hBestDist.resize(N);
    for (int i = 0; i < N; i++) {
        int matchIdx = hBestIdx1[i];
        // Check if j->i matches back to i
        if (matchIdx >= 0 && matchIdx < N && hBestIdx2[matchIdx] == i) {
            hBestIdx[i] = matchIdx;
            hBestDist[i] = hBestDist1[i];
        } else {
            hBestIdx[i] = -1;  // No mutual match
            hBestDist[i] = INT_MAX;
        }
    }

    // Cleanup
    cudaFree(dDes1);
    cudaFree(dDes2);
    cudaFree(dBestIdx1);
    cudaFree(dBestDist1);
    cudaFree(dBestIdx2);
    cudaFree(dBestDist2);
}