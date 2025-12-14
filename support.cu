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

    // Launch matching kernel - KERNEL TIME ONLY
    const int BLOCK_SIZE = 256;
    dim3 grid(N);
    dim3 block(BLOCK_SIZE);
    size_t sharedMemBytes = 2 * BLOCK_SIZE * sizeof(int);

    auto kernelStart = std::chrono::high_resolution_clock::now();
    matchKernelHamming<<<grid, block, sharedMemBytes>>>(
        dDes1, dDes2, N, N, dim, dBestIdx, dBestDist
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    auto kernelEnd = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> kernelTime = kernelEnd - kernelStart;
    elapsedMs = kernelTime.count();  // Return ONLY kernel execution time

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