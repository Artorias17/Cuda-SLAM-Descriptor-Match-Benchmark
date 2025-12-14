#ifndef __SUPPORT_H__
#define __SUPPORT_H__

#include <vector>
#include <cstdint>
#include <string>
#include <tuple>
#include <iostream>

// Simple CUDA error check macro
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    }


// Match result structure
struct MatchResult {
    int frameI;
    int frameJ;
    int numMatches;
    double timeMs;
    std::vector<int> bestIdx;
    std::vector<int> bestDist;
};

// Load metadata: returns (num_images, num_features, dim)
std::tuple<int, int, int> loadMetaInfo(const std::string& metaPath);

// Load binary descriptors (.bin) into a flat vector<uint8_t>
std::vector<uint8_t> loadDescriptorBinary(const std::string& path, size_t expectedBytes);

// Load all descriptor sets from .bin files
// Returns: (descriptors, file_numbers) where file_numbers[i] is the number from "des{N}.bin"
std::pair<std::vector<std::vector<uint8_t>>, std::vector<int>> loadAllDescriptors(const std::string& descriptorDir);

// Match two descriptor sets on GPU
void matchDescriptorsGPU(const uint8_t* hDes1, const uint8_t* hDes2,
                         int N, int dim,
                         std::vector<int>& hBestIdx, 
                         std::vector<int>& hBestDist,
                         double& elapsedMs);

#endif