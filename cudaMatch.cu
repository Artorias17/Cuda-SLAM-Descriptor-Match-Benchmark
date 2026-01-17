#include <iostream>
#include <vector>
#include <cstdint>
#include <string>
#include "support.h"

void matchSequentialFrames(const std::vector<std::vector<uint8_t>>& descriptors,
                           const std::vector<int>& fileNumbers,
                           int numFeatures, int dim) {
    int n = descriptors.size();
    
    if (n < 2) {
        std::cerr << "Need at least 2 frames for sequential matching" << std::endl;
        return;
    }
    
    std::cout << "\nMatching " << (n - 1) << " consecutive frame pairs..." << std::endl;
    std::cout << "=================================================" << std::endl;
    
    std::vector<MatchResult> results;
    double totalTime = 0.0;
    int totalMatches = 0;
    
    for (int i = 0; i < n - 1; i++) {
        int frameI = fileNumbers[i];     // Use actual file number
        int frameJ = fileNumbers[i + 1]; // Use actual file number
        
        std::vector<int> bestIdx, bestDist;
        double elapsedMs;
        
        // Match descriptors[i] with descriptors[i+1] using cross-check for fair CPU/GPU comparison
        matchDescriptorsGPUCrossCheck(descriptors[i].data(), descriptors[i + 1].data(),
                                      numFeatures, dim,
                                      bestIdx, bestDist, elapsedMs);
        
        // Count valid matches (non-negative indices)
        int numMatches = 0;
        for (int idx : bestIdx) {
            if (idx >= 0) numMatches++;
        }
        
        std::cout << "Frame " << frameI << " <-> Frame " << frameJ 
                  << ": " << numMatches << " matches, " 
                  << elapsedMs << " ms" << std::endl;
        
        totalTime += elapsedMs;
        totalMatches += numMatches;
        
        results.push_back({frameI, frameJ, numMatches, elapsedMs, 
                          std::move(bestIdx), std::move(bestDist)});
    }
    
    // Summary
    std::cout << "=================================================" << std::endl;
    std::cout << "SEQUENTIAL MATCHING SUMMARY:" << std::endl;
    std::cout << "Total frames: " << n << std::endl;
    std::cout << "Total pairs matched: " << results.size() << std::endl;
    std::cout << "Total GPU time: " << totalTime << " ms" << std::endl;
    std::cout << "Average time per pair: " << (totalTime / results.size()) << " ms" << std::endl;
    std::cout << "Average matches per pair: " << (totalMatches / static_cast<double>(results.size())) << std::endl;
}

int main(int argc, char** argv) {
    std::string descriptorDir = "descriptors";
    
    // Load metadata
    auto [numImages, numFeatures, dim] = loadMetaInfo(descriptorDir + "/meta.txt");
    
    std::cout << "Metadata: numImages=" << numImages 
              << ", numFeatures=" << numFeatures 
              << ", dim=" << dim << std::endl;
    
    // Load all descriptor sets and their file numbers
    auto [descriptors, fileNumbers] = loadAllDescriptors(descriptorDir);
    
    std::cout << "\nLoaded " << descriptors.size() << " descriptor sets:" << std::endl;
    for (size_t i = 0; i < descriptors.size(); ++i) {
        std::cout << "  des" << fileNumbers[i] << ": " 
                  << numFeatures << " x " << dim << " bytes" << std::endl;
    }
    
    // Match all consecutive frames sequentially (ORB-SLAM style)
    matchSequentialFrames(descriptors, fileNumbers, numFeatures, dim);
    
    return 0;
}
