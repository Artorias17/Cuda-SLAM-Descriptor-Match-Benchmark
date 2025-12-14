# ORB Descriptor Matching Benchmark: CPU vs GPU

A performance comparison framework for ORB (Oriented FAST and Rotated BRIEF) descriptor matching between CPU and GPU implementations. This project benchmarks sequential frame matching patterns commonly used in visual SLAM systems like ORB-SLAM.

## Overview

This benchmark compares:
- **CPU**: OpenCV's BFMatcher with highly optimized SIMD instructions (SSE4.2, AVX2, POPCNT)
- **GPU**: Custom CUDA kernel with parallel Hamming distance computation

The benchmark uses sequential frame matching (frame 1↔2, 2↔3, etc.) to simulate real-world ORB-SLAM workflows.

## Features

- **Modular Architecture**: Clean separation between descriptor extraction, matching, and benchmarking
- **Automatic File Discovery**: Uses glob patterns to automatically discover and process images/descriptors
- **File Number Preservation**: Maintains consistent numbering between images and descriptors (e.g., `img5.jpg` → `des5.npy/bin`)
- **Sequential Matching**: Matches consecutive frames only (ORB-SLAM pattern), not all pairwise combinations
- **Dual Format Support**: Saves descriptors in both `.npy` (NumPy) and `.bin` (binary) formats
- **Progress Tracking**: Uses `tqdm` for visual progress indication
- **Kernel-Only Timing**: GPU benchmarks measure pure kernel execution time, excluding memory transfer overhead

## Project Structure

```
.
├── extract_descriptors.py    # Extract ORB features from images
├── cpu_baseline.py            # CPU matching benchmark (OpenCV BFMatcher)
├── cudaMatch.cu               # Main CUDA application
├── kernel.cu                  # CUDA kernel implementation
├── support.cu                 # Support functions (loading, GPU matching)
├── support.h                  # Header with declarations and structs
├── Makefile                   # Build configuration
├── images/                    # Input images (img*.jpg)
└── descriptors/               # Generated descriptors (des*.npy, des*.bin, meta.txt)
```

## Requirements

### Python
- Python 3.11+
- OpenCV (`opencv-python>=4.12.0.88`)
- tqdm (`tqdm>=4.67.1`)

### CUDA
- NVIDIA GPU with CUDA support
- CUDA Toolkit (tested with CUDA 11+)
- C++17 compiler (for `std::filesystem`)

## Installation

### Python Setup

Using `uv` (recommended):
```bash
uv sync
```

Or using pip:
```bash
pip install opencv-python tqdm
```

### CUDA Setup

Ensure CUDA toolkit is installed and `nvcc` is in your PATH:
```bash
nvcc --version
```

## Usage

### 1. Extract Descriptors

Extract ORB features from images in the `images/` directory:

```bash
python extract_descriptors.py
```

**Options:**
- `--max-features`: Maximum number of features per image (default: 2000)
- `--images-dir`: Directory containing input images (default: "images")
- `--output-dir`: Directory for output descriptors (default: "descriptors")

**Example with custom settings:**
```bash
python extract_descriptors.py --max-features 10000 --images-dir my_images
```

**Input:** Images named `img*.jpg` (e.g., `img1.jpg`, `img2.jpg`, `img5.jpg`)

**Output:**
- `descriptors/des{N}.npy` - NumPy format descriptors
- `descriptors/des{N}.bin` - Binary format descriptors
- `descriptors/meta.txt` - Metadata: `num_images num_features descriptor_dim`

### 2. Run CPU Benchmark

```bash
python cpu_baseline.py
```

**Features:**
- Uses OpenCV's BFMatcher with `crossCheck=True` for mutual best matches
- Optimized with SIMD instructions (POPCNT for Hamming distance)
- Reports actual matched pairs (not just nearest neighbors)

### 3. Run GPU Benchmark

Build and run the CUDA implementation:

```bash
make clean
make
./cudaMatch
```

**Features:**
- Custom CUDA kernel with parallel Hamming distance computation
- One block per descriptor in the first frame
- 256 threads per block scan the second frame in parallel
- Shared memory reduction to find minimum distance
- Reports kernel execution time only (excludes memory transfer)

## How It Works

### Descriptor Extraction (`extract_descriptors.py`)

1. **Auto-discovers images** using `Path.glob("img*.jpg")` sorted by numeric suffix
2. **Extracts ORB features** using `cv2.ORB_create(max_features)`
3. **Limits descriptors** to minimum available across all images (ensures consistency)
4. **Preserves image numbering**: `img3.jpg` → `des3.npy` and `des3.bin`
5. **Saves metadata** in single-line format: `num_images num_features dim`

### CPU Matching (`cpu_baseline.py`)

1. **Loads descriptors** using glob pattern, sorted by file number
2. **Matches consecutive frames** sequentially (i↔i+1)
3. **Uses BFMatcher** with Hamming distance and cross-checking
4. **Reports per-pair and summary statistics**

### GPU Matching (`cudaMatch.cu`, `kernel.cu`, `support.cu`)

1. **Loads descriptors** from binary files using filesystem iteration
2. **Allocates GPU memory** and copies descriptors (Host→Device)
3. **Launches kernel** with timing around kernel execution only
4. **CUDA Kernel** (`matchKernelHamming`):
   - Each block processes one descriptor from frame 1
   - 256 threads scan all descriptors in frame 2 with stride
   - Computes Hamming distance using `__popc()` intrinsic
   - Parallel reduction in shared memory to find best match
5. **Copies results back** (Device→Host)
6. **Reports per-pair and summary statistics**

### Meta Format

The `meta.txt` file contains a single line:
```
num_images num_features descriptor_dim
```

Example: `2 2000 32` means 2 image frames, 2000 features each, 32-byte descriptors.

## Technical Details

### CPU Optimizations (OpenCV)
- **SIMD Instructions**: SSE4.2, AVX2, AVX512 vectorization
- **POPCNT**: Hardware bit counting for Hamming distance
- **Multi-threading**: pthread-based parallelism
- **Cross-Check**: Mutual best match filtering (A→B and B→A)

### GPU Implementation
- **Kernel Strategy**: One block per descriptor in frame 1
- **Thread Count**: 256 threads per block
- **Memory Pattern**: Coalesced global memory access
- **Reduction**: Shared memory parallel reduction for min distance
- **Bit Counting**: `__popc()` intrinsic for Hamming distance
- **Grid Size**: N blocks (N = number of descriptors)

## Key Differences: CPU vs GPU Output

### Match Counts
- **CPU (460 matches)**: Returns mutual best matches (cross-checked)
- **GPU (2000 matches)**: Returns all nearest neighbors (one per descriptor)

The GPU implementation finds the best match for each descriptor in frame 1, while CPU's cross-check ensures bidirectional consistency.

### Timing Methodology
- **CPU**: Pure matching computation (excludes I/O)
- **GPU**: Kernel execution only (excludes H2D, D2H transfers and allocation)

Both report fair computation-only timings for accurate comparison.

## Future Enhancements

- [ ] Add cross-checking to GPU kernel for fair match count comparison
- [ ] Implement ratio test (Lowe's ratio) for both CPU and GPU
- [ ] Add support for CUDA streams for overlapping computation and transfer
- [ ] Benchmark with varying descriptor counts (1K, 5K, 10K, 50K)
- [ ] Generate performance plots automatically
- [ ] Support for other descriptor types (SIFT, SURF, AKAZE)
