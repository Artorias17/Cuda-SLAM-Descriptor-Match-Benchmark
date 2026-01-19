# ORB Descriptor Matching Benchmark: CPU vs GPU

Performance comparison of ORB descriptor matching on CPU (OpenCV with SIMD) vs GPU (CUDA kernel) for sequential frame matching patterns used in visual SLAM.

## Project Structure

```text
extract_video_frames.py       # Extract frames from video
extract_descriptors.py        # Extract ORB features from images
cpu_match.py                  # CPU matching (OpenCV BFMatcher)
generate_results.py           # Benchmark suite with scalability testing
cudaMatch.cu / kernel.cu      # CUDA implementation
support.cu / support.h        # GPU support functions
images/                       # Input images (img*.jpg)
descriptors/                  # Output descriptors (des*.npy, des*.bin, meta.txt)
```

## Requirements

- Python 3.11+ with opencv-python, tqdm
- NVIDIA GPU with CUDA Toolkit 11+ and nvcc compiler
- C++17 compiler

## Installation

Install Python dependencies using uv:

```bash
uv sync
source .venv/bin/activate
```

Verify CUDA setup:

```bash
nvcc --version
```

## Quick Start

**1. Extract frames from video** (optional):

```bash
python extract_video_frames.py video.mp4 --target-fps 30 [--start-time 10] [--end-time 60]
```

**2. Build and benchmark**:

```bash
make
python generate_results.py
```

Outputs `summary.png` with CPU vs GPU performance across feature counts (50 to 5000 features).

## Running Scripts Individually

If you want to run scripts separately:

**Extract frames from video:**

```bash
python extract_video_frames.py video.mp4 --target-fps 30 --start-time 10 --end-time 60 --output-dir images
```

Arguments: `--target-fps` (default 30), `--start-time` (seconds), `--end-time` (seconds), `--output-dir` (default "images")

**Extract descriptors:**

```bash
python extract_descriptors.py --max-features 2000 --images-dir images --output-dir descriptors
```

Arguments: `--max-features` (default 2000), `--images-dir` (default "images"), `--output-dir` (default "descriptors")

**Run CPU matching only:**

```bash
python cpu_match.py
```

Uses descriptors from `descriptors/` directory by default.

**Run GPU matching only:**

```bash
./build/cudaMatch
```

Uses descriptors from `descriptors/` directory.

## Details

### extract_video_frames.py

- Extracts frames at target FPS (default 30)
- Options: `--start-time`, `--end-time` (seconds), `--output-dir`
- Works with game footage (needs texture/detail)
- Outputs: `images/img0.jpg`, `img1.jpg`, etc.

### extract_descriptors.py

- Extracts ORB features from image sequence
- Default max 2000 features per image
- Outputs: `descriptors/des{N}.npy`, `des{N}.bin`, `meta.txt`

### generate_results.py

- Tests feature counts: 50, 100, 500, 1K, 2K, 3K, 4K, 5K (adaptive)
- Automatically stops when images can't provide more features
- Runs CPU (OpenCV) and GPU (CUDA) benchmarks for each count
- Generates performance plot with absolute times and speedup curves
- Outputs: `summary.png` with dual-axis visualization

## Implementation Details

### CPU: OpenCV BFMatcher

- Bidirectional cross-checking (A→B and B→A match)
- SIMD optimized (SSE4.2, AVX2, POPCNT)
- Sequential frame matching (i↔i+1)

### GPU: CUDA Kernel  

- Bidirectional cross-checking (forward + backward kernel launches)
- One block per descriptor, 256 threads per block
- Shared memory parallel reduction
- CUDA events for GPU-synchronized timing (excludes CPU overhead)

Both CPU and GPU use the same cross-checking logic for fair comparison. Expected agreement: 99%+.

## Meta Format

`descriptors/meta.txt` contains: `num_images num_features descriptor_dim`

Example: `2 2000 32` means 2 frames, 2000 features each, 32-byte ORB descriptors.
