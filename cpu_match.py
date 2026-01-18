import cv2
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm


def load_meta_info(descriptor_dir: str = "descriptors") -> tuple[int, int, int]:
    """Load metadata about descriptors.
    
    Returns:
        num_images: Number of descriptor sets
        num_features: Number of features per descriptor set (same for all)
        dim: Descriptor dimension (32 for ORB)
    """
    meta_path = Path(descriptor_dir) / "meta.txt"
    with open(meta_path, "r") as f:
        line = f.readline().strip()
        num_images, num_features, dim = map(int, line.split())

    return num_images, num_features, dim


def load_all_descriptors(descriptor_dir: str = "descriptors") -> tuple[list[np.ndarray], list[int]]:
    """Load all descriptor sets from .npy files using glob pattern (matches image numbering).
    
    Returns:
        descriptors: List of descriptor arrays
        file_numbers: List of file numbers extracted from filenames (e.g., [1, 2, 5] from des1.npy, des2.npy, des5.npy)
    """
    desc_path = Path(descriptor_dir)
    
    # Use glob to find all des*.npy files, sorted by numeric suffix (like images)
    descriptor_files = sorted(
        desc_path.glob("des*.npy"),
        key=lambda p: int(p.stem[3:])  # Extract number from "des{N}.npy"
    )
    
    if not descriptor_files:
        raise FileNotFoundError(f"No descriptor files found in {descriptor_dir}")
    
    descriptors = []
    file_numbers = []
    for desc_file in tqdm(descriptor_files, desc="Loading descriptors", unit="file"):
        # Extract file number from filename
        file_num = int(desc_file.stem[3:])
        file_numbers.append(file_num)
        
        des = np.load(desc_file)
        des = des.astype(np.uint8)
        descriptors.append(des)

    return descriptors, file_numbers


def match_descriptors_cpu(
    des1: np.ndarray, des2: np.ndarray, warmup: bool = True
) -> tuple[list, float]:
    """Match two descriptor sets using CPU BFMatcher with Hamming distance."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Warmup run
    if warmup:
        _ = bf.match(des1, des2)

    # Timed matching
    t0 = time.perf_counter()
    matches = bf.match(des1, des2)
    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) * 1000.0

    return matches, elapsed_ms


def match_sequential_frames(descriptors: list[np.ndarray], file_numbers: list[int]) -> list[dict]:
    """Match consecutive frames sequentially (ORB-SLAM style).
    
    Args:
        descriptors: List of descriptor arrays
        file_numbers: List of actual file numbers (e.g., [1, 2, 5])
    
    Returns list of results for each consecutive pair.
    """
    results = []
    n = len(descriptors)
    
    if n < 2:
        raise ValueError("Need at least 2 frames for sequential matching")
    
    print(f"\nMatching {n-1} consecutive frame pairs...")
    
    for i in tqdm(range(n - 1), desc="Matching frames", unit="pair"):
        frame_i = file_numbers[i]       # Use actual file number
        frame_j = file_numbers[i + 1]   # Use actual file number
        
        matches, elapsed_ms = match_descriptors_cpu(descriptors[i], descriptors[i+1])
        
        result = {
            'pair': (frame_i, frame_j),
            'matches': len(matches),
            'time_ms': elapsed_ms,
            'match_objects': matches
        }
        results.append(result)
    
    return results


def main(descriptor_dir: str = "descriptors"):
    """Main CPU matching benchmark for ORB-SLAM style sequential matching.

    Args:
        descriptor_dir: Directory containing descriptor files
    """

    # Load all descriptors
    print("Loading descriptors...")
    descriptors, file_numbers = load_all_descriptors(descriptor_dir)

    print(f"Loaded {len(descriptors)} descriptor sets:")
    for file_num, des in zip(file_numbers, descriptors):
        print(f"  des{file_num}: {des.shape}")

    # Match all consecutive frames
    results = match_sequential_frames(descriptors, file_numbers)
    
    # Summary
    print("=" * 50)
    print("SEQUENTIAL MATCHING SUMMARY:")
    total_time = sum(r['time_ms'] for r in results)
    avg_matches = sum(r['matches'] for r in results) / len(results)
    print(f"Total frames: {len(descriptors)}")
    print(f"Total pairs matched: {len(results)}")
    print(f"Total CPU time: {total_time:.3f} ms")
    print(f"Average time per pair: {total_time / len(results):.3f} ms")
    print(f"Average matches per pair: {avg_matches:.1f}")
    
    return results



if __name__ == "__main__":
    # Match all consecutive frames (des0↔des1, des1↔des2, ...)
    main()

