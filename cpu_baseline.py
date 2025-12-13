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


def load_all_descriptors(descriptor_dir: str = "descriptors") -> list[np.ndarray]:
    """Load all descriptor sets from .npy files."""
    num_images, num_features, dim = load_meta_info(descriptor_dir)
    desc_path = Path(descriptor_dir)

    descriptors = []
    for i in tqdm(
        range(1, num_images + 1), desc="Loading descriptors", unit="file"
    ):
        des = np.load(desc_path / f"des{i}.npy")
        des = des.astype(np.uint8)
        descriptors.append(des)

    return descriptors


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


def match_sequential_frames(descriptors: list[np.ndarray]) -> list[dict]:
    """Match consecutive frames sequentially (ORB-SLAM style).
    
    Returns list of results for each consecutive pair.
    """
    results = []
    n = len(descriptors)
    
    if n < 2:
        raise ValueError("Need at least 2 frames for sequential matching")
    
    print(f"\nMatching {n-1} consecutive frame pairs...")
    
    for i in tqdm(range(n - 1), desc="Matching frames", unit="pair"):
        frame_i = i + 1
        frame_j = i + 2
        
        matches, elapsed_ms = match_descriptors_cpu(descriptors[i], descriptors[i+1])
        
        result = {
            'pair': (frame_i, frame_j),
            'matches': len(matches),
            'time_ms': elapsed_ms,
            'match_objects': matches
        }
        results.append(result)
    
    return results


def main(descriptor_dir: str = "descriptors", pair_index: tuple[int, int] = None):
    """Main CPU matching benchmark for ORB-SLAM style sequential matching.

    Args:
        descriptor_dir: Directory containing descriptor files
        pair_index: Optional tuple (i, j) to match specific des_i with des_j (1-indexed).
                   If None, matches all consecutive frames sequentially.
    """

    # Load all descriptors
    print("Loading descriptors...")
    descriptors = load_all_descriptors(descriptor_dir)

    print(f"Loaded {len(descriptors)} descriptor sets:")
    for i, des in enumerate(descriptors, start=1):
        print(f"  des{i}: {des.shape}")

    if pair_index:
        # Match specific pair
        i, j = pair_index
        if i < 1 or j < 1 or i > len(descriptors) or j > len(descriptors):
            raise ValueError(f"Invalid pair index: ({i}, {j})")

        print(f"\nMatching des{i} <-> des{j}...")
        matches, elapsed_ms = match_descriptors_cpu(descriptors[i - 1], descriptors[j - 1])

        print(f"Number of matches: {len(matches)}")
        print(f"CPU matching time: {elapsed_ms:.3f} ms")

        return matches, elapsed_ms
    else:
        # Match all consecutive frames
        results = match_sequential_frames(descriptors)
        
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
    # Match all consecutive frames (des1↔des2, des2↔des3, ...)
    main()
    
    # Or match a specific pair:
    # main(pair_index=(1, 2))

