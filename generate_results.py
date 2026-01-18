from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
from extract_descriptors import main as extract_descriptors_main
from cpu_match import main as cpu_match_main
import subprocess
import re

# Feature counts to benchmark
FEATURE_COUNTS = [50, 100, 500, 1000, 2000, 3000, 4000, 5000]

def plot_summary(results: Dict):
    """Plot and save benchmark results using matplotlib."""
    
    plt.figure(figsize=(10, 6))
    plt.plot(results["features"], results["cpu"], marker='o', linewidth=2, markersize=8, label='CPU Total Time (ms)')
    plt.plot(results["features"], results["gpu"], marker='o', linewidth=2, markersize=8, label='GPU Total Time (ms)')
    plt.xlabel('Number of Features', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('Descriptor Matching Performance Benchmark', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('summary.png', dpi=300)
    plt.close()
    print("Summary plot saved as summary.png")

def benchmark_suite() -> Dict[int, Dict]:
    """Run complete benchmark suite across all feature counts."""
    results = {
        "features": [],
        "cpu": [],
        "gpu": [],
    }
    
    for max_features in FEATURE_COUNTS:
        # Extract descriptors
        extract_descriptors_main(max_features=max_features)
        
        num_features = max_features
        with open(Path("descriptors") / "meta.txt", "r") as f:
            _, num_features, _ = map(int, f.readline().split())
        
        print(f"\n{'#'*60}")
        print(f"# Testing with {num_features} features")
        print(f"{'#'*60}")
        
        print(f"\n{'='*60}")
        print(f"# Running CPU Test")
        print(f"{'='*60}")
        # Run CPU baseline
        cpu_result = cpu_match_main()
        
        results["features"].append(num_features)
        cpu_times = [r['time_ms'] for r in cpu_result]
        results["cpu"].append(sum(cpu_times))  # Total CPU time for all pairs
        
        print(f"\n{'='*60}")
        print(f"# Running GPU Test")
        print(f"{'='*60}")
        
        # Run GPU
        gpu_result = subprocess.run(
            ["./build/cudaMatch"],
            cwd="/home/abhishek/projects/descriptor_benchmark_cuda",
            capture_output=True,
            text=True
        )
        
        print(gpu_result.stdout)
        if gpu_result.stderr:
            print(gpu_result.stderr)
        
        match = re.search(r'Total GPU time:\s+(\d+\.?\d*)\s+ms', gpu_result.stdout)
        if match:
            gpu_time = float(match.group(1))
            results["gpu"].append(gpu_time)
        
        if num_features < max_features:
            break
    
    return results

def main():
    """Main entry point."""
    print("Starting Benchmark")
    print(f"{'='*60}")
    
    # Run benchmarks
    results = benchmark_suite()
    
    if not results:
        print("No results collected. Exiting.")
        return
    
    # Plot and save summary
    plot_summary(results)


if __name__ == "__main__":
    main()
