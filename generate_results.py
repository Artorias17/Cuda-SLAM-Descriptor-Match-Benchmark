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
    """Plot and save benchmark results with speedup analysis."""
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Left axis: Absolute times
    ax1.plot(results["features"], results["cpu"], marker='o', linewidth=2.5, markersize=8, label='CPU Time', color='#1f77b4')
    ax1.plot(results["features"], results["gpu"], marker='s', linewidth=2.5, markersize=8, label='GPU Time', color='#ff7f0e')
    ax1.set_xlabel('Number of Features', fontsize=12)
    ax1.set_ylabel('Time (ms)', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)
    
    # Right axis: Speedup ratio
    ax2 = ax1.twinx()
    speedup = [cpu / gpu for cpu, gpu in zip(results["cpu"], results["gpu"])]
    ax2.plot(results["features"], speedup, marker='^', linewidth=2.5, markersize=8, label='Speedup', color='#2ca02c', linestyle='--')
    ax2.set_ylabel('GPU Speedup (CPU time / GPU time)', fontsize=12, color='#2ca02c')
    ax2.tick_params(axis='y', labelcolor='#2ca02c')
    ax2.axhline(y=1, color='red', linestyle=':', linewidth=1.5, alpha=0.6, label='No speedup (1x)')
    
    plt.title('Descriptor Matching: Performance & Speedup Analysis', fontsize=14, fontweight='bold')
    
    # Combined legend from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    fig.tight_layout()
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
