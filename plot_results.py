import matplotlib.pyplot as plt

# Data
features = [500, 1000, 1500, 2000]
cpu = [7.40, 6.20, 6.82, 6.67]
gpu = [0.90, 0.99, 0.89, 0.90]

plt.figure(figsize=(8, 5))

plt.plot(features, cpu, marker='o', markersize=8, linewidth=2.5, label="CPU Runtime (ms)")
plt.plot(features, gpu, marker='o', markersize=8, linewidth=2.5, label="GPU Runtime (ms)")

plt.xlabel("Number of ORB Features", fontsize=12)
plt.ylabel("Matching Time (ms)", fontsize=12)
plt.title("CPU vs GPU ORB Feature Matching Performance", fontsize=14)

plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()

plt.savefig("matching_performance.png", dpi=300)
plt.show()
