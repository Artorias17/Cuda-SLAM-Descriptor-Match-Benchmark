import matplotlib.pyplot as plt

features = [500, 1000, 1500, 2000]
speedup = [8.2, 6.3, 7.7, 7.4]

plt.figure(figsize=(7,5))
plt.plot(features, speedup, marker='s', markersize=8, linewidth=2.5, color="purple")

plt.xlabel("Number of ORB Features", fontsize=12)
plt.ylabel("Speedup (CPU Runtime / GPU Runtime)", fontsize=12)
plt.title("GPU Speedup Over CPU", fontsize=14)

plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

plt.savefig("speedup_plot.png", dpi=300)
plt.show()
