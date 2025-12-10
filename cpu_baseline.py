import cv2
import numpy as np
import time

# Load descriptors from file .npy
des1 = np.load("descriptors/des1.npy")
des2 = np.load("descriptors/des2.npy")

print("des1 shape:", des1.shape)
print("des2 shape:", des2.shape)

# Convert data to the same unit size
des1 = des1.astype(np.uint8)
des2 = des2.astype(np.uint8)

# Create BFMatcher dby using Hamming 
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Measure the first time
_ = bf.match(des1, des2)

# Measuring matching time
t0 = time.perf_counter()
matches = bf.match(des1, des2)
t1 = time.perf_counter()

elapsed_ms = (t1 - t0) * 1000.0

print(f"Number of matches: {len(matches)}")
print(f"CPU matching time: {elapsed_ms:.3f} ms")
