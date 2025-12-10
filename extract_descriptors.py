import cv2
import numpy as np

# Load images
img1 = cv2.imread("images/img1.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("images/img2.jpg", cv2.IMREAD_GRAYSCALE)

# Resize the images
img1 = cv2.resize(img1, (1280, 720))
img2 = cv2.resize(img2, (1280, 720))

# ORB extractor
orb = cv2.ORB_create(2000)

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

print("des1:", des1.shape)
print("des2:", des2.shape)

# Save as .npy 
np.save("descriptors/des1.npy", des1)
np.save("descriptors/des2.npy", des2)

# Convert to uint8
des1 = des1.astype(np.uint8)
des2 = des2.astype(np.uint8)

# Save to binary for C++/CUDA
des1.tofile("descriptors/des1.bin")
des2.tofile("descriptors/des2.bin")

# Record metadata: N1 N2 DIM
with open("descriptors/meta.txt", "w") as f:
    f.write(f"{des1.shape[0]} {des2.shape[0]} {des1.shape[1]}\n")

print("Saved .npy, .bin, and meta.txt")
