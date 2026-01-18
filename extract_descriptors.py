import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import shutil


def load_and_resize_image(
    image_path: str, target_size: tuple[int, int] = (1280, 720)
) -> np.ndarray:
    """Load an image in grayscale and resize it."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return cv2.resize(img, target_size)


def extract_orb_features(
    img: np.ndarray, max_features: int = 2000
) -> tuple[list, np.ndarray]:
    """Extract ORB features from an image."""
    orb = cv2.ORB_create(max_features)
    keypoints, descriptors = orb.detectAndCompute(img, None)
    if descriptors is None:
        raise ValueError("No descriptors found in image")
    return keypoints, descriptors


def limit_descriptors(
    descriptors_list: list[np.ndarray], max_feats: int
) -> list[np.ndarray]:
    """Limit all descriptor sets to the same maximum number of features."""
    # Find minimum available features across all descriptor sets
    min_available = min(des.shape[0] for des in descriptors_list)
    max_feats = min(max_feats, min_available)

    return [des[:max_feats] for des in descriptors_list]


def save_descriptors(
    descriptors_list: list[np.ndarray],
    image_numbers: list[int],
    output_dir: str = "descriptors",
):
    """Save multiple descriptor sets in both .npy and .bin formats, along with metadata.
    
    Args:
        descriptors_list: List of descriptor arrays
        image_numbers: List of image numbers corresponding to each descriptor (e.g., [1, 2, 3])
        output_dir: Output directory path
    """
    output_path = Path(output_dir)
    # Remove existing directory to clear old descriptors
    if output_path.exists():
        shutil.rmtree(output_path)
        print(f"Cleared existing {output_dir} directory")
    output_path.mkdir(exist_ok=True)

    # Ensure uint8 type for all descriptors
    descriptors_list = [des.astype(np.uint8) for des in descriptors_list]

    # Get dimensions (all should have same dim, just different N)
    descriptor_dim = descriptors_list[0].shape[1]
    feature_counts = [des.shape[0] for des in descriptors_list]

    # Save each descriptor set with matching image number
    for img_num, des in tqdm(
        zip(image_numbers, descriptors_list),
        desc="Saving descriptors",
        total=len(descriptors_list),
        unit="file",
    ):
        np.save(output_path / f"des{img_num}.npy", des)
        des.tofile(output_path / f"des{img_num}.bin")

    # Save meta info: num_images, num_features (same for all), descriptor_dim
    num_features = feature_counts[0]  # All have same count after limiting
    with open(output_path / "meta.txt", "w") as f:
        f.write(f"{len(descriptors_list)} {num_features} {descriptor_dim}\n")

    return feature_counts, descriptor_dim


def extract_descriptors_from_images(
    image_paths: list[Path],
    max_features: int = 2000,
    resize_to: Optional[tuple[int, int]] = None,
    output_dir: str = "descriptors",
):
    """Extract descriptors from multiple images."""

    if len(image_paths) < 2:
        raise ValueError("Need at least 2 images for descriptor extraction")

    descriptors_list = []
    image_numbers = []

    # Load and extract features from each image
    for img_path in tqdm(image_paths, desc="Processing images", unit="image"):
        # Extract image number from filename (e.g., img3.jpg -> 3)
        img_num = int(img_path.stem[3:])
        image_numbers.append(img_num)
        
        # Load and optionally resize
        if resize_to:
            img = load_and_resize_image(str(img_path), resize_to)
        else:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")

        # Extract features
        kp, des = extract_orb_features(img, max_features)
        descriptors_list.append(des)

    # Limit to consistent feature count across all images
    print(f"\nLimiting all descriptors to max {max_features} features...")
    descriptors_list = limit_descriptors(descriptors_list, max_features)

    print(f"\nFinal feature count per image: {descriptors_list[0].shape[0]}")
    for img_num, des in zip(image_numbers, descriptors_list):
        print(f"  des{img_num}: {des.shape}")

    # Save descriptors
    feature_counts, dim = save_descriptors(descriptors_list, image_numbers, output_dir)
    print(f"\nSaved {len(descriptors_list)} descriptor sets successfully!")
    print(f"Meta: {len(descriptors_list)} images, dim={dim}, features={feature_counts}")

    return descriptors_list


def main(
    max_features: int = 2000,
    images_dir: str = "images",
    output_dir: str = "descriptors",
    resize_to: tuple[int, int] = None
):
    """Main extraction pipeline for sequential frames.
    
    Args:
        max_features: Maximum number of ORB features to extract per image
        images_dir: Directory containing input images
        output_dir: Directory for output descriptors
        resize_to: Tuple of (width, height) for descriptors
    """

    # Generate sequential image paths: img1.jpg, img2.jpg, ...
    image_paths = sorted(
        Path(images_dir).glob("img*.jpg"), key=lambda x: int(x.stem[3:])
    )

    if not image_paths:
        raise FileNotFoundError(f"No images found matching 'img*.jpg' in {images_dir}")

    print(f"Found {len(image_paths)} images in {images_dir}")

    # Extract descriptors
    extract_descriptors_from_images(
        image_paths=image_paths,
        max_features=max_features,
        resize_to=resize_to,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract ORB descriptors from sequential image frames"
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=2000,
        help="Maximum number of ORB features per image",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="images",
        help="Directory containing images (img1.jpg, img2.jpg, ...)",
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        default=None,
        help="Resize descriptors to width and height",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="descriptors",
        help="Output directory for descriptors",
    )

    args = parser.parse_args()
    resize_to = None if args.resize is None else (args.resize[0], args.resize[1])
    
    main(
        max_features=args.max_features,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        resize_to=resize_to
    )
