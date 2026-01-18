#!/usr/bin/env python3
"""
Extract frames from a video file at specified FPS and save as image sequence.
Frames are saved as img0.jpg, img1.jpg, ... in the output directory.
"""

import argparse
import cv2
from pathlib import Path
from tqdm import tqdm
import shutil


def extract_frames_from_video(
    video_path: str,
    output_dir: str = "images",
    target_fps: int = 30,
    start_time: float = None,
    end_time: float = None,
) -> None:
    """Extract frames from video at target FPS and save as images.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames (created if not exists)
        target_fps: Target frames per second (e.g., 30, 24, 15)
        start_time: Start time in seconds (default: 0)
        end_time: End time in seconds (default: video duration)
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / video_fps
    
    if video_fps <= 0:
        raise ValueError(f"Invalid video FPS: {video_fps}")
    
    # Convert times to frame numbers
    if start_time is None:
        start_time = 0
    if end_time is None:
        end_time = video_duration
    
    start_frame = int(start_time * video_fps)
    end_frame = int(end_time * video_fps)
    
    # Clamp to valid range
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(start_frame + 1, min(end_frame, total_frames))
    
    # Calculate frame interval for target FPS
    frame_interval = max(1, round(video_fps / target_fps))
    actual_fps = video_fps / frame_interval
    
    print(f"Video properties:")
    print(f"  Original FPS: {video_fps:.1f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {video_duration:.2f}s")
    print(f"  Target FPS: {target_fps}")
    print(f"  Frame interval: every {frame_interval} frame(s)")
    print(f"  Actual output FPS: {actual_fps:.1f}")
    print(f"  Extract range: frame {start_frame} - {end_frame} ({start_time:.2f}s - {end_time:.2f}s)")
    
    # Create output directory
    output_path = Path(output_dir)
    # Remove existing directory to clear old frames
    if output_path.exists():
        shutil.rmtree(output_path)
        print(f"  Cleared existing {output_dir} directory")
    output_path.mkdir(exist_ok=True)
    print(f"  Output directory: {output_path.absolute()}")
    
    # Extract frames
    frame_count = 0
    img_num = 0
    
    frames_to_process = end_frame - start_frame
    with tqdm(total=frames_to_process, desc="Extracting frames", unit="frame") as pbar:
        while frame_count < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames before start_frame
            if frame_count < start_frame:
                frame_count += 1
                continue
            
            # Extract every nth frame
            if (frame_count - start_frame) % frame_interval == 0:
                filename = output_path / f"img{img_num}.jpg"
                cv2.imwrite(str(filename), frame)
                img_num += 1
            
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    
    print(f"\n✓ Successfully extracted {img_num} frames")
    print(f"  Saved to: {output_path.absolute()}")
    print(f"  Files: img0.jpg - img{img_num - 1}.jpg")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Extract frames from video at specified FPS"
    )
    
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to input video file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="images",
        help="Output directory for frames (default: images)"
    )
    
    parser.add_argument(
        "--target-fps",
        type=int,
        default=30,
        help="Target FPS for extracted frames (default: 30)"
    )
    
    parser.add_argument(
        "--start-time",
        type=int,
        default=None,
        help="Start time in seconds (default: 0)"
    )
    
    parser.add_argument(
        "--end-time",
        type=int,
        default=None,
        help="End time in seconds (default: end of video)"
    )
    
    args = parser.parse_args()
    
    # Validate FPS
    if args.target_fps <= 0:
        parser.error("Target FPS must be positive")
    
    # Extract frames
    try:
        extract_frames_from_video(
            video_path=args.video_path,
            output_dir=args.output_dir,
            target_fps=args.target_fps,
            start_time=args.start_time,
            end_time=args.end_time,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
