#!/usr/bin/env python3
"""
Camera Depth Analysis Script
Extracts RGB and depth frames from video and HDF5 files
"""

import cv2
import h5py
import numpy as np
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def create_output_directory(output_dir):
    """Create output directory if it doesn't exist"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")


def extract_rgb_frames(video_path, output_dir, max_frames=1000, crop_top_half=True):
    """Extract RGB frames from video file, optionally cropping to top half"""
    print(f"\n=== Extracting RGB frames from {video_path} ===")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {frame_count} frames, {fps} FPS, {width}x{height}")
    
    if crop_top_half:
        target_height = 720  # Match depth image height
        print(f"Will crop RGB frames to top half: {width}x{target_height}")
    
    frames_to_extract = min(max_frames, frame_count)
    print(f"Extracting first {frames_to_extract} RGB frames...")
    
    rgb_dir = os.path.join(output_dir, "rgb")
    create_output_directory(rgb_dir)
    
    for i in range(frames_to_extract):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {i}")
            break
        
        # Crop to top half to match depth size (1280x720)
        if crop_top_half:
            frame = frame[:720, :, :]  # Keep top 720 pixels of height
        
        # Save RGB frame
        rgb_filename = f"rgb_frame_{i:06d}.jpg"
        rgb_path = os.path.join(rgb_dir, rgb_filename)
        cv2.imwrite(rgb_path, frame)
        
        if i % 100 == 0:
            print(f"  Processed {i} RGB frames...")
    
    cap.release()
    print(f"RGB extraction complete: {i+1} frames saved")
    return i+1


def extract_depth_frames(h5_path, output_dir, max_frames=1000):
    """Extract depth frames from HDF5 file"""
    print(f"\n=== Extracting depth frames from {h5_path} ===")
    
    with h5py.File(h5_path, 'r') as f:
        if 'frames' not in f:
            raise ValueError(f"No 'frames' dataset found in {h5_path}")
        
        depth_data = f['frames']
        total_frames = depth_data.shape[0]
        height, width = depth_data.shape[1], depth_data.shape[2]
        
        print(f"Depth data info: {total_frames} frames, {width}x{height}")
        
        frames_to_extract = min(max_frames, total_frames)
        print(f"Extracting first {frames_to_extract} depth frames...")
        
        depth_dir = os.path.join(output_dir, "depth")
        create_output_directory(depth_dir)
        
        for i in range(frames_to_extract):
            # Get depth frame
            depth_frame = depth_data[i]
            
            # Normalize depth values for visualization (0-255)
            depth_normalized = ((depth_frame - np.min(depth_frame)) / 
                              (np.max(depth_frame) - np.min(depth_frame)) * 255).astype(np.uint8)
            
            # Save original depth data as numpy file
            depth_raw_filename = f"depth_frame_{i:06d}.npy"
            depth_raw_path = os.path.join(depth_dir, depth_raw_filename)
            np.save(depth_raw_path, depth_frame)
            
            # Save normalized depth as image for visualization
            depth_vis_filename = f"depth_frame_{i:06d}.png"
            depth_vis_path = os.path.join(depth_dir, depth_vis_filename)
            cv2.imwrite(depth_vis_path, depth_normalized)
            
            if i % 100 == 0:
                print(f"  Processed {i} depth frames...")
        
        print(f"Depth extraction complete: {i+1} frames saved")
        return i+1


def create_depth_colormap(depth_frame, colormap=cv2.COLORMAP_JET, mask_zeros=True):
    """Create colorized depth visualization"""
    # Handle zero values (invalid depth measurements)
    if mask_zeros:
        # Create a mask for valid depth values (non-zero)
        valid_mask = depth_frame > 0
        
        if np.any(valid_mask):
            # Normalize only valid depth values
            valid_depths = depth_frame[valid_mask]
            depth_normalized = np.zeros_like(depth_frame, dtype=np.uint8)
            
            # Normalize valid depths to 1-255 (reserve 0 for invalid areas)
            if np.max(valid_depths) > np.min(valid_depths):
                normalized_valid = ((valid_depths - np.min(valid_depths)) / 
                                  (np.max(valid_depths) - np.min(valid_depths)) * 254 + 1).astype(np.uint8)
                depth_normalized[valid_mask] = normalized_valid
        else:
            depth_normalized = np.zeros_like(depth_frame, dtype=np.uint8)
    else:
        # Original normalization (includes zeros)
        depth_normalized = ((depth_frame - np.min(depth_frame)) / 
                           (np.max(depth_frame) - np.min(depth_frame)) * 255).astype(np.uint8)
    
    # Apply colormap
    depth_colored = cv2.applyColorMap(depth_normalized, colormap)
    
    # Make invalid areas (zeros) black if masking is enabled
    if mask_zeros:
        invalid_mask = depth_frame == 0
        depth_colored[invalid_mask] = [0, 0, 0]  # Black for invalid depth
    
    return depth_colored


def create_combined_visualization(rgb_frame, depth_frame, save_path, colormap=cv2.COLORMAP_JET):
    """Create combined RGB and depth visualization"""
    # Create depth colormap with masking for better visualization
    depth_colored = create_depth_colormap(depth_frame, colormap, mask_zeros=True)
    
    # Ensure both images have same size
    h, w = depth_frame.shape
    if rgb_frame.shape[:2] != (h, w):
        rgb_frame = cv2.resize(rgb_frame, (w, h))
    
    # Create side-by-side visualization
    combined = np.hstack([rgb_frame, depth_colored])
    
    # Add text labels with background for better visibility
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    
    # RGB label
    cv2.rectangle(combined, (5, 5), (100, 35), (0, 0, 0), -1)  # Black background
    cv2.putText(combined, 'RGB', (10, 25), font, font_scale, (255, 255, 255), thickness)
    
    # Depth label  
    cv2.rectangle(combined, (w + 5, 5), (w + 120, 35), (0, 0, 0), -1)  # Black background
    cv2.putText(combined, 'Depth', (w + 10, 25), font, font_scale, (255, 255, 255), thickness)
    
    # Add depth info
    zero_count = np.sum(depth_frame == 0)
    zero_pct = zero_count / depth_frame.size * 100
    info_text = f'Invalid: {zero_pct:.1f}%'
    cv2.rectangle(combined, (w + 5, 40), (w + 180, 65), (0, 0, 0), -1)
    cv2.putText(combined, info_text, (w + 10, 57), font, 0.5, (255, 255, 255), 1)
    
    # Save combined image
    cv2.imwrite(save_path, combined)


def create_visualizations(video_path, h5_path, output_dir, max_frames=1000, vis_every=10):
    """Create combined RGB+Depth visualizations"""
    print(f"\n=== Creating combined visualizations ===")
    
    # Create visualization directory
    vis_dir = os.path.join(output_dir, "vis")
    create_output_directory(vis_dir)
    
    # Open video and HDF5
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    with h5py.File(h5_path, 'r') as h5f:
        depth_data = h5f['frames']
        
        frames_to_process = min(max_frames, depth_data.shape[0])
        vis_count = 0
        
        print(f"Creating visualizations for {frames_to_process} frames (every {vis_every} frames)...")
        
        for i in range(frames_to_process):
            # Read RGB frame
            ret, rgb_frame = cap.read()
            if not ret:
                print(f"Warning: Could not read RGB frame {i}")
                break
            
            # Crop RGB to match depth size
            rgb_frame = rgb_frame[:720, :, :]
            
            # Get depth frame
            depth_frame = depth_data[i]
            
            # Create visualization every vis_every frames
            if i % vis_every == 0:
                vis_filename = f"combined_frame_{i:06d}.jpg"
                vis_path = os.path.join(vis_dir, vis_filename)
                create_combined_visualization(rgb_frame, depth_frame, vis_path, cv2.COLORMAP_JET)
                vis_count += 1
            
            if i % 100 == 0:
                print(f"  Processed {i} frames...")
        
        cap.release()
        print(f"Visualization complete: {vis_count} combined images saved")
        return vis_count


def analyze_depth_statistics(h5_path, num_samples=100):
    """Analyze depth data statistics"""
    print(f"\n=== Analyzing depth statistics ===")
    
    with h5py.File(h5_path, 'r') as f:
        depth_data = f['frames']
        total_frames = depth_data.shape[0]
        
        # Sample frames for statistics
        sample_indices = np.linspace(0, total_frames-1, num_samples, dtype=int)
        
        min_vals, max_vals, mean_vals = [], [], []
        
        for idx in sample_indices:
            frame = depth_data[idx]
            min_vals.append(np.min(frame))
            max_vals.append(np.max(frame))
            mean_vals.append(np.mean(frame))
        
        print(f"Depth statistics from {num_samples} sample frames:")
        print(f"  Min depth range: {np.min(min_vals):.3f} - {np.max(min_vals):.3f}")
        print(f"  Max depth range: {np.min(max_vals):.3f} - {np.max(max_vals):.3f}")
        print(f"  Mean depth range: {np.min(mean_vals):.3f} - {np.max(mean_vals):.3f}")
        print(f"  Overall min: {np.min(min_vals):.3f}")
        print(f"  Overall max: {np.max(max_vals):.3f}")
        print(f"  Overall mean: {np.mean(mean_vals):.3f}")


def main():
    parser = argparse.ArgumentParser(description='Extract RGB and depth frames from video and HDF5 files')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--h5', type=str, required=True, help='Path to HDF5 file')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--max-frames', type=int, default=1000, help='Maximum frames to extract (default: 1000)')
    parser.add_argument('--stats-only', action='store_true', help='Only analyze depth statistics')
    parser.add_argument('--no-vis', action='store_true', help='Skip creating visualizations')
    parser.add_argument('--vis-every', type=int, default=10, help='Create visualization every N frames (default: 10)')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video file not found: {args.video}")
    if not os.path.exists(args.h5):
        raise FileNotFoundError(f"HDF5 file not found: {args.h5}")
    
    print(f"Input video: {args.video}")
    print(f"Input HDF5: {args.h5}")
    print(f"Output directory: {args.output}")
    print(f"Max frames to extract: {args.max_frames}")
    
    # Analyze depth statistics
    analyze_depth_statistics(args.h5)
    
    if not args.stats_only:
        # Create main output directory
        create_output_directory(args.output)
        
        # Extract frames
        rgb_count = extract_rgb_frames(args.video, args.output, args.max_frames)
        depth_count = extract_depth_frames(args.h5, args.output, args.max_frames)
        
        # Create visualizations
        vis_count = 0
        if not args.no_vis:
            vis_count = create_visualizations(args.video, args.h5, args.output, 
                                            args.max_frames, args.vis_every)
        
        print(f"\n=== Extraction Summary ===")
        print(f"RGB frames extracted: {rgb_count}")
        print(f"Depth frames extracted: {depth_count}")
        if not args.no_vis:
            print(f"Visualizations created: {vis_count}")
        print(f"Output directory structure:")
        print(f"  {args.output}/rgb/     - RGB frames (.jpg) - cropped to 1280x720")
        print(f"  {args.output}/depth/   - Depth frames (.npy + .png)")
        if not args.no_vis:
            print(f"  {args.output}/vis/     - Combined RGB+Depth visualizations (.jpg)")


if __name__ == "__main__":
    main()