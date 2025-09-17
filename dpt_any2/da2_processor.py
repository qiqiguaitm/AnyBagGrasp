#!/usr/bin/env python3
"""
Depth-Anything-V2 Processor for RGB to Depth Estimation
Supports MPS (Metal Performance Shaders) acceleration on Mac
"""

import torch
import cv2
import numpy as np
import os
from pathlib import Path

from depth_anything_v2.dpt import DepthAnythingV2

class DA2Processor:
    """Depth Anything V2 processor for monocular depth estimation"""
    
    def __init__(self, model_path=None, encoder='vitg', device=None):
        """
        Initialize DA2 processor
        
        Args:
            model_path: Path to .safetensors or .pth model file
            encoder: Model size ('vits', 'vitb', 'vitl', 'vitg')
            device: Device to run on ('mps', 'cuda', 'cpu', or None for auto)
        """
        # Auto-select device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = 'mps'
                print("Using MPS (Metal Performance Shaders) for DA2")
            elif torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        self.encoder = encoder
        self.model = None
        self.initialized = False
        
        # Model configurations
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        # Default model path if not provided
        if model_path is None:
            # Check for both .safetensors and .pth formats
            safetensors_path = f"/Users/didi/models/depth_anything_v2/depth_anything_v2_{encoder}_fp32.safetensors"
            pth_path = f"/Users/didi/models/depth_anything_v2/depth_anything_v2_{encoder}.pth"
            
            if os.path.exists(safetensors_path):
                model_path = safetensors_path
            elif os.path.exists(pth_path):
                model_path = pth_path
            else:
                model_path = safetensors_path  # Default to safetensors path for error message
        
        self.model_path = Path(model_path)
        
        if self.model_path.exists():
            self._load_model()
        else:
            print(f"Warning: Model file not found: {self.model_path}")
    
    def _load_model(self):
        """Load DA2 model"""
        try:
            print(f"Loading DA2 model from {self.model_path}")
            
            # Create model
            self.model = DepthAnythingV2(**self.model_configs[self.encoder])
            
            # Load checkpoint
            if self.model_path.suffix == '.safetensors':
                # Load safetensors format
                try:
                    from safetensors.torch import load_file
                    state_dict = load_file(str(self.model_path))
                except ImportError:
                    print("Warning: safetensors not installed, trying torch.load")
                    state_dict = torch.load(self.model_path, map_location='cpu', weights_only=False)
            else:
                # Load regular pytorch format
                checkpoint = torch.load(self.model_path, map_location='cpu')
                if isinstance(checkpoint, dict) and 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            
            # Load state dict
            self.model.load_state_dict(state_dict, strict=True)
            
            # Move to device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.initialized = True
            print(f"DA2 model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading DA2 model: {e}")
            self.model = None
            self.initialized = False
    
    def process_rgb(self, rgb_image, input_size=518):
        """
        Process RGB image to generate depth map
        
        Args:
            rgb_image: RGB image (H, W, 3) in BGR or RGB format
            input_size: Model input size (default 518)
            
        Returns:
            depth_map: Normalized depth map (H, W) in range [0, 1]
        """
        if not self.initialized or self.model is None:
            print("Model not initialized")
            return None
        
        try:
            with torch.no_grad():
                # Prepare image
                h, w = rgb_image.shape[:2]
                
                # Convert BGR to RGB if needed
                if len(rgb_image.shape) == 3 and rgb_image.shape[2] == 3:
                    # Assume BGR input for OpenCV compatibility
                    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                
                # Use model's inference method
                depth_raw = self.model.infer_image(rgb_image, input_size)
                
                # Resize to original resolution first
                depth_raw = cv2.resize(depth_raw, (w, h))
                
                # DA2 outputs inverse depth (similar to disparity)
                # Apply robust reciprocal transformation based on muggled_dpt insights
                
                # 1. Clip extreme values using percentiles for robustness
                p_low, p_high = 2, 98  # Use 2nd and 98th percentiles
                inv_low = np.percentile(depth_raw, p_low)
                inv_high = np.percentile(depth_raw, p_high)
                depth_clipped = np.clip(depth_raw, inv_low, inv_high)
                
                # 2. Convert inverse depth to depth
                # Larger raw values = closer objects (inverse depth)
                # After inversion: larger values = farther objects (depth)
                depth = 1.0 / (depth_clipped + 1e-6)
                
                # 3. Handle sky/infinite depth regions
                # Very small inverse depth values (< 1% of p_low) likely represent sky/infinity
                sky_mask = depth_raw < inv_low * 0.01
                if np.any(sky_mask):
                    # Set sky regions to maximum depth
                    depth[sky_mask] = np.percentile(depth, 99.9) * 1.5
                
                # 4. Final normalization to [0, 1] for relative depth
                # Remove outliers before normalization
                valid_depth = depth[~sky_mask] if np.any(sky_mask) else depth
                depth_min = np.percentile(valid_depth, 1)
                depth_max = np.percentile(valid_depth, 99)
                
                depth_normalized = (depth - depth_min) / (depth_max - depth_min + 1e-8)
                depth_normalized = np.clip(depth_normalized, 0, 1)
                
                return depth_normalized
                
        except Exception as e:
            print(f"Error processing image: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def is_available(self):
        """Check if processor is available"""
        return self.initialized and self.model is not None

# Global instance
_da2_processor = None

def get_da2_processor(encoder='vitg'):
    """Get global DA2 processor instance
    Default to vitg (giant) model for best quality
    """
    global _da2_processor
    if _da2_processor is None:
        _da2_processor = DA2Processor(encoder=encoder)
    return _da2_processor

def process_rgb_with_da2(rgb_image, input_size=518):
    """Convenience function to process RGB with DA2"""
    processor = get_da2_processor()
    if processor.is_available():
        return processor.process_rgb(rgb_image, input_size)
    return None