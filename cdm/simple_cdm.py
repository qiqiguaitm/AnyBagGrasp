#!/usr/bin/env python3
"""
Simple CDM processor for integration with visualization scripts
"""

import torch
import cv2
import numpy as np
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from rgbddepth.dpt import RGBDDepth
    CDM_AVAILABLE = True
except ImportError:
    CDM_AVAILABLE = False
    RGBDDepth = None

class SimpleCDMProcessor:
    """Simplified CDM processor for integration"""
    
    def __init__(self, model_path=None, device=None):
        """Initialize CDM processor"""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.initialized = False
        
        # Temporarily disable CDM due to output range issues
        # TODO: Fix model weights or calibration
        print("Warning: CDM temporarily disabled - using original depth")
        return
        
        if not CDM_AVAILABLE:
            print("Warning: CDM not available")
            return
        
        if model_path is None:
            model_path = Path(__file__).parent / "cdm_d435.ckpt"
        
        self.model_path = Path(model_path)
        
        if self.model_path.exists():
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize CDM model"""
        try:
            print(f"Loading CDM model from {self.model_path}")
            
            # Create model
            self.model = RGBDDepth(encoder='vitl', features=256)
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # Remove module prefix if present
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    cleaned_state_dict[key[7:]] = value
                else:
                    cleaned_state_dict[key] = value
            
            self.model.load_state_dict(cleaned_state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            
            self.initialized = True
            print(f"CDM model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Warning: Failed to load CDM model: {e}")
            self.model = None
            self.initialized = False
    
    def process_rgbd(self, rgb_image, depth_image, target_size=518):
        """Process RGB-D pair with CDM model"""
        if not self.initialized or self.model is None:
            return None
        
        try:
            # Prepare inputs
            rgb_tensor = self._prepare_rgb(rgb_image, target_size)
            depth_tensor = self._prepare_depth(depth_image, target_size)
            
            # Run inference
            with torch.no_grad():
                # Concatenate RGB and depth tensors
                rgbd_input = torch.cat([rgb_tensor, depth_tensor], dim=1)
                print(f"[DEBUG] RGBD input shape: {rgbd_input.shape}")
                print(f"[DEBUG] RGB channel range: {rgb_tensor.min():.3f} - {rgb_tensor.max():.3f}")
                print(f"[DEBUG] Depth channel range: {depth_tensor.min():.3f} - {depth_tensor.max():.3f}")
                print(f"[DEBUG] RGBD input range: {rgbd_input.min():.3f} - {rgbd_input.max():.3f}")
                
                enhanced_depth = self.model(rgbd_input)
                print(f"[DEBUG] Model output shape: {enhanced_depth.shape}")
                print(f"[DEBUG] Model output range: {enhanced_depth.min():.6f} - {enhanced_depth.max():.6f}")
                
                # Add channel dimension if needed
                if len(enhanced_depth.shape) == 3:
                    enhanced_depth = enhanced_depth.unsqueeze(1)
                
                # Resize back to original size
                original_size = (depth_image.shape[0], depth_image.shape[1])
                enhanced_depth = torch.nn.functional.interpolate(
                    enhanced_depth, size=original_size, 
                    mode='bilinear', align_corners=False
                )
                
                # Convert back to numpy
                enhanced_depth_raw = enhanced_depth.squeeze().cpu().numpy()
                
                print(f"[DEBUG] CDM raw output range: {enhanced_depth_raw.min():.6f} - {enhanced_depth_raw.max():.6f}")
                print(f"[DEBUG] CDM raw output mean: {enhanced_depth_raw.mean():.6f}")
                print(f"[DEBUG] CDM raw output non-zero: {np.sum(enhanced_depth_raw != 0)}")
                
                # The model might output depth directly or in similarity space
                # Based on the output range (0.085-0.11), this looks like similarity depth
                # that needs to be inverted, but the result (9-11m) is too large
                
                # Try treating it as direct depth that needs scaling
                # Assumption: Model outputs relative depth that needs calibration
                # Let's scale it to match the input depth range
                
                if np.sum(enhanced_depth_raw != 0) > 0:
                    # Get valid regions from input
                    input_valid = depth_image > 0
                    if np.any(input_valid):
                        input_mean = np.mean(depth_image[input_valid])
                        output_mean = np.mean(enhanced_depth_raw[enhanced_depth_raw > 0])
                        
                        # Scale factor to match input range
                        if output_mean > 0:
                            scale_factor = input_mean / output_mean
                            enhanced_depth = enhanced_depth_raw * scale_factor
                            print(f"[DEBUG] Scale factor: {scale_factor:.3f}")
                            print(f"[DEBUG] Scaled depth range: {enhanced_depth[enhanced_depth>0].min():.3f} - {enhanced_depth[enhanced_depth>0].max():.3f}")
                        else:
                            enhanced_depth = enhanced_depth_raw
                    else:
                        enhanced_depth = enhanced_depth_raw
                else:
                    enhanced_depth = enhanced_depth_raw
                
                return enhanced_depth.astype(np.float32)
                
        except Exception as e:
            print(f"Warning: CDM processing failed: {e}")
            return None
    
    def _prepare_rgb(self, rgb_image, target_size):
        """Prepare RGB image tensor"""
        if len(rgb_image.shape) == 3 and rgb_image.shape[2] == 3:
            rgb = rgb_image.copy()
        else:
            rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        rgb_tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        rgb_tensor = torch.nn.functional.interpolate(
            rgb_tensor, size=(target_size, target_size), 
            mode='bilinear', align_corners=False
        )
        
        return rgb_tensor.to(self.device)
    
    def _prepare_depth(self, depth_image, target_size):
        """Prepare depth tensor (similarity depth)"""
        simi_depth = np.zeros_like(depth_image)
        valid_mask = depth_image > 0
        simi_depth[valid_mask] = 1.0 / depth_image[valid_mask]
        
        depth_tensor = torch.from_numpy(simi_depth).float().unsqueeze(0).unsqueeze(0)
        depth_tensor = torch.nn.functional.interpolate(
            depth_tensor, size=(target_size, target_size), 
            mode='bilinear', align_corners=False
        )
        
        return depth_tensor.to(self.device)
    
    def is_available(self):
        """Check if CDM processing is available"""
        return self.initialized and self.model is not None

# Global CDM processor instance
_cdm_processor = None

def get_cdm_processor():
    """Get global CDM processor instance"""
    global _cdm_processor
    if _cdm_processor is None:
        _cdm_processor = SimpleCDMProcessor()
    return _cdm_processor

def process_depth_with_cdm(rgb_image, depth_image):
    """Convenience function to process depth with CDM"""
    processor = get_cdm_processor()
    if processor.is_available():
        return processor.process_rgbd(rgb_image, depth_image)
    return None