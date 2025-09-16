#!/usr/bin/env python3
"""
Fixed CDM processor using correct inference method
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

class FixedCDMProcessor:
    """Fixed CDM processor using infer_image method"""
    
    def __init__(self, model_path=None, device=None):
        """Initialize CDM processor"""
        # Try to use MPS if available
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
                print("Using MPS (Metal Performance Shaders) for acceleration")
            else:
                self.device = "cpu"
        else:
            self.device = device
        self.model = None
        self.initialized = False
        
        if not CDM_AVAILABLE:
            print("Warning: CDM not available - rgbddepth module not found")
            return
        
        if model_path is None:
            model_path = Path(__file__).parent / "cdm_d435.ckpt"
        
        self.model_path = Path(model_path)
        
        if self.model_path.exists():
            self._initialize_model()
        else:
            print(f"Warning: Model file not found: {self.model_path}")
    
    def _initialize_model(self):
        """Initialize CDM model"""
        try:
            print(f"Loading CDM model from {self.model_path}")
            
            # Create model (D435 camera configuration - using vitl)
            self.model = RGBDDepth(
                encoder='vitl', 
                features=256,
                out_channels=[256, 512, 1024, 1024]
            )
            
            # Load checkpoint - always load to CPU first
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                # Handle checkpoints that wrap state dict in 'model' key
                # Remove 'module.' prefix if present (from DataParallel training)
                state_dict = {k[7:] if k.startswith('module.') else k: v 
                             for k, v in checkpoint['model'].items()}
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                # Remove 'pipeline.' prefix if present
                state_dict = {k[9:] if k.startswith('pipeline.') else k: v 
                             for k, v in state_dict.items()}
            else:
                state_dict = checkpoint
            
            # Load state dict
            self.model.load_state_dict(state_dict, strict=False)
            
            # Move model to device (MPS or CPU)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Ensure all parameters and buffers are on the correct device
            for param in self.model.parameters():
                param.data = param.data.to(self.device)
            for buffer in self.model.buffers():
                buffer.data = buffer.data.to(self.device)
            
            self.initialized = True
            print(f"CDM model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Warning: Failed to load CDM model: {e}")
            self.model = None
            self.initialized = False
    
    def _prepare_inputs(self, rgb_image, depth, input_size=518):
        """Prepare inputs for the model (based on model's image2tensor method)"""
        from torchvision.transforms import Compose
        from util.transform import Resize, NormalizeImage, PrepareForNet
        
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=True,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
        h, w = rgb_image.shape[:2]
        
        # Normalize RGB to [0, 1]
        image = rgb_image.astype(np.float32) / 255.0
        
        # Apply transforms
        prepared = transform({"image": image, "depth": depth})
        image = prepared["image"]
        image = torch.from_numpy(image).unsqueeze(0)
        
        depth = prepared["depth"]
        depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
        
        # Concatenate RGB and depth
        inputs = torch.cat((image, depth), dim=1)
        
        # Move to device (CPU in our case)
        inputs = inputs.to(self.device)
        
        return inputs, (h, w)
    
    def process_rgbd(self, rgb_image, depth_image, input_size=518):
        """Process RGB-D pair with CDM model using custom inference to handle device issues"""
        if not self.initialized or self.model is None:
            return None
        
        try:
            # Ensure RGB format
            if len(rgb_image.shape) == 3 and rgb_image.shape[2] == 3:
                # Assume BGR input, convert to RGB
                rgb_for_model = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_for_model = rgb_image
            
            # Create similarity depth (inverse depth)
            simi_depth = np.zeros_like(depth_image)
            valid_mask = depth_image > 0
            if np.any(valid_mask):
                simi_depth[valid_mask] = 1.0 / depth_image[valid_mask]
            
            # Custom inference to handle device placement properly
            with torch.no_grad():
                # Prepare inputs and ensure they're on the correct device
                inputs, (h, w) = self._prepare_inputs(rgb_for_model, simi_depth, input_size)
                
                # Ensure inputs are on the same device as model
                inputs = inputs.to(self.device)
                
                # Forward pass
                pred_depth = self.model(inputs)
                
                # Resize to original resolution
                pred_depth = torch.nn.functional.interpolate(
                    pred_depth[:, None], 
                    (h, w), 
                    mode="nearest"
                )[0, 0]
                
                # Convert to numpy (always move to CPU first)
                pred_simi_depth = pred_depth.cpu().numpy()
            
            # Convert from similarity depth back to meters
            # The model outputs similarity depth (1/depth), so we invert it
            pred_depth = np.zeros_like(pred_simi_depth)
            valid_pred = pred_simi_depth > 0
            if np.any(valid_pred):
                pred_depth[valid_pred] = 1.0 / pred_simi_depth[valid_pred]
                
                # Apply reasonable limits for desktop grasping scenarios
                # Filter out unreasonable values
                pred_depth[pred_depth > 10.0] = 0  # Remove far outliers
                pred_depth[pred_depth < 0.1] = 0   # Remove near outliers
            
            return pred_depth.astype(np.float32)
            
        except Exception as e:
            print(f"Warning: CDM processing failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def is_available(self):
        """Check if CDM processing is available"""
        return self.initialized and self.model is not None

# Global CDM processor instance
_cdm_processor = None

def get_cdm_processor():
    """Get global CDM processor instance"""
    global _cdm_processor
    if _cdm_processor is None:
        _cdm_processor = FixedCDMProcessor()
    return _cdm_processor

def process_depth_with_cdm(rgb_image, depth_image):
    """Convenience function to process depth with CDM"""
    processor = get_cdm_processor()
    if processor.is_available():
        return processor.process_rgbd(rgb_image, depth_image)
    return None

def test_cdm():
    """Test CDM with sample data"""
    processor = get_cdm_processor()
    if not processor.is_available():
        print("CDM not available for testing")
        return
    
    # Create dummy data
    rgb_dummy = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    depth_dummy = np.random.uniform(0.2, 1.5, (720, 1280)).astype(np.float32)
    
    print(f"Test input - RGB: {rgb_dummy.shape}, Depth: {depth_dummy.shape}")
    print(f"Test depth range: {depth_dummy.min():.2f} - {depth_dummy.max():.2f}")
    
    # Process
    result = processor.process_rgbd(rgb_dummy, depth_dummy)
    
    if result is not None:
        valid = result > 0
        if np.any(valid):
            print(f"Result depth range: {result[valid].min():.2f} - {result[valid].max():.2f}")
            print(f"Result mean depth: {result[valid].mean():.2f}")
        else:
            print("No valid depth in result")
    else:
        print("Processing returned None")

if __name__ == "__main__":
    test_cdm()