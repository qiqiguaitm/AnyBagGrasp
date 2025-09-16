#!/usr/bin/env python3
"""
CDM (Camera Depth Models) processor for depth enhancement
Standalone validation and testing module
"""

import torch
import cv2
import numpy as np
import os
import sys
from pathlib import Path

# Add CDM path to Python path
CDM_DIR = Path(__file__).parent
sys.path.insert(0, str(CDM_DIR))

def test_cdm_imports():
    """Test CDM module imports"""
    print("Testing CDM imports...")
    try:
        from rgbddepth.dpt import RGBDDepth
        print("✓ RGBDDepth imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import RGBDDepth: {e}")
        return False

def test_cdm_model_loading():
    """Test CDM model loading"""
    print("\nTesting CDM model loading...")
    
    # Check model file exists
    model_path = CDM_DIR / "cdm_d435.ckpt"
    if not model_path.exists():
        print(f"✗ Model file not found: {model_path}")
        return False, None
    
    print(f"✓ Model file found: {model_path}")
    print(f"  Size: {model_path.stat().st_size / (1024**3):.2f} GB")
    
    try:
        from rgbddepth.dpt import RGBDDepth
        
        # Create model
        print("Creating RGBDDepth model...")
        model = RGBDDepth(encoder='vitl', features=256)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {param_count:,}")
        
        # Load checkpoint
        print("Loading checkpoint...")
        device = "cpu"  # Use CPU for validation
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("  Using 'state_dict' from checkpoint")
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
            print("  Using 'model' from checkpoint")
        else:
            state_dict = checkpoint
            print("  Using checkpoint directly")
        
        print(f"  Checkpoint keys: {len(state_dict)}")
        
        # Remove module prefix if present
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                cleaned_state_dict[key[7:]] = value
            else:
                cleaned_state_dict[key] = value
        
        # Load state dict
        missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
        if missing_keys:
            print(f"  Missing keys: {len(missing_keys)} (first 5: {missing_keys[:5]})")
        if unexpected_keys:
            print(f"  Unexpected keys: {len(unexpected_keys)} (first 5: {unexpected_keys[:5]})")
        
        model.to(device)
        model.eval()
        
        print("✓ Model loaded successfully")
        return True, model
        
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_cdm_processing():
    """Test CDM processing with dummy data"""
    print("\nTesting CDM processing with dummy data...")
    
    success, model = test_cdm_model_loading()
    if not success:
        return False
    
    try:
        # Create dummy RGB and depth data
        rgb_dummy = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        depth_dummy = np.random.uniform(0.2, 1.5, (720, 1280)).astype(np.float32)
        
        print(f"Dummy RGB shape: {rgb_dummy.shape}")
        print(f"Dummy depth shape: {depth_dummy.shape}")
        print(f"Depth range: {depth_dummy.min():.2f} - {depth_dummy.max():.2f}")
        
        # Prepare tensors
        rgb_tensor = torch.from_numpy(rgb_dummy.copy()).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        
        # Create similarity depth
        simi_depth = np.zeros_like(depth_dummy)
        valid_mask = depth_dummy > 0
        simi_depth[valid_mask] = 1.0 / depth_dummy[valid_mask]
        depth_tensor = torch.from_numpy(simi_depth).float().unsqueeze(0).unsqueeze(0)
        
        # Resize to model input size
        target_size = 518
        with torch.no_grad():
            rgb_resized = torch.nn.functional.interpolate(
                rgb_tensor, size=(target_size, target_size), 
                mode='bilinear', align_corners=False
            )
            depth_resized = torch.nn.functional.interpolate(
                depth_tensor, size=(target_size, target_size), 
                mode='bilinear', align_corners=False
            )
            
            print(f"Resized RGB tensor: {rgb_resized.shape}")
            print(f"Resized depth tensor: {depth_resized.shape}")
            
            # Run inference
            print("Running inference...")
            # Concatenate RGB and depth tensors along channel dimension
            rgbd_input = torch.cat([rgb_resized, depth_resized], dim=1)
            print(f"RGBD input shape: {rgbd_input.shape}")
            enhanced_depth = model(rgbd_input)
            
            print(f"Enhanced depth shape: {enhanced_depth.shape}")
            
            # Add channel dimension if needed
            if len(enhanced_depth.shape) == 3:  # (B, H, W)
                enhanced_depth = enhanced_depth.unsqueeze(1)  # (B, 1, H, W)
            
            # Resize back
            enhanced_depth = torch.nn.functional.interpolate(
                enhanced_depth, size=(720, 1280), 
                mode='bilinear', align_corners=False
            )
            
            # Convert back to numpy
            enhanced_np = enhanced_depth.squeeze().cpu().numpy()
            
            # Convert from similarity back to depth
            enhanced_depth_meters = np.where(enhanced_np > 0, 1.0 / enhanced_np, 0)
            
            print(f"Final enhanced depth shape: {enhanced_depth_meters.shape}")
            print(f"Enhanced depth range: {enhanced_depth_meters[enhanced_depth_meters>0].min():.2f} - {enhanced_depth_meters[enhanced_depth_meters>0].max():.2f}")
            
            print("✓ CDM processing successful")
            return True
            
    except Exception as e:
        print(f"✗ CDM processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_real_data():
    """Test with real data from existing depth files"""
    print("\nTesting with real data...")
    
    # Find real depth data
    depth_files = list(Path("test_cdm_vis/depth").glob("*.npy")) if Path("test_cdm_vis/depth").exists() else []
    rgb_files = list(Path("test_cdm_vis/rgb").glob("*.jpg")) if Path("test_cdm_vis/rgb").exists() else []
    
    if not depth_files or not rgb_files:
        print("✗ No real data found, skipping real data test")
        return False
    
    print(f"Found {len(depth_files)} depth files and {len(rgb_files)} RGB files")
    
    # Load first pair
    depth_file = depth_files[0]
    rgb_file = rgb_files[0]
    
    print(f"Testing with: {rgb_file.name} and {depth_file.name}")
    
    try:
        # Load data
        depth_data = np.load(depth_file)
        rgb_data = cv2.imread(str(rgb_file))
        
        print(f"RGB shape: {rgb_data.shape}")
        print(f"Depth shape: {depth_data.shape}")
        print(f"Depth range: {depth_data[depth_data>0].min():.2f} - {depth_data[depth_data>0].max():.2f}")
        
        # Test CDM processing
        success, model = test_cdm_model_loading()
        if not success:
            return False
        
        # Convert BGR to RGB
        rgb_rgb = cv2.cvtColor(rgb_data, cv2.COLOR_BGR2RGB)
        
        # Process with CDM (simplified version of processor)
        rgb_tensor = torch.from_numpy(rgb_rgb.copy()).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        
        # Create similarity depth
        simi_depth = np.zeros_like(depth_data)
        valid_mask = depth_data > 0
        simi_depth[valid_mask] = 1.0 / depth_data[valid_mask]
        depth_tensor = torch.from_numpy(simi_depth).float().unsqueeze(0).unsqueeze(0)
        
        # Process
        with torch.no_grad():
            target_size = 518
            rgb_resized = torch.nn.functional.interpolate(
                rgb_tensor, size=(target_size, target_size), 
                mode='bilinear', align_corners=False
            )
            depth_resized = torch.nn.functional.interpolate(
                depth_tensor, size=(target_size, target_size), 
                mode='bilinear', align_corners=False
            )
            
            # Concatenate RGB and depth tensors along channel dimension
            rgbd_input = torch.cat([rgb_resized, depth_resized], dim=1)
            enhanced_depth = model(rgbd_input)
            
            # Add channel dimension if needed
            if len(enhanced_depth.shape) == 3:  # (B, H, W)
                enhanced_depth = enhanced_depth.unsqueeze(1)  # (B, 1, H, W)
            
            enhanced_depth = torch.nn.functional.interpolate(
                enhanced_depth, size=(depth_data.shape[0], depth_data.shape[1]), 
                mode='bilinear', align_corners=False
            )
            
            enhanced_np = enhanced_depth.squeeze().cpu().numpy()
            enhanced_meters = np.where(enhanced_np > 0, 1.0 / enhanced_np, 0)
            
            print(f"Enhanced depth range: {enhanced_meters[enhanced_meters>0].min():.2f} - {enhanced_meters[enhanced_meters>0].max():.2f}")
            
            # Save result for inspection
            output_path = "cdm_test_result.npy"
            np.save(output_path, enhanced_meters)
            print(f"✓ Saved enhanced depth to {output_path}")
            
            return True
            
    except Exception as e:
        print(f"✗ Real data test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all CDM validation tests"""
    print("=" * 60)
    print("CDM Processor Validation")
    print("=" * 60)
    
    # Test 1: Imports
    imports_ok = test_cdm_imports()
    
    if not imports_ok:
        print("\n❌ CDM imports failed. Cannot proceed.")
        return 1
    
    # Test 2: Model loading
    model_ok, _ = test_cdm_model_loading()
    
    if not model_ok:
        print("\n❌ CDM model loading failed. Cannot proceed.")
        return 1
    
    # Test 3: Processing with dummy data
    processing_ok = test_cdm_processing()
    
    # Test 4: Processing with real data (optional)
    real_data_ok = test_with_real_data()
    
    print("\n" + "=" * 60)
    print("CDM Validation Summary")
    print("=" * 60)
    print(f"Imports: {'✓' if imports_ok else '✗'}")
    print(f"Model Loading: {'✓' if model_ok else '✗'}")
    print(f"Dummy Data Processing: {'✓' if processing_ok else '✗'}")
    print(f"Real Data Processing: {'✓' if real_data_ok else '✗ (optional)'}")
    
    if imports_ok and model_ok and processing_ok:
        print("\n🎉 CDM processor is ready for integration!")
        return 0
    else:
        print("\n❌ CDM processor has issues. Check errors above.")
        return 1

if __name__ == "__main__":
    exit(main())