# -*- coding: utf-8 -*-
import cv2
import numpy as np
from mmengine.config import Config
from dino_any_percept_api import DetectionAPI

def test_detection_api():
    """Test DetectionAPI on sample.bag.png with prompt 'handle.bag' and visualize results"""
    
    # Configure DetectionAPI
    cfg = Config()
    cfg.uri = r'/v2/task/dinox/detection'
    cfg.status_uri = r'/v2/task_status'  
    cfg.token = 'c4cdacb48bc4d1a1a335c88598a18e8c'
    cfg.model_name = 'DINO-X-1.0'
    
    # Initialize DetectionAPI
    print("Initializing DetectionAPI...")
    detection_api = DetectionAPI(cfg)
    
    # Read test image
    img_path = 'sample.bag.png'
    print(f"Reading image: {img_path}")
    rgb = cv2.imread(img_path)
    if rgb is None:
        print(f"Error: Cannot read image {img_path}")
        return
    
    print(f"Image shape: {rgb.shape}")
    
    # Set detection parameters
    prompt_text = "handle.bag"
    bbox_threshold = 0.25
    iou_threshold = 0.8
    
    print(f"Using prompt: '{prompt_text}'")
    print(f"Detection thresholds: bbox_threshold={bbox_threshold}, iou_threshold={iou_threshold}")
    
    try:
        # Execute detection
        print("Executing detection...")
        result = detection_api.detect_objects(
            rgb=rgb,
            prompt_text=prompt_text,
            bbox_threshold=bbox_threshold,
            iou_threshold=iou_threshold
        )
        
        print("Detection completed!")
        print(f"Detection result: {result}")
        
        # Check results
        if result and 'objects' in result:
            objects = result['objects']
            print(f"Detected {len(objects)} objects")
            
            for i, obj in enumerate(objects):
                bbox = obj.get('bbox', [])
                score = obj.get('score', 0.0)
                category = obj.get('category', 'unknown')
                print(f"  Object {i+1}: category={category}, score={score:.3f}, bbox={bbox}")
            
            # Visualize detection results (bounding boxes)
            print("Generating detection visualization...")
            detection_img = detection_api.visualize_detection_results(
                rgb=rgb, 
                detection_result=result, 
                save_path='detection_result_sample_bag.jpg'
            )
            
            # Visualize segmentation results (masks)
            print("Generating segmentation visualization...")
            segmentation_img = detection_api.visualize_segmentation_results(
                rgb=rgb,
                detection_result=result,
                save_path='segmentation_result_sample_bag.jpg'
            )
            
            print("Visualization completed! Results saved to detection_result_sample_bag.jpg and segmentation_result_sample_bag.jpg")
            
        else:
            print("No objects detected")
            
    except Exception as e:
        print(f"Error during detection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_detection_api()