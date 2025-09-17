import numpy as np
import cv2
import os
import sys

def visualize_depth(depth_file_path):
    """
    Visualize depth data by putting depth values in red channel * 255
    """
    # Load depth data
    depth_data = np.load(depth_file_path)
    print(f"Depth data shape: {depth_data.shape}")
    print(f"Depth range: [{np.min(depth_data):.3f}, {np.max(depth_data):.3f}]")
    
    # Create turbo colormap visualization
    h, w = depth_data.shape
    # Normalize depth to 0-255 range for colormap
    depth_normalized = np.clip(depth_data * 255, 0, 255).astype(np.uint8)
    # Apply turbo colormap
    vis_img = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)
    display_img = vis_img.copy()
    
    # Mouse callback function to display RGB values on image
    def mouse_callback(event, x, y, flags, param):
        nonlocal display_img
        if event == cv2.EVENT_MOUSEMOVE:
            display_img = vis_img.copy()
            rgb = vis_img[y, x]
            depth_value = depth_data[y, x]
            # Create text to display
            text = f"({x}, {y}) RGB: ({rgb[2]}, {rgb[1]}, {rgb[0]}) Depth: {depth_value:.3f}"
            # Add text to image with background for better visibility
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 0.6, 1)[0]
            # Draw background rectangle
            cv2.rectangle(display_img, (10, 10), (20 + text_size[0], 35), (0, 0, 0), -1)
            # Draw text
            cv2.putText(display_img, text, (15, 30), font, 0.6, (255, 255, 255), 1)
            cv2.imshow(window_name, display_img)
    
    # Display the image
    window_name = 'Depth Visualization (Turbo Colormap)'
    cv2.imshow(window_name, display_img)
    cv2.setMouseCallback(window_name, mouse_callback)
    print("Hover mouse to see RGB values. Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return vis_img

if __name__ == "__main__":
    # Default path
    #depth_file = "./data/bag/0912_topdownbags/images/depth/depth_frame_000000.npy"
    #depth_file = "./data/bag/0912_topdownbags/images/da2/da2_depth_000000.npy"
    depth_file = "./data/bag/0912_topdownbags/images/cdm/cdm_depth_000000.npy"
    
    # Allow custom path from command line
    if len(sys.argv) > 1:
        depth_file = sys.argv[1]
    
    if not os.path.exists(depth_file):
        print(f"Error: Depth file not found at {depth_file}")
        sys.exit(1)
    
    visualize_depth(depth_file)