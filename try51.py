import os
import sys
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Polygon
import numpy as np

try:
    from pycocotools import mask as coco_mask
    from pycocotools.coco import COCO
    HAS_PYCOCOTOOLS = True
except ImportError:
    print("Warning: pycocotools not available, will use basic JSON loading")
    HAS_PYCOCOTOOLS = False

debug = False


fp = r"ann2/val.C2.mo.rle.shape.json"
fp = "ann2/val.C2.mo.rle.shape.fs.aligned.json"
fp = r"ann2/grasp_shot1k.coco.rle.C100.json"
fp = r"ann3/all_s100.shape.rbc0.75.fix.json"
fp = r"ann3/val.C100.json"
fp = r"data/bag/0912/anno/backfront_01.json"
# fp = r"/comp_robot/dino-3D/grasp/Grasp-Anything/ann3/all_s100.shape.rbc0.75.rbc0.75.fix.json"

data_root='data'


if not os.path.isfile(fp) or not os.path.isdir(data_root):
    print(f"{fp} not found or {data_root} not found")
    sys.exit(1)

width, height = 416, 416
def visualize_annotations(fp, data_root, width=416, height=416, max_samples=5):
    """Visualize COCO annotations using matplotlib instead of FiftyOne"""
    
    if not HAS_PYCOCOTOOLS:
        with open(fp, 'r') as f:
            data = json.load(f)
        images = data.get('images', [])
        annotations = data.get('annotations', [])
        print(f"Loaded {len(images)} images and {len(annotations)} annotations")
        return
    
    coco = COCO(annotation_file=fp)
    
    # Get first few images for visualization
    img_ids = list(coco.imgs.keys())[:max_samples]
    
    fig, axes = plt.subplots(1, min(len(img_ids), 3), figsize=(15, 5))
    if len(img_ids) == 1:
        axes = [axes]
    
    for idx, img_id in enumerate(img_ids[:3]):
        if idx >= len(axes):
            break
            
        info = coco.imgs[img_id]
        img_path = os.path.join(data_root, info["file_name"])
        
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
            
        ax = axes[idx] if len(axes) > 1 else axes[0]
        
        # Load and display image
        try:
            img = plt.imread(img_path)
            ax.imshow(img)
        except Exception as e:
            print(f"Could not load image {img_path}: {e}")
            continue
        
        # Get annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # Draw bounding boxes and affordances
        for i, ann in enumerate(anns):
            # Get category id from cat field (fallback to category_id if cat field doesn't exist)
            cat_id = ann.get("cat", ann.get("category_id", 1))
            cat_info = coco.loadCats(cat_id)[0] if cat_id in coco.cats else {"name": "unknown"}
            cat = cat_info["name"]
            
            # Draw bounding box
            if "bbox" in ann:
                x, y, w, h = ann["bbox"]
                rect = Rectangle((x, y), w, h, linewidth=2, 
                               edgecolor='red', facecolor='none', alpha=0.7)
                ax.add_patch(rect)
                ax.text(x, y-5, cat, fontsize=8, color='red', 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
            
            # Draw affordances if available
            if "affordance" in ann:
                affs = ann["affordance"]
                for j, aff in enumerate(affs):
                    if len(aff) >= 5:
                        xc, yc, w2, h2, theta = aff[:5]
                        # Create rotated rectangle for affordance
                        cos_theta = np.cos(theta)
                        sin_theta = np.sin(theta)
                        
                        # Calculate rectangle corners relative to center
                        dx = w2 / 2
                        dy = h2 / 2
                        corners = np.array([
                            [-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy], [-dx, -dy]
                        ])
                        
                        # Apply rotation
                        rotation_matrix = np.array([[cos_theta, -sin_theta], 
                                                   [sin_theta, cos_theta]])
                        rotated_corners = corners @ rotation_matrix.T
                        
                        # Translate to center position
                        rotated_corners[:, 0] += xc
                        rotated_corners[:, 1] += yc
                        
                        # Draw the rotated rectangle
                        ax.plot(rotated_corners[:, 0], rotated_corners[:, 1], 
                               'g-', linewidth=2, alpha=0.8, label='affordance' if j == 0 else '')
                        
                        # Draw center point
                        ax.plot(xc, yc, 'go', markersize=4, alpha=0.8)
        
        ax.set_title(f"Image {idx+1}: {os.path.basename(info['file_name'])}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('annotation_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()  # Close instead of show to avoid blocking
    
    print(f"Processed {len(img_ids)} images from {fp}")
    print("Visualization saved as annotation_visualization.png")

if __name__ == "__main__":
    dn = os.path.basename(fp)
    print(f"Processing dataset: {dn}")
    
    visualize_annotations(fp, data_root, width, height)