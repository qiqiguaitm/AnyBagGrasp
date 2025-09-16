import os
import sys

import fiftyone as fo
from fiftyone.utils.coco import COCODetectionDatasetImporter
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

# Configuration
debug = False
data_root = 'data'

# Annotation file path - change this to visualize different datasets
# fp = r"data/bag/0912/anno/backfront_01.json"
fp = r"data/bag/0915_layerbags/anno/backfront_grasp01.json"

# Validation
if not os.path.isfile(fp) or not os.path.isdir(data_root):
    print(f"Error: {fp} not found or {data_root} not found")
    sys.exit(1)

# Dataset management
dataset_name = os.path.basename(fp)
existing_datasets = fo.list_datasets()
if dataset_name in existing_datasets:
    fo.delete_dataset(dataset_name)

# Visualization mode (0: built-in COCO support, 1: custom with affordance support)
vis_mode = 1

if vis_mode == 0:  # Built-in COCO support (simple but cannot visualize affordance)
    data_dir = "."
    images_path = f"{data_dir}/image"
    annotations_path = f"{data_dir}/{fp}"

    importer = COCODetectionDatasetImporter(
        dataset_dir=data_dir,
        data_path=data_dir,
        labels_path=annotations_path,
        include_id=True,
        shuffle=False,
    )

    ds = fo.Dataset.from_importer(importer, name=dataset_name)

elif vis_mode == 1:  # Custom visualization with affordance support
    # Configuration
    max_samples = 200  # Maximum number of samples to load (set to None for all)
    
    # Create dataset
    ds = fo.Dataset(dataset_name)
    coco = COCO(annotation_file=fp)
    
    # Process each image in the COCO dataset
    for idx, (img_id, info) in enumerate(coco.imgs.items()):
        # Get image dimensions
        width = info.get("width", 416)
        height = info.get("height", 416)
        
        # Create sample with image path
        image_path = os.path.join(data_root, info["file_name"])
        sample = fo.Sample(filepath=image_path, text=info.get("text", ""))
        
        # Load annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # Collections for detections and shapes
        detections = []
        tags = []
        shapes = []
        
        # Process each annotation
        for ann_idx, ann in enumerate(anns):
            # Get category information
            cat_id = ann["cat"]
            if cat_id in coco.cats:
                category = coco.cats[cat_id]["name"]
            else:
                category = str(cat_id)
            tags.append(category)

            # Extract bounding box (COCO format: x, y, width, height in pixels)
            x, y, w, h = ann["bbox"]
            # Convert to normalized coordinates for FiftyOne
            x_norm, y_norm, w_norm, h_norm = x / width, y / height, w / width, h / height
            x_int, y_int, w_int, h_int = int(x), int(y), int(w), int(h)
            
            # Extract mask patch
            mask = coco_mask.decode(ann["segmentation"])
            mask_patch = mask[y_int : y_int + h_int + 1, x_int : x_int + w_int + 1]
            
            # Create detection with mask
            detection = fo.Detection(
                bounding_box=(x_norm, y_norm, w_norm, h_norm), 
                mask=mask_patch, 
                label=category
            )
            detections.append(detection)

            # Process affordances (grasp points)
            affordances = ann["affordance"]
            rotated_boxes = []
            for aff_idx, affordance in enumerate(affordances):
                # Extract rotated bbox parameters (x_center, y_center, width, height, theta)
                xc, yc, w_aff, h_aff, theta = affordance
                # Create rotated box polyline for visualization
                rot_box = fo.Polyline.from_rotated_box(
                    xc, yc, w_aff, h_aff, theta, 
                    frame_size=(width, height), 
                    label=category
                )
                rotated_boxes.append(rot_box)
            
            # Add affordances as a separate field
            sample[f"aff{ann_idx}"] = fo.Polylines(polylines=rotated_boxes)

            # Process shape annotations if available
            if "shapes" in ann or "shape" in ann:
                offset = 0
                shape_list = [ann["shape"]] if "shape" in ann else ann["shapes"]
                
                for shape_data in shape_list:
                    # Decode shape mask
                    rle = shape_data["guess_mask"]
                    shape_name = shape_data["name"]
                    
                    # Calculate expanded bounding box for shape
                    x3 = max(x_int - offset, 0)
                    x4 = min(x_int + w_int + 1 + offset, width)
                    y3 = max(y_int - offset, 0)
                    y4 = min(y_int + h_int + 1 + offset, height)

                    # Decode and crop shape mask
                    decoded_mask = coco_mask.decode(rle)
                    shape_mask = decoded_mask[y3:y4, x3:x4]
                    
                    # Create shape detection
                    shape_detection = fo.Detection(
                        bounding_box=(
                            x3 / width,
                            y3 / height,
                            (x4 - x3) / width,
                            (y4 - y3) / height,
                        ),
                        mask=shape_mask,
                        label=shape_name,
                    )
                    shapes.append(shape_detection)
        
        # Add all annotations to sample
        if len(shapes) > 0:
            sample["shape"] = fo.Detections(detections=shapes)
        sample["bbox"] = fo.Detections(detections=detections)
        sample.tags = tags
        ds.add_sample(sample)

        # Check if we've reached the maximum number of samples
        if max_samples and idx >= max_samples:
            break
        if debug and idx > 10:
            break

# Launch FiftyOne app
print(f"\n{'='*50}")
print(f"Loading dataset: {dataset_name}")
print(f"Total samples: {len(ds)}")
print(f"{'='*50}\n")

print("Launching FiftyOne app...")
session = fo.launch_app(ds, port=5151)  # Local app on port 5151
print(f"✓ App launched successfully!")
print(f"✓ Open your browser and navigate to: http://localhost:5151")
print(f"\nPress Ctrl+C to stop the app...")

# Keep the app running
session.wait()