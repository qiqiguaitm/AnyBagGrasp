import os
import sys

import fiftyone as fo
from fiftyone.utils.coco import COCODetectionDatasetImporter
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

debug = False



fp = r"data/bag/0912/anno/backfront_01.json"
fp = r"data/bag/0915_layerbags/anno/backfront_grasp01.json"
# fp = r"/comp_robot/dino-3D/grasp/Grasp-Anything/ann3/all_s100.shape.rbc0.75.rbc0.75.fix.json"

data_root='data'


if not os.path.isfile(fp) or not os.path.isdir(data_root):
    print(f"{fp} not found or {data_root} not found")
    sys.exit(1)

dn = os.path.basename(fp)
existing = fo.list_datasets()
if dn in existing:
    fo.delete_dataset(dn)


if 0:  # built-in support for COCO, simple but cannot visualize afforadance
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

    ds = fo.Dataset.from_importer(importer, name=dn)


if 1:  #
    shot = 200
    debug = False
    ds = fo.Dataset(dn)
    coco = COCO(annotation_file=fp)
    for idx, (img_id, info) in enumerate(coco.imgs.items()):
        # Get dimensions from image info, fallback to defaults if not present
        width = info.get("width", 416)
        height = info.get("height", 416)
        
        fp = os.path.join(data_root, info["file_name"])
        sample = fo.Sample(filepath=fp, text=info.get("text", ""))
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        dts = []
        tags = []
        shps = []
        for i, ann in enumerate(anns):
            cat_id = ann["cat"]
            # Convert category ID to name if available
            if cat_id in coco.cats:
                cat = coco.cats[cat_id]["name"]
            else:
                cat = str(cat_id)  # Fallback to string ID if name not found
            tags.append(cat)

            # for bbox and mask
            x, y, w, h = ann["bbox"]  # xywh in pixel
            x1, y1, w1, h1 = x / width, y / height, w / width, h / height
            x, y, w, h = int(x), int(y), int(w), int(h)
            mask = coco_mask.decode(ann["segmentation"])
            mask_patch = mask[y : y + h + 1, x : x + w + 1]
            dt = fo.Detection(bounding_box=(x1, y1, w1, h1), mask=mask_patch, label=cat)
            dts.append(dt)

            # for affordance
            affs = ann["affordance"]
            rbs = []
            for j in range(len(affs)):
                xc, yc, w2, h2, theta = affs[j]  # xc,y, w, h in pixel, theta in radian
                theta = (
                    theta  # Positive angle means rotate anti-clockwise from horizontal.
                )
                rb = fo.Polyline.from_rotated_box(
                    xc, yc, w2, h2, theta, frame_size=(width, height), label=cat
                )
                rbs.append(rb)
            sample[f"aff{i}"] = fo.Polylines(polylines=rbs)

            # for shape

            if "shapes" in ann or "shape" in ann:
                offset = 0
                if "shape" in ann:
                    shapes = [ann["shape"]]
                else:
                    shapes = ann["shapes"]
                for shape in shapes:
                    rle = shape["guess_mask"]
                    x3 = x - offset
                    x3 = max(x3, 0)
                    x4 = x + w + 1 + offset
                    x4 = min(x4, width)
                    y3 = y - offset
                    y3 = max(y3, 0)
                    y4 = y + h + 1 + offset
                    y4 = min(y4, height)

                    guess_mask = coco_mask.decode(rle)[y3:y4, x3:x4]
                    # guess_mask = coco_mask.decode(rle)
                    shape_name = shape["name"]
                    fill = fo.Detection(
                        bounding_box=(
                            x3 / width,
                            y3 / height,
                            (x4 - x3) / width,
                            (y4 - y3) / height,
                        ),
                        mask=guess_mask,
                        label=shape_name,
                    )
                    # print("add shape")
                    shps.append(fill)
        if len(shps) > 0:
            sample["shape"] = fo.Detections(detections=shps)
        sample["bbox"] = fo.Detections(detections=dts)
        sample.tags = tags
        ds.add_sample(sample)

        if shot > 0 and idx >= shot:
            break
        if idx > 10 and debug:
            break


print(f"to start app: {dn}")
session = fo.launch_app(ds, port=5151)  # Local app on port 5151, will auto-open browser
print(f"started app: {dn}")
session.wait()