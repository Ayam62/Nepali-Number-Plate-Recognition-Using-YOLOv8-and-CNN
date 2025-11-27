# """Robust training script for YOLOv8 (ultralytics).

# Usage examples (macOS zsh):
#   python train.py --data vehicle_number_plate_detection/data.yaml --epochs 100 --batch 16

# This script validates that the train/val folders exist and provides a
# small helper to create a quick 80/20 train/val split if you only have
# one images/ + labels/ pair.
# """

# import argparse
# import os
# import random
# import shutil
# from pathlib import Path

# from ultralytics import YOLO


# def check_and_prepare(data_yaml: str):
#     """Ensure the paths in data_yaml exist. If train/val folders don't exist,
#     offer to create a simple split from `images/` and `labels/`.
#     """
#     import yaml

#     with open(data_yaml, "r") as f:
#         data = yaml.safe_load(f)

#     train_p = Path(data.get("train"))
#     val_p = Path(data.get("val"))

#     # If both train and val are present, assume OK
#     if train_p.exists() and val_p.exists():
#         return True

#     # Try fallback: single images/ and labels/ at parent
#     parent = Path(data_yaml).parent
#     images_dir = parent / "images"
#     labels_dir = parent / "labels"

#     if images_dir.exists() and labels_dir.exists():
#         # create train/val split directories
#         print(f"Creating train/val split from {images_dir} (80/20)")
#         img_files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in [".jpg", ".png", ".jpeg"]])
#         random.shuffle(img_files)
#         split = int(len(img_files) * 0.8)
#         train_imgs = img_files[:split]
#         val_imgs = img_files[split:]

#         def copy_subset(img_list, subset_name):
#             dest_img = images_dir / subset_name
#             dest_lbl = labels_dir / subset_name
#             dest_img.mkdir(parents=True, exist_ok=True)
#             dest_lbl.mkdir(parents=True, exist_ok=True)
#             for img_path in img_list:
#                 label_name = img_path.with_suffix('.txt').name
#                 shutil.copy(img_path, dest_img / img_path.name)
#                 lbl_src = labels_dir / label_name
#                 if lbl_src.exists():
#                     shutil.copy(lbl_src, dest_lbl / label_name)

#         copy_subset(train_imgs, "train")
#         copy_subset(val_imgs, "val")

#         # Update data_yaml to point to new folders
#         data["train"] = str(images_dir / "train")
#         data["val"] = str(images_dir / "val")
#         with open(data_yaml, "w") as f:
#             yaml.safe_dump(data, f)

#         print("Created train/ and val/ directories and updated data.yaml")
#         return True

#     print("Could not find required image/label folders. Please create the following structure:\n"
#           "vehicle_number_plate_detection/images/train, images/val and corresponding labels/train, labels/val")
#     return False


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data", default="open_source_nepali_plates/data.yaml", help="Path to data.yaml")
#     parser.add_argument("--epochs", type=int, default=50)
#     parser.add_argument("--batch", type=int, default=16)
#     parser.add_argument("--imgsz", type=int, default=640)
#     parser.add_argument("--weights", default="yolov8n.pt")
#     args = parser.parse_args()

#     if not Path(args.data).exists():
#         raise SystemExit(f"Data manifest not found: {args.data}")

#     # Lazy import yaml here; ensure requirements installed
#     try:
#         import yaml
#     except Exception as e:
#         raise SystemExit("Please install pyyaml (pip install pyyaml) to use this script")

#     ok = check_and_prepare(args.data)
#     if not ok:
#         raise SystemExit("Dataset not ready. See the README for required structure.")

#     model = YOLO(args.weights)

#     print(f"Starting training with data={args.data}, epochs={args.epochs}, batch={args.batch}, imgsz={args.imgsz}")
#     model.train(data=args.data, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz)


# if __name__ == "__main__":
#     main()
