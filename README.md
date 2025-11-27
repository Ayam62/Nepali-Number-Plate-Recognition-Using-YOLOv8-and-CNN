# Training YOLOv8 on your Nepali license plate dataset

Quick summary of what I added and how to train:

- `vehicle_number_plate_detection/data.yaml` — dataset manifest used by ultralytics YOLO.train
- `main.py` — minimal example entrypoint that calls YOLO.train
- `train.py` — CLI script that validates dataset, can create a quick 80/20 split, and runs training
- `requirements.txt` — minimal Python dependencies

Expected dataset layout (recommended):

vehicle_number_plate_detection/
  images/
    train/
    val/
  labels/
    train/
    val/
  data.yaml

Each image must have a same-named .txt file in the corresponding labels folder using YOLO format:
class x_center y_center width height  (normalized values 0..1)

If you only have `images/` and `labels/` (flat), `train.py` will create an 80/20 split for you automatically.

Install dependencies (zsh):

```bash
python -m pip install -r requirements.txt
```

Run training:

```bash
python train.py --data vehicle_number_plate_detection/data.yaml --epochs 100 --batch 16 --imgsz 640
```

Or quick run with the minimal example:

```bash
python main.py
```

Notes:
- Adjust `nc` and `names` in `data.yaml` if you have more classes.
- For production or better accuracy, prefer `yolov8s.pt` or larger backbones and tune hyperparameters.
