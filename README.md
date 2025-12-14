# Nepali License Plate Detection & Recognition

This project is an end-to-end system for detecting Nepali license plates from vehicle images and recognizing the characters (OCR) on them. It utilizes **YOLOv8** for object detection (locating the plate and characters) and a custom **CNN (Convolutional Neural Network)** for recognizing specific Nepali characters.

##  Project Structure

```text
├── vehicle_number_plate_detection/ # Dataset for YOLOv8 (Plate Detection)
├── open_source_nepali_plates/      # Dataset for YOLOv8 (Character Segmentation)
├── character_ocr/                  # Dataset for CNN (Character Recognition)
├── models/                         # Saved models (best.pt, nepali_character_model.h5)
├── main.py                         # Main inference script
├── train.py                        # Script to train YOLOv8 (with auto-split)
├── bounding_box_training.py        # Minimal YOLOv8 training script
├── nep_letter_ocr_training.py      # Script to train CNN for OCR
├── utils.py                        # Image preprocessing utilities
├── model_prediction.py             # Helper for CNN inference
└── requirements.txt                # Python dependencies
```

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Ayam62/Nepali-Number-Plate-Recognition-Using-YOLOv8-and-CNN.git
    cd Nepali-Number-Plate-Recognition-Using-YOLOv8-and-CNN
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

##  Training

### 1. Train YOLOv8 (Plate/Character Detection)
To train the object detection model to find plates or characters:

1.  Prepare your dataset in YOLO format (images and txt labels).
2.  Update `data.yaml` with your paths.
3.  Run the training script:
    ```bash
    python train.py --data open_source_nepali_plates/data.yaml --epochs 50
    ```

### 2. Train CNN (Character Recognition)
To train the OCR model on individual Nepali characters:

2.  Update `DATA_PATH` in `nep_letter_ocr_training.py`.
3.  Run the training script:
    ```bash
    python nep_letter_ocr_training.py
    ```
    *This saves `nepali_character_model.h5` and `class_indices.json`.*

### Alternative: Minimal Training Script
If you prefer a simple script without command-line arguments, use `bounding_box_training.py`.

1.  Open `bounding_box_training.py` and ensure the `data_yaml` variable points to your config file.
2.  Run the script:
    ```bash
    python bounding_box_training.py
    ```

##  Inference (Running the System)

To run the full pipeline on a folder of images:

1.  Ensure you have your trained models (`best.pt` and `nepali_character_model.h5`) ready in the `models/` directory.
2.  Update the `IMAGE_DIR` path in `main.py` to point to your test images.
3.  Run the script:
    ```bash
    python main.py
    ```

### How it works:
1.  **YOLOv8** detects the bounding boxes for characters on the plate.
2.  The image is cropped to the character's bounding box.
3.  **Preprocessing** (Resize -> Grayscale -> Otsu Threshold) is applied via `utils.py`.
4.  The **CNN** predicts the character class.
5.  The result is printed to the console.

##  Model Architecture

### OCR Model (CNN)
*   **Input:** 64x64 Grayscale images
*   **Layers:**
    *   **Conv2D:** Feature extraction with ReLU activation.
    *   **MaxPooling2D:** Downsampling to reduce dimensionality.
    *   **Flatten:** Converts 2D matrices to a 1D vector.
    *   **Dense (Fully Connected):** Classification layers with Dropout (0.5) to prevent overfitting.
    *   **Output Layer:** Softmax activation for multi-class classification.

##  Requirements
*   Python 3.8+
*   Ultralytics (YOLOv8)
*   TensorFlow / Keras
*   OpenCV
*   NumPy
*   Scikit-learn

##  Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

##  License
[MIT](https://choosealicense.com/licenses/mit/)
