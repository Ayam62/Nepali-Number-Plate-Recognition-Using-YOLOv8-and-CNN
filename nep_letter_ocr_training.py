import os
import cv2
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# --- Configuration ---
DATA_PATH = '/content/drive/MyDrive/character_ocr'  # Main folder name
IMG_HEIGHT = 64
IMG_WIDTH = 64

def pre_process_img(image_path):
    """
    Reads an image from a path and applies the specific preprocessing:
    Resize -> Grayscale -> Otsu Threshold
    """
    # Read the image
    img = cv2.imread(image_path)

    if img is None:
        return None

    # 1. Resize to 64x64
    # Note: cv2.resize expects (width, height)
    resized_img_char = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)

    # 2. Convert to Grayscale
    gray_scaled_img = cv2.cvtColor(resized_img_char, cv2.COLOR_BGR2GRAY)

    # 3. Otsu Thresholding
    ret, binary_image_otsu = cv2.threshold(
        gray_scaled_img,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Normalization (0-1 range)
    binary_image_otsu = binary_image_otsu / 255.0

    return binary_image_otsu

def load_data():
    images = []
    labels = []

    print("Scanning folder for classes...")

    # 1. Automatically find all subfolders (classes)
    if not os.path.exists(DATA_PATH):
        print(f"Error: Directory '{DATA_PATH}' not found.")
        return np.array([]), np.array([]), {}, 0

    # Get list of folders, ignoring hidden files like .DS_Store
    folder_names = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])

    num_classes = len(folder_names)
    print(f"Found {num_classes} classes: {folder_names}")

    # 2. Create a mapping (Folder Name -> Integer ID)
    # Example: {'ka': 0, 'kha': 1, ...}
    class_mapping = {name: idx for idx, name in enumerate(folder_names)}

    print("Starting image loading...")

    for folder_name in folder_names:
        folder_path = os.path.join(DATA_PATH, folder_name)
        class_id = class_mapping[folder_name]

        img_files = os.listdir(folder_path)

        # Limit print output
        print(f"Loading {len(img_files)} images from folder: {folder_name}")

        for file_name in img_files:
            try:
                img_path = os.path.join(folder_path, file_name)

                # Process the image
                processed_img = pre_process_img(img_path)

                if processed_img is not None:
                    images.append(processed_img)
                    labels.append(class_id)
            except Exception as e:
                # Silently skip non-image files or errors to keep output clean
                pass

    print(f"Data loading complete.")
    print(f"Total images: {len(images)}")

    return np.array(images), np.array(labels), class_mapping, num_classes

def build_model(input_shape, num_classes):
    """
    Constructs the model based on your architecture.
    """
    model = Sequential()

    # Layer 1: Conv 32, 3x3 + MaxPool
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    # Layer 2: Conv 64, 3x3 + MaxPool
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    # Layer 3: Conv 128, 3x3
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))

    # Layer 4: Conv 128, 3x3
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))

    # Layer 5: Conv 64, 3x3 + MaxPool
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    # Flatten
    model.add(Flatten())

    # Layer 7: FC 4096
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    # Layer 8: FC 4096
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))

    return model

def main():
    # 1. Load Data
    X, y, class_mapping, num_classes = load_data()

    if len(X) == 0:
        print("No images found. Exiting.")
        return

    # 2. Reshape for CNN (Batch, 64, 64, 1)
    X = X.reshape(X.shape[0], IMG_HEIGHT, IMG_WIDTH, 1)

    # 3. Save the class mapping so you can interpret predictions later
    # This creates a file "class_indices.json" that tells you 0='ka', 1='kha', etc.
    with open('class_indices.json', 'w') as f:
        json.dump(class_mapping, f)
    print("Class mapping saved to 'class_indices.json'")

    # 4. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Build Model
    model = build_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1), num_classes=num_classes)

    # 6. Compile
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 7. Train
    print("\nStarting Training...")
    history = model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=32,
        validation_data=(X_test, y_test)
    )

    # 8. Save Model
    model.save('nepali_character_model.h5')
    print("Model saved as nepali_character_model.h5")

if __name__ == "__main__":
    main()