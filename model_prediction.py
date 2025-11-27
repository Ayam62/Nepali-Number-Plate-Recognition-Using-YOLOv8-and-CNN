import tensorflow as tf
import numpy as np
import os # Necessary if you use file system operations
from utils import pre_process_img

# Load the entire model
def predict_output(processed_img):
    try:
        model = tf.keras.models.load_model('/Users/ayamkattel/Desktop/YOLO_PROJECTS/Nepali_Liscence_Plate/models/nepali_character_model.h5')
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
    # Assuming 'test_image_path' points to the new image file

    # Assuming pre_process_img is defined as it was during your training script

    # The model expects a batch of images, even if it's just one.
    # We add an extra dimension (batch dimension) to the image array.
    # Example: (28, 28, 1) -> (1, 28, 28, 1)
    processed_img = np.expand_dims(processed_img, axis=0)
    
    class_mapping={
    'क': 0, 'को': 1, 'ख': 2, 'ग': 3, 'च': 4, 'ज': 5,
    'झ': 6, 'ञ': 7, 'डि': 8, 'त': 9, 'ना': 10, 'प': 11,
    'प्र': 12, 'ब': 13, 'बा': 14, 'भे': 15, 'म': 16, 'मे': 17,
    'य': 18, 'लु': 19, 'सी': 20, 'सु': 21, 'से': 22, 'ह': 23,
    '०': 24, '१': 25, '२': 26, '३': 27, '४': 28, '५': 29,
    '६': 30, '७': 31, '८': 32, '९': 33
    }
    reverse_mapping = {v: k for k, v in class_mapping.items()}
    # 1. Run the prediction
    predictions = model.predict(processed_img)

    # 2. Get the index (ID) of the highest probability
    predicted_class_id = np.argmax(predictions)

    # 3. Use the reverse mapping to get the character
    predicted_character = reverse_mapping[predicted_class_id]

    # print(f"\n--- Inference Result ---")
    # print(f"Predicted Class ID: {predicted_class_id}")
    # print(f"Predicted Character: **{predicted_character}**")
    # print(f"Confidence Score (Probability): {predictions[0][predicted_class_id]:.4f}")
    return predicted_character
    
    
    
        