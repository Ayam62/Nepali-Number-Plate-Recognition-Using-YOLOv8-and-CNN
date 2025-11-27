

from ultralytics import YOLO
import cv2
import numpy as np
from utils import pre_process_img
import tensorflow as tf
from model_prediction import predict_output
import os


def main():
	# Path to the dataset manifest - adjust if you saved it elsewhere
    data_yaml = "vehicle_number_plate_detection/data.yaml"
	# Load a pretrained YOLOv8 nano model (weights will be downloaded by ultralytics if needed)
 
coco_model = YOLO("yolov8n.pt")

license_plate_detection=YOLO("/Users/ayamkattel/Desktop/YOLO_PROJECTS/Nepali_Liscence_Plate/models/best.pt")

#instance of one image 
# license_plate_labels = license_plate_detection("/Users/ayamkattel/Desktop/YOLO_PROJECTS/Nepali_Liscence_Plate/Plates_Data/Images/train/2021-03-23_05_49_30.jpg")
# img_path="/Users/ayamkattel/Desktop/YOLO_PROJECTS/Nepali_Liscence_Plate/Plates_Data/Images/train/2021-03-23_05_49_30.jpg"
# img=cv2.imread(img_path)
IMAGE_DIR = "/Users/ayamkattel/Desktop/YOLO_PROJECTS/Nepali_Liscence_Plate/Plates_Data/Images/val"


count=0
for filename in os.listdir(IMAGE_DIR):
    count+=1
    if(count==20):
        break
    # Create the full path to the file
    img_path = os.path.join(IMAGE_DIR, filename)
    license_plate_labels=license_plate_detection(img_path)
    number_plate_list=[]
    
    
    img=cv2.imread(img_path)
    for i in range(5):
        x1,y1,x2,y2=license_plate_labels [0][i].boxes.xyxy.tolist()[0] # bounding box
        number_roi=img[int(y1):int(y2),int(x1):int(x2)]
        if number_roi is None or number_roi.size == 0 or number_roi.shape[0] == 0 or number_roi.shape[1] == 0:
            print("⚠️ Warning: Skipping invalid or empty character ROI due to bad coordinates.")
            continue # Skip the rest of the loop for this bad character/ROI
        binary_img_otsu=pre_process_img(number_roi)
        
        character=predict_output(binary_img_otsu)
        number_plate_list.append(character)
    print(number_plate_list)
    print("\n")
    print("\n")
    print("_____________________________________________")
        
    # print(license_plate_labels [0][i].boxes.conf)   # confidence score
    # print(license_plate_labels [0][i].boxes.cls)    # class id



if __name__ == "__main__":
	main()

