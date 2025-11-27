
from ultralytics import YOLO
def main():
	# Path to the dataset manifest - adjust if you saved it elsewhere
	data_yaml = "./data.yaml"

	# Load a pretrained YOLOv8 nano model (weights will be downloaded by ultralytics if needed)
	model = YOLO("yolov8n.pt")

	# Train - adjust epochs, imgsz, batch as needed
	model.train(data=data_yaml, epochs=60, imgsz=640, batch=16)


if __name__ == "__main__":
	main()

