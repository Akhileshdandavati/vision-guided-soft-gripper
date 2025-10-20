"""
Filtered Food Detection with PLC Communication
Author: Akhilesh Dandavati

Description:
 - Detects food items using your custom YOLOv8 model
 - Filters detections based on dataset classes (yolo.yaml)
 - Looks up pressure mapping from item_data.json
 - Sends detected item index and pressure to Allen-Bradley PLC (EtherNet/IP)
"""

from ultralytics import YOLO
from pylogix import PLC
import cv2
import yaml
import json
import os
import time

# ------------------ USER SETTINGS ------------------
MODEL_PATH = "yolov8s.pt"        # your trained YOLO model
DATASET_YAML = "yolo.yaml"       # dataset with class names
ITEM_DATA_FILE = "item_data.json" # custom item-pressure mapping
CAMERA_INDEX = 0                 # webcam index
CONF_THRESHOLD = 0.6             # minimum detection confidence
PLC_IP = "192.168.1.20"          # Allen-Bradley PLC IP address
SEND_TO_PLC = False               # set False to test without PLC
# ----------------------------------------------------

# ------------------ LOAD MODEL ------------------
print("Loading model and dataset...")
model = YOLO(MODEL_PATH)

# Load dataset-defined classes
if os.path.exists(DATASET_YAML):
    with open(DATASET_YAML, "r") as f:
        data = yaml.safe_load(f)
        dataset_classes = data.get("names", [])
else:
    dataset_classes = list(model.names.values())

print(f"Model loaded: {MODEL_PATH}")
print(f"Dataset classes: {dataset_classes}")

# ------------------ LOAD CUSTOM ITEM DATA ------------------
if not os.path.exists(ITEM_DATA_FILE):
    print(f"{ITEM_DATA_FILE} not found. Please create it (object: pressure).")
    exit()

with open(ITEM_DATA_FILE, "r") as f:
    item_data = json.load(f)

print(f"Loaded item-pressure map: {item_data}\n")

# Create item order (index reference)
item_names = list(item_data.keys())

# ------------------ DETECTION FUNCTION ------------------
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Camera not found or can't be opened.")
        return

    print("🎥 Camera started. Press 'q' to quit.\n")
    detected_items = []

    # optional: establish PLC connection
    plc = PLC() if SEND_TO_PLC else None
    if plc:
        plc.IPAddress = PLC_IP
        print(f"🔌 Connected to PLC at {PLC_IP}\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=CONF_THRESHOLD, verbose=False)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                conf = float(box.conf[0])

                if label not in dataset_classes:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}",
                            (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Add to list if new
                if label not in detected_items:
                    detected_items.append(label)
                    print(f"New item detected: {label}")

                    # Get pressure from mapping (default 50)
                    pressure = item_data.get(label, 50)
                    item_index = item_names.index(label) + 1 if label in item_names else 0

                    # Send to PLC
                    if SEND_TO_PLC and item_index > 0:
                        plc.Write('Vision_Item_Index', item_index)
                        plc.Write('Vision_Pressure', float(pressure))
                        plc.Write('Vision_NewData', True)
                        print(f"Sent to PLC: {label} | Index={item_index} | Pressure={pressure} kPa\n")
                        time.sleep(0.5)
                        plc.Write('Vision_NewData', False)

        cv2.imshow("YOLOv8 Food Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if plc:
        plc.Close()

    print("\nProgram ended successfully.")
    print("Final Detected Items:")
    for i, item in enumerate(detected_items, start=1):
        pressure = item_data.get(item, 50)
        print(f"[{i}] {item}  ->  {pressure} kPa")

if __name__ == "__main__":
    main()
