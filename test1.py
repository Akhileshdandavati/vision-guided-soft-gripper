"""
Food/Item Detection using YOLOv8 (VS Code version)
Author: Akhilesh Dandavati
Description:
 - Uses YOLOv8 pre-trained model to detect objects via webcam
 - Displays bounding boxes and labels in real-time
 - Optional: Sends detection data (object name + coordinates) to robot over UDP
"""

# ------------------- IMPORTS -------------------
from ultralytics import YOLO
import cv2
import socket
import json

# ------------------- SETTINGS -------------------
# Load pre-trained YOLOv8 model (general object detection)
model = YOLO("yolov8n.pt")   # pre-trained on COCO dataset

# Camera index: 0 = default webcam; change if needed
CAMERA_INDEX = 0

# Optional: Robot UDP setup (set to False if not using)
SEND_TO_ROBOT = False
ROBOT_IP = "192.168.1.10"   # Change to your robot or PC IP
ROBOT_PORT = 5000

# Initialize socket if sending data
if SEND_TO_ROBOT:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ------------------- MAIN PROGRAM -------------------
def main():
    # Open camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ Camera not found or can't be opened.")
        return

    print("✅ Camera started. Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame.")
            break

        # Run YOLO detection
        results = model(frame, verbose=False)

        # Parse results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0]
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                # Draw bounding box & info
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                # Print results
                print(f"Detected {label} at ({cx}, {cy}) | Confidence: {conf:.2f}")

                # Optional: send data to robot
                if SEND_TO_ROBOT:
                    data = {"object": label, "confidence": conf, "cx": cx, "cy": cy}
                    sock.sendto(json.dumps(data).encode(), (ROBOT_IP, ROBOT_PORT))

        # Show camera window
        cv2.imshow("YOLOv8 Food Detection", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    if SEND_TO_ROBOT:
        sock.close()
    print("\n✅ Program ended successfully.")

# ------------------- ENTRY POINT -------------------
if __name__ == "__main__":
    main()
