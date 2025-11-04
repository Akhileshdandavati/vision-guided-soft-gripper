"""Vision-Based Object Presence Detection with Modbus Communication.

This script runs a YOLOv8 model on CPU to determine whether *any* object is
present in the camera feed. If the confidence threshold is met for at least one
bounding box, a boolean coil is written to a PLC over Modbus TCP. The coil can
be used to trigger downstream automation, such as a robot routine.

Author: Akhilesh Dandavati
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
from pymodbus.client import ModbusTcpClient
from ultralytics import YOLO

# ----------------------------- CONFIGURATION -----------------------------
MODEL_PATH = "yolov8s.pt"        # YOLO model path
CAMERA_INDEX = 0                 # Camera index (0 for default webcam)
CONF_THRESHOLD = 0.6             # Minimum detection confidence
PLC_IP = "127.0.0.7"             # PLC IP address
PLC_PORT = 502                   # Modbus TCP port (default: 502)
PLC_COIL_ADDRESS = 1             # Coil/register address for writing detection state
SEND_TO_PLC = True               # Enable/disable PLC communication
# -------------------------------------------------------------------------


def initialize_model() -> YOLO:
    """Load the YOLO model in CPU mode."""
    print("Loading YOLO model on CPU...")
    model = YOLO(MODEL_PATH)
    print(f"Model loaded successfully (CPU mode): {MODEL_PATH}")
    return model


def initialize_camera() -> cv2.VideoCapture:
    """Initialize the camera feed."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Camera not found or could not be opened.")
    print("Camera initialized successfully.")
    return cap


def initialize_plc() -> Optional[ModbusTcpClient]:
    """Establish a Modbus TCP connection to the PLC if enabled."""
    if not SEND_TO_PLC:
        return None

    plc = ModbusTcpClient(PLC_IP, port=PLC_PORT)
    if plc.connect():
        print(f"Connected to PLC at {PLC_IP}:{PLC_PORT}")
        return plc
    msg = f"Unable to connect to PLC at {PLC_IP}:{PLC_PORT}"
    raise ConnectionError(msg)


def update_plc(plc: ModbusTcpClient, state: bool) -> None:
    """Write the presence state to the configured Modbus coil."""
    if not SEND_TO_PLC or plc is None:
        return

    try:
        plc.write_coil(PLC_COIL_ADDRESS, state)
        print(f"PLC updated: Coil {PLC_COIL_ADDRESS} -> {int(state)}")
    except Exception as exc:  # pragma: no cover - depends on hardware
        print(f"PLC write error: {exc}")


def main() -> None:
    """Main execution loop."""
    model = initialize_model()
    cap = initialize_camera()

    try:
        plc = initialize_plc()
    except Exception as exc:  # pragma: no cover - depends on hardware
        print(f"PLC connection error: {exc}")
        plc = None

    print("\nSystem ready. Press 'Q' to terminate.\n")
    last_state: Optional[bool] = None  # Track last detection state to prevent redundant writes

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera frame could not be read; exiting loop.")
                break

            # Perform object detection (CPU-only)
            results = model(frame, conf=CONF_THRESHOLD, verbose=False, device="cpu")
            object_detected = any(len(r.boxes) > 0 for r in results)

            # Update PLC only if the detection state changes
            if object_detected != last_state:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                state_text = "Object detected" if object_detected else "No object detected"
                print(f"[{timestamp}] {state_text}")
                update_plc(plc, object_detected)
                last_state = object_detected

            # Display video with bounding boxes
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{conf:.2f}",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

            cv2.imshow("Vision-Based Presence Detection", frame)
            if cv2.waitKey(1) & 0xFF in [ord("q"), ord("Q")]:
                print("'Q' pressed; exiting loop.")
                break
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

        if plc:
            plc.close()
            print("PLC connection closed.")

        print("\nProgram terminated successfully.")


if __name__ == "__main__":
    main()
