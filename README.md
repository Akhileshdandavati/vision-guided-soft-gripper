# Vision-Guided Soft Gripper

This repository demonstrates how to drive a soft gripper (or any external system) from vision feedback by combining YOLOv8 object detection with optional PLC / robot communications.

- `food_detection.py` — filters YOLO detections to a curated list of foods, translates them to grip pressures, and can publish the results to an Allen-Bradley PLC over EtherNet/IP.
- `test1.py` — lightweight webcam viewer that streams raw YOLOv8 detections and can optionally forward the centroid and confidence data to a robot via UDP.
- `item_data.json` — JSON map from each recognized food item to the pressure (kPa) the soft gripper should apply.
- `detected_items.json` — example enumeration of default indices/pressures that informed the PLC tags used in `food_detection.py`.
- `yolo.yml` — Conda environment specification for reproducing the training/inference environment.

The sections below explain how to set up the software, prepare calibration data, and run each script.

## 1. Prerequisites

1. **Python & Conda**
   - Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda.
   - Create the project environment:  
     ```bash
     conda env create -f yolo.yml
     conda activate yolo
     ```
2. **YOLOv8 model files**
   - Place your trained model weights (e.g., `yolov8s.pt`) in the repository root. The example scripts expect the file to exist locally.
3. **Hardware (optional)**
   - Allen-Bradley PLC reachable on the same network if you intend to test PLC communication.
   - Robot controller or PC listening for UDP packets when using `test1.py` with `SEND_TO_ROBOT=True`.
   - A USB webcam or other video device recognized by OpenCV.

## 2. Configure item-to-pressure mapping

Edit `item_data.json` to list every food class you plan to detect and the target gripping pressure in kilopascals. For example:

```json
{
  "apple": 60,
  "banana": 45,
  "carrot": 55
}
```

The keys also establish the PLC item index order in `food_detection.py`. Any detected label that is not present defaults to `50` kPa.

## 3. Running `food_detection.py`

This script is designed for a PLC-connected workflow.

1. **Update the configuration block** near the top of the file:
   - `MODEL_PATH`: YOLOv8 weight file to load.
   - `DATASET_YAML`: optional dataset YAML with `names` matching the classes in your training set.
   - `ITEM_DATA_FILE`: path to your item-pressure JSON.
   - `CAMERA_INDEX`: integer index used by OpenCV (0 = default webcam).
   - `CONF_THRESHOLD`: detection confidence threshold.
   - `PLC_IP`: IP address of your Allen-Bradley PLC.
   - `SEND_TO_PLC`: set to `True` to enable PLC writes.
2. **Run the script**:
   ```bash
   python food_detection.py
   ```
3. **What it does**
   - Opens the camera stream and loads the YOLO model.
   - Filters detections to classes listed in the dataset YAML (or the model defaults if the YAML is absent).
   - Looks up each new label in `item_data.json` to fetch the desired grip pressure.
   - When PLC output is enabled, writes the following tags via EtherNet/IP using `pylogix`:
     - `Vision_Item_Index`
     - `Vision_Pressure`
     - `Vision_NewData` (pulsed `True`/`False` to signal new data)
   - Draws bounding boxes and labels on the video stream and logs a summary list of unique detections when the program exits.

Press `q` in the OpenCV window to quit. The script automatically releases the camera and closes the PLC session.

## 4. Running `test1.py`

`test1.py` is a smaller demo that skips PLC logic and optionally sends detection data over UDP.

1. Adjust the configuration constants at the top:
   - `model`: path to a YOLOv8 weight file (defaults to the COCO pre-trained `yolov8n.pt`).
   - `CAMERA_INDEX`: webcam index.
   - `SEND_TO_ROBOT`: `True` to enable UDP messages.
   - `ROBOT_IP`, `ROBOT_PORT`: address of the robot or PC that should receive detection packets.
2. Run the script:
   ```bash
   python test1.py
   ```
3. Each detection prints its centroid and confidence to the console. When UDP is enabled the script broadcasts a JSON payload like:
   ```json
   {"object": "banana", "confidence": 0.91, "cx": 240, "cy": 180}
   ```

## 5. Troubleshooting

- **Camera not found**: Verify `CAMERA_INDEX` is correct and no other process is using the camera.
- **Model fails to load**: Confirm the weights file exists and matches the Ultralytics version installed in the environment.
- **PLC writes fail**: Ensure the IP address is reachable, the PLC tags match the ones in `food_detection.py`, and the PLC firewall allows EtherNet/IP traffic.
- **UDP receiver missing data**: Confirm `SEND_TO_ROBOT=True`, the target IP/port are correct, and the receiver is listening on the specified port.

## 6. Extending the project

- Update `item_data.json` with new items and pressures as you calibrate the gripper.
- Customize the PLC tag names in `food_detection.py` to match your ladder logic.
- Integrate additional communication layers (MQTT, REST, ROS2) by following the detection loop patterns in `test1.py`.
- Train new YOLOv8 models and update `MODEL_PATH` / `DATASET_YAML` to match your datasets.

For any questions or improvements, feel free to open an issue or submit a pull request.
