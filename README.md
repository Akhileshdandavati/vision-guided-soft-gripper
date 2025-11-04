# Vision-Guided Soft Gripper

## Project goals
- Detect relevant items with a YOLOv8 model and show the results on a live camera feed.
- Translate each detection into a target grip pressure for a pneumatic soft gripper.
- Publish the item index and pressure to external hardware (Allen-Bradley PLC or any PLC that mirrors the same tags).
- Provide a lightweight demo that streams detections over UDP to a robot or visualization PC.

The repository contains a minimal, hardware-friendly reference implementation so you can adapt the detection and communication layers to your lab or production cell.

## System architecture at a glance
```
┌────────────┐      ┌──────────────────────┐      ┌───────────────────┐
│ OpenCV     │      │ Ultralytics YOLOv8   │      │ Output interfaces │
│ video feed │ ───► │ detection / filtering│ ───► │  • PLC (pylogix)   │
└────────────┘      └──────────────────────┘      │  • UDP socket      │
                                                  │  • Console overlay │
                                                  └───────────────────┘
```
1. `food_detection.py` acquires frames, runs YOLOv8, filters detections to the classes defined in your dataset, and looks up the desired pressure in `item_data.json`.
2. Each new label is written to the PLC tags `Vision_Item_Index`, `Vision_Pressure`, and a `Vision_NewData` handshake pulse. You can use the same pattern for other protocols if pylogix is not available.
3. `test1.py` demonstrates the same detection loop without PLC dependencies and can optionally emit JSON packets over UDP.

## Repository layout
| Path | Purpose |
| ---- | ------- |
| `food_detection.py` | PLC-focused detection loop with pressure lookups and EtherNet/IP writes. |
| `test1.py` | Lightweight detection viewer that can stream centroids over UDP. |
| `item_data.json` | Ordered mapping from detection label to gripping pressure (kPa). The key order defines PLC item indices. |
| `detected_items.json` | Example enumeration of indices/pressures for PLC testing and ladder-logic prototyping. |
| `yolo.yml` | Conda environment specification that installs Ultralytics YOLO, OpenCV, pylogix, and supporting packages. |
| `yolov8n.pt`, `yolov8s.pt` | Sample YOLO weights (replace with your trained model). |

## Quick start
1. **Create the environment**
   ```bash
   conda env create -f yolo.yml
   conda activate yolo
   ```
2. **Place your trained weights** in the repository root (update `MODEL_PATH` in the scripts if the filename differs).
3. **Review `item_data.json`** to match the foods or parts you intend to grip. The JSON keys must match the labels produced by your model.
4. *(Optional)* **Provide a dataset YAML** describing your class names. Set `DATASET_YAML` in `food_detection.py` to the YAML path. If you skip this step, the script falls back to the model's built-in label list.

## Configuring the detection pipeline
1. Edit `food_detection.py`:
   - `MODEL_PATH`: YOLOv8 weights to load.
   - `DATASET_YAML`: dataset definition (`names:` list). Set to `None` or an empty string to use the model defaults.
   - `ITEM_DATA_FILE`: JSON file with label → pressure mappings.
   - `CAMERA_INDEX`: OpenCV device index (0 for the default webcam).
   - `CONF_THRESHOLD`: discard detections below this confidence.
   - `PLC_IP` / `SEND_TO_PLC`: target PLC address and toggle for EtherNet/IP writes.
2. Update `item_data.json` with every class you want the PLC to recognize. Because the PLC tag expects a numeric index, the array order of the JSON keys defines `Vision_Item_Index`. Any detection missing from the JSON defaults to 50 kPa and index 0.
3. (Optional) Duplicate `item_data.json` per recipe or product run and swap the filename in the configuration block.

## Running the PLC-integrated loop (`food_detection.py`)
1. Connect the camera and ensure no other program is using it.
2. If `SEND_TO_PLC=True`, verify the PLC is reachable from the PC (ping) and that the `pylogix` dependency is installed in the environment.
3. Start the script:
   ```bash
   python food_detection.py
   ```
4. Watch the OpenCV window for overlays and the console for newly detected items. For each new label, the script performs the following PLC handshake:
   1. Look up the pressure in `item_data.json`.
   2. Write the 1-based item index to `Vision_Item_Index`.
   3. Write the pressure (float) to `Vision_Pressure`.
   4. Pulse `Vision_NewData` to `True` for ~0.5 s, then reset to `False`.
5. Press `q` in the window to end the session. The script closes the camera and PLC connection automatically and prints a summary list of detections with their pressures.

**Tip:** If you want every frame to update the PLC (not just the first time an item appears), remove the `detected_items` guard in the loop and write on every iteration.

## Running the UDP demo (`test1.py`)
1. Set `SEND_TO_ROBOT=True` if you want to transmit detections over the network and specify `ROBOT_IP`/`ROBOT_PORT`.
2. Launch the script:
   ```bash
   python test1.py
   ```
3. The console prints the label, centroid, and confidence for each detection. When UDP is enabled, the script sends a JSON payload such as `{"object": "banana", "confidence": 0.91, "cx": 240, "cy": 180}` to the configured listener.

## Calibrating pressures and extending the system
- Start with conservative pressures in `item_data.json`, then gradually tune values while watching the gripper response.
- Mirror the PLC tag names in your ladder logic or adapt the script to your PLC's naming convention.
- Use `detected_items.json` as a quick lookup table when drafting PLC routines or HMI displays.
- To add logging, wrap the detection loop with CSV or database writes—every detection already includes the label, confidence, and bounding box coordinates.
- To support other protocols (MQTT, REST, ROS 2), reuse the detection loop and plug in your preferred client libraries where the PLC writes occur.

## Troubleshooting
| Symptom | Suggested fix |
| ------- | ------------- |
| Camera fails to open | Confirm the correct `CAMERA_INDEX` and that no other program owns the device. On Linux, check `/dev/video*` permissions. |
| Model fails to load | Ensure the weights file exists, matches the Ultralytics version installed, and that your GPU drivers (if any) are compatible. |
| PLC writes time out | Check network connectivity, verify `PLC_IP`, and confirm the PLC tags are not aliased or protected. Temporarily set `SEND_TO_PLC=False` to isolate camera/model issues. |
| No PLC updates for a known label | Make sure the label string matches the key in `item_data.json` exactly (case sensitive). |
| UDP receiver gets nothing | Confirm `SEND_TO_ROBOT=True`, firewall rules allow UDP on the selected port, and the listener binds to the correct interface. |

## Contributing and next steps
Pull requests are welcome for additional communication backends, model-training utilities, or deployment scripts. Feel free to open an issue with questions or improvement ideas.
