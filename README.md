# PyBullet YOLO + Laya Color Sorter

> The cheap man's local JEV.

This project contains a simulated Franka Panda arm that sorts three cups into red, green, and blue trays.

## How It Works

The sorter:

1. Creates one red, one green, and one blue cup at randomized, camera-visible positions. Use `--seed` to repeat a layout.
2. Renders the PyBullet scene through its CPU camera renderer into an OpenCV window. This single-window setup avoids a graphics-driver conflict between separate PyBullet and OpenCV windows on some Linux systems.
3. Uses YOLO11 to detect cups, HSV pixel values to classify their colors, and the simulated depth image to estimate their positions in the scene.
4. Sends each color to Laya as a typed destination question. The terminal shows the question, input state, selected tray, and probabilities.
5. Moves the Panda to pick and place each cup. It verifies placements, asks Laya a final `complete` or `retry` question, and retries failed placements once.

Laya's response is not trusted on its own: a destination must match the measured cup color and pass the confidence threshold, and the final success check uses the physical cup positions.

## Requirements

- Python 3.10 or newer
- Node.js 20 or newer and npm
- A desktop graphical session for the OpenCV preview window
- About 2 GB of available RAM for Laya
- Internet access on the first run to download model weights

The scripts have been tested with Python 3.12 and Node.js 21.

## Install

From the project directory, create and activate a Python environment, then install the simulation and vision packages:

```sh
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pybullet numpy opencv-python ultralytics
```

Install the Laya TypeScript worker dependencies from the same directory:

```sh
npm install
```

On first launch, Ultralytics downloads the YOLO11n weights. Laya downloads about 1.7 GB of ONNX model weights and caches them under `~/.cache/receptron-laya`.

## Run the Color Sorter

```sh
source venv/bin/activate
python yolo_color_sorter.py
```

To reproduce the same randomized cup layout:

```sh
python yolo_color_sorter.py --seed 42
```

The sorter prints each Laya question and response, placement results, and the final verification. Press `Q` in the camera preview window to quit.

## Other Demos

- `yolo_robotic_arm.py` runs YOLO11 while the Panda picks up and moves a mug.
- `test_robotic_arm.py` demonstrates a basic Panda pick-and-place.

Check the TypeScript worker with:

```sh
npm run typecheck
```