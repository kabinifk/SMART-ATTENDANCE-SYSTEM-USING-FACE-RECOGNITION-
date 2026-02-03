# Smart Attendance System Using Face Recognition

## Overview
This project provides a **real-time automated attendance system** built with **Python, OpenCV, and a CNN-based face classifier**. The workflow follows a standard pipeline:

1. **Image Collection** – Capture face images from a webcam and build a dataset.
2. **Model Training** – Train a lightweight CNN to recognize known faces.
3. **Face Detection** – Detect faces in real-time video streams.
4. **Attendance Storage** – Log attendance in a CSV file with timestamps.
5. **Final Application** – Run live recognition and record attendance.

## Project Structure
```
.
├── app.py                 # Final application (real-time recognition)
├── attendance_storage.py  # CSV-based attendance logging
├── face_detection.py      # OpenCV-based face detection utilities
├── image_collection.py    # Capture and store face images
├── model_training.py      # CNN training pipeline
├── requirements.txt       # Python dependencies
└── data/
    └── dataset/           # Captured images by person name
```

## Setup
1. Create a virtual environment (recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Run in VS Code
1. Open the repository folder in VS Code.
2. Ensure the **Python** extension is installed.
3. Open the command palette (`Ctrl+Shift+P` / `Cmd+Shift+P`) and select **Python: Create Environment** (or **Python: Select Interpreter**) to choose your virtual environment.
4. Open an integrated terminal and install dependencies if you haven't already:
   ```bash
   pip install -r requirements.txt
   ```
5. Run a script from the terminal, for example:
   ```bash
   python image_collection.py --name "Alice" --count 50
   ```
6. Train the model:
   ```bash
   python model_training.py --epochs 10
   ```
7. Start the application:
   ```bash
   python app.py
   ```

## Usage

### 1. Collect Images
```bash
python image_collection.py --name "Alice" --count 50
```
Images are saved under `data/dataset/Alice/`.

### 2. Train the CNN Model
```bash
python model_training.py --epochs 10
```
Saves the trained model to `models/face_cnn.h5` and labels to `models/labels.json`.

### 3. Run Real-Time Attendance
```bash
python app.py
```
Attendance is logged to `attendance.csv`.

### 4. Process a Single Image & Share Attendance
```bash
python app.py --image path/to/photo.jpg --output-image output.jpg --share-attendance
```
This exports attendance to `attendance_share.json` for easy sharing.

## Notes
- Ensure good lighting during image collection.
- Expand the dataset for better recognition accuracy.
- This is a baseline example; consider augmentations and deeper models for production.
