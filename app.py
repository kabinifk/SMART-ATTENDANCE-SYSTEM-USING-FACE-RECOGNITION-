import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras

from attendance_storage import AttendanceStore
from face_detection import detect_faces


def load_labels(labels_path):
    with open(labels_path, "r", encoding="utf-8") as file:
        label_map = json.load(file)
    return {int(k): v for k, v in label_map.items()}


def preprocess_face(face):
    resized = cv2.resize(face, (64, 64))
    normalized = resized.astype("float32") / 255.0
    return np.expand_dims(normalized, axis=0)


def annotate_faces(frame, model, labels, attendance_store, confidence_threshold):
    faces = detect_faces(frame)
    for (x, y, w, h) in faces:
        face = frame[y : y + h, x : x + w]
        processed = preprocess_face(face)
        predictions = model.predict(processed, verbose=0)[0]
        label_index = int(np.argmax(predictions))
        confidence = float(predictions[label_index])
        name = labels.get(label_index, "Unknown")

        if confidence < confidence_threshold:
            name = "Unknown"

        if name != "Unknown":
            attendance_store.mark_attendance(name)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(
            frame,
            f"{name} ({confidence:.2f})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )
    return frame


def run_attendance(model_path, labels_path, camera_index=0, confidence_threshold=0.7):
    model = keras.models.load_model(model_path)
    labels = load_labels(labels_path)
    attendance_store = AttendanceStore()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not access webcam.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        annotate_faces(
            frame,
            model,
            labels,
            attendance_store,
            confidence_threshold,
        )

        cv2.imshow("Smart Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_attendance_on_image(
    model_path,
    labels_path,
    image_path,
    output_path=None,
    confidence_threshold=0.7,
    share_attendance=False,
    attendance_share_path="attendance_share.json",
):
    model = keras.models.load_model(model_path)
    labels = load_labels(labels_path)
    attendance_store = AttendanceStore()

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    annotated = annotate_faces(
        frame,
        model,
        labels,
        attendance_store,
        confidence_threshold,
    )

    if output_path:
        cv2.imwrite(str(output_path), annotated)

    if share_attendance:
        attendance_store.export_json(attendance_share_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Smart attendance system runner.")
    parser.add_argument("--model-path", default="models/face_cnn.h5")
    parser.add_argument("--labels-path", default="models/labels.json")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--image", help="Path to a single image for attendance.")
    parser.add_argument(
        "--output-image", help="Optional path to save annotated image."
    )
    parser.add_argument(
        "--share-attendance",
        action="store_true",
        help="Export attendance to a JSON share file.",
    )
    parser.add_argument(
        "--attendance-share-path",
        default="attendance_share.json",
        help="Path for exported attendance JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    labels_path = Path(args.labels_path)

    if not model_path.exists() or not labels_path.exists():
        raise FileNotFoundError("Model or labels not found. Train the model first.")

    if args.image:
        run_attendance_on_image(
            model_path=model_path,
            labels_path=labels_path,
            image_path=Path(args.image),
            output_path=Path(args.output_image)
            if args.output_image
            else None,
            confidence_threshold=args.confidence_threshold,
            share_attendance=args.share_attendance,
            attendance_share_path=args.attendance_share_path,
        )
    else:
        run_attendance(
            model_path,
            labels_path,
            camera_index=args.camera_index,
            confidence_threshold=args.confidence_threshold,
        )


if __name__ == "__main__":
    main()
