import argparse
from pathlib import Path

import cv2

from face_detection import detect_faces


def collect_images(name, count, output_dir, camera_index):
    output_path = Path(output_dir) / name
    output_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not access webcam.")

    saved = 0
    while saved < count:
        ret, frame = cap.read()
        if not ret:
            continue

        faces = detect_faces(frame)
        for (x, y, w, h) in faces:
            face = frame[y : y + h, x : x + w]
            face_resized = cv2.resize(face, (64, 64))
            file_path = output_path / f"{name}_{saved + 1:04d}.jpg"
            cv2.imwrite(str(file_path), face_resized)
            saved += 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Captured: {saved}/{count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            if saved >= count:
                break

        cv2.imshow("Image Collection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Collect face images for a person.")
    parser.add_argument("--name", required=True, help="Person name / label")
    parser.add_argument("--count", type=int, default=50, help="Number of images")
    parser.add_argument(
        "--output-dir", default="data/dataset", help="Dataset output directory"
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect_images(args.name, args.count, args.output_dir, args.camera_index)
