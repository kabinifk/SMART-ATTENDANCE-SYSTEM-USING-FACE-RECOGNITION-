import csv
from datetime import datetime
import json
from pathlib import Path


class AttendanceStore:
    def __init__(self, file_path="attendance.csv"):
        self.file_path = Path(file_path)
        self.seen_today = set()

        if not self.file_path.exists():
            with self.file_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["Name", "Date", "Time"])

    def mark_attendance(self, name):
        date_str = datetime.now().strftime("%Y-%m-%d")
        time_str = datetime.now().strftime("%H:%M:%S")
        key = (name, date_str)

        if key in self.seen_today:
            return False

        with self.file_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([name, date_str, time_str])

        self.seen_today.add(key)
        return True

    def read_records(self):
        if not self.file_path.exists():
            return []
        with self.file_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            return list(reader)

    def export_json(self, output_path="attendance_share.json"):
        records = self.read_records()
        output_path = Path(output_path)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(records, file, indent=2)
        return output_path
