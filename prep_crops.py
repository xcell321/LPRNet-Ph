"""Build plate crops from the labeled real-photo dataset.

For every labeled photo in downloads/, locate the plate with the same
classical-CV detector used by detect_and_read.py, crop it with a small
margin, and save it as crops/<LABEL>.jpg. Writes manifest.csv with the
label, class, source file and box used for every crop.

Multi-plate photos (the LTO group photo) are skipped here and cropped
manually - see MANUAL list.
"""
import csv
import importlib.util
import os
import sys

import cv2

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.expanduser("~/workspace/lprnet-real-dataset")
DOWNLOADS = os.path.join(DATASET, "downloads")
CROPS = os.path.join(DATASET, "crops")
os.makedirs(CROPS, exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "detect_and_read", os.path.expanduser("~/workspace/lprnet-test/detect_and_read.py"))
dar = importlib.util.module_from_spec(spec)
sys.argv = ["detect_and_read.py"]  # module reads argv at import time
spec.loader.exec_module(dar)

# file -> (label, class/condition note) from LABELS.md
LABELS = {
    "2018 Philippine license plate.jpg": ("NDA8962", "current private, low-res"),
    "Government plate PHI.jpg": ("SEE665", "1981 government, worn/dirty"),
    "Licence plate Philippines Bohol.jpg": ("GXK980", "1981 private"),
    "Motorcycle PHI.jpg": ("2236NG", "current motorcycle, small/blurry"),
    "Newgovernmentplate.jpg": ("SAB1812", "current government"),
    "Philippine license plate (Philippines 2000 series).jpg": ("RFR888", "1981 private"),
    "Philippine license plate (2024, electric vehicles).jpg": ("DNW1509", "current EV"),
    "Philippine license plate (2024, government).jpg": ("SND2569", "current government"),
    "Philippineslicenseplate.jpg": ("EVN729", "1981 PUV"),
    "Miriam protocol plate 7G2.jpg": ("7G2", "protocol, tiny"),
    "Private Vehicle Plate in Philippines.jpg": ("ZGE470", "1981 private"),
    "Private vehicle plates of the Philippines (2019 Series).jpg": ("CBC2080", "current private"),
    "Private vehicle plates of the Philippines (2020 Series).jpg": ("CAX3200", "current private"),
    "Private vehicle plates of the Philippines (2021 Series).jpg": ("CBT2794", "current private"),
    "Private vehicle plates of the Philippines (2021 Series) (1).jpg": ("CDK2649", "current private, tilt"),
    "Private vehicle plates of the Philippines (2018 Series).jpg": ("CCK6781", "current private"),
    "Private vehicle plates of the Philippines (2022 Series).jpg": ("DAR7662", "current private"),
    "Private vheicle plates of the Philippines (2008 Series).jpg": ("ZMC445", "1981 private"),
    "Private vheicle plates of the Philippines (2015 Series).jpg": ("TGM803", "2014 private ABC123"),
    "Private hybrid electric vehicle plates of the Philippines (2024 Series).jpg": ("NOW8263", "current EV"),
    "For Hire vehicle plates of the Philippines CVR-846 (2025 Series).jpg": ("CVR846", "current PUV"),
    "Private motorcycle plates of the Philippines 888-RJZ (2022 Series).jpg": ("888RJZ", "current motorcycle"),
    "Private motorcycle plates of the Philippines 958-RWF (2022 Series).jpg": ("958RWF", "current motorcycle"),
    "Private tricycle plates of the Philippines W6358P (2025 Series).jpg": ("W6358P", "tricycle"),
    "Private vehicle plates of the Philippines NGP3944 (2020 Series).jpg": ("NGP3944", "current private, STANDING TEST"),
    "Private vehicle plates of the Philippines CBN8658 (2019 Series).jpg": ("CBN8658", "current private"),
    "Private vehicle plates of the Philippines MAV-6198 (2021 Series).jpg": ("MAV6198", "current private"),
    "Private vehicle plates of the Philippines ABV-1463 (2015 Series).jpg": ("ABV1463", "current private"),
    "Private vehicle plates of the Philippines YCR-238 (2007 Series).jpg": ("YCR238", "1981 private"),
    "Private vehicle plates of the Philippines ZKY-932 (2015 Series).jpg": ("ZKY932", "2014 private ABC123"),
}
MANUAL = {"Philippine license plates LTO.jpg"}  # 3 plates in one photo


def main():
    rows, failed = [], []
    for fname, (label, note) in sorted(LABELS.items()):
        path = os.path.join(DOWNLOADS, fname)
        img = cv2.imread(path)
        if img is None:
            failed.append((label, fname, "cannot read file"))
            continue
        box = dar.find_plate(img)
        if box is None:
            # close-up photos: the plate fills the frame, use the full image
            h, w = img.shape[:2]
            box = (0, 0, w, h)
            src = "full-frame fallback"
        else:
            src = "detected"
        x1, y1, x2, y2 = box
        mx, my = int(0.04 * (x2 - x1)), int(0.08 * (y2 - y1))
        h, w = img.shape[:2]
        cx1, cy1 = max(0, x1 - mx), max(0, y1 - my)
        cx2, cy2 = min(w, x2 + mx), min(h, y2 + my)
        crop = img[cy1:cy2, cx1:cx2]
        out = os.path.join(CROPS, label + ".jpg")
        cv2.imwrite(out, crop)
        rows.append([label, note, fname, src, f"{cx1},{cy1},{cx2},{cy2}", os.path.basename(out)])
        print(f"OK   {label:8s} {src:18s} box={box} -> {os.path.basename(out)}")
    for label, fname, why in failed:
        print(f"FAIL {label:8s} {why}: {fname}")
    with open(os.path.join(DATASET, "manifest.csv"), "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["label", "note", "source_file", "box_source", "box", "crop_file"])
        wcsv.writerows(rows)
    print(f"\n{len(rows)} crops written, {len(failed)} failed, {len(MANUAL)} manual pending")


if __name__ == "__main__":
    main()
