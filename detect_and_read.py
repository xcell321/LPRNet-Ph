"""Automatic plate detection + reading (PH LPRNet).

No manual crop: the plate is located automatically with classical
computer vision (edge detection + contour filtering for a wide
rectangle), cropped, and read by the PH-trained LPRNet.

Usage: python detect_and_read.py <photo>
"""
import sys, os
import numpy as np
import cv2
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = BASE
sys.path.insert(0, REPO)

from data.load_data import CHARS
from model.LPRNet import build_lprnet

def find_plate(img):
    """Return (x1, y1, x2, y2) of the most plate-like rectangle, or None."""
    h, w = img.shape[:2]
    scale = 1000.0 / w
    small = cv2.resize(img, (1000, int(h * scale)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(gray, 30, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:40]
    best, best_score = None, -1
    img_area = small.shape[0] * small.shape[1]
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if ch == 0:
            continue
        aspect = cw / ch
        area_frac = (cw * ch) / img_area
        if 2.0 <= aspect <= 5.5 and 0.02 <= area_frac <= 0.9:
            score = area_frac * (1 - abs(aspect - 3.4) / 3.4)
            if score > best_score:
                best, best_score = (x, y, x + cw, y + ch), score
    if best is None:
        return None
    inv = 1.0 / scale
    return tuple(int(v * inv) for v in best)

def read_plate(net, crop):
    x = cv2.resize(crop, (94, 24)).astype("float32")
    x = (x - 127.5) * 0.0078125
    x = np.transpose(x, (2, 0, 1))[None, :, :, :]
    with torch.no_grad():
        preb = net(torch.from_numpy(x)).cpu().numpy()[0]
    seq = [int(np.argmax(preb[:, j])) for j in range(preb.shape[1])]
    blank = len(CHARS) - 1
    res, prev = [], seq[0] if seq else blank
    if prev != blank:
        res.append(prev)
    for c in seq:
        if c == prev or c == blank:
            if c == blank:
                prev = c
            continue
        res.append(c); prev = c
    return "".join(CHARS[i] for i in res)

def main():
    photo = sys.argv[1]
    img = cv2.imread(photo)
    if img is None:
        raise SystemExit(f"Cannot read image: {photo}")
    box = find_plate(img)
    if box is None:
        print("DETECTION: no plate-like region found")
        return
    x1, y1, x2, y2 = box
    pad_x, pad_y = int((x2 - x1) * 0.02), int((y2 - y1) * 0.05)
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(img.shape[1], x2 + pad_x), min(img.shape[0], y2 + pad_y)
    crop = img[y1:y2, x1:x2]
    cv2.imwrite(os.path.join(BASE, "auto_crop.jpg"), crop)
    print(f"DETECTION: plate found at ({x1},{y1})-({x2},{y2}) of {img.shape[1]}x{img.shape[0]} -> auto_crop.jpg")

    weights = os.path.join(REPO, "weights", "PH_LPRNet_model.pth")
    net = build_lprnet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    net.eval()
    print("PREDICTION:", read_plate(net, crop))

if __name__ == "__main__":
    main()
