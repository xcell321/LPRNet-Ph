"""Test the PH-trained LPRNet on one real plate photo.

LPRNet reads a tight plate crop, it does not detect the plate in a full
photo. If your photo shows the whole vehicle, crop to the plate first —
either in an image editor, or with --crop X1 Y1 X2 Y2 (relative 0-1
coordinates of the plate region).

Usage:
    python test_real_plate.py plate_photo.jpg
    python test_real_plate.py vehicle.jpg --crop 0.10 0.24 0.93 0.67
"""
import os, argparse
import numpy as np
import cv2
import torch
from data.load_data import CHARS
from model.LPRNet import build_lprnet

BASE = os.path.dirname(os.path.abspath(__file__))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="photo containing a Philippine plate")
    ap.add_argument("--crop", nargs=4, type=float, default=None,
                    metavar=("X1", "Y1", "X2", "Y2"),
                    help="relative plate region, e.g. --crop 0.10 0.24 0.93 0.67")
    ap.add_argument("--weights", default=os.path.join(BASE, "weights", "PH_LPRNet_model.pth"))
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Cannot read image: {args.image}")
    if args.crop:
        h, w, _ = img.shape
        x1, y1, x2, y2 = args.crop
        img = img[int(h * y1):int(h * y2), int(w * x1):int(w * x2)]

    # same preprocessing as the data loader: resize to 94x24, normalize
    x = cv2.resize(img, (94, 24)).astype("float32")
    x = (x - 127.5) * 0.0078125
    x = np.transpose(x, (2, 0, 1))[None, :, :, :]

    net = build_lprnet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
    net.load_state_dict(torch.load(args.weights, map_location="cpu"))
    net.eval()
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
    print("PREDICTION:", "".join(CHARS[i] for i in res))

if __name__ == "__main__":
    main()
