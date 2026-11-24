"""Evaluate the PH-trained LPRNet on the synthetic test sets.

Expects ./weights/PH_LPRNet_model.pth (from train_ph.py) and the test
folders ./ph_data/test_clean, ./ph_data/test_dirty, ./ph_data/test_old
(from gen_ph_plates.py). Reports exact-match accuracy, character
accuracy, and average forward time per plate.
"""
import os, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from data.load_data import CHARS, LPRDataLoader
from model.LPRNet import build_lprnet

BASE = os.path.dirname(os.path.abspath(__file__))

def collate_fn(batch):
    imgs, labels, lengths = [], [], []
    for img, label, length in batch:
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    return torch.stack(imgs, 0), np.asarray(labels), lengths

def greedy_decode(prebs):
    out, blank = [], len(CHARS) - 1
    for i in range(prebs.shape[0]):
        preb = prebs[i]
        seq = [int(np.argmax(preb[:, j])) for j in range(preb.shape[1])]
        res, prev = [], seq[0] if seq else blank
        if prev != blank:
            res.append(prev)
        for c in seq:
            if c == prev or c == blank:
                if c == blank:
                    prev = c
                continue
            res.append(c); prev = c
        out.append(res)
    return out

def evaluate(net, split, show=5):
    ds = LPRDataLoader([os.path.join(BASE, "ph_data", split)], [94, 24], 8)
    loader = DataLoader(ds, batch_size=100, shuffle=False, num_workers=2, collate_fn=collate_fn)
    tp = wrong = 0
    char_ok = char_tot = 0
    fwd = 0.0
    shown = 0
    with torch.no_grad():
        for images, labels, lengths in loader:
            targets, start = [], 0
            for L in lengths:
                targets.append(labels[start:start + L]); start += L
            t0 = time.time()
            prebs = net(images).cpu().numpy()
            fwd += time.time() - t0
            for pred, tgt in zip(greedy_decode(prebs), targets):
                ptxt = "".join(CHARS[int(i)] for i in pred)
                ttxt = "".join(CHARS[int(i)] for i in tgt)
                if shown < show:
                    print(f"    target={ttxt}  pred={ptxt}  {'OK' if ptxt == ttxt else 'X'}")
                    shown += 1
                if ptxt == ttxt:
                    tp += 1
                else:
                    wrong += 1
                for a, b in zip(ptxt, ttxt):
                    char_tot += 1
                    if a == b:
                        char_ok += 1
                char_tot += abs(len(ptxt) - len(ttxt))
    total = tp + wrong
    print(f"[{split}] exact-match: {tp}/{total} = {tp/total:.3f} | char-acc: {char_ok/max(char_tot,1):.3f} | {1000*fwd/total:.1f} ms/plate")
    return tp / total

def main():
    net = build_lprnet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
    net.load_state_dict(torch.load(os.path.join(BASE, "weights", "PH_LPRNet_model.pth"), map_location="cpu"))
    net.eval()
    print(f"PH model loaded. classes={len(CHARS)}")
    for split in ["test_clean", "test_dirty", "test_old"]:
        evaluate(net, split)

if __name__ == "__main__":
    main()
