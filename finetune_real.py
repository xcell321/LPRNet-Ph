"""Fine-tune PH LPRNet on REAL plate crops and evaluate honestly.

Design (documented for the study):
- Split is FIXED (below): 23 train / 10 held-out test plate instances.
  NGP3944 is always in test (standing test case, never trained on).
- Baseline = current PH weights (trained on synthetic plates only).
- Fine-tune = same weights, trained further on the real train crops with
  augmentation (brightness/contrast, blur, noise, small rotation) plus a
  small replay of synthetic plates each epoch (standard rehearsal against
  catastrophic forgetting of characters rare in the small real set).
- Metric = exact-match accuracy on the held-out test crops, per condition,
  reported for baseline and fine-tuned side by side. No test data is ever
  used in training.
"""
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(BASE, "LPRNet_PH")
sys.path.insert(0, REPO)
from data.load_data import CHARS  # noqa: E402
from model.LPRNet import build_lprnet  # noqa: E402

DATASET = os.path.expanduser("~/workspace/lprnet-real-dataset")
CROPS = os.path.join(DATASET, "crops")
DEVICE = torch.device("cpu")
BLANK = len(CHARS) - 1
CHAR2IDX = {c: i for i, c in enumerate(CHARS)}

TEST = ["NGP3944", "CAX3200", "DAR7662", "SND2569", "ZKY932",
        "SEE665", "YCR238", "DNW1509", "888RJZ", "CVR846"]
SUPP = ["0561000", "XLS887", "AA5490"]  # supplementary held-out test (added later, never trained on)
ALL = sorted(f[:-4] for f in os.listdir(CROPS) if f.endswith(".jpg"))
TRAIN = [lb for lb in ALL if lb not in TEST and lb not in SUPP]

# condition per label (from LABELS.md notes)
COND = {}
import csv  # noqa: E402
with open(os.path.join(DATASET, "manifest.csv")) as f:
    for row in csv.DictReader(f):
        COND[row["label"]] = row["note"]


def load_crop(label):
    img = cv2.imread(os.path.join(CROPS, label + ".jpg"))
    return img


def to_tensor(img):
    x = cv2.resize(img, (94, 24)).astype("float32")
    x = (x - 127.5) * 0.0078125
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x)


def decode(net, img):
    net.eval()
    with torch.no_grad():
        preb = net(to_tensor(img)[None].to(DEVICE)).cpu().numpy()[0]
    seq = [int(np.argmax(preb[:, j])) for j in range(preb.shape[1])]
    res, prev = [], None
    for c in seq:
        if c != prev and c != BLANK:
            res.append(CHARS[c])
        prev = c
    return "".join(res)


def evaluate(net, labels):
    out = {}
    for lb in labels:
        pred = decode(net, load_crop(lb))
        out[lb] = {"true": lb, "pred": pred, "exact": pred == lb,
                   "condition": COND.get(lb, "")}
    return out


def augment(img, rng):
    out = img.copy()
    # small rotation
    if rng.random() < 0.7:
        ang = rng.uniform(-4, 4)
        h, w = out.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        out = cv2.warpAffine(out, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    # brightness / contrast
    if rng.random() < 0.8:
        alpha = rng.uniform(0.75, 1.25)
        beta = rng.uniform(-25, 25)
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
    # blur
    if rng.random() < 0.3:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)
    # noise
    if rng.random() < 0.3:
        noise = np.random.normal(0, rng.uniform(3, 10), out.shape)
        out = np.clip(out.astype("float32") + noise, 0, 255).astype("uint8")
    return out


def encode(label):
    return [CHAR2IDX[c] for c in label]


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    print(f"train={len(TRAIN)} test={len(TEST)}")
    print("TRAIN:", TRAIN)
    print("TEST :", TEST)

    weights = os.path.join(REPO, "weights", "PH_LPRNet_model.pth")
    base_net = build_lprnet(lpr_max_len=8, phase="test", class_num=len(CHARS), dropout_rate=0)
    base_net.load_state_dict(torch.load(weights, map_location="cpu"))
    base_net.to(DEVICE)

    t0 = time.time()
    base_test = evaluate(base_net, TEST)
    base_supp = evaluate(base_net, SUPP)
    base_acc = np.mean([v["exact"] for v in base_test.values()])
    print(f"\nBASELINE test exact-match: {base_acc:.3f}")
    for lb, v in base_test.items():
        print(f"  {lb:8s} -> {v['pred'] or '(empty)':10s} {'OK ' if v['exact'] else 'MISS'} [{v['condition']}]")

    # ---- fine-tune ----
    net = build_lprnet(lpr_max_len=8, phase="train", class_num=len(CHARS), dropout_rate=0)
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    net.to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)

    synth_dir = os.path.join(BASE, "ph_data", "train")
    synth_files = [f for f in os.listdir(synth_dir) if f.endswith(".jpg")] if os.path.isdir(synth_dir) else []
    print(f"\nsynthetic replay pool: {len(synth_files)}")

    rng = random.Random(123)
    EPOCHS, REAL_COPIES, REPLAY = 60, 25, 300
    # Pre-downscale the real crops once: augmenting full-resolution photos
    # in bulk exhausted memory (OOM). Working size is 4x the model input.
    train_small = {lb: cv2.resize(load_crop(lb), (376, 96)) for lb in TRAIN}
    for ep in range(EPOCHS):
        net.train()
        descs = []
        for lb in TRAIN:
            descs += [("real", lb)] * REAL_COPIES
        for fn in rng.sample(synth_files, min(REPLAY, len(synth_files))):
            descs.append(("synth", fn))
        rng.shuffle(descs)
        tot, nb = 0.0, 0
        BATCH = 16
        for i in range(0, len(descs), BATCH):
            chunk = descs[i:i + BATCH]
            xs, lbs = [], []
            for kind, key in chunk:
                if kind == "real":
                    xs.append(to_tensor(augment(train_small[key], rng)))
                    lbs.append(key)
                else:
                    xs.append(to_tensor(cv2.imread(os.path.join(synth_dir, key))))
                    lbs.append(key[:-4])
            xs = torch.stack(xs).to(DEVICE)
            logits = net(xs)  # (B, C, T)
            logp = logits.log_softmax(dim=1).permute(2, 0, 1)  # (T, B, C)
            targets = torch.cat([torch.tensor(encode(lb), dtype=torch.long) for lb in lbs])
            tgt_lens = torch.tensor([len(lb) for lb in lbs], dtype=torch.long)
            in_lens = torch.tensor([logp.shape[0]] * len(chunk), dtype=torch.long)
            loss = ctc(logp, targets, in_lens, tgt_lens)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        if (ep + 1) % 10 == 0 or ep == 0:
            acc = np.mean([v["exact"] for v in evaluate(net, TEST).values()])
            print(f"epoch {ep+1:3d}  loss {tot/max(nb,1):.4f}  test exact-match {acc:.3f}")

    out_weights = os.path.join(REPO, "weights", "PH_LPRNet_finetuned_real.pth")
    torch.save(net.state_dict(), out_weights)
    print("\nsaved", out_weights)

    ft_test = evaluate(net, TEST)
    ft_supp = evaluate(net, SUPP)
    ft_acc = np.mean([v["exact"] for v in ft_test.values()])
    print(f"\nSUPPLEMENTARY test (diplomatic / angled / damaged):")
    for lb in SUPP:
        b, v = base_supp[lb], ft_supp[lb]
        print(f"  {lb:8s} baseline {b['pred'] or '(empty)':10s} -> finetuned {v['pred'] or '(empty)':10s} {'OK' if v['exact'] else 'MISS'} [{v['condition']}]")
    print(f"\nFINE-TUNED test exact-match: {ft_acc:.3f}  (baseline {base_acc:.3f})")
    for lb, v in ft_test.items():
        b = base_test[lb]["pred"]
        print(f"  {lb:8s} -> {v['pred'] or '(empty)':10s} {'OK ' if v['exact'] else 'MISS'} (baseline: {b or '(empty)'}) [{v['condition']}]")

    report = {
        "train_labels": TRAIN, "test_labels": TEST, "supp_labels": SUPP,
        "baseline": {"accuracy": float(base_acc), "results": base_test, "supp_results": base_supp},
        "finetuned": {"accuracy": float(ft_acc), "results": ft_test, "supp_results": ft_supp},
        "epochs": EPOCHS, "real_copies_per_epoch": REAL_COPIES, "synthetic_replay_per_epoch": REPLAY,
        "weights_out": out_weights, "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(DATASET, "finetune_results.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\nreport ->", os.path.join(DATASET, "finetune_results.json"))


if __name__ == "__main__":
    main()
