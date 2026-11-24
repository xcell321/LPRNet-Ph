"""Train LPRNet on Philippine plates (PH character set).

Uses the model and data loader in this repo (data/load_data.py has the PH
CHARS set). Training images are read from ./ph_data/train — generate them
first with gen_ph_plates.py, or replace that folder with real labeled
plate photos (filename = plate text, e.g. ABC1234.jpg).

CTC loss with blank = last entry of CHARS ('-'), same setup as the
original LPRNet_Pytorch training script.
"""
import os, time, argparse
import numpy as np
import torch
import torch.nn as nn
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
    return torch.stack(imgs, 0), torch.from_numpy(np.asarray(labels).astype(np.int64)), lengths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--train_dir", default=os.path.join(BASE, "ph_data", "train"))
    args = ap.parse_args()

    torch.manual_seed(7)
    net = build_lprnet(lpr_max_len=8, phase=True, class_num=len(CHARS), dropout_rate=0.2)
    print(f"PH model built. classes={len(CHARS)} params={sum(p.numel() for p in net.parameters()):,}")
    ds = LPRDataLoader([args.train_dir], [94, 24], 8)
    print("train images:", len(ds))
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=2, collate_fn=collate_fn)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    ctc = nn.CTCLoss(blank=len(CHARS) - 1, reduction="mean", zero_infinity=True)

    t0 = time.time()
    for epoch in range(args.epochs):
        net.train()
        tot, nb = 0.0, 0
        for images, labels, lengths in loader:
            preds = net(images)                      # [N, C, T]
            T = preds.shape[2]
            logp = preds.permute(2, 0, 1).log_softmax(2)
            input_lengths = torch.full((images.size(0),), T, dtype=torch.long)
            loss = ctc(logp, labels, input_lengths, torch.tensor(lengths, dtype=torch.long))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch+1}/{args.epochs}  loss={tot/max(nb,1):.4f}  elapsed={time.time()-t0:.0f}s", flush=True)
    out = os.path.join(BASE, "weights", "PH_LPRNet_model.pth")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(net.state_dict(), out)
    print("saved:", out, f"({os.path.getsize(out)/1e6:.2f} MB)  total_train_time={time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
