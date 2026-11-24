# Weights

This folder contains the trained Philippine model weights: `PH_LPRNet_model.pth` (1.34 MB, trained November 25, 2026 — see `results/TEST_REPORT.md` for its verified accuracy).

To regenerate the weights yourself:

```bash
python gen_ph_plates.py
python train_ph.py
```

This trains the model (30 epochs, ~27 minutes on CPU) and saves `PH_LPRNet_model.pth` into this folder.
