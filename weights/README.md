# Weights

The trained Philippine model weights (`PH_LPRNet_model.pth`, 1.34 MB) are not stored in this folder yet — binary files are added to this repository separately from the code.

To generate the weights yourself:

```bash
python gen_ph_plates.py
python train_ph.py
```

This trains the model (30 epochs, ~27 minutes on CPU) and saves `PH_LPRNet_model.pth` into this folder.
