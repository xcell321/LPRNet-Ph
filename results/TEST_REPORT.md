# LPRNet Philippine Plate Test Report

**Date:** November 25, 2026
**Task:** Edsiel's model assignment (Model 3) for Team AUTHENTIC's study — *"Performance Evaluation of Three License Plate Recognition Models for Philippine Vehicle License Plates"*
**Repo used:** `sirius-ai/LPRNet_Pytorch` (based on LPRNet, Zherzdev & Gruzdev, Intel 2018, arXiv:1806.10447)
**Test environment:** CPU-only sandbox, PyTorch 2.15.0+cpu

---

## 1. What was changed to make it Philippine-based

| Item | Original repo | PH version |
|---|---|---|
| Character set (`CHARS` in `data/load_data.py`) | 68 classes — 31 Chinese province characters + digits + letters | **37 classes — digits 0–9 + letters A–Z + blank** |
| Model size | 446,976 parameters | **326,541 parameters** (1.34 MB weights file) |
| Training data | Chinese plate dataset | **1,520 synthetic Philippine plates** generated for this test (1,000 train / 520 test, all unique, format: 3 letters + 4 digits) |

The change is made in one place only — the `CHARS` list — and the model's output layer automatically adjusts because the code builds it from `len(CHARS)`. No other architecture changes were needed.

## 2. Training

- 1,000 synthetic PH training plates: 800 new-style (white background, black text) + 200 old-style (green background, white text), with small random rotation, blur, and light dirt in some samples
- 30 epochs, batch 100, Adam optimizer, CTC loss — **loss went from 5.46 down to 0.0013**
- Total training time: ~27 minutes on CPU

## 3. Results — synthetic Philippine test plates (PH model only)

| Test set | Condition | Exact-match accuracy | Character accuracy | Speed |
|---|---|---|---|---|
| test_clean (200 plates) | New plates, clean | **98.0%** (196/200) | 99.0% | 22.7 ms/plate |
| test_dirty (200 plates) | New plates + mud smudges, noise, blur | **97.5%** (195/200) | 98.1% | 25.2 ms/plate |
| test_old (120 plates) | Old green plates, clean | **96.7%** (116/120) | 98.3% | 26.7 ms/plate |

Speed note: ~23–27 ms/plate on a small shared sandbox CPU. The original paper reports 1.3 ms/plate on a desktop Intel i7 CPU, so a real PC will be much faster.

## 4. Result — one real Philippine plate photo (PH model)

Edsiel provided a real plate photo (plate: **NGP 3944**, new white plate on a vehicle).

- **PH model prediction: `3244`** — **wrong.** The digits were only partly read and all letters were missed.
- **Honest finding:** the model scored 96–98% on synthetic plates but failed on the real photo. The reason is the **font and appearance gap** — the synthetic plates were drawn with a standard computer font (DejaVu Sans Bold), while real LTO plates use a different, narrower plate font, embossed characters, bolts, and real lighting/reflection.
- **Conclusion for the study:** LPRNet *can* be converted to Philippine plates and learns them well, but **real-world accuracy requires training on real Philippine plate photos** (or synthetic plates rendered with the actual plate font). Training-data realism matters more than the model architecture at this stage.

## 5. Privacy note (requirement added by Edsiel)

The testing used **synthetic plates only** plus one plate photo provided by Edsiel — no owner information was collected or stored, so no privacy violation. For the proposed real application, plate numbers linked to owner information must be **encrypted** (e.g., AES-256 at rest, hashed plate numbers for matching, HTTPS in transit) with **access limited to authorized personnel**, in line with the Data Privacy Act (RA 10173).

## 6. Files

- PH-modified code: `` (character set changed in `data/load_data.py`)
- Trained PH weights: `weights/PH_LPRNet_model.pth`
- Synthetic dataset generator: `gen_ph_plates.py` (dataset generated into `ph_data/`)
- Scripts: `train_ph.py`, `eval_ph.py`, `test_real_plate.py` (repo root)

## 7. Next steps (if the team continues)

1. Collect real PH plate photos (with permission) across conditions — clean, dirty, old green, faded, angled, night — and label them.
2. Retrain/fine-tune LPRNet on the real dataset and re-measure accuracy, precision, recall, and speed per condition.
3. Compare against the team's other two models (YOLO-based pipeline and the OCR model) using the same test set.
