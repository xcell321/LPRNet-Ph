# LPRNet-Ph

LPRNet adapted for **Philippine license plates** — the original Chinese character set was replaced with the Philippine plate character set (0–9, A–Z) and the model was retrained on Philippine-style plates.

This project is one of three models in a comparative study by **Team AUTHENTIC**: *"Performance Evaluation of Three License Plate Recognition Models for Philippine Vehicle License Plates"* — measuring recognition accuracy, precision, recall, processing speed, and reliability under different image conditions (clean, dirty, old plates), to identify the best model for vehicle identification and traffic violation documentation in the Philippines.

## What was changed

The original [LPRNet_Pytorch](https://github.com/sirius-ai/LPRNet_Pytorch) recognizes Chinese plates: its character set (`CHARS` in `data/load_data.py`) has 68 classes, starting with 31 Chinese province characters.

In this repo, that list is the only structural change:

```python
CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
         'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', '-']   # '-' is the CTC blank
```

37 classes instead of 68. The network's output layer is built from `len(CHARS)`, so it adjusts automatically — the model shrinks from 446,976 to **326,541 parameters** (weights file: 1.34 MB). The LPRNet architecture itself is unchanged.

## Results (verified November 25, 2026)

Trained for 30 epochs on 1,000 synthetic Philippine plates (new white style + old green style), CTC loss, Adam optimizer, CPU only. Training loss: 5.46 → 0.0013.

| Test set | Condition | Exact-match accuracy | Character accuracy | Speed |
|---|---|---|---|---|
| test_clean (200 plates) | New plates, clean | **98.0%** | 99.0% | 22.7 ms/plate |
| test_dirty (200 plates) | New plates + mud smudges, noise, blur | **97.5%** | 98.1% | 25.2 ms/plate |
| test_old (120 plates) | Old green plates, clean | **96.7%** | 98.3% | 26.7 ms/plate |

Speed was measured on a small shared sandbox CPU; the original paper reports 1.3 ms/plate on a desktop Intel i7, so a real PC runs much faster. Full details: [results/TEST_REPORT.md](results/TEST_REPORT.md).

### Honest limitation

On one **real** Philippine plate photo (plate `NGP 3944`), the model predicted only `3244` — the letters were missed. The synthetic training plates were drawn with a standard computer font, which does not match the real LTO plate font (narrower, embossed characters, bolts, reflections). **The model architecture works for Philippine plates, but real-world accuracy requires training on real Philippine plate photos.** Closing that dataset gap is the next step of the study.

## How to run

Requirements: Python 3.10+, and a DejaVu Sans Bold font for the dataset generator (default path is the standard Linux location; edit `FONT` in `gen_ph_plates.py` if yours differs).

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python-headless numpy Pillow imutils

python gen_ph_plates.py     # generate the synthetic PH dataset into ph_data/
python train_ph.py          # train (30 epochs) -> weights/PH_LPRNet_model.pth
python eval_ph.py           # evaluate on clean / dirty / old test sets
python test_real_plate.py your_photo.jpg --crop 0.10 0.24 0.93 0.67
```

Notes:

- LPRNet reads a **tight plate crop**, not a full photo. `test_real_plate.py` takes `--crop X1 Y1 X2 Y2` (relative coordinates) when the photo shows the whole vehicle.
- To train on real plates instead, replace `ph_data/train/` with labeled photos — the filename is the ground truth (e.g. `ABC1234.jpg`).
- The trained weights are included at `weights/PH_LPRNet_model.pth`; running `train_ph.py` regenerates them.

## Privacy note

All testing used **synthetic plates** plus one plate photo provided by the author — no owner information was collected or stored. In a real deployment, a plate number linked to its owner's information is personal data: it must be **encrypted** (e.g. AES-256 at rest, hashed plate numbers for matching, HTTPS in transit) with **access limited to authorized personnel**, in line with the Philippine Data Privacy Act (RA 10173).

## Credits

- Original code: [sirius-ai/LPRNet_Pytorch](https://github.com/sirius-ai/LPRNet_Pytorch) (Apache-2.0, see `LICENSE` and `NOTICE.md`)
- Paper: S. Zherzdev & A. Gruzdev, *LPRNet: License Plate Recognition via Deep Neural Networks*, arXiv:1806.10447 (Intel, 2018) — a lightweight CNN that recognizes plates end-to-end without character segmentation, using CTC decoding.
- Philippine adaptation, training scripts, dataset generator, and evaluation: Edsiel Ursal, Team AUTHENTIC.
