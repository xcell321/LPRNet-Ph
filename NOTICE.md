# Notice

This repository is a modified version of **LPRNet_Pytorch** by sirius-ai:

- Original repository: https://github.com/sirius-ai/LPRNet_Pytorch
- License: Apache License 2.0 (see `LICENSE` in this repository)
- Based on the paper: S. Zherzdev & A. Gruzdev, *LPRNet: License Plate Recognition via Deep Neural Networks*, arXiv:1806.10447 (Intel, 2018)

Modifications made in this repository:

- `data/load_data.py` — the character set (`CHARS`) was changed from the original Chinese plate set (68 classes) to a Philippine plate set (37 classes: digits 0–9, letters A–Z, and the CTC blank).
- Added by the modifier: `train_ph.py`, `eval_ph.py`, `gen_ph_plates.py`, `test_real_plate.py`, `results/TEST_REPORT.md`, and this notice.

All other original code is unchanged from the upstream repository.
