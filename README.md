## miniJSRT Warm-Up (Tasks 0–3) — PyTorch + ResNet-18

Chest X-ray warm-ups on the miniJSRT teaching datasets:
Task 0 (I/O for PNG/JPG/DICOM), Task 1 (Orientation), Task 2 (Gender), Task 3 (Age regression).
Everything runs from a single notebook/script with lightweight PyTorch utilities.

⚠️ These datasets are for coursework/demo only—not for clinical use.

###  What’s inside

Task 0 – I/O: Read/write PNG, JPG, DICOM; robust percentile windowing; safe format conversion.

Task 1 – Orientation: Up/Down/Left/Right classifier with ResNet-18 (ImageNet init, frozen backbone).

Task 2 – Gender: Female/Male classifier (same pipeline as Task 1).

Task 3 – Age: Single-output regression with ResNet-18 head; MAE/RMSE evaluation.

Reusable helpers: dataset downloader, ImageNet preprocessing for grayscale, tiny training loops.

###  Tasks & commands (what each cell does)
#### Task 0 — Read/Write CXR (PNG/JPG/DICOM)

Downloads Practice_PNGandJPG.zip and Practice_DICOM.zip.

Functions:

png_to_jpg, jpg_to_png (PIL)

dcm_to_png (pydicom → percentile windowing → PNG)

Saves non-destructive outputs under:

Practice_PNGandJPG/_converted/

Practice_DICOM/_png/

#### Task 1 — Orientation (Up/Down/Left/Right)

Dataset: Directions01/train|test.

Pipeline: grayscale→3-ch, ImageNet preprocessing, ResNet-18 head (backbone frozen).

Optimizer: AdamW, Loss: CrossEntropy.

Prints train/val accuracy per epoch and final test accuracy.

#### Task 2 — Gender (Female/Male)

Dataset: Gender01/train|test.

Same pipeline as Task 1; 2-class head.

#### Task 3 — Age regression

Dataset: XPAge01_RGB + CSV.

Robust CSV/filename discovery; resolves nested paths.

Model: ResNet-18 with 1-output head (MSELoss).

Metrics: MAE and RMSE on test split.

Optional short fine-tune: unfreeze backbone at low LR.

###  Expected results (guideposts)

Task 0: Visual inspection only (contrast looks reasonable, no clipping/banding).

Task 1: High test accuracy in a few epochs (global orientation cues).

Task 2: Reasonable accuracy but smaller dataset—consider light unfreeze if needed.

Task 3: MAE/RMSE vary by split; start with frozen backbone, then brief fine-tune.

###  License & data

Code: MIT

Data: miniJSRT practice sets and derived splits belong to their respective owners and are intended for educational use. Check original dataset terms before redistribution.

### Acknowledgments

miniJSRT practice materials and task splits.

PyTorch & torchvision for model and preprocessing utilities.
