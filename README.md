## miniJSRT Warm-Up (Tasks 0–3) and Otsu Thresholding — PyTorch + ResNet-18 + OpenCV

This repository contains coursework experiments on chest X-rays from the miniJSRT teaching datasets, as well as an additional implementation of Otsu’s thresholding method for radiograph segmentation.

⚠️ These datasets and code are for coursework/demo only — not for clinical use.

---

### What’s Inside

- **Task 0 – I/O:**  
  Read/write PNG, JPG, and DICOM; robust percentile windowing; safe format conversion.

- **Task 1 – Orientation:**  
  Up/Down/Left/Right classifier with ResNet-18 (ImageNet init, frozen backbone).

- **Task 2 – Gender:**  
  Female/Male classifier (same pipeline as Task 1).

- **Task 3 – Age regression:**  
  Single-output regression with ResNet-18 head; MAE/RMSE evaluation.

- **Otsu Thresholding (New):**  
  Implementation of Otsu’s method (manual and OpenCV) on JSRT chest radiographs.  
  Includes Gaussian preprocessing, histogram analysis (raw, normalized, and post-threshold), and visualization of binary segmentation masks.

Reusable helpers: dataset downloader, ImageNet preprocessing for grayscale, tiny training loops.

---

### Tasks \& Commands (What Each Cell Does)

#### Task 0 — Read/Write CXR (PNG/JPG/DICOM)
- Downloads: `Practice_PNGandJPG.zip`, `Practice_DICOM.zip`.  
- Functions:  
  - `png_to_jpg`, `jpg_to_png` (PIL)  
  - `dcm_to_png` (pydicom → percentile windowing → PNG)  
- Saves non-destructive outputs under:  
  - `Practice_PNGandJPG/_converted/`  
  - `Practice_DICOM/_png/`

#### Task 1 — Orientation (Up/Down/Left/Right)
- Dataset: `Directions01/train|test`  
- Pipeline: grayscale → 3-channel → ImageNet preprocessing → ResNet-18 head (backbone frozen)  
- Optimizer: AdamW; Loss: CrossEntropy  
- Outputs: train/val accuracy per epoch, final test accuracy

#### Task 2 — Gender (Female/Male)
- Dataset: `Gender01/train|test`  
- Same pipeline as Task 1; binary classification head

#### Task 3 — Age Regression
- Dataset: `XPAge01_RGB + CSV`  
- Robust CSV/filename discovery with nested path handling  
- Model: ResNet-18 with 1-output head (MSELoss)  
- Metrics: MAE, RMSE on test split  
- Optional fine-tune: unfreeze backbone at low LR

#### Otsu Thresholding on JSRT Radiographs
- Dataset: JSRT practice set (e.g., `JPCLN002.png`)  
- Preprocessing: Gaussian blur ($5\times5$ kernel)  
- Manual Otsu: between-class variance maximization (NumPy)  
- OpenCV Otsu: `cv2.threshold(..., THRESH_OTSU)`  
- Outputs:  
  - Histograms (raw counts, normalized probabilities, binary 0/255)  
  - Thresholded masks for original and blurred images  
  - Montages of all results saved under `jsrt_otsu_outputs/`

---

### Expected Results

- **Task 0:** Visual inspection only (contrast looks reasonable, no clipping/banding).  
- **Task 1:** High test accuracy in a few epochs (global orientation cues).  
- **Task 2:** Reasonable accuracy but limited by dataset size.  
- **Task 3:** MAE/RMSE vary; frozen backbone is a good start, brief fine-tuning improves results.  
- **Otsu Thresholding:** Threshold $\sim 110$ for CXR images, segmenting dark lung fields from brighter bone/mediastinum. Histograms show complex multimodal distributions (not strictly bimodal), yet Otsu selects a useful partition.

---

### License \& Data

- Code: MIT  
- Data: miniJSRT practice sets and derived splits belong to their respective owners and are intended for educational use. Check dataset terms before redistribution.

---

### Acknowledgments

- miniJSRT practice materials and task splits  
- PyTorch \& torchvision for model and preprocessing utilities  
- OpenCV and NumPy for Otsu thresholding implementation
