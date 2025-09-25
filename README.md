# miniJSRT Warm-Up (Tasks 0–3), Otsu Thresholding, and Active Contours (Snakes) — PyTorch + ResNet-18 + OpenCV + NumPy

This repository contains coursework experiments on chest X-rays from the miniJSRT teaching datasets, an implementation of Otsu’s thresholding method, and a classic reproduction of the **Snakes: Active Contour Models** algorithm.

⚠️ These datasets and code are for coursework/demo only — not for clinical use.

---

## What’s Inside

- **Task 0 – I/O:**  
  Read/write PNG, JPG, and DICOM; robust percentile windowing; safe format conversion.

- **Task 1 – Orientation:**  
  Up/Down/Left/Right classifier with ResNet-18 (ImageNet init, frozen backbone).

- **Task 2 – Gender:**  
  Female/Male classifier (same pipeline as Task 1).

- **Task 3 – Age Regression:**  
  Single-output regression with ResNet-18 head; MAE/RMSE evaluation.

- **Otsu Thresholding:**  
  Implementation of Otsu’s method (manual and OpenCV) on JSRT chest radiographs.  
  Includes Gaussian preprocessing, histogram analysis (raw, normalized, and post-threshold), and visualization of binary segmentation masks.

- **Snakes (Active Contours):**  
  Implementation of Kass–Witkin–Terzopoulos snakes (1988) with:  
  - Semi-implicit solver for contour evolution  
  - Image energies (line, edge, termination)  
  - Bilinear force interpolation and arc-length resampling  
  - Multi-scale schedule with Gaussian pyramids  
  - Edge snapping and radial windowing for stable convergence  
  - Example demo on the `scikit-image` coins dataset (`example.py`)  

Reusable helpers: dataset downloader, ImageNet preprocessing for grayscale, tiny training loops.

---

## Tasks & Commands (What Each Cell Does)

### Task 0 — Read/Write CXR (PNG/JPG/DICOM)
- Downloads: `Practice_PNGandJPG.zip`, `Practice_DICOM.zip`.  
- Functions:  
  - `png_to_jpg`, `jpg_to_png` (PIL)  
  - `dcm_to_png` (pydicom → percentile windowing → PNG)  
- Saves non-destructive outputs under:  
  - `Practice_PNGandJPG/_converted/`  
  - `Practice_DICOM/_png/`

### Task 1 — Orientation (Up/Down/Left/Right)
- Dataset: `Directions01/train|test`  
- Pipeline: grayscale → 3-channel → ImageNet preprocessing → ResNet-18 head (backbone frozen)  
- Optimizer: AdamW; Loss: CrossEntropy  
- Outputs: train/val accuracy per epoch, final test accuracy

### Task 2 — Gender (Female/Male)
- Dataset: `Gender01/train|test`  
- Same pipeline as Task 1; binary classification head

### Task 3 — Age Regression
- Dataset: `XPAge01_RGB + CSV`  
- Robust CSV/filename discovery with nested path handling  
- Model: ResNet-18 with 1-output head (MSELoss)  
- Metrics: MAE, RMSE on test split  
- Optional fine-tune: unfreeze backbone at low LR

### Otsu Thresholding on JSRT Radiographs
- Dataset: JSRT practice set (e.g., `JPCLN002.png`)  
- Preprocessing: Gaussian blur ($5\times5$ kernel)  
- Manual Otsu: between-class variance maximization (NumPy)  
- OpenCV Otsu: `cv2.threshold(..., THRESH_OTSU)`  
- Outputs:  
  - Histograms (raw counts, normalized probabilities, binary 0/255)  
  - Thresholded masks for original and blurred images  
  - Montages of all results saved under `jsrt_otsu_outputs/`

### Snakes (Active Contours)
- Files: `snakes.py`, `example.py`  
- Based on Kass et al. (1988), with refinements inspired by blog/code resources (see References).  
- Workflow in `example.py`:  
  - Load the `scikit-image` coins dataset  
  - Initialize snake as a circle around a chosen coin  
  - Apply multi-scale evolution with decreasing $\sigma$  
  - Plot panels: original image, edge strength map, overlay of initial vs. final contour  
- Output: `snake_multiscale.png` showing convergence of snake to coin boundary.

---

## Expected Results

- **Task 0:** Visual inspection only (contrast looks reasonable, no clipping/banding).  
- **Task 1:** High test accuracy in a few epochs (global orientation cues).  
- **Task 2:** Reasonable accuracy but limited by dataset size.  
- **Task 3:** MAE/RMSE vary; frozen backbone is a good start, brief fine-tuning improves results.  
- **Otsu Thresholding:** Threshold $\sim 110$ for CXR images, segmenting dark lung fields from brighter bone/mediastinum. Histograms show complex multimodal distributions (not strictly bimodal), yet Otsu selects a useful partition.  
- **Snakes:** Snakes successfully lock onto coin boundaries; multi-scale evolution improves stability and edge alignment.

---

## License & Data

- Code: MIT  
- Data: miniJSRT practice sets and derived splits belong to their respective owners and are intended for educational use. Check dataset terms before redistribution.

---

## Acknowledgments

- miniJSRT practice materials and task splits  
- PyTorch & torchvision for model and preprocessing utilities  
- OpenCV and NumPy for Otsu thresholding implementation  
- Kass, Witkin, and Terzopoulos for the original Snakes model [1]  
- Cris Larsson’s blog on simple snake implementations [2]  
- Matthancock’s GitHub repo on snakes [3]  

---

## References

[1] M. Kass, A. Witkin, D. Terzopoulos. *Snakes: Active contour models*. IJCV, 1988.  
[2] C. Larsson. *A simple implementation of snakes*. <http://www.cb.uu.se/~cris/blog/index.php/archives/217>  
[3] M. Hancock. *snakes/example.py*. <https://github.com/notmatthancock/snakes/blob/master/example.py>
