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
# README — Z-algorithm Pattern Matcher

### Overview
This program finds all exact occurrences of a pattern p in a text t using the Z-algorithm and reports:
- the 1-based starting positions of every match (one per line, ascending), then
- three tallies on separate lines:
    - Number of comparisons: <int>
    - Number of matches: <int>
    - Number of mismatches: <int>
A “comparison” is a single character == check performed while extending Z-boxes. Each comparison is counted as either a match or a mismatch.

### Files
exact_pattern_matching.py — main program (Z-algorithm implementation + CLI)

### Requirements
- Python 3.8+ (no external packages)
- POSIX or Windows shell

### Input format
Each input file must contain exactly two lines:
- Line 1: the text t
- Line 2: the pattern p

Example:
ATATTGATGATATG...
ATTGATGATA

### Output format
- One line per match position (1-based indices into t)
- Then the three tallies, each on its own line

Example:
3
166
538
688
1111
Number of comparisons: 1879
Number of matches: 522
Number of mismatches: 1357

### How to run
Print to stdout
python exact_pattern_matching.py <input_file>

Example: python zmatch.py samples/sample_0

Redirect to a file if needed:
python exact_pattern_matching.py samples/sample_0 > sol_0

Write directly to an output file
python exact_pattern_matching.py <input_file> <output_file>

Example:
python exact_pattern_matching.py samples/sample_0 sol_0

The program does not create any directories, therefore the path to <output_file> must exist.

### Batch over many samples
From a shell inside the directory that contains samples/:

for f in samples/sample_*; do
  base=$(basename "$f")
  out="sol_${base#sample_}"
  python zmatch.py "$f" "$out"
done

### Notes on correctness
- Positions are 1-based: first character of t is position 1; last is position |t|.
- The algorithm uses the standard pattern + '$' + text construction (the sentinel $ is not in the DNA alphabet), so there are no separator collisions.
- We skip unnecessary extensions when a Z value is fully determined inside the current [L, R] window; this keeps comparison counts tight.

### Implementation details
- Z-array construction maintains [L, R]. When i ≤ R:
    - If Z[i-L] < R-i+1, we reuse Z[i] = Z[i-L] and do not extend (no new comparisons).
    - Otherwise, we set Z[i] = R-i+1 and extend from R+1.
- Every time we compare t[a] with t[i+a], we increment the global comparison counter and classify it as a match or mismatch.
