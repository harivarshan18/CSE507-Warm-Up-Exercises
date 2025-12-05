# miniJSRT Warm-Up Tasks (1–9) — Consolidated PyTorch + CV Pipeline

This repository contains **all nine warm-up exercises for CSE 507** implemented in two Jupyter notebooks:

- **`CSE507_WarmUp1-7.ipynb`** — Tasks 1–7 (I/O, orientation, gender, age regression, binary segmentation, multi-class segmentation, organ localization)  
- **`CSE507_WarmUp8_9.ipynb`** — Tasks 8–9 (unsupervised anomaly detection & image clustering)

A full written summary of the methods and results appears in the accompanying **Final Warm-Up Report** (PDF).

⚠️ *All datasets (miniJSRT and course-provided splits) are for educational use only and not intended for clinical workflows.*

---

## 📂 Notebook Overview

### **`CSE507_WarmUp1-7.ipynb`**
This notebook implements the complete supervised pipeline described in the final report:

#### **Task 1 — Robust Image I/O (PNG/JPG/DICOM)**
- Read/write PNG/JPG via PIL  
- DICOM loading via pydicom  
- Percentile windowing (0.5–99.5%) for 16-bit CXRs  
- Safe conversion to 8-bit PNG

#### **Task 2 — Orientation Classification (Up/Down/Left/Right)**
- ResNet-18 backbone (ImageNet init, frozen)
- Linear classification head
- AdamW training, cross-entropy loss  
- Achieves extremely high accuracy on the Directions01 split

#### **Task 3 — Gender Classification**
- Same backbone as Task 2  
- Experiments with unweighted and class-weighted CE  
- Includes per-class F1 analysis

#### **Task 4 — Age Regression**
- ResNet-18 regression head (Smooth L1)  
- Z-scored targets, two-stage training (head-only → light fine-tuning)  
- MAE ≈ 6.9 years with TTA improvements  
- Includes scatter plots and error histograms

#### **Task 5 — Binary Segmentation (Organs vs. Background)**
- FCN32s-style head over a frozen ResNet-18
- Weighted CE + Dice loss  
- Morphological post-processing (cleanup & hole filling)

#### **Task 6 — Multi-class Segmentation (Lungs + Heart)**
- DeepLabV3-ResNet101  
- Combined loss: Focal CE + Soft Dice + Lovasz-Softmax  
- TTA inference (scales + flips)  
- Semantic cleaning (remove speckles, enforce connected lungs)  
- Achieves strong IoU (~0.91 on example validation prints)

#### **Task 7 — Three-Organ Localization (Boxes)**
- Derived from cleaned segmentation masks  
- Steps include:  
  - Largest-component lung filtering  
  - k-means (k=2) to separate fused lungs  
  - Geometric heart constraints (width 20–45% of thorax)  
  - Always returns 3 boxes (left lung, right lung, heart)

---

### **`CSE507_WarmUp8_9.ipynb`**
This notebook covers the two unsupervised tasks:

#### **Task 8 — Anomaly Detection via Denoising Autoencoder**
- Lightweight convolutional autoencoder  
- Trained on *normal* images (flips are anomalies)  
- Loss: 0.7 · (1 – SSIM) + 0.3 · L1  
- Outputs  
  - Reconstruction error maps  
  - AUROC/AP  
  - Top anomaly visualizations

#### **Task 9 — Image Clustering (Directions01)**
- Frozen ResNet-18 feature extraction  
- PCA whitening  
- k-means clustering (k = 5 for {up, down, left, right, flip})  
- Metrics: NMI, ARI, Purity, Silhouette  
- Nearly perfect reproduction of ground-truth clusters (NMI ≈ 0.98)

---

## 📑 Report

A detailed written summary of motivation, methodology, metrics, visualizations, and discussion for Tasks 1–9 is available in:

**`CSE507_FinalWarmup.pdf`** :contentReference[oaicite:1]{index=1}

The report includes:
- Plots for regression performance  
- Example segmentation masks  
- Bounding box overlays  
- PR curves for anomaly detection  
- Clustering scatterplots and tables

---

## 🚀 Environment & Requirements
The notebooks rely on:

- Python 3.8+
- PyTorch & torchvision
- NumPy, SciPy, scikit-learn
- OpenCV
- pydicom
- matplotlib  
- scikit-image

Install dependencies (example):

```bash
pip install torch torchvision numpy opencv-python pydicom scikit-learn scikit-image matplotlib
```
---

## 📚 References

- Chen, L.-C., Papandreou, G., Schroff, F., & Adam, H.  
  **Rethinking Atrous Convolution for Semantic Image Segmentation.**  
  arXiv:1706.05587, 2017.

- Berman, M., Triki, A., & Blaschko, M.  
  **The Lovász-Softmax Loss: A Tractable Surrogate for the Optimization of the Intersection-over-Union.**  
  CVPR, 2018.

- miniJSRT Teaching Dataset — Educational subset of JSRT chest radiographs.

- PyTorch & Torchvision Documentation  
  https://pytorch.org

- scikit-image Documentation  
  https://scikit-image.org

- scikit-learn Documentation  
  https://scikit-learn.org


---

## 🙏 Acknowledgments

- **Arizona State University — CSE 507** for providing the warm-up task framework and datasets.  
- **miniJSRT creators** for the educational chest X-ray subsets used throughout the exercises.  
- **PyTorch**, **Torchvision**, **OpenCV**, **NumPy**, **scikit-image**, and **scikit-learn** for the core tooling that enabled model training and visualization.  
- **DeepLabV3** and **Lovasz-Softmax** research authors for foundational segmentation methods used in Tasks 5–7.  
- All open-source contributors whose libraries and tools made these experiments possible.

---

