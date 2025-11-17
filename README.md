# APTOS Blindness Detection — Deep Learning Approaches

This project implements multiple computer vision models to classify diabetic retinopathy severity from retinal fundus images (APTOS 2019 dataset).  
The goal is to compare classical baselines, simple CNNs, and modern deep architectures on a consistent preprocessing and evaluation pipeline.

---

## 1. Project Overview

Diabetic Retinopathy (DR) is classified into 5 severity levels:

0 — No DR  
1 — Mild  
2 — Moderate  
3 — Severe  
4 — Proliferative DR  

This project investigates how different model families perform on this task:

- Null model (majority class baseline)  
- Logistic Regression on hand-crafted features  
- Simple CNN  
- ResNet-50 (transfer learning)  
- DenseNet-121 (transfer learning)

Performance is evaluated using accuracy, precision, recall, F1-score, confusion matrices, and Grad-CAM visualizations.

---

## 2. Dataset

**Source:** APTOS 2019 Blindness Detection (Kaggle)  
**Total images:** 3662  
- Training: 2929 images  
- Validation: 733 images  

Each sample includes:  
- `id_code`: image identifier  
- `diagnosis`: integer label (0–4)  
- Retinal image (1024x1024 PNG)

Dataset is not included in the repository.

---

## 3. Preprocessing Pipeline

We apply medically-motivated preprocessing steps:

1. CLAHE (Adaptive Histogram Equalization)  
2. Circular Crop (remove black borders)  
3. Resize  
   - SimpleCNN: 128×128  
   - ResNet-50: 224×224  
   - DenseNet-121: 320×320  
4. Albumentations augmentations for ResNet and DenseNet:
   - RandomRotate90  
   - Horizontal/Vertical Flip  
   - ShiftScaleRotate  
   - RandomBrightnessContrast  
   - Gaussian/Motion Blur  
   - Normalize (ImageNet stats)

All dataloaders and class weights are cached via pickle.

---

## 4. Models Implemented

### 4.1 Null Model  
Baseline predicting the most frequent class.

### 4.2 Logistic Regression  
Features extracted:  
- Brightness  
- Blurriness  
- Entropy  
- Width, height  
- RGB channel means and standard deviations  

Trained using scikit-learn with StandardScaler.

### 4.3 Simple CNN  
A lightweight 3-layer convolutional network.  
Input size: 128×128  
Parameters: 2.12M

### 4.4 ResNet-50 (Transfer Learning)  
Pretrained on ImageNet.  
Fine-tuned end-to-end.  
Input size: 224×224  
Parameters: 25.5M

### 4.5 DenseNet-121 (Transfer Learning)  
Pretrained on ImageNet.  
Fine-tuned end-to-end.  
Input size: 320×320  
Parameters: 7.97M

DenseNet-121 achieved the highest validation accuracy.

---

## 5. Training Details

| Model            | Training Time | Image Size | Parameters |
|------------------|--------------:|-----------:|-----------:|
| Null Model        | < 1 sec       | N/A        | N/A        |
| Logistic Regression | < 1 sec    | N/A        | N/A        |
| Simple CNN       | 24 min 54 s   | 128×128    | 2.12M      |
| ResNet-50        | 57 min 16 s   | 224×224    | 25.5M      |
| DenseNet-121     | 50 min 28 s   | 320×320    | 7.97M      |

### Hardware and System Configuration

All deep learning experiments in this project were conducted on the
**University of Georgia — School of Computing GPU cluster**, using the
following hardware and software configuration:

### GPU Resources
- **2 × NVIDIA GeForce RTX 3090 GPUs**
- **24 GB GDDR6X VRAM per GPU**
- **CUDA Version:** 12.2
- **NVIDIA Driver:** 535.161.07

### CPU & Memory
- **CPU:** Intel Core i9-10920X (12 cores, 24 threads)
- **Clock Speed:** 3.50 GHz (Turbo up to 4.80 GHz)
- **System Memory:** 256 GB RAM
- **Swap:** 8 GB

### Operating System
- **Ubuntu 22.04.5 LTS (Jammy)**

### Software Environment
- **PyTorch (CUDA-enabled)** within the project conda environment  
  (Training performed using GPU; CPU-only PyTorch was detected only in the base shell environment.)
- **Conda environment:** `aptos-env`
- **CUDA toolkit:** 12.2
- **Python:** 3.10

---

## 6. Results

### 6.1 Validation Accuracy

| Model              | Accuracy |
|--------------------|---------:|
| Null Model         | 0.4925   |
| Logistic Regression| 0.6317   |
| Simple CNN         | 0.7100   |
| ResNet-50          | 0.7271   |
| DenseNet-121       | 0.7735   |

DenseNet-121 performs best overall.

### 6.2 Classification Reports
(See `notebooks/` for complete metrics and confusion matrices.)

---


## 7. Repository Structure
aptos-blindness-detection/
│
├── data/                  # (excluded from repo)
├── models/                # Trained model weights
├── notebooks/             # All Jupyter notebooks
├── scripts/               # .py versions of notebooks
│
├── README.md
├── requirements.txt
└── environment.yml

---
## 8. How to Run


### 1. Dataset Preparation

Download the *APTOS 2019 Blindness Detection* dataset from Kaggle and place the files in the following directory structure:

```
data/raw/train.csv
data/raw/train_images/
```

The image filenames must correspond to the `id_code` field in `train.csv`.

---

### 2. Environment Setup

#### Using Conda:
```bash
conda env create -f environment.yml
conda activate aptos-env
```

#### Using pip:
```bash
pip install -r requirements.txt
```

This environment includes all dependencies required for preprocessing, model training, inference, and evaluation.

---

### 3. Using Pre-Trained Models (Recommended)

Pre-trained model checkpoints are included in the repository under:

```
models/best_resnet50.pth
models/best_densenet121_320.pth
```

These models can be evaluated directly without retraining.

#### Inference and Explanation Notebooks:
- **ResNet-50 Inference and Grad-CAM**
  ```
  notebooks/05_ResNet50_Inference_GradCAM.ipynb
  ```
- **DenseNet-121 Inference**
  ```
  notebooks/07_Densenet121_Inference.ipynb
  ```

These notebooks automatically:
- Load preprocessed dataloaders  
- Restore pretrained weights  
- Compute validation metrics  
- Generate interpretability visualizations (Grad-CAM)

---

### 4. (Optional) Training Models from Scratch

To reproduce training results, the following sequence should be followed:

#### Step 1 — Preprocessing
```
notebooks/01_Preprocessing.ipynb
```
This notebook constructs and serializes dataloaders into:
```
data/processed/loaders.pkl
```

#### Step 2 — Model Training
The project includes training notebooks for all models:

| Model               | Notebook Path                                   |
|---------------------|--------------------------------------------------|
| Simple CNN          | `02_SimpleCNN_Training.ipynb`                    |
| Logistic Regression | `03_Logistic_Regression.ipynb`                   |
| ResNet-50           | `04_ResNet50_Training.ipynb`                     |
| DenseNet-121        | `06_Densenet121_Training.ipynb`                  |

Each notebook outputs a trained checkpoint to the `models/` directory.

---

### 5. Python Script Versions

In addition to notebooks, executable `.py` scripts are provided for all major stages.  
Scripts are located under:

```
scripts/
```

Examples include:

- `preprocessing.py`
- `simplecnn_training.py`
- `logistic_regression.py`
- `resnet50_training.py`
- `densenet121_training.py`
- `resnet50_inference_gradcam.py`
- `densenet121_inference.py`



To execute a script:
```bash
python scripts/resnet50_inference_gradcam.py
```

---

### 6. Model Comparison

A comprehensive comparative analysis of all models (Simple CNN, Logistic Regression, ResNet-50, DenseNet-121) is provided in:

```
notebooks/Model_Comparison.ipynb
```

This includes:
- Accuracy, Precision, Recall, F1-scores
- Per-class metrics
- Confusion matrices
- Training time comparison
- Parameter count comparison

---
## 9. Key Findings

- **Classical baselines are insufficient:** The null model and logistic regression perform poorly on minority classes due to limited feature richness and the complexity of retinal pathology.
- **Deep learning improves performance substantially:** A lightweight Simple CNN already outperforms classical models, demonstrating the importance of spatial feature extraction.
- **Transfer learning provides the strongest gains:** Pretrained architectures such as ResNet-50 and DenseNet-121 significantly boost accuracy due to their rich learned feature hierarchies.
- **DenseNet-121 achieves the best overall accuracy (0.7735):** Its densely connected layers promote stronger gradient flow and feature reuse, which improves discriminative ability.
- **Grad-CAM visualizations validate model behavior:** Deep models focus on clinically relevant regions such as hemorrhages, microaneurysms, and exudates, increasing trust in predictions.
- **Preprocessing plays a critical role:** Higher-resolution images (320×320) combined with CLAHE and circular cropping improve the quality of retinal features available to the model.