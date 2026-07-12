# GeoHam: Geometric Hamiltonian Network for Remote Sensing Change Detection

<p align="center">

**Physics-Inspired Deep Learning for High-Resolution Remote Sensing Change Detection**

</p>

---

## Overview

**GeoHam** is a physics-inspired deep learning framework for binary change detection in high-resolution remote sensing imagery.

Unlike conventional change detection networks that rely solely on feature differencing or attention mechanisms, GeoHam models temporal image evolution as a **Hamiltonian dynamical system** using a **symplectic Störmer–Verlet integrator**. This preserves geometric properties during feature evolution and produces a physically meaningful representation of change.

The framework combines

- EfficientNet-B2 encoder
- Hamiltonian feature evolution
- U-Net decoder
- Physics-informed loss
- Topological persistence regularization
- Energy conservation constraint
- Exponential Moving Average (EMA)
- Test Time Augmentation (TTA)

to achieve accurate and stable change detection.

---

# Architecture

```
            Image T1                     Image T2
               │                            │
               └────────────┬───────────────┘
                            │
                Shared EfficientNet-B2 Encoder
                            │
        ┌──────────┬──────────┬──────────┬──────────┐
        │          │          │          │
     Skip 1     Skip 2     Skip 3   Bottleneck Features
        │          │          │          │
        │          │          │          ▼
        │          │          │   Symplectic Hamiltonian
        │          │          │      Integrator
        │          │          │          │
        └──────────┴──────────┴──────────┘
                            │
                      U-Net Decoder
                            │
                     Change Probability Map
```

---

# Key Features

- Physics-inspired feature evolution
- Symplectic Hamiltonian integration
- Energy conservation regularization
- Topological persistence loss
- EfficientNet-B2 pretrained encoder
- Lightweight U-Net decoder
- EMA training
- Test Time Augmentation
- Automatic threshold optimization
- Mixed Precision (AMP) training

---

# Model Components

## 1. Shared Encoder

The encoder extracts hierarchical multi-scale representations from the two temporal images using **EfficientNet-B2**.

Features are extracted at four different spatial resolutions.

---

## 2. Hamiltonian Feature Evolution

Instead of directly subtracting encoder features, GeoHam interprets deep features as the state of a Hamiltonian system.

The bottleneck features are evolved using a **Störmer–Verlet leapfrog integrator**, producing

- feature divergence
- energy sequence

which serve as the basis for change detection.

---

## 3. Decoder

The decoder follows a U-Net style architecture.

Skip connections are constructed using the absolute feature differences

```
|F(T1) − F(T2)|
```

while the bottleneck receives the Hamiltonian divergence representation.

---

## 4. Physics-Informed Loss

The total loss consists of three components

```
L = Lfocal+dice
  + λtopology × Ltopology
  + λenergy × Lenergy
```

where

- **Focal Dice Loss** improves segmentation performance.
- **Topological Persistence Loss** preserves structural consistency.
- **Energy Conservation Loss** encourages physically consistent Hamiltonian evolution.

---

# Training Pipeline

```
Input Images
      │
      ▼
Data Augmentation
      │
      ▼
EfficientNet-B2 Encoder
      │
      ▼
Hamiltonian Integrator
      │
      ▼
Decoder
      │
      ▼
Prediction
      │
      ▼
Physics Loss
      │
      ▼
Backpropagation
      │
      ▼
EMA Update
```

---

# Data Augmentation

Training uses Albumentations with

- Horizontal Flip
- Vertical Flip
- Random Rotation
- Color Jitter
- CLAHE
- Gaussian Blur
- Gaussian Noise
- Random Scaling
- Random Crop
- Image Normalization

---

# Supported Datasets

The code currently supports

| Dataset | Status |
|----------|--------|
| LEVIR-CD | ✅ |
| LEVIR-CD+ | ✅ |
| WHU-CD | ✅ |
| CDD | ✅ |

Dataset paths are configured inside the `Config.DATASETS` dictionary.

---

# Configuration

Main hyperparameters

| Parameter | Value |
|-----------|------:|
| Patch Size | 256 |
| Batch Size | 16 |
| Epochs | 100 |
| Learning Rate | 2e-4 |
| Optimizer | AdamW |
| Scheduler | OneCycleLR |
| Mixed Precision | Yes |
| EMA | Yes |
| TTA | Yes |

---

# Installation

Clone the repository

```bash
git clone https://github.com/your_username/GeoHam.git

cd GeoHam
```

Install dependencies

```bash
pip install -r requirements.txt
```

Main dependencies

```
torch
torchvision
timm
albumentations
numpy
Pillow
opencv-python
tqdm
```

---

# Dataset Structure

```
Dataset/

├── train/
│   ├── A/
│   ├── B/
│   └── label/
│
├── val/
│   ├── A/
│   ├── B/
│   └── label/
│
└── test/
    ├── A/
    ├── B/
    └── label/
```

If no validation folder exists, the code automatically performs a **90/10 train-validation split**.

---

# Training

Example

Train on LEVIR-CD

```python
train("LEVIR")
```

Train on CDD

```python
train("CDD")
```

Train on WHU-CD

```python
train("WHU")
```

---

# Evaluation

During evaluation the framework performs

- Exponential Moving Average inference
- Test Time Augmentation
- Threshold search on validation data
- Frozen threshold evaluation on test data

Reported metrics include

- F1 Score
- IoU
- Precision
- Recall
- Overall Accuracy

---

# Project Structure

```
GeoHam/

├── datasets/
├── models/
├── outputs/
├── train.py
├── README.md
└── requirements.txt
```

---

# Highlights

✔ Physics-guided feature evolution

✔ Symplectic Hamiltonian dynamics

✔ Topology-aware segmentation loss

✔ Energy conservation regularization

✔ Lightweight architecture

✔ End-to-end trainable

✔ Multi-dataset support

---

# Citation

If you use this work in your research, please cite

```bibtex
@article{GeoHam2026,
  title={GeoHam: A Physics-Inspired Hamiltonian Network for Remote Sensing Change Detection},
  author={Shailendra},
  journal={Under Review},
  year={2026}
}
```

---

# License

This repository is released under the MIT License.

---

# Acknowledgements

This work makes use of the following open-source libraries:

- PyTorch
- timm
- Albumentations
- NumPy
- Pillow
- tqdm

We thank the authors of the LEVIR-CD, LEVIR-CD+, WHU-CD, and CDD datasets for making their datasets publicly available.

---

# Contact

For questions, suggestions, or collaborations, please open an issue or contact the repository author.
