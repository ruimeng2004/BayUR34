# BayUR34-- a Chest X-ray Image Classification Model

This repository contains a hybrid deep learning model for medical image classification, combining a UNet-based feature extractor with a ResNet34 classifier. The system is designed for binary classification of chest X-ray/CT images (Normal vs. Abnormal) with enhanced feature extraction capabilities. This model was developed as a course project for ECE4513 — Computer Vision and Image Processing.

## 📂 Dataset

This project utilizes the **Shenzhen Chest X-ray Set**, a publicly available dataset designed for tuberculosis diagnosis using chest radiographs.

### 📌 Dataset Overview

The Shenzhen Chest X-ray Set is a tuberculosis digital imaging dataset created by the **Lister Hill National Center for Biomedical Communications (LHNCBC)** at the **U.S. National Library of Medicine (NLM)** in collaboration with the **Third People's Hospital of Shenzhen** and **Guangdong Medical College** in China.

- 📸 **Total Images**: 662 chest X-ray images  
- 🧍 **Normal Cases**: 326  
- ⚠️ **Abnormal (TB) Cases**: 336  
- 🖼️ **Image Format**: PNG  
- 📐 **Resolution**: Up to 3000×3000  
- 📄 **Clinical Info**: Accompanying `.txt` files include age, gender, and diagnostic remarks  

> This dataset has been de-identified and is exempt from IRB (Institutional Review Board) review.

---

### 📈 Meta Statistics

| Property           | Value                                      |
|--------------------|--------------------------------------------|
| Total samples      | 662 (Normal 326;Abnormal 336)       |
| Abnormal rate      | 336 / 662 ≈ 50.75%                         |
| Image resolution   | Min: (1130, 948); Max: (3001, 3001); Median: ~2730×2940 |
| Format             | PNG images + TXT clinical data             |
| Size               | ~3.6 GB                                    |

---

| Preprocessing Method | Accuracy (ACC) | Precision (Pre) | Recall (Rec) | F1 Score |
|----------------------|---------------|----------------|--------------|---------|
| None (Baseline)      | 0.8195        | 0.8243         | 0.8472       | 0.8356  |
| Spatial              | 0.8346        | 0.8235         | 0.8485       | 0.8358  |
| Frequency            | 0.8496        | 0.8548         | 0.8281       | 0.8413  |
| **Bayesian**         | **0.9023**    | **0.9155**     | **0.9028**   | **0.9091** |

### 🔗 Dataset Links

- 🔹 **Official Site**: [LHNCBC TB Image Data Sets](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#tuberculosis-image-data-sets)  
- 🔹 **Download ZIP**: [ChinaSet_AllFiles.zip](https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip)  
- 🔹 **Related Publication**: [NIH Article on PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/)


## Image Preprocessing -- Bayesian Wavelet Denoising

- **Method**: Adaptive wavelet thresholding with Bayesian estimation
- **Key Features**:
- Automatic threshold calculation（Uses Bayesian statistics to determine optimal thresholds）
- Multi-scale processing（Default 4-level wavelet decomposition）
- Adaptive denoising （Level-specific soft-thresholding）
- **Recommended wavelets**: 
  - `bior3.3` (Biorthogonal 3.3)
  - `sym4` (Symlet 4)

### Technical Specifications
| Parameter        | Default Value | Description                          |
|------------------|---------------|--------------------------------------|
| `wavelet`        | `bior3.3`     | Wavelet basis function               |
| `level`          | 4             | Decomposition levels                 |
| `threshold_mode` | `soft`        | Thresholding method (soft/hard)      |

## 🏗️ Architecture Design
The system employs a novel two-stage architecture:
### ​​1.UNet Encoder with Attention​​：
- Lightweight encoder with 3 convolutional blocks
- Integrated attention mechanism
- Outputs 128-channel feature maps at 1/4 resolution
### ​​2.Adapted ResNet34 Classifier​​
- Modified input layer to accept UNet features
- Retained pretrained weights from ImageNet
- Custom binary classification head

## 🛠️ Key Components

### 1.Feature Extraction Module
- Hierarchical feature learning​​ through progressive downsampling
- Instance normalization​​ for contrast invariance
- Spatial attention gate​​ for adaptive feature refinement
### 2.Classification Module
- ​​Differential learning rates​​ for each component
- Focal loss​​ for handling class imbalance
- Early stopping​​ with configurable patience

## Run The Model

### 1. Preprocess images

```bash
python preprocess_images.py --input_dir data/raw --output_dir data/processed_X --mode X
```
Available modes: `bayesian`,` none`

### 2. Train the model

```bash
python train.py --data_dir data/processed_X --batch_size 32 --epochs 300 --lr 0.0001
```
### Evaluation Metrics
- Training and validation loss and accuracy
- Validation precision, recall, and F1-score

## Environment Dependencies
- Python 3.7+
- PyTorch
- torchvision
- scikit-learn
- OpenCV (cv2)
- Pillow (PIL)

## Project Structure
```bash
├── dataset.py              # Dataset class
├── preprocess_images.py    # Image preprocessing script
├── train.py                # Training script
├── model.py                # Structure of the classification model
├── data/                   # Data directory
│   ├── raw/                # Raw images
│   └── processed_X/    # Processed images
```
## Contact
- Feel Free to contact the author if you have any questions.
  
