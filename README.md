# Emotion Analysis via Deep Learning on Facial Videos

## 🎯 Introduction

This project aims to classify human emotions from facial video sequences using two deep learning architectures:  
a **Long Short-Term Memory (LSTM)** model and an **Attention-based Transformer** model.

The dataset used is **RAVDESS**, containing video recordings of 10 actors expressing 8 emotions:

> Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised

### Dataset Details
- Emotional Intensity: `01 = normal`, `02 = strong` (neutral only has normal)
- Two statements × Two repetitions per emotion
- Truncation: **First 60 frames** per video
- Image size: **64×64**
- Data split: **80% train / 20% validation**


## 🗃️ Dataset Preparation

Before training, videos are preprocessed using a custom pipeline:

1. **Download & unzip** video archives from Zenodo (10 actors).
2. **Extract frames & detect faces** using MTCNN.
3. **Crop and save** face images.
4. **Organize** images by actor and video ID for training.

Dataset preparation is made in this file `Dataset_preparation.py`


## 🧪 Model Architectures

### Common CNN Feature Extractor
- `Conv2D(16)` → ReLU → MaxPool → **(16×32×32)**
- `Conv2D(32)` → ReLU → MaxPool → **(32×16×16)**
- Flatten → **8192 features per frame**


### LSTM-Based Model
- **LSTM** processes 60 frames (hidden size: 128)
- Final hidden state → Fully connected layer → **8 emotion classes**

LSTM training is done here `Lstm_video_classification.ipynb`


### Attention-Based Model
- Same CNN backbone
- Project features into embedding space
- Add **positional encoding**
- **Transformer Encoder** (1 layer, multi-head attention)
- Global average pooling → Fully connected → **8 emotion classes**

Attention-based training is done here `Attention-Based_video_classification.ipynb`

## ⚙️ Training Parameters

Shared training configuration:

- **Optimizer**: Adam (`lr=1e-3`)
- **Loss**: CrossEntropy
- **Epochs**: 30


## 📊 Results

### LSTM Results
- **Training Accuracy**: 97%
- **Validation Accuracy**: 80%
- **Training Loss**: 11.87
- **Validation Loss**: 9.94

⚠️ Confusion on **neutral** class due to class imbalance.


### Attention-Based Results
- **Training Accuracy**: 88%
- **Validation Accuracy**: 73%
- **Training Loss**: 18.71
- **Validation Loss**: 14.20

⚠️ Misclassification on **neutral** and **angry** classes.


## 🔍 Comparison Table

| **Aspect**              | **LSTM**           | **Attention-Based**          |
|-------------------------|--------------------|-------------------------------|
| Validation Accuracy     | 80%              | 73%                        |
| Training Speed          | Fast             | Slower                     |
| Memory Handling         | Temporal memory  | Global attention           |
| Computational Cost      | Linear           | Quadratic (sequence-wise)  |
| Interpretability        | Limited          | Attention maps possible     |


## 🎥 Online Inference Interface

A real-time, CLI-based pipeline is available for emotion recognition:

#### 🔄 Pipeline Steps
1. Load pre-trained model
2. Detect faces (MTCNN)
3. Buffer 60 frames
4. Predict emotion
5. Save annotated output video

#### 🛠️ Usage

```bash
python online_inference.py --video path_to_video.mp4 \
                           --model_type [lstm|attention] \
                           --model_path path_to_model.pth
