# COMSYS-Hackathon-5,2025
---
##  Challenge Overview 

### TASK A: GENDER CLASSIFICATION👩‍🦰🧓
- **Objective:** 
Develop a robust gender classification model capable of accurately predicting male/female labels from facial images captured under adverse environmental conditions (e.g., low light, shadows, rain, or haze)
- **Dataset Structure:**
```
dataset/
├── train/
│ ├── male/ # 1532 images
│ └── female/ # 394 images
└── val/
├── male/ # 317 images
└── female/ # 105 images
```
- **Model Goal:** 
Train a model to predict gender from faces that generalizes well to non-ideal images—low light, motion blur, or weather effects (binary Classifier).
---
## 🎬 Demo Preview 
<div align="center">
  <img src="https://github.com/dolly677/COMSYS-Hackathon-5-2025/blob/3cf4147860a06f467b63d6499a2f1f792af9ce1b/Image/projectdemo01-ezgif.com-video-to-gif-converter.gif?raw=true" 
       width="80%"
       style="border-radius: 8px;"
       alt="Looping animation">
       <p><em> Gender Classification </em></p>
</div>

- **PROJECT STRUCTURE**
```
Gender_classification 
│
├── .venv/ # Virtual environment directory
│
├── best_model/ # Directory containing the best trained model
│ └── gender_classification_transfer_learning_with_ResNet18.pth
│
├── Comsys_Hackathon5/ # Main project directory
│ └── Task_A/ # Task specific directory
│      ├── train/ # Training data
│      └── val/ # Validation data
│
├── previous_used_models/ # Directory containing previously used models
│      ├── mobilenet_gender_final_v2.keras
│      ├── model.pth
│      └── xception_v5_03_0.939.h5
│
├── app.py # Main application file
│
├── config.py # Configuration settings
├── data_loader.py # Data loading utilities
├── model.py # Model architecture definitions
├── requirement.txt # Project dependencies
├── train.py # Training script
└── utils.py # Utility functions
```
### 🧠 MODEL DESCRIPTION : ResNet-18 for Gender Classification

We employed **ResNet-18**, a deep convolutional neural network introduced in the paper _["Deep Residual Learning for Image Recognition"](https://arxiv.org/abs/1512.03385)_ by He et al., for the task of **gender classification**.ResNet-18 is a lightweight and efficient deep residual network with 18 layers, well-suited for image-based tasks with limited data and resources.

#### 🔍 Why ResNet-18?

- **Lightweight & Efficient**: With only 18 layers, ResNet-18 is ideal for tasks requiring real-time or resource-efficient inference.
- **Residual Connections**: These help avoid vanishing gradients and enable better feature learning in deeper networks.
- **Pretrained Strength**: Leveraging ImageNet-pretrained weights allows faster convergence and improved performance on smaller datasets.

>Works well on small-to-medium datasets without overfitting.
![Alt text](https://github.com/dolly677/COMSYS-Hackathon-5-2025/blob/3511035c45b6724f4c5f3b445856cfcff35b6145/resnet-architectures-34-101.png)

#### ⚙️ Model Overview

| Model     | ImageNet Top-1 Error | ImageNet Top-5 Error |
|-----------|----------------------|-----------------------|
| ResNet-18 | 30.24%               | 10.92%                |

In our project:
- We replaced the final fully connected layer with a layer having **2 output nodes** corresponding to the classes: `Male` and `Female`.
- We fine-tuned the network on a custom dataset of face images.

#### 📈 Performance

✅ Our trained ResNet-18 model achieved an impressive **98% accuracy** on the validation/test set.

> 📌 _Note: This high accuracy highlights the suitability of ResNet-18 for gender classification tasks with well-curated datasets._

**In this project, ResNet-18 was selected for its lightweight design, fast training, and solid performance on small to medium image datasets such as gender classification.**

**🧩 Model Pipeline**
```
Input Image (224x224 RGB)
      ↓
Data Augmentation (flip, rotate, color jitter)
      ↓
Pretrained ResNet-18 (on ImageNet)
      ↓
Custom Classification Head:
    - Flatten
    - Fully Connected Layer (2 output neurons: [Male, Female])
      ↓
Softmax Activation
      ↓
Predicted Label
```
## 📊 Dataset Statistics
| Set      | Male | Female | Total | Imbalance Ratio |
|----------|------|--------|-------|------------------|
| Train    | 1532 | 394    | 1926  | 3.9:1            |
| Val      | 317  | 105    | 422   | 3.0:1            |
| **Total**| 1849 | 499    | 2348  | 3.7:1            |


### 📈 Dataset  Overview

<div align="center">
  <img src="https://github.com/dolly677/COMSYS-Hackathon-5-2025/blob/03e7855810638d392a5d46787ec8af507fb7737e/Image/full_dataset_class_distribution.png" alt="Full Dataset Distribution" width="48%" />
  &nbsp;&nbsp;
  <img src="https://github.com/dolly677/COMSYS-Hackathon-5-2025/blob/03e7855810638d392a5d46787ec8af507fb7737e/Image/datasize.png" alt="Dataset Size" width="48%" />

  <br><br>
  <b>Left:</b> Full Dataset Distribution 
  &nbsp;&nbsp;&nbsp;&nbsp;
  <b>Right:</b> Overall Datasplit 
</div>

---

**Augmentation Techniques Applied:**
- Horizontal Flip (p=0.5)
- Random Rotation (±15°)
- Color Jitter (brightness=0.2, contrast=0.2)
- Random Grayscale (p=0.1)
  
### 🏋️‍♀️ Training Details

| Parameter            | Value                              |
|----------------------|------------------------------------|
| Loss Function        | Cross Entropy Loss                 |
| Optimizer            | Adam                               |
| Learning Rate        | 0.001 (set via `config.py`)        |
| Epochs               | 10–30 (with early stopping)        |
| Evaluation Metrics   | Accuracy, Precision, Recall        |



### 🧪 Evaluation Metrics

| Metric    | Value   |
|-----------|---------|
| Accuracy  | 0.9869  |
| Precision | 0.9849  |
| Recall    | 0.9855  |

**🛠 Fine-tuning**

The ResNet-18 model was loaded with pretrained ImageNet weights, and only the final fully connected (FC) layer was replaced and fine-tuned for binary classification (male, female).

### 🧪 Sample Predictions

Below are example outputs from the trained model:

![Alt text](https://github.com/dolly677/COMSYS-Hackathon-5-2025/blob/5b8dfcbb680675160aa6b94aaa7bfb3814d7c6d5/Sample%20Results.png)
![Alt text](https://github.com/dolly677/COMSYS-Hackathon-5-2025/blob/4e59478cf5ed25023ca3127636cec13ecfc12491/sampleresult2.png)
![Alt text](https://github.com/dolly677/COMSYS-Hackathon-5-2025/blob/4e59478cf5ed25023ca3127636cec13ecfc12491/sampleresult3.png)
> These predictions were made using the inference script on  test images.


---

### 📊 Confusion Matrices — **Testing Set**

<div align="center">
  <img src="https://github.com/dolly677/COMSYS-Hackathon-5-2025/blob/03e7855810638d392a5d46787ec8af507fb7737e/Image/conmatrix1test.png" alt="Testing Set Confusion Matrix" >
  &nbsp;&nbsp;
  <br><br>
  <b>Testing Set Confusion Matrix </b> 
</div>


### 📊 Confusion Matrices — **Validation Set**

<div align="center">
  <img src="https://github.com/dolly677/COMSYS-Hackathon-5-2025/blob/03e7855810638d392a5d46787ec8af507fb7737e/Image/conmatrix2test.png" alt="Validation Set Confusion Matrix" width="45%" />
  <b> Validation Set Confusion Matrix </b> 
</div>

---

### 💻 Hardware Requirements
**Minimum Configuration:**
- **GPU:** NVIDIA GTX 1060 (4GB VRAM) or equivalent
- **CPU:** 4-core processor (Intel i5 or AMD Ryzen 5)
- **RAM:** 8GB
- **Storage:** 5GB available space

**Recommended Configuration:**
- **GPU:** RTX 3060/T4 (8GB+ VRAM) for faster training
- **CPU:** 8-core processor
- **RAM:** 16GB+
- **Storage:** SSD preferred

---
## 🚀How to Reproduce the Results 
**Follow the steps below to reproduce the results of the gender classification model:**

**1. Clone the Repository**
```
git clone https://github.com/dolly677/COMSYS-Hackathon-5-2025.git
cd Gender_classification
```
**2. Set Up the Environment**

**Option A: Virtual Environment (Recommended)**
```
python -m venv .venv          # Create virtual environment
source .venv/bin/activate     # Linux/Mac
.\.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```
**Option B: Conda**
```
conda create -n gender_classification python=3.9
conda activate gender_classification
pip install -r requirements.txt
```
**3. Prepare the Dataset**
Organize the dataset into the following structure:
```
Task_A/
├── train/
│   ├── male/
│   └── female/
└── val/
    ├── male/
    └── female/
```
Place appropriate images of males and females into each folder. Supported formats: .jpg, .png, etc.

**4. Train the Model**

To train the model using the training dataset, run:
```
python train.py
```
**Configurations:** Modify hyperparameters in config.py or via command line.

**Logs:** Training progress will be logged (check logs/).

**5. Launch the Web App**

If you want to test the model using a browser interface, run the Streamlit app:
```
  streamlit run app.py
```
**✨Features:**
- **Image Upload**: Drag-and-drop support for JPG/PNG/JPEG
- **Real-time Prediction**: Gender classification + confidence scores
- **Responsive UI**: Works on desktop and mobile
This will launch a web UI for uploading images and displaying gender prediction results.

**6.Access the Interface**

![Alt text](https://github.com/dolly677/COMSYS-Hackathon-5-2025/blob/dc7b6666508f49a70598f49089c5e48614b1d110/app.png)

---

## 🤝 Acknowledgements

Developed by [AI-dentifiers](https://github.com/dolly677/COMSYS-Hackathon-5-2025.git) and contributor.  
For academic/educational use only.
