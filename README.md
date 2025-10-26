# 🧪 Deepfake Face Generator (DCGAN + TensorFlow)

A **Deep Convolutional Generative Adversarial Network (DCGAN)** built with **TensorFlow** to generate realistic human face images from random noise.  
This project treats Generative AI like real engineering — from model design and training to deployment readiness.

---

## 🧠 Overview

This project implements a **DCGAN** that learns to synthesize human faces by training on real-world face datasets and enhancing the generated output resolution.

### ✨ Key Features
- Generates **synthetic human faces** from random noise  
- Built with **TensorFlow/Keras** using a **DCGAN architecture**  
- Produces **512×512 resolution** images  
- Images are **upscaled using the Lanczos4 interpolation method** for superior visual quality  
- Saves **model weights** and **sample outputs** automatically after each epoch  
- Supports both **CPU and GPU** training environments  

---

## 🖼️ Sample Outputs

> Example generated faces after **Epoch 50**  
> *(Generated and upscaled using Lanczos4 interpolation)*  
>  
(https://raw.githubusercontent.com/Tanishk237/Deepfake-Face-GAN/main/Tests/Test1.jpeg)
(https://raw.githubusercontent.com/Tanishk237/Deepfake-Face-GAN/main/Tests/Test2.jpeg)

---

## 🚀 Project Highlights

- ✅ **TensorFlow-based DCGAN** implementation  
- ✅ Compact generator model (**~19 MB**)  
- ✅ Experiment tracking with **MLflow**  
- ✅ Automatic saving of training samples and checkpoints  
- ✅ Optimized for efficient training on **CPU and GPU**  
- ✅ Ready for integration with **FastAPI**, **Streamlit**, or **Gradio** for serving  

---

## 🌍 Dataset

- Trained on the **50K Celebrity Faces Image Dataset** from Kaggle  
- Dataset Source: [CelebFaces Kaggle Dataset](https://www.kaggle.com/datasets/farzadnekouei/50k-celebrity-faces-image-dataset)  
- Each image is preprocessed, normalized, and aligned for consistent training performance  

---

## 🛠️ Tech Stack

- **Python**  
- **TensorFlow / Keras**  
- **MLflow**  
- **NumPy**  
- **OpenCV**  
- **Matplotlib**

---

## 📦 Output

- Generated face samples (`/outputs/samples/`)  
- Model weights (`/outputs/models/`)  
- MLflow experiment logs (`/mlruns/`)

---

## 🧩 Future Work

- Enhance diversity and realism with **StyleGAN-based architecture**  
- Add **conditional generation** for attribute control (e.g., gender, hair color, age)  
- Integrate a **web-based interactive demo**  

---

> 💡 *A TensorFlow-powered Deepfake Face Generator that scales up realism — one face at a time.*
