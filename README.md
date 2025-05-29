# 👗 Fashion Recommender System

A content-based image recommendation system built with **Deep Learning**, **Computer Vision**, and **Streamlit**. Upload an image of a clothing item and get visually similar fashion recommendations using **ResNet50** and **K-Nearest Neighbors (KNN)**.

---

## 📌 Overview

The Fashion Recommender System helps users discover clothing items that are visually similar to the one they upload. This can be particularly useful in e-commerce platforms, virtual wardrobes, or personal style assistants.

This project extracts deep image features using a pre-trained ResNet50 CNN model and compares the uploaded image's features to a dataset of fashion images using vector similarity.

---

## 🔍 Key Features

- 🖼️ Upload fashion images through an intuitive Streamlit interface
- 🤖 Automatic feature extraction using **ResNet50 (ImageNet pretrained)**
- 📌 5 nearest neighbor recommendation using **Euclidean Distance**
- 🔄 Efficient feature comparison using **scikit-learn’s NearestNeighbors**
- ⚡ Instant visual results displayed in columns

---

## 🏗️ Tech Stack

| Area           | Technology / Library              |
|----------------|-----------------------------------|
| Language       | Python 3.7                        |
| Deep Learning  | TensorFlow, Keras                 |
| ML Algorithms  | scikit-learn                      |
| Data Handling  | NumPy, Pickle                     |
| Visualization  | Streamlit                         |
| Others         | tqdm, Pillow (PIL), OS, zipfile   |

---

## 📁 Project Structure

Fashion_Recommender_System/
├── app.py # Streamlit app: image upload, recommendation logic
├── extract_embeddings.py # One-time feature extractor for dataset images
├── embeddings.pkl # Pre-computed feature vectors of dataset images
├── filenames.pkl # Corresponding image file paths
├── uploads/ # Uploaded user images
├── requirements.txt # Python dependencies
└── README.md # Project documentation

yaml
Copy
Edit

---

## 🔧 Setup Instructions

### 📌 Prerequisites

- Python **3.7.x** (Use `py -3.7` to activate if installed alongside other versions)
- Ensure you have `pip` installed and available

### ⚙️ Install Dependencies

```bash
pip install -r requirements.txt
