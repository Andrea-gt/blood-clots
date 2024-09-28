
# 🧬 Blood Clot Etiology Classification

Welcome to the **Blood Clot Etiology Classification** project! This repository aims to classify blood clots depicted in high-resolution whole-slide digital pathology images as either **Cardioembolic (CE)** or **Large Artery Atherosclerosis (LAA)**. The dataset is derived from patients who have experienced acute ischemic strokes, and your task is to identify the etiology (cause) of the clots for each patient.

---

## 🗂️ Project Overview

This project involves deep learning techniques to analyze digital pathology images of blood clots and classify their origin. The dataset contains:
- **Training & Test Sets**: Slides with known CE or LAA etiologies.
- **Supplemental Slides**: Additional slides with unknown or other etiologies not used in training.

---

## 📊 Exploratory Data Analysis (EDA)

The EDA provides insights into the dataset before moving into modeling. Key analyses include:

- **Data Description**: Overview of the dataset's structure and content.
- **Null Value & Duplicate Analysis**: Checked for missing or repeated data to ensure data integrity.
- **Image Size Distribution**: A visualization of varying image sizes across the dataset.
- **Variable Classification**: Categorizing key variables for better understanding.
- **Label Distribution**: Analysis of the proportion of CE and LAA labels in the training data.

and plenty of other analysis.

---

## 🔧 Image Preprocessing

Before feeding the images into a Convolutional Neural Network (CNN), preprocessing steps ensure they are standardized:

1. **Resizing**: Each image is resized to 5% of its original dimensions to maintain consistency and reduce computational load.
2. **Grayscale Conversion**: Depending on the CNN model requirements, images are converted to grayscale for single-channel input.
3. **Normalization**: Pixel values are normalized to the range `[0, 1]` to ensure consistent intensity levels.
4. **Gaussian Blur**: Noise reduction is applied using a Gaussian blur filter to smoothen the images and eliminate unnecessary details.
5. **Image Dimension Adjustment**: Images are reshaped for compatibility with CNN input layers.

---

## Built Using 
![Static Badge](https://img.shields.io/badge/%20-brightgreen?style=flat&logo=jupyter&logoColor=violet&label=Jupyter&color=violet)

## Authors
- [Daniel Valdez](https://github.com/Danval-003)
- [Emilio Solano](https://github.com/emiliosolanoo21)
- [Adrian Flores](https://github.com/adrianRFlores)
- [Andrea Ramírez](https://github.com/Andrea-gt)
