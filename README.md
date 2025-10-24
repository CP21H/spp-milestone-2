# Student Proud Project, Milestone 2

This project implements and evaluates two Deep Learning (DL) models—**VGG-16 Standalone** and **Hybrid VGG-16 + XGBoost**—to classify kidney CT scan images into four categories: Normal, Cyst, Stone, and Tumor. The goal is to replicate and exceed the performance benchmarks from the reference paper, **[Enhanced Automatic Identification of Kidney Cyst, Stone and Tumor using Deep Learning](https://ieeexplore.ieee.org/document/10617000).**

## 1. Project Overview

The core challenge of this project was to overcome severe class imbalance and model underfitting/overfitting issues to achieve high, robust classification accuracy, measured by **Accuracy**.

| Model | Achieved Accuracy | Paper's Target Accuracy | Result |
| :--- | :--- | :--- | :--- |
| **VGG-16 Standalone** | **96.01%** | 90.99% | **Exceeded Benchmark** |
| **Hybrid VGG-16 + XGBoost** | **97.11%** | 99.76% | **Currently Below Target** (Requires tuning in-post) |

## 2. Setup and Installation

### Prerequisites

To run the provided Google Colab Notebook (`milestone2.ipynb`), you need the following:

1.  **Google Account and Google Colab:** The notebook is designed to run in a Google Colab environment, utilizing its GPU capabilities.
2.  **Dataset:** The raw dataset must be downloaded and placed into your Google Drive.

### 2.1 Dataset Download

The project relies on the following Kaggle dataset:

> **[CT Kidney Dataset: Normal, Cyst, Tumor, and Stone](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone/data)**

**Before running any code, download this dataset, and place the unzipped contents into your Google Drive.**

### 2.2 Running the Notebook

1.  Open the `milestone2.ipynb` file in Google Colab.
2.  Ensure your Colab runtime is set to **GPU** (Runtime -> Change runtime type).
3.  Execute the first code cell to mount your Google Drive. This is **essential** for accessing the dataset:

    ```python
    !cp -r /content/drive/MyDrive/CTKidneyDataset /content/
    ```

    **(Note: If your dataset path is different, adjust this copy command accordingly.)**
4.  Run all subsequent code cells sequentially.

## 3. Key Model Implementation Details

### VGG-16 Standalone Model

This model achieved a heightened performance compared to the target.

* **Architecture:** Fine-tuned VGG-16 pre-trained on ImageNet, followed by custom Dense layers for classification.
* **Key Techniques for Success:**
    * **Stratified Data Split:** Used to ensure proportional representation of the four classes in training and validation sets.
    * **Class Weighting:** Applied during training (`model.fit`) to counter class imbalance.
    * **Very Low Learning Rate (1e-6):** Essential for stable fine-tuning of the frozen VGG-16 weights.

### Hybrid VGG-16 + XGBoost Model

This model is designed to leverage VGG-16 as a feature extractor and XGBoost as a highly efficient classifier.

* **Process:**
    1.  VGG-16 (frozen base) extracts $\mathbf{25,088}$ features per image.
    2.  XGBoost trains on these feature vectors.
* **Tuning Status:** The model was aggressively tuned (`n_estimators=1000`, `max_depth=10`) with application of **sample weighting** to achieve as close to the target accuracy as possible.

## 4. Final Performance Visualization

The project generated detailed metrics and visualizations, which can be found in the notebook's output cells:

* **Confusion Matrices:** Show the true number of misclassifications for each model.
* **Classification Reports:** Provide Precision, Recall, and F1-scores for each of the four classes.
* **Precision-Recall Curves:** Offer a robust visualization of the model's performance on minority classes.
