# Dataset Description

## Overview

The dataset used in this project was manually collected and prepared specifically for this work. It consists of hand gesture samples representing the 26 letters of the English alphabet (A–Z), using **MediaPipe** for hand landmark detection and feature extraction.

## Dataset Characteristics

- **Data Type:** Structured Data
- **Data Collection:** Manually collected
- **Feature Extraction:** MediaPipe
- **Classes:** 26 classes (A–Z)
- **Target Label:** The corresponding alphabet letter.

Each sample represents a single-hand sign. MediaPipe is used to detect and extract hand landmarks, which are represented as structured numerical features. Each sample is then associated with its corresponding alphabet letter as the target label.

## Data Privacy

The original dataset is privately collected and is not publicly available or included in this repository due to data privacy considerations.

Therefore, the dataset files are intentionally excluded from version control.

## Reference Dataset

For reference, a publicly available Kaggle dataset with a similar concept and structure is provided below:

- **[Kaggle – Sign Language MNIST / Alphabet Dataset](https://www.kaggle.com/datasets/eraakash/indian-sign-language-hand-landmarks-dataset)**

This dataset is shared for reference purposes only and is not the actual dataset used in this project.

## Notes

To reproduce the experiments, users need access to a dataset with the same structure and feature representation, containing MediaPipe hand landmark features and their corresponding alphabet labels.