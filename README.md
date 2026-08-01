Medical Image Segmentation with U-Net
A Python deep learning project that automatically segments lung boundaries in chest X-ray images using a U-Net convolutional neural network.

Overview
Accurate lung segmentation is a key preprocessing step in computer-aided diagnosis systems. This project implements the U-Net architecture — an encoder–decoder network with skip connections designed for biomedical image segmentation — and trains it to produce pixel-level lung masks from chest X-rays.

Pipeline
Data preprocessing — images are converted to grayscale, resized to a fixed input size, and pixel values are normalized (OpenCV, NumPy)
Dataset partitioning — the dataset is split into training, validation, and test sets using Scikit-learn to ensure unbiased evaluation
Model training — the U-Net model is trained on image–mask pairs
Evaluation — accuracy metrics are computed on the held-out test set to validate generalization on unseen data
Tech Stack
Language: Python 3
Deep learning: TensorFlow/Keras (U-Net architecture)
Image processing: OpenCV, NumPy
Evaluation & splitting: Scikit-learn

What I Learned
Implementing an encoder–decoder segmentation architecture with skip connections
Building a reproducible data processing pipeline for medical imaging data
Designing proper train/validation/test splits and evaluating model generalization
