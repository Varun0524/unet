# Polyp Segmentation with U-Net

A deep learning project that segments polyps in colonoscopy images using a U-Net convolutional neural network, built with TensorFlow/Keras.

## Overview

Polyps are growths in the colon that can become cancerous. This project trains a U-Net model to mark, pixel by pixel, where a polyp is in a colonoscopy image. The input is a 256 x 256 colour image and the output is a 256 x 256 mask.

I built this by following a tutorial, to learn how encoder-decoder segmentation networks work.

## Model

- 4 encoder levels (64, 128, 256, 512 filters), each with two 3x3 convolutions, batch normalization, ReLU, and 2x2 max pooling
- Bottleneck with 1024 filters
- 4 decoder levels using transposed convolutions, with skip connections from the matching encoder level
- Final 1x1 convolution with sigmoid, giving a probability per pixel

## Data and training

- 400 colonoscopy images with 400 polyp masks
- Images resized to 256 x 256 and pixel values scaled to 0-1 (OpenCV, NumPy); masks read in grayscale
- 80/20 train-validation split with Scikit-learn (320 training, 80 validation)
- Loss: binary cross-entropy. Optimizer: Adam. 25 epochs, batch size 8
- Data loaded in batches with a Keras `Sequence` generator

## Results

On the 80 validation images:

| Metric | Value |
| --- | --- |
| Pixel accuracy | 0.87 |
| Dice score | 0.53 |

Accuracy is high mainly because most pixels are background. The Dice score shows the overlap between predicted and true masks is moderate.

## Planned improvements

- Data augmentation (flips, rotations)
- Dice loss, or a combined cross-entropy and Dice loss
- A separate held-out test set
- Thresholded masks and RGB colour order in the prediction display

## Files

| File | Purpose |
| --- | --- |
| `unet_model.py` | U-Net architecture |
| `data_generator.py` | Loads and preprocesses image-mask batches |
| `train_unet.py` | Splits the data and trains the model |
| `predict_show.py` | Shows an image, its true mask, and the predicted mask |
| `evaluate.py` | Computes validation accuracy and Dice score |
| `check_gpu.py` | Lists the devices TensorFlow can use |

## How to run

```
pip install tensorflow opencv-python numpy matplotlib scikit-learn
python train_unet.py
python evaluate.py
python predict_show.py
```

The dataset goes in `dataset/images` and `dataset/masks`. The dataset and the trained model file are not included in this repository.
