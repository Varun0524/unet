import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

model = load_model("polyp_unet_model.h5")

def show_sample(image_path, mask_path):
    image = cv2.resize(cv2.imread(image_path), (256, 256)) / 255.0
    pred_mask = model.predict(np.expand_dims(image, axis=0))[0]

    true_mask = cv2.resize(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE), (256, 256)) / 255.0

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Image")

    plt.subplot(1, 3, 2)
    plt.imshow(true_mask, cmap='gray')
    plt.title("True Mask")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask.squeeze(), cmap='gray')
    plt.title("Predicted Mask")
    plt.show()

# Test with one sample
show_sample("dataset/images/0b792e26-e1dd-4fb7-a6b7-5d76f227a677.jpg", 
            "dataset/masks/0b556d02-f9ca-4270-b568-3200335c7d08.jpg")

