import os
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, image_paths, mask_paths, batch_size=8, image_size=(256, 256)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, idx):
        batch_x = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.mask_paths[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = [cv2.resize(cv2.imread(x), self.image_size) / 255.0 for x in batch_x]
        masks = [cv2.resize(cv2.imread(y, cv2.IMREAD_GRAYSCALE), self.image_size) / 255.0 for y in batch_y]
        masks = [np.expand_dims(m, axis=-1) for m in masks]

        return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)
