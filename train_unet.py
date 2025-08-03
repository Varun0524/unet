import glob
from sklearn.model_selection import train_test_split
from unet_model import build_unet
from data_generator import DataGenerator
import tensorflow as tf

# Load paths
image_paths = sorted(glob.glob("dataset/images/*.jpg"))
mask_paths = sorted(glob.glob("dataset/masks/*.jpg"))

# Train/Val split
train_images, val_images, train_masks, val_masks = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42)

# Generators
train_gen = DataGenerator(train_images, train_masks, batch_size=8)
val_gen = DataGenerator(val_images, val_masks, batch_size=8)

# Model
model = build_unet((256, 256, 3))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Training
model.fit(train_gen, validation_data=val_gen, epochs=25)

# Save model
model.save("polyp_unet_model.h5")
