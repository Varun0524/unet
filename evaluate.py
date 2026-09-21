import glob, numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from data_generator import DataGenerator

images = sorted(glob.glob("dataset/images/*.jpg"))
masks = sorted(glob.glob("dataset/masks/*.jpg"))
print("Images:", len(images), " Masks:", len(masks))
_, val_x, _, val_y = train_test_split(images, masks, test_size=0.2, random_state=42)

model = load_model("polyp_unet_model.h5")
gen = DataGenerator(val_x, val_y, batch_size=8)
acc, dice = [], []
for i in range(len(gen)):
    x, y = gen[i]
    p = (model.predict(x, verbose=0) > 0.5).astype(np.float32)
    y = (y > 0.5).astype(np.float32)
    acc.append((p == y).mean())
    dice.append(2 * (p * y).sum() / (p.sum() + y.sum() + 1e-7))
print("Validation accuracy:", round(float(np.mean(acc)), 4))
print("Validation Dice:", round(float(np.mean(dice)), 4))