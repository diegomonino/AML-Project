import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.utils import shuffle
from scipy.stats import ks_2samp


# === 1. Configuration ===
STD_DIR = "/ER_label_0"      # <-- set path to standard images folder
AMBE_DIR = "/NR_label_1"    # <-- set path to AmBe images folder
IMG_SIZE = (128, 128)
VALIDATION_SPLIT = 0.2
SEED = 42
BATCH_SIZE = 64


# Reproducibility
tf.random.set_seed(SEED)
np.random.seed(SEED)


def load_all_images(folder, img_size=(128,128)):
    """Load grayscale images from folder, normalize to [0,1], skip corrupted."""
    exts = ['*.png', '*.PNG', '*.jpg', '*.jpeg', '*.bmp', '*.gif']
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, e)))
    paths = sorted(paths)
    imgs = []
    for p in paths:
        try:
            im = Image.open(p).convert("L")  # grayscale
            im = im.resize(img_size, Image.BILINEAR)
            arr = np.array(im, dtype=np.float32) / 255.0  # scale to [0,1]
            imgs.append(arr[..., None])  # add channel dim
        except Exception as e:
            print(f"[load_all_images] skipping {p}: {e}")
    if len(imgs) == 0:
        raise RuntimeError(f"No images loaded from {folder}")
    return np.stack(imgs, axis=0)  # (N,H,W,1)

def normalize_per_image(x):
    """Per-image mean/std normalization (avoid division by zero)."""
    mean = np.mean(x, axis=(1,2,3), keepdims=True)
    std = np.std(x, axis=(1,2,3), keepdims=True)
    return (x - mean) / (std + 1e-6)

def augment_tf(img):
    """Simple augmentation: random flips + 90° rotations."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    img = tf.image.rot90(img, k)
    return img

def visualize_grid(images, titles=None, ncol=5, figsize=(12,3), cmap="gray"):
    n = len(images)
    nrow = int(np.ceil(n / ncol))
    plt.figure(figsize=figsize)
    for i in range(n):
        plt.subplot(nrow, ncol, i+1)
        plt.imshow(images[i].squeeze(), cmap=cmap, vmin=0, vmax=1)
        plt.axis("off")
        if titles:
            plt.title(titles[i], fontsize=8)
    plt.tight_layout()
    plt.show()



def build_cwola_classifier(input_shape=(128,128,1)):
    #""" CNN classifier for CWoLa."""
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, strides=1, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, strides=1, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, strides=1, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, out, name="cwola_clf_improved")
    return model



std_images = load_all_images(STD_DIR, img_size=IMG_SIZE)   # label 0
ambe_images = load_all_images(AMBE_DIR, img_size=IMG_SIZE) #label 1


# Create labels and combine
X_std = std_images
y_std = np.zeros(len(std_images), dtype=np.float32)
X_ambe = ambe_images
y_ambe = np.ones(len(ambe_images), dtype=np.float32)

X = np.concatenate([X_std, X_ambe], axis=0)
y = np.concatenate([y_std, y_ambe], axis=0)

# Shuffle and split into train/validation
X, y = shuffle(X, y, random_state=SEED)
n_total = len(X)
val_count = int(n_total * VALIDATION_SPLIT)
X_val, y_val = X[:val_count], y[:val_count]
X_train, y_train = X[val_count:], y[val_count:]

# Build tf.data datasets
def make_cwola_dataset(X, y, batch_size, augment=False, shuffle_data=True):
    ds = tf.data.Dataset.from_tensor_slices((X.astype("float32"), y.astype("float32")))
    if shuffle_data:
        ds = ds.shuffle(buffer_size=len(X), seed=SEED)
    if augment:
        ds = ds.map(lambda im, label: (augment_tf(im), label), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_cwola_dataset(X_train, y_train, batch_size=BATCH_SIZE, augment=True)
val_ds = make_cwola_dataset(X_val, y_val, batch_size=BATCH_SIZE, augment=False, shuffle_data=False)


#TRAINING

clf = build_cwola_classifier(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1))
clf.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.AUC(name="roc_auc")]
)
clf.summary()

# Callbacks
early_stop = keras.callbacks.EarlyStopping(monitor="val_roc_auc", patience=10, mode="max", restore_best_weights=True)
lr_schedule = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1)
checkpoint = keras.callbacks.ModelCheckpoint(
    "bigger_bandcwola_classifier_ckp.keras", monitor="val_roc_auc", save_best_only=True, save_weights_only=False
)


history = clf.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=[early_stop, lr_schedule, checkpoint],
    verbose=1,
)


# Immediately after training finished (when best weights are restored in RAM)
clf.save("final_model_CNN_PURE.keras")  # overwrites with the actual best weights
clf.save_weights("final_model_CNN_PURE.weights.h5")  # optional safety copy
