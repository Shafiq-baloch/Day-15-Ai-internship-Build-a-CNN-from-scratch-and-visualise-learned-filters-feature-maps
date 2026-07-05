# ============================================================
# Day 15 - Build CNN from Scratch
# Step 1: Import Libraries & Load Dataset
# ============================================================

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Check TensorFlow Version
# ============================================================

print("=" * 60)
print("TensorFlow Version:", tf.__version__)
print("=" * 60)

# ============================================================
# Load CIFAR-10 Dataset
# ============================================================

print("\nLoading CIFAR-10 Dataset...")
print("=" * 60)

(train_ds, test_ds), ds_info = tfds.load(
    "cifar10",
    split=["train", "test"],
    as_supervised=True,
    with_info=True
)

# ============================================================
# Dataset Information
# ============================================================

print(f"Training Samples : {ds_info.splits['train'].num_examples}")
print(f"Testing Samples  : {ds_info.splits['test'].num_examples}")
print(f"Number of Classes: {ds_info.features['label'].num_classes}")

# ============================================================
# Class Names
# ============================================================

class_names = ds_info.features["label"].names

print("\nClass Names:")
print(class_names)

# ============================================================
# Display Sample Images
# ============================================================

plt.figure(figsize=(8, 8))

for i, (image, label) in enumerate(train_ds.take(9)):
    plt.subplot(3, 3, i + 1)
    plt.imshow(image)
    plt.title(class_names[int(label)])
    plt.axis("off")

plt.tight_layout()
plt.show()

# ============================================================
# Preprocessing Function
# ============================================================

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# ============================================================
# Prepare Dataset
# ============================================================

BATCH_SIZE = 64
AUTOTUNE = tf.data.AUTOTUNE

train_ds = (
    train_ds
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .shuffle(10000)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

test_ds = (
    test_ds
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

# ============================================================
# Verify Dataset
# ============================================================

for images, labels in train_ds.take(1):
    print("\nBatch Shape:", images.shape)
    print("Labels Shape:", labels.shape)
    print("Pixel Value Range:",
          tf.reduce_min(images).numpy(),
          "to",
          tf.reduce_max(images).numpy())

print("\nDataset is ready for CNN training!")
print("=" * 60)



# ============================================================
# Step 3: Compile the CNN Model
# ============================================================

print("\nCompiling Model...")
print("=" * 60)

print("\nBuilding CNN Model...")
print("=" * 60)

model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(32, 32, 3)),

    tf.keras.layers.Conv2D(
        32,
        (3,3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(
        64,
        (3,3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(
        128,
        (3,3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(128, activation="relu"),

    tf.keras.layers.Dense(10, activation="softmax")

])

model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Model Compiled Successfully!")

# ============================================================
# Train the CNN
# ============================================================

EPOCHS = 20

print("\nTraining CNN...")
print("=" * 60)

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS
)

# ============================================================
# Evaluate the Model
# ============================================================

print("\nEvaluating Model...")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(test_ds)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# ============================================================
# Save the Model
# ============================================================

model.save("cnn_cifar10.keras")

print("\nModel saved successfully!")
print("File Name: cnn_cifar10.keras")


# ============================================================
# Plot Accuracy and Loss
# ============================================================

plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("CNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()


# ============================================================
# Save Training History
# ============================================================

import pandas as pd

history_df = pd.DataFrame(history.history)

history_df.to_csv("cnn_training_history.csv", index=False)

print("Training history saved successfully!")


# ============================================================
# Step 4: Visualize Learned Filters
# ============================================================

print("\nVisualizing Learned Filters...")
print("=" * 60)

# Get weights from the first Conv2D layer
filters, biases = model.layers[0].get_weights()

print("Filter Shape:", filters.shape)

# Normalize filter values to 0-1 for visualization
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# Plot all 32 filters
plt.figure(figsize=(12, 12))

for i in range(32):
    plt.subplot(4, 8, i + 1)

    # Each filter has RGB channels
    plt.imshow(filters[:, :, :, i])

    plt.title(f"F{i+1}", fontsize=8)
    plt.axis("off")

plt.suptitle("Learned Filters of First Conv2D Layer", fontsize=16)
plt.tight_layout()
plt.show()


# ============================================================
# Step 5: Visualize Feature Maps
# ============================================================

print("\nVisualizing Feature Maps...")
print("=" * 60)

# Create a model that outputs the feature maps from the first Conv2D layer
feature_map_model = tf.keras.Model(
    inputs=model.input,
    outputs=model.layers[1].output   # First Conv2D layer
)

# Get 3 test images
sample_images = []
sample_labels = []

for images, labels in test_ds.take(1):
    sample_images = images[:3]
    sample_labels = labels[:3]

# Generate feature maps
feature_maps = feature_map_model.predict(sample_images)

# Display feature maps for each image
for img_index in range(3):

    plt.figure(figsize=(12, 12))

    # Original Image
    plt.subplot(5, 7, 1)
    plt.imshow(sample_images[img_index])
    plt.title(f"Original\n{class_names[int(sample_labels[img_index])]}")
    plt.axis("off")

    # Display all 32 feature maps
    for i in range(32):
        plt.subplot(5, 7, i + 2)
        plt.imshow(feature_maps[img_index, :, :, i], cmap="viridis")
        plt.title(f"F{i+1}", fontsize=8)
        plt.axis("off")

    plt.suptitle(
        f"Feature Maps for Image {img_index + 1}",
        fontsize=16
    )

    plt.tight_layout()
    plt.show()


    # ============================================================
# Step 6: Compare CNN vs MLP Accuracy
# ============================================================

print("\nComparing CNN vs MLP")
print("=" * 60)

# Replace with your actual MLP accuracy on CIFAR-10
mlp_accuracy = 0.58

# CNN accuracy obtained after evaluation
cnn_accuracy = test_accuracy

print(f"MLP Accuracy : {mlp_accuracy:.4f}")
print(f"CNN Accuracy : {cnn_accuracy:.4f}")

# ============================================================
# Step 6: Compare CNN vs MLP Accuracy
# ============================================================

print("\nComparing CNN vs MLP")
print("=" * 60)

# Replace with your actual MLP accuracy on CIFAR-10
mlp_accuracy = 0.58

# CNN accuracy obtained after evaluation
cnn_accuracy = test_accuracy

print(f"MLP Accuracy : {mlp_accuracy:.4f}")
print(f"CNN Accuracy : {cnn_accuracy:.4f}")

# ============================================================
# Accuracy Comparison Graph
# ============================================================

plt.figure(figsize=(6,5))

plt.bar(
    comparison_df["Model"],
    comparison_df["Test Accuracy"]
)

plt.title("CNN vs MLP Accuracy")
plt.xlabel("Model")
plt.ylabel("Test Accuracy")

for i, value in enumerate(comparison_df["Test Accuracy"]):
    plt.text(i, value + 0.01, f"{value:.2f}", ha="center")

plt.ylim(0, 1)

plt.show()