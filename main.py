import os
import shutil
import kagglehub
import tensorflow as tf
import openpyxl
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime

# Paths
train_dir = "/kaggle/input/fruit-classification-dataset/Fruit_dataset/train1"
val_dir = "/kaggle/input/fruit-classification-dataset/Fruit_dataset/val1"
test_dir = "/kaggle/input/fruit-classification-dataset/Fruit_dataset/test1"

IMG_SIZE = 256
BATCH_SIZE = 32

# Load datasets
train_data = tf.keras.utils.image_dataset_from_directory(
    directory=train_dir,
    labels="inferred",
    label_mode='int',
    batch_size=BATCH_SIZE,
    image_size=(IMG_SIZE, IMG_SIZE),
)

val_data = tf.keras.utils.image_dataset_from_directory(
    directory=val_dir,
    labels="inferred",
    label_mode='int',
    batch_size=BATCH_SIZE,
    image_size=(IMG_SIZE, IMG_SIZE),
)

test_data = tf.keras.utils.image_dataset_from_directory(
    directory=test_dir,
    labels="inferred",
    label_mode='int',
    batch_size=BATCH_SIZE,
    image_size=(IMG_SIZE, IMG_SIZE),
)

class_names = train_data.class_names
num_classes = len(class_names)

# Normalize images
def preprocess(image, label):
    image = tf.cast(image/255.0, tf.float32)
    return image, label

train_data_norm = train_data.map(preprocess)
val_data_norm = val_data.map(preprocess)
test_data_norm = test_data.map(preprocess)

def show_variety_images(data, class_names, split_name="Dataset", n_images=20):
    images = []
    labels = []
    for img, lbl in data.unbatch():
        images.append(img.numpy())
        labels.append(lbl.numpy())
    total = len(images)
    print(f"{split_name}: {total} images available.")
    n_show = min(n_images, total)
    idxs = random.sample(range(total), n_show)
    plt.figure(figsize=(20, 10))
    for i, idx in enumerate(idxs):
        plt.subplot(4, 5, i+1)
        plt.imshow(images[idx].astype("uint8"))
        plt.title(class_names[labels[idx]])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

show_variety_images(test_data, class_names, split_name="Test Images", n_images=20)
show_variety_images(val_data, class_names, split_name="Validation Images", n_images=20)

# --- SANITY TEST: Overfit on a tiny subset of train_data (NO data augmentation!) ---
def do_sanity_test(train_data, class_names):
    wanted_classes = [0, 1]
    max_per_class = 10
    images = []
    labels = []
    class_counts = {c: 0 for c in wanted_classes}
    for img, lbl in train_data.unbatch():
        label = int(lbl.numpy())
        if label in wanted_classes and class_counts[label] < max_per_class:
            images.append(img.numpy())
            labels.append(label)
            class_counts[label] += 1
        if all(class_counts[c] == max_per_class for c in wanted_classes):
            break
    images = np.stack(images)
    labels = np.array(labels)
    images = images.astype("float32") / 255.0
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.layers.Flatten()(inp)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    out = tf.keras.layers.Dense(2, activation='softmax')(x)
    sanity_model = tf.keras.Model(inp, out)
    sanity_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    relabel = {c: i for i, c in enumerate(wanted_classes)}
    labels_re = np.array([relabel[l] for l in labels])
    sanity_history = sanity_model.fit(
        images, labels_re, epochs=30, batch_size=4, verbose=0
    )
    if sanity_history.history["accuracy"][-1] <= 0.95:
        raise RuntimeError("Sanity test FAILED: Model did NOT overfit, check label/image pipeline!")

do_sanity_test(train_data, class_names)

# --- DATA AUGMENTATION PIPELINE ---
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomBrightness(0.2),
    tf.keras.layers.RandomTranslation(0.2, 1.0),
])

# --- TRANSFER LEARNING MODEL (EfficientNetB0) ---
base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # Freeze base for initial training

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = tf.keras.applications.efficientnet.preprocess_input(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(base_model(x, training=False))
x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# --- INITIAL TRAINING: NO AUGMENTATION, FROZEN BASE ---
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
checkpoint = tf.keras.callbacks.ModelCheckpoint("best_model.keras", monitor="val_loss", save_best_only=True, verbose=1)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

# --- FINE-TUNE THE LAST 20 LAYERS, ADD DATA AUGMENTATION ---
for layer in base_model.layers[:-20]:
    layer.trainable = False
for layer in base_model.layers[-20:]:
    layer.trainable = True

inputs_aug = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
aug = data_augmentation(inputs_aug)
aug = tf.keras.applications.efficientnet.preprocess_input(aug)
x = base_model(aug, training=True)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
model_aug = tf.keras.Model(inputs_aug, outputs)

model_aug.set_weights(model.get_weights())  # Transfer weights

model_aug.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
history_fine = model_aug.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

model_aug = tf.keras.models.load_model("best_model.keras")

# --- EVALUATION ON TEST SET ---
y_true = []
y_pred = []

for images, labels in test_data:
    preds = model_aug.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

# --- SAVE CONFUSION MATRIX IMAGE ---
now = datetime.now().strftime("%Y%m%d_%H%M%S")
model_type = "EfficientNetB0"
filename = f"confusion_matrix_{model_type}_img{IMG_SIZE}_bs{BATCH_SIZE}_ep{len(history.epoch)+len(history_fine.epoch)}_{now}.png"

plt.figure(figsize=(16, 14))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(filename, dpi=300)
plt.close()
print(f"Confusion matrix saved as {filename}")

report_str = classification_report(y_true, y_pred, target_names=class_names)
test_loss, test_acc = model_aug.evaluate(test_data, verbose=1)
print(f"Test accuracy: {test_acc:.4f}")

model_aug.save("fruit_classifier_model.keras")

# --- SAVE RESULTS TO EXCEL ---
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Results"

ws.append(["Metric", "Value"])
ws.append(["Test Accuracy", test_acc])
ws.append(["Test Loss", test_loss])
ws.append(["Epochs", len(history.epoch) + len(history_fine.epoch)])
ws.append(["Confusion Matrix Image", filename])

ws.append([])
ws.append(["Model Summary", ""])
model_summary = []
model_aug.summary(print_fn=lambda x: model_summary.append(x))
for line in model_summary:
    ws.append([line])

ws.append([])
ws.append(["Confusion Matrix"])
ws.append([""] + class_names)
for i, row in enumerate(cm):
    ws.append([class_names[i]] + row.tolist())

ws.append([])
ws.append(["Classification Report"])
for line in report_str.strip().split("\n"):
    ws.append([line])

wb.save("fruit_classification_results.xlsx")
print("Results saved to fruit_classification_results.xlsx")