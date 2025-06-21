import os
import kagglehub
import tensorflow as tf
import openpyxl
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


# Download the dataset from Kaggle
path = kagglehub.dataset_download("icebearogo/fruit-classification-dataset")
root = os.path.join(path, "Fruit_dataset")

# Load images as tensorflow datasets
train_data = tf.keras.utils.image_dataset_from_directory(
    os.path.join(root, "train1"),
    image_size=(224, 224),
    batch_size=32
)

val_data = tf.keras.utils.image_dataset_from_directory(
    os.path.join(root, "val1"),
    image_size=(224, 224),
    batch_size=32
)

test_data = tf.keras.utils.image_dataset_from_directory(
    os.path.join(root, "test1"),
    image_size=(224, 224),
    batch_size=32
)

# Define the model
num_classes = len(train_data.class_names)

data_augmentation=tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),              # flips images horizontally
    tf.keras.layers.RandomRotation(0.2),                   # rotates images up to 20%
    tf.keras.layers.RandomZoom(0.2),                       # zooms images up to 20%
    tf.keras.layers.RandomContrast(0.2),                   # changes image contrast
    tf.keras.layers.RandomBrightness(0.2),                 # changes image brightness 
    tf.keras.layers.RandomTranslation(0.2, 1.0),            # translates images
])

# Build new model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
    tf.keras.layers.Rescaling(1./255),
    data_augmentation,
    
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding="same"),
    tf.keras.layers.MaxPooling2D(2, 2,),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding="same"),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding="same"),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding="same"),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding="same"),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax'),
])

# Compile the model
model.compile(
    loss="sparse_categorical_crossentropy", 
    optimizer=tf.keras.optimizers.Adam(1e-5),
    metrics=["accuracy"]
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5,
)

# Generate confusion matrix
y_true = []
y_pred = []

for images, labels in test_data:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

cm = confusion_matrix(y_true, y_pred)
class_names = test_data.class_names

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

print(classification_report(y_true, y_pred, target_names=class_names))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_data)
print(f"Test accuracy: {test_acc:.2f}")

# Save the model
model.save("fruit_classifier_model.keras")

# Save results to excel
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Results"

# Write headers
ws.append(["Metric", "Value"])
ws.append(["Test Accuracy", test_acc])
ws.append(["Test Loss", test_loss])
ws.append(["Epochs", len(history.epoch)])

# Add model summary as text
ws.append([])
ws.append(["Model Summary", ""])
model_summary = []
model.summary(print_fn=lambda x: model_summary.append(x))

for line in model_summary:
    row = [cell for cell in line.split(" ") if cell.strip() != ""]
    ws.append(row)
    
# Add confusion matrix
ws.append([])
ws.append(["Confusion Matrix"])

# Add class names as header row
ws.append([""] + class_names)

# Add each row of the confusion matrix
for i, row in enumerate(cm):
    ws.append([class_names[i]] + row.tolist())
    
# Get classification report as text
report_str = classification_report(y_true, y_pred, target_names=class_names)
report_lines = report_str.strip().split("\n")

ws.append([])
ws.append(["Classification Report"])

for line in report_lines:
    # Split line by whitespace and clean up
    row = [cell for cell in line.strip().split() if cell]
    ws.append(row)

# Save to file
wb.save("fruit_classification_results.xlsx")
print("Results saved to fruit_classification_results.xlsx")