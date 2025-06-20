import os
import kagglehub
import tensorflow as tf
import openpyxl

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

# Create base model
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False, 
    weights='imagenet'
)

base_model.trainable = False    # freeze all base layers

# Build new model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
    tf.keras.layers.Rescaling(1./255),
    data_augmentation,
    base_model,  
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(
    loss="sparse_categorical_crossentropy", 
    optimizer=tf.keras.optimizers.Adam(1e-5),
    metrics=["accuracy"]
)

# Train the model
callback = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
    tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)
]

# Unfreeze the last 20 layers of the base model
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Recompile with a lower learning rate
model.compile(
    loss="sparse_categorical_crossentropy", 
    optimizer=tf.keras.optimizers.Adam(1e-5),
    metrics=["accuracy"]
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,
    callbacks=[callback]
)

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

# Add dataset information
ws.append([])
ws.append(["Dataset", "Num Classes", "Num Images"])
ws.append(["Train", num_classes, train_data.cardinality().numpy() * train_data.batch_size])
ws.append(["Validation", num_classes, val_data.cardinality().numpy() * val_data.batch_size])
ws.append(["Test", num_classes, test_data.cardinality().numpy() * test_data.batch_size])

# Save to file
wb.save("fruit_classification_results.xlsx")
print("Results saved to fruit_classification_results.xlsx")