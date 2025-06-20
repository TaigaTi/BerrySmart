import os
import kagglehub
import tensorflow as tf

# Download the dataset from Kaggle
path = kagglehub.dataset_download("icebearogo/fruit-classification-dataset")
root = os.path.join(path, "Fruit_dataset")

# Load images as tensorflow datasets
train_data = tf.keras.utils.image_dataset_from_directory(
    os.path.join(root, "train1"),
    image_size=(128, 128),
    batch_size=32
)

val_data = tf.keras.utils.image_dataset_from_directory(
    os.path.join(root, "val1"),
    image_size=(128, 128),
    batch_size=32
)

test_data = tf.keras.utils.image_dataset_from_directory(
    os.path.join(root, "test1"),
    image_size=(128, 128),
    batch_size=32
)

# Define the model
num_classes = len(train_data.class_names)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(128, 128, 3)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(
    loss="sparse_categorical_crossentropy", 
    optimizer="adam",
    metrics=["accuracy"]
)

# Train the model
callback = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
    tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
]
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=[callback]
)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_data)
print(f"Test accuracy: {test_acc:.2f}")

# Save the model
model.save("fruit_classifier_model")