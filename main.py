import os
import kagglehub
import tensorflow as tf

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
    tf.keras.layers.RandomTranslation(0.2, 1.0)
])

# Create base model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False, 
    weights='imagenet'
)

base_model.trainable = False    # freeze all base layers

# Build new model
inputs = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.layers.Rescaling(1./255)(inputs)
x = data_augmentation(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

# Compile the model
model.compile(
    loss="sparse_categorical_crossentropy", 
    optimizer="adam",
    metrics=["accuracy"]
)

# Train the model
callback = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
    tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)
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
model.save("fruit_classifier_model.keras")