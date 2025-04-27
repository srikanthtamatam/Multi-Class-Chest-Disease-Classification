import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define dataset paths
train_dir = "/kaggle/input/train-dataset-new/train"
test_dir = "/kaggle/input/test-1/test_1"

# Data Preprocessing (No validation split used here)
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

# Load test data
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Get class names
classes = list(train_generator.class_indices.keys())

# Load pre-trained base model
base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
for layer in base_model.layers:
    layer.trainable = False  # Freeze base model

# Add custom classifier layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(len(classes), activation='softmax')(x)

# Build full model
model = Model(inputs=base_model.input, outputs=output_layer)

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
)

# Callbacks
early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='loss', patience=3, factor=0.5, verbose=1)

# Train model (only on training set)
history = model.fit(
    train_generator,
    epochs=30,
    callbacks=[early_stop, reduce_lr]
)

# Unfreeze last 30 layers for fine-tuning
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Recompile with lower learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
)

# Fine-tune model (still only on training set)
fine_tune_history = model.fit(
    train_generator,
    epochs=10,
    callbacks=[early_stop, reduce_lr]
)

# Save model
model.save("/kaggle/working/chest_disease_xception8.h5")
print("Training complete. Model saved as chest_disease_xception8.h5")

# ---------- Evaluation on Test Set Only ----------

# Evaluate model on test set
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_generator)


# Predictions for classification report & confusion matrix
y_true = test_generator.classes
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=classes))

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# ---------- Training Metric Plot (No val_ metrics) ----------
def plot_training_metrics(histories):
    metrics = ['accuracy', 'loss', 'precision', 'recall']
    plt.figure(figsize=(12, 8))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        for name, hist in histories.items():
            plt.plot(hist.history[metric], label=f'{name} {metric}')
        plt.legend()
        plt.title(f'{metric.capitalize()} per Epoch')
    plt.tight_layout()
    plt.show()

plot_training_metrics({'initial': history, 'fine_tune': fine_tune_history})


import cv2
from tensorflow.keras.preprocessing import image

def get_img_array(img_path, size):
    img = image.load_img(img_path, target_size=size)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = arr / 255.0
    return arr

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (299, 299))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)
    
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title("Grad-CAM")
    plt.axis('off')
    plt.show()

# Find the last conv layer of Xception (usually 'block14_sepconv2_act')
last_conv_layer_name = 'block14_sepconv2_act'

# Show Grad-CAM for one image per class
print("\nGrad-CAM Visualization Per Class:")
for class_name in classes:
    class_path = os.path.join(test_dir, class_name)
    sample_image = None
    for fname in os.listdir(class_path):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            sample_image = os.path.join(class_path, fname)
            break
    if sample_image:
        img_array = get_img_array(sample_image, size=(299, 299))
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        print(f"\nClass: {class_name}")
        save_and_display_gradcam(sample_image, heatmap)
    else:
        print(f"No image found for class {class_name}")
