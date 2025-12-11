import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os

# --- 0. GPU CONFIGURATION (Crucial for RTX 4050) ---
# This prevents the GPU from crashing by allocating memory slowly
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ POWER UP: Running on {len(gpus)} GPU(s)! (RTX 4050 Detected)")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ WARNING: GPU not detected. Running on CPU (Slower).")

# --- CONFIGURATION ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = "dataset"
CLASSES = 5  # Acne, Eczema, Psoriasis, Melanoma, Normal

# --- 1. DATA PREPARATION ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,      # Increased rotation for better learning
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

print("Loading Training Data...")
train_generator = train_datagen.flow_from_directory(
    directory=os.path.join(DATA_DIR, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

print("Loading Validation Data...")
val_generator = train_datagen.flow_from_directory(
    directory=os.path.join(DATA_DIR, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# --- 2. BUILD MODEL (MobileNetV2) ---
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Unfreeze the last 20 layers for better accuracy (Fine Tuning)
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)  # Increased dropout slightly to prevent overfitting
predictions = Dense(CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Use a smaller learning rate because we are fine-tuning
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- 3. CALLBACKS (The "Smart" Features) ---
callbacks = [
    # Stop if validation loss doesn't improve for 5 epochs
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    
    # Save the BEST model, not just the last one
    ModelCheckpoint("skin_disease_best_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1),
    
    # Reduce learning rate if stuck (helps get unstuck)
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
]

# --- 4. TRAIN ---
print("Starting Training with RTX 4050 Power...")
history = model.fit(
    train_generator,
    epochs=100,  # As requested
    validation_data=val_generator,
    callbacks=callbacks
)

# --- 5. SAVE ---
# We save the final one too, just in case
model.save("skin_disease_final_model.h5")
print("Training Complete. Best Model saved as 'skin_disease_best_model.h5'")