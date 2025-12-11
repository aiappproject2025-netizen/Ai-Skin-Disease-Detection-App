import tensorflow as tf

# Load the heavy model
model = tf.keras.models.load_model("skin_disease_best_model.h5")

# Convert to TFLite (Lightweight)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the lightweight file
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("Success! model.tflite created.")