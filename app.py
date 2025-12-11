import os
import numpy as np
import cv2  # OpenCV for image processing
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io

app = Flask(__name__)
MODEL_PATH = "skin_disease_model.h5"
CLASSES = ['Acne', 'Eczema', 'Psoriasis', 'Melanoma', 'Normal']

# --- MEDICAL DATABASE (From Project PDF) ---
REMEDIES = {
    "Acne": {
        "Mild": "Use Aloe Vera gel and Neem paste. Drink 3L water daily.",
        "Moderate": "Use Salicylic Acid face wash. Avoid oily food.",
        "Severe": "Consult a Dermatologist. Do not pop pimples!"
    },
    "Eczema": {
        "Mild": "Apply Coconut Oil or Shea Butter to moisturize.",
        "Moderate": "Use mild, fragrance-free moisturizers.",
        "Severe": "Requires prescription steroid creams. Visit a doctor."
    },
    "Psoriasis": {
        "Mild": "Expose skin to morning sunlight (Vitamin D).",
        "Moderate": "Use coal tar salicylic acid shampoo/soap.",
        "Severe": "Autoimmune condition. Urgent Dermatologist visit required."
    },
    "Melanoma": {
        "Mild": "CRITICAL ALERT: Visit an Oncologist immediately.",
        "Moderate": "CRITICAL ALERT: Visit an Oncologist immediately.",
        "Severe": "CRITICAL ALERT: Visit an Oncologist immediately."
    },
    "Normal": {
        "Mild": "You are healthy! Use Sunscreen (SPF 50) daily.",
        "Moderate": "You are healthy! Use Sunscreen (SPF 50) daily.",
        "Severe": "You are healthy! Use Sunscreen (SPF 50) daily."
    }
}

print("Loading AI Model...")
model = load_model(MODEL_PATH)
print("✅ Smart Server Ready!")

def calculate_severity(image_array):
    """
    Uses OpenCV to find 'Redness' in the image.
    More Redness = Higher Severity.
    """
    # 1. Convert to OpenCV format (BGR)
    # The image comes in as (1, 224, 224, 3) normalized float. We need 0-255 int.
    img = (image_array[0] * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # 2. Convert to HSV (Hue Saturation Value) to find red color
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define range for 'Red' color (Inflammation)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Create mask (Only keep red pixels)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2
    
    # 3. Calculate Percentage of Redness
    red_pixels = cv2.countNonZero(mask)
    total_pixels = img.shape[0] * img.shape[1]
    redness_ratio = (red_pixels / total_pixels) * 100
    
    # 4. Determine Logic
    if redness_ratio < 5:
        return "Mild"
    elif redness_ratio < 15:
        return "Moderate"
    else:
        return "Severe"

def prepare_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    
    try:
        # 1. Process Image
        image = Image.open(io.BytesIO(file.read()))
        processed_image = prepare_image(image, target_size=(224, 224))
        
        # 2. AI Prediction (Which Disease?)
        prediction = model.predict(processed_image)
        class_index = np.argmax(prediction, axis=1)[0]
        disease_name = CLASSES[class_index]
        confidence = float(prediction[0][class_index]) * 100
        
        # 3. OpenCV Logic (How Bad is it?)
        severity_status = calculate_severity(processed_image)
        
        # 4. Fetch Remedy
        advice = REMEDIES.get(disease_name, {}).get(severity_status, "Consult a doctor.")

        # 5. Send JSON back to App
        return jsonify({
            "disease": disease_name,
            "confidence": f"{confidence:.2f}%",
            "severity": severity_status,
            "remedy": advice
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)