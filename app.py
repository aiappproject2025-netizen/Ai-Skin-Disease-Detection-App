import os
import numpy as np
import cv2
from flask import Flask, request, jsonify
import tensorflow.lite as tflite
from PIL import Image
import io

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = "model1.tflite"
CLASSES = ['Acne', 'Eczema', 'Psoriasis', 'Melanoma', 'Normal']

# --- DATABASE ---
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

print("Loading Lite Model...")
try:
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ Smart Server Ready! (Lite Version)")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# --- FINAL SEVERITY LOGIC (Universal Skin Tone + Custom Thresholds) ---
def calculate_severity(image):
    # Convert PIL Image to OpenCV format
    img_array = np.array(image)
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # 1. Resize (Standardize size for consistent math)
    img = cv2.resize(img, (500, 500))

    # 2. Preprocess (CLAHE + LAB for Universal Skin Tone Support)
    # This separates "Redness" (A-channel) from "Darkness" (L-channel)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply Contrast Limited Adaptive Histogram Equalization to 'A' channel
    # This makes acne "pop" regardless of skin color (White/Asian/African)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced_a = clahe.apply(a)

    # 3. Auto-Threshold (Otsu's Method)
    # Automatically finds the best limit for the specific image lighting
    _, mask = cv2.threshold(enhanced_a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Remove Noise (Pores/Hair)
    # Removes tiny isolated pixels (< 3x3)
    noise_kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, noise_kernel)
    
    # 5. Merge Nearby Spots (Distance Check)
    # Merges spots if they are close (within 5px).
    # Scattered spots remain separate. Clustered spots become one "Giant Blob".
    distance_kernel = np.ones((5, 5), np.uint8)
    mask_clustered = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, distance_kernel)

    # 6. Calculate Metrics
    total_pixels = img.shape[0] * img.shape[1]
    
    # Metric A: Total Infection (Sum of all damaged areas)
    total_infection_pixels = cv2.countNonZero(mask_clustered)
    total_infection_ratio = (total_infection_pixels / total_pixels) * 100
    
    # Metric B: Largest Blob (The single biggest contiguous patch)
    contours, _ = cv2.findContours(mask_clustered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    largest_blob_ratio = 0.0
    if contours:
        largest_blob_area = cv2.contourArea(max(contours, key=cv2.contourArea))
        largest_blob_ratio = (largest_blob_area / total_pixels) * 100

    print(f"DEBUG: Total: {total_infection_ratio:.2f}% | Blob: {largest_blob_ratio:.2f}%")

    # --- FINAL THRESHOLDS (Calibrated to your Data) ---
    
    # CASE 1: MILD
    # Logic: Little total coverage (< 13%)
    if total_infection_ratio < 13.0:
        return "Mild"

    # CASE 2: SEVERE
    # Logic: Contains a giant infected patch/blob (> 35%)
    if largest_blob_ratio > 35.0:
        return "Severe"

    # CASE 3: MODERATE
    # Logic: High coverage but scattered (No giant blobs) OR Medium blobs (13-35%)
    return "Moderate"

# --- HOME ROUTE ---
@app.route('/', methods=['GET'])
def home():
    return "✅ Skin Doctor AI is Running!", 200

# --- PREDICTION ROUTE ---
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        file = request.files["image"]
        
        # 1. Read Image
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        # 2. AI Prediction (Resize to 224x224 for Model)
        img_resized = image.resize((224, 224))
        input_data = np.expand_dims(img_resized, axis=0)
        input_data = (np.float32(input_data) / 255.0)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        class_index = np.argmax(output_data[0])
        disease_name = CLASSES[class_index]
        confidence = float(output_data[0][class_index]) * 100
        
        # 3. Calculate Severity (Using High-Res Image + Updated Logic)
        severity_status = calculate_severity(image)
        
        # 4. Fetch Remedy
        advice = REMEDIES.get(disease_name, {}).get(severity_status, "Consult a doctor.")

        return jsonify({
            "disease": disease_name,
            "confidence": f"{confidence:.2f}%",
            "severity": severity_status,
            "remedy": advice
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
