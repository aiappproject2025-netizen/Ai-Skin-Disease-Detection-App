import os
import numpy as np
import cv2
from flask import Flask, request, jsonify
import tensorflow.lite as tflite
from PIL import Image
import io

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = "model1.tflite"  # Using the lightweight model
CLASSES = ['Acne', 'Eczema', 'Psoriasis', 'Melanoma', 'Normal']

# --- DATABASE (From your Project) ---
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
    # Load TFLite Model (Uses very little RAM)
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ Smart Server Ready! (Lite Version)")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# --- UPDATED SEVERITY LOGIC (Contour Based) ---
def calculate_severity(image):
    # Convert PIL Image to OpenCV format (numpy array)
    img_array = np.array(image)
    
    # Convert RGB to BGR (OpenCV standard)
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # 1. Blur the image to remove noise/small reddish dots 
    # This prevents counting tiny imperfections as "Severe"
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Convert to HSV for red detection
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # Define Red Ranges (Slightly adjusted to avoid normal skin tones)
    lower_red1 = np.array([0, 60, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 60, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2
    
    # 2. Find Contours (distinct shapes) instead of just pixel count
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return "Mild"  # No distinct red shapes found

    # 3. Find the LARGEST red spot only (The actual Lesion)
    largest_contour = max(contours, key=cv2.contourArea)
    lesion_area = cv2.contourArea(largest_contour)
    total_area = img.shape[0] * img.shape[1]
    
    # 4. Calculate Percentage of the Lesion relative to image size
    lesion_ratio = (lesion_area / total_area) * 100
    
    # 5. New Logic Thresholds
    # < 1.0% = Mild (Small spot)
    # 1.0% - 8.0% = Moderate (Visible patch)
    # > 8.0% = Severe (Large area)
    if lesion_ratio < 1.0:
        return "Mild"
    elif lesion_ratio < 8.0:
        return "Moderate"
    else:
        return "Severe"

# --- HOME ROUTE (CRITICAL FOR CRON JOB) ---
@app.route('/', methods=['GET'])
def home():
    return "✅ Skin Doctor AI is Running! Use the App to analyze.", 200

# --- PREDICTION ROUTE ---
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        file = request.files["image"]
        
        # 1. Read Image
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        # 2. Preprocess for AI (Resize & Normalize)
        img_resized = image.resize((224, 224))
        input_data = np.expand_dims(img_resized, axis=0)
        input_data = (np.float32(input_data) / 255.0)

        # 3. Run Inference (TFLite Way)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # 4. Get Prediction
        class_index = np.argmax(output_data[0])
        disease_name = CLASSES[class_index]
        confidence = float(output_data[0][class_index]) * 100
        
        # 5. Calculate Severity (Using original high-res image for accuracy)
        severity_status = calculate_severity(image)
        
        # 6. Fetch Remedy
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
    # Render uses port 10000 by default
    app.run(host="0.0.0.0", port=10000)
