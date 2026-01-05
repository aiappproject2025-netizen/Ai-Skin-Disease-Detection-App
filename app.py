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

# --- DATABASE (GEMINI STYLE DETAILED CONTENT) ---
REMEDIES = {
    "Acne": {
        "Mild": "🟢 **MILD ACNE DETECTED**\n\n💡 **ADVICE:**\nYour skin is congested but it's early stage. No harsh chemicals needed.\n\n🌿 **REMEDY:**\nApply Aloe Vera gel at night. Use Neem paste on spots.\n\n🥗 **DIET:**\nDrink 3L water. Eat Cucumber & Carrots to reduce body heat.\n\n✅ **ROUTINE:**\nWash face twice a day. Do not touch your face.",
        
        "Moderate": "🟡 **MODERATE ACNE DETECTED**\n\n💡 **ADVICE:**\nOil production is high. You need active ingredients.\n\n🧪 **RECOMMENDATION:**\nUse a Face Wash with **Salicylic Acid** or **Niacinamide**.\n\n🥗 **DIET:**\nAvoid Oily food, Milk & Sweets for 1 week.\n\n🛍️ **PRODUCT:**\nBuy 'Minimalist Salicylic Acid' on Amazon.",
        
        "Severe": "🔴 **SEVERE ACNE (CYSTIC)**\n\n⚠️ **CLINICAL WARNING:**\nThis stage causes scarring. Home remedies will NOT work.\n\n🩺 **NEXT STEP:**\nConsult a Dermatologist immediately.\n\n🚫 **DON'T:**\nDo NOT pop pimples. Do NOT use lemon/toothpaste.\n\n💡 **INFO:**\nDoctors may suggest Carbon Peels for this."
    },
    "Eczema": {
        "Mild": "🟢 **MILD ECZEMA (DRYNESS)**\n\n💡 **ADVICE:**\nSkin barrier is dry. Lock in the moisture.\n\n🌿 **REMEDY:**\nApply Coconut Oil immediately after bath.\n\n🥗 **DIET:**\nEat Omega-3 rich foods like Walnuts & Fish.\n\n✅ **ROUTINE:**\nUse lukewarm water for bathing (Not hot!).",
        
        "Moderate": "🟡 **MODERATE ECZEMA**\n\n💡 **ADVICE:**\nRedness and itching detected. Skin needs repair.\n\n🧪 **RECOMMENDATION:**\nUse creams with **Ceramides** or **Oatmeal**.\n\n🥗 **DIET:**\nAvoid Eggs & Citric fruits (Lemon) temporarily.\n\n🛍️ **PRODUCT:**\nUse 'Aveeno Dermexa' or 'Cetaphil' moisturizer.",
        
        "Severe": "🔴 **SEVERE ECZEMA**\n\n⚠️ **CLINICAL WARNING:**\nSkin may crack or bleed. Risk of infection.\n\n🩺 **NEXT STEP:**\nVisit a Doctor for Steroid Creams or UV Therapy.\n\n🚫 **DON'T:**\nDo not scratch! Wear cotton clothes only."
    },
    "Psoriasis": {
        "Mild": "🟢 **MILD PSORIASIS**\n\n💡 **ADVICE:**\nSmall scales detected. Keep skin hydrated.\n\n🌿 **REMEDY:**\nExpose skin to **Morning Sunlight** (Vit D) for 15 mins.\n\n🥗 **DIET:**\nAvoid Red Meat. Eat more leafy vegetables.\n\n✅ **ROUTINE:**\nApply thick moisturizer or Vaseline.",
        
        "Moderate": "🟡 **MODERATE PSORIASIS**\n\n💡 **ADVICE:**\nPatches are thickening. Need keratolytic agents.\n\n🧪 **RECOMMENDATION:**\nUse **Coal Tar** or **Salicylic Acid** based soap/shampoo.\n\n🥗 **DIET:**\nAvoid Alcohol and Spicy foods.\n\n🛍️ **PRODUCT:**\nSearch for 'Coal Tar Lotion' online.",
        
        "Severe": "🔴 **SEVERE PSORIASIS**\n\n⚠️ **CLINICAL WARNING:**\nWidespread scaling. Needs systemic treatment.\n\n🩺 **NEXT STEP:**\nConsult a Dermatologist for Biologics/Laser treatment.\n\n🚫 **DON'T:**\nDo not peel off the scales forcefully."
    },
    "Melanoma": {
        "Mild": "⚠️ **CRITICAL ALERT: MELANOMA**\n\n🚨 **ACTION REQUIRED:**\nAI has detected irregular mole patterns indicative of Skin Cancer.\n\n🏥 **NEXT STEP:**\nThis cannot be treated at home. Visit an **Oncologist** immediately.\n\n📍 **GPS:**\nClick 'Find Dermatologist' button below.",
        "Moderate": "⚠️ **CRITICAL ALERT: MELANOMA**\n\n🚨 **ACTION REQUIRED:**\nAI has detected irregular mole patterns indicative of Skin Cancer.\n\n🏥 **NEXT STEP:**\nThis cannot be treated at home. Visit an **Oncologist** immediately.\n\n📍 **GPS:**\nClick 'Find Dermatologist' button below.",
        "Severe": "⚠️ **CRITICAL ALERT: MELANOMA**\n\n🚨 **ACTION REQUIRED:**\nAI has detected irregular mole patterns indicative of Skin Cancer.\n\n🏥 **NEXT STEP:**\nThis cannot be treated at home. Visit an **Oncologist** immediately.\n\n📍 **GPS:**\nClick 'Find Dermatologist' button below."
    },
    "Normal": {
        "Mild": "✨ **HEALTHY SKIN DETECTED**\n\n🎉 **STATUS:**\nNo diseases found! Your skin is glowing.\n\n🧴 **MAINTENANCE:**\nApply **Sunscreen (SPF 50)** daily to prevent aging.\n\n💧 **ROUTINE:**\nCleanse -> Tone -> Moisturize.\n\n🥗 **DIET:**\nEat Vitamin C fruits (Orange/Guava).",
        "Moderate": "✨ **HEALTHY SKIN DETECTED**\n\n🎉 **STATUS:**\nNo diseases found! Your skin is glowing.\n\n🧴 **MAINTENANCE:**\nApply **Sunscreen (SPF 50)** daily to prevent aging.\n\n💧 **ROUTINE:**\nCleanse -> Tone -> Moisturize.\n\n🥗 **DIET:**\nEat Vitamin C fruits (Orange/Guava).",
        "Severe": "✨ **HEALTHY SKIN DETECTED**\n\n🎉 **STATUS:**\nNo diseases found! Your skin is glowing.\n\n🧴 **MAINTENANCE:**\nApply **Sunscreen (SPF 50)** daily to prevent aging.\n\n💧 **ROUTINE:**\nCleanse -> Tone -> Moisturize.\n\n🥗 **DIET:**\nEat Vitamin C fruits (Orange/Guava)."
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

    # --- FINAL THRESHOLDS (Calibrated for Exam Demo) ---
    
    # CASE 1: MILD
    # Logic: Coverage must be less than 40% to be Mild.
    if total_infection_ratio < 40.0:
        return "Mild"

    # CASE 2: SEVERE
    # Logic: Only if MORE THAN 80% is covered (Very rare).
    # This prevents False Alarms.
    if total_infection_ratio > 80.0:
        return "Severe"

    # CASE 3: MODERATE
    # Everything in between (40% - 80%)
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
        
        # 3. Calculate Severity (Using Updated Logic)
        severity_status = calculate_severity(image)
        
        # 4. Fetch Remedy (From New Gemini DB)
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
