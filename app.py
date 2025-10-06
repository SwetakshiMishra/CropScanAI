from flask import Flask, render_template, request, jsonify, url_for, redirect, session
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
from flask_babel import Babel

# 1. CREATE the Flask app
app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

# 2. Add configurations to the 'app'
app.config['SECRET_KEY'] = 'a-very-secret-key-for-sessions'
app.config['LANGUAGES'] = {'en': 'English', 'hi': 'हिन्दी'}

# --- THIS IS THE CORRECTED SECTION ---
# 3. Define your language selection function normally
def get_locale():
    # Get the language from the session, defaulting to English
    return session.get('language', 'en')

# 4. Initialize Babel and pass the function directly
babel = Babel(app, locale_selector=get_locale)
# --- END OF CORRECTION ---

# This route handles the actual language switch
@app.route('/language/<language>')
def set_language(language=None):
    session['language'] = language
    return redirect(request.referrer)

# --- The rest of your code is perfect ---

MODEL_PATH = "pearl_millet_ergot_model.h5"
model = load_model(MODEL_PATH)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -------------------- ROUTES --------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about_ergot")
def about_ergot():
    return render_template("about_ergot.html")

@app.route("/identify")
def identify():
    return render_template("identify.html")

@app.route("/faq")
def faq():
    return render_template("faq.html")

# -------------------- PREDICTION ROUTE --------------------

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    upload_dir = os.path.join(app.static_folder, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    file.save(file_path)

    img = preprocess_image(file_path)
    pred = model.predict(img)[0][0]
    
    if pred < 0.5:
        result = "Healthy"
        confidence = round((1 - float(pred)) * 100, 2)
        return render_template(
            "results_healthy.html",
            result=result,
            confidence=confidence,
            filename=file.filename
        )
    else:
        result = "Diseased: Ergot"
        confidence = round(float(pred) * 100, 2)
        return render_template(
            "ergot-detected.html",
            result=result,
            confidence=confidence,
            filename=file.filename
        )

# -------------------- RUN APP --------------------

if __name__ == "__main__":
    app.run(debug=True)