from flask import Flask, render_template, request, jsonify, url_for, redirect
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

# Flask app setup
app = Flask(
    __name__,
    template_folder="templates",  # HTML files inside /templates
    static_folder="static"        # CSS, JS, Images inside /static
)

MODEL_PATH = "pearl_millet_ergot_model.h5"

# Load model once at startup
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
    return render_template("index.html")   # Home page

@app.route("/about_ergot")
def about_ergot():
    return render_template("about_ergot.html")   # About Ergot

@app.route("/identify")
def identify():
    return render_template("identify.html")   # Identify disease

@app.route("/faq")
def faq():
    return render_template("faq.html")   # FAQs page


# -------------------- PREDICTION ROUTE --------------------

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]

    # Save uploaded file inside /static/uploads
    upload_dir = os.path.join(app.static_folder, "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)
    file.save(file_path)

    # Preprocess for prediction
    img = preprocess_image(file_path)

    # Predict
    pred = model.predict(img)[0][0]
    confidence = round(float(pred) * 100, 2)  # confidence in %

    if pred > 0.5:
        result = "Healthy"
        return render_template(
            "results_healthy.html",
            result=result,
            confidence=confidence,
            filename=file.filename
        )
    else:
        result = "Diseased: Ergot"
        return render_template(
            "ergot-detected.html",  # create another template for healthy
            result=result,
            confidence=100 - confidence,
            filename=file.filename
        )


# -------------------- RUN APP --------------------

if __name__ == "__main__":
    app.run(debug=True)
