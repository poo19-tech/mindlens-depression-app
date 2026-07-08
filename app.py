"""
app.py -- Flask backend
------------------------
STEP-BY-STEP EXPLANATION:

1. This is a small web server. When someone visits your site, it sends
   them the HTML page (templates/index.html).

2. The browser's JavaScript captures a photo (from webcam or an uploaded
   file) and sends it to this server as a base64-encoded image via the
   "/predict" endpoint.

3. This server:
   a. Decodes the image
   b. Uses OpenCV's Haar Cascade to find the face in the photo
   c. Crops just the face, resizes it to 96x96 (same size the model
      was trained on)
   d. Feeds it to the trained MobileNetV2 model
   e. Sends back JSON: {"label": "Depressed" or "Not Depressed",
                         "confidence": 0.87}

4. Why Flask + browser JS instead of a desktop app?
   -> This makes it "platform independent": it runs in ANY browser on
      Windows, Mac, Linux, Android, iPhone -- no installation needed for
      the end user. Only the person hosting the server needs Python.

HOW TO RUN LOCALLY:
   1. pip install -r requirements.txt
   2. Put your trained depression_model.h5 file in backend/model/
   3. python app.py
   4. Open http://127.0.0.1:5000 in your browser
"""

import os
import json
import base64
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
app = Flask(__name__)

# ------------------------------------------------------------------
# Load the trained model ONCE when the server starts (not per-request,
# which would be slow)
# ------------------------------------------------------------------
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "model", "depression_model.weights.h5")
IMG_SIZE = 96
CLASS_NAMES = ["Not Depressed", "Depressed"]

print("Rebuilding model architecture...")

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights=None
)

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

print("Loading trained weights...")
model.load_weights(WEIGHTS_PATH)
print("Model loaded successfully.")
# Load OpenCV's built-in, free, pre-trained face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def decode_base64_image(base64_string):
    """Converts a base64 string (sent from the browser) into an OpenCV image."""
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]  # strip "data:image/png;base64,"
    img_bytes = base64.b64decode(base64_string)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


def detect_and_crop_face(img):
    """
    Finds the largest face in the image and returns the cropped face.
    Returns None if no face is found.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) == 0:
        return None

    # If multiple faces are detected, use the largest one
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]
    face_crop = img[y:y + h, x:x + w]
    return face_crop


@app.route("/")
def home():
    """Serves the main webpage."""
    return render_template("index.html")


@app.route("/accuracy")
def accuracy():
    """
    Returns the real test accuracy saved by train_model.py (metrics.json),
    so the 'About' tab on the website shows YOUR actual trained number
    instead of a placeholder.
    """
    metrics_path = os.path.join(os.path.dirname(__file__), "model", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return jsonify(json.load(f))
    return jsonify({"accuracy": None})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Receives an image (base64) from the browser, detects the face,
    runs it through the model, and returns the prediction as JSON.
    """
    data = request.get_json()

    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        img = decode_base64_image(data["image"])
    except Exception:
        return jsonify({"error": "Could not decode image"}), 400

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    face = detect_and_crop_face(img)
    if face is None:
        return jsonify({"error": "No face detected. Please face the camera clearly."}), 200

    # Preprocess exactly like during training
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = np.expand_dims(face, axis=0).astype(np.float32)
    face = preprocess_input(face)

    # Model outputs a single number between 0 and 1
    # (probability of being "Depressed", since that's class 1)
    prediction = model.predict(face, verbose=0)[0][0]

    label_index = 1 if prediction >= 0.5 else 0
    label = CLASS_NAMES[label_index]
    confidence = float(prediction if label_index == 1 else 1 - prediction)

    return jsonify({
        "label": label,
        "confidence": round(confidence * 100, 2)
    })


if __name__ == "__main__":
    # host="0.0.0.0" makes it accessible from other devices on your network too
    app.run(host="0.0.0.0", port=5000, debug=True)
