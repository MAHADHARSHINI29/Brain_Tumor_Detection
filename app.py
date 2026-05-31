import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

cnn_model = load_model("model_vgg_full.h5")

IMAGE_SIZE = 150

# Preprocess an image
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image /= 255.0  # normalize to [0,1] range
    return image

# Read the image from path and preprocess
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

# Predict & classify image
def classify(model, image_path):
    preprocessed_image = load_and_preprocess_image(image_path)
    preprocessed_image = tf.reshape(preprocessed_image, (1, IMAGE_SIZE, IMAGE_SIZE, 3))
    prob = model.predict(preprocessed_image)[0]
    print(prob)
    # Get the index of the maximum probability
    predicted_label_index = np.argmax(prob)
    # Mapping index to label name
    label_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
    label = label_names[predicted_label_index]
    classified_prob = prob[predicted_label_index]
    return label, classified_prob

# home page
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/classify", methods=["POST", "GET"])
def upload_file():
    if request.method == "GET":
        return render_template("home.html")
    else:
        file = request.files["image"]
        upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        print(upload_image_path)
        file.save(upload_image_path)
        # BUGFIX: Use cnn_model instead of undefined 'model'
        label, prob = classify(cnn_model, upload_image_path)
        prob = round((prob * 100), 2)
    return render_template(
        "classify.html", image_file_name=file.filename, label=label, prob=prob
    )

@app.route("/classify/<filename>")
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(port=5000)
