# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(page_title="Digit Recognizer", layout="centered")

# Load model only once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_ann.h5")

model = load_model()

# Preprocessing: convert to grayscale, invert, resize to 28x28, normalize
def preprocess_image(image: Image.Image):
    image = image.convert("L")  # Convert to grayscale
    image = ImageOps.invert(image)  # MNIST is white on black
    image = image.resize((28, 28))
    image_array = np.array(image) / 255.0
    return image_array.reshape(1, 28, 28)

# Title
st.title("🧠 Handwritten Digit Recognizer")
st.markdown("Upload **one or more images** of handwritten digits. The model will predict the digit in each image.")

# Upload widget
uploaded_files = st.file_uploader("Upload digit image(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.divider()

        # Load and display the image
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Uploaded: {uploaded_file.name}", width=150)

        # Preprocess and predict
        processed = preprocess_image(image)
        prediction = model.predict(processed, verbose=0)
        predicted_class = np.argmax(prediction)

        # Show result
        st.success(f"✅ Predicted Digit: **{predicted_class}**")
