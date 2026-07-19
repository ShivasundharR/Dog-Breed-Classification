import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import tf_keras as tfk
import numpy as np
from PIL import Image

# Load the pre-trained model from TensorFlow Hub

IMG_SIZE = 224
MODEL_PATH = "models/Dog Breed Identifier Mobilenetv2 Model.h5"
LABELS_PATH = "unique_labels.txt"

@st.cache_resource
def load_model():
  """
  Loads a model from the specified path
  """
  model = tfk.models.load_model(MODEL_PATH, custom_objects={"KerasLayer":hub.KerasLayer})

  return model

@st.cache_resource
def load_breed_names():
    """
    loads the breed names from the specified path
    """
    with open(LABELS_PATH) as f:
        return [line.strip() for line in f]


def process_image(pil_image):
    """
    Preprocesses the input image for prediction
    """
    image = np.array(pil_image.convert("RGB"), dtype=np.float32)
    image = image / 255.0
    image = tf.image.resize(image, size=(IMG_SIZE, IMG_SIZE))
    image = np.expand_dims(image, axis=0)  # add batch dimension
    return image

model = load_model()
breed_names = load_breed_names()
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', width="stretch")

    with st.spinner('Classifying...'):
        processed_image = process_image(img)
        predictions = model.predict(processed_image)[0]

    top_idx = np.argmax(predictions)
    top_breed = breed_names[top_idx]
    top_confidence = predictions[top_idx] * 100

    st.subheader(f"Prediction: {top_breed}")
    st.write(f"Confidence: {top_confidence:.1f}%")

    st.markdown("### Top 5 predictions")
    top_5_idx = np.argsort(predictions)[-5:][::-1]
    for idx in top_5_idx:
        st.write(f"- {breed_names[idx]}: {predictions[idx] * 100:.1f}%")


