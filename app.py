import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

# Load the pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    return MobileNetV2(weights="imagenet")

model = load_model()

# Title and description
st.title("Image Classification with MobileNetV2")
st.write("Upload an image, and the model will classify it.")

# Sidebar for settings
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# Uploading an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

# Display image and make a prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Classifying..."):
        time.sleep(2)  # Simulate a delay for processing
        
        # Preprocess the image
        img_array = np.array(image.resize((224, 224)))  # Resize image to match model's input size
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Make predictions
        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        
        # Filter predictions based on confidence threshold
        filtered_predictions = [pred for pred in decoded_predictions if pred[2] >= confidence_threshold]
        
    st.success("Done!")
    
    if filtered_predictions:
        # Display predictions
        st.write("Top Predictions:")
        for i, (imagenet_id, label, score) in enumerate(filtered_predictions):
            st.write(f"{i+1}. {label}: {score:.2f}")
    else:
        st.write("No predictions met the confidence threshold.")
    
    # Graph
    labels = [label for (_, label, _) in decoded_predictions]
    scores = [score for (_, _, score) in decoded_predictions]
    
    fig, ax = plt.subplots()
    ax.barh(labels, scores, color='skyblue')
    ax.set_xlabel("Confidence")
    ax.set_title("Top Predictions")
    ax.invert_yaxis()  # Highest confidence at the top
    
    st.pyplot(fig)
