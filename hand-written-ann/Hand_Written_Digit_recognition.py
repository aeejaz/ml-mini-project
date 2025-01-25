
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt

@st.cache_resource
def load_trained_model():
    try:
        model = load_model('project.h5', compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_trained_model()

# Interface
st.title("Handwritten Digit Recognition")
st.write("Draw a digit on the canvas and let the model predict it.")

# Create a drawable canvas for digit input
canvas = st_canvas(
    fill_color="rgb(0, 0, 0)",
    stroke_width=30,            
    stroke_color="rgb(255, 255, 255)", 
    background_color="rgb(0, 0, 0)", 
    width=350,
    height=350,
    drawing_mode="freedraw",
    key="canvas"
)

# STeps for preprocess the image drawn on the canvas
def preprocess_image(image):
    img = Image.fromarray(image.astype('uint8')).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28 pixels (for MNIST model input)
    img = np.array(img) / 255.0  # Normalize pixel values
    img = img.reshape((1, 28, 28, 1))  # Reshape for the model input (batch_size, height, width, channels)
    
    #confrimation
    st.image(img[0, :, :, 0], use_column_width=True, caption='Preprocessed Image')
    
    return img

# What should happen whbn 'Predict' button is pressed
if st.button("Predict"):
    if model:
        if canvas.image_data is not None and np.any(canvas.image_data):  # Ensure there's something drawn on the canvas
            img = preprocess_image(canvas.image_data[:, :, 0])  # Preprocess the image for model input
            prediction = model.predict(img)  # Make prediction using the model
            predicted_class = np.argmax(prediction)  # Get the predicted class (0-9 digit)

            # Display the prediction
            st.subheader("Prediction")
            st.write(f"The model predicts the digit as: {predicted_class}")

            # Display the drawn digit image
            st.subheader("Digit Image")
            fig, ax = plt.subplots()
            ax.imshow(canvas.image_data[:, :, 0], cmap='gray')
            ax.axis('off')  # Hide axis
            st.pyplot(fig)
        else:
            st.warning("Please draw a digit on the canvas.")
    else:
        st.warning("Model not loaded. Please check the error message above.")


if st.button("Clear"):
    # Reseting the  canvas to blank
    canvas.image_data = np.zeros((150, 150, 3), dtype=np.uint8)
