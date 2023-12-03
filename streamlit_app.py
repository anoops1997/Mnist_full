
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Assuming you have a model loaded, replace this with your actual model
model = load_model('mnist_model.keras')

def mnist_digit_classification_app():
    st.title("MNIST Digit Classification App")
    uploaded_file = st.file_uploader("Choose a digit image...", type="jpg", key="file")

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        # Preprocess the image for the model
        img = Image.open(uploaded_file)
        img = img.resize((28, 28))
        img = np.array(img)
        img = img.reshape((1, 28, 28, 1)).astype('float32') / 255.0

        # Make prediction
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)

        st.write("")
        st.write("Classifying...")

        # Display the predicted digit and confidence
        st.write(f"Predicted Digit: {predicted_class}")
        st.write(f"Confidence: {prediction[0][predicted_class]:.2%}")
        
mnist_digit_classification_app()
