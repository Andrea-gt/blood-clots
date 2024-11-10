import streamlit as st
from utils.simple_preprocessing import preprocess
from PIL import Image

st.title("Ischemic Stroke Blood Clot Classification")

# File uploader for image submission
uploaded_image = st.file_uploader("Upload an image (JPEG or PNG format)", type=["jpg", "jpeg", "png"])

# Display the uploaded image
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)

# Buttons to select the AI model in a row
st.write("Choose an AI model:")
col1, col2, col3 = st.columns([1, 1, 3])

model_dict = {
    'ENB0': {
        'model': None,
        'preprocess': None,
        'resolution': 512
    },
    'ENB1': {
        'model': None,
        'preprocess': preprocess,
        'resolution': 512
    },
    'Swin': {
        'model': None,
        'preprocess': preprocess,
        'resolution': 384
    },

}

with col1:
    if st.button("EfficientNet B0"):
        st.write("Submitting the image to EfficientNet B0...")
        # Add logic to send the image to Model 1 for processing

with col2:
    if st.button("EfficientNet B1"):
        st.write("Submitting the image to EfficientNet B1...")
        # Add logic to send the image to Model 2 for processing

with col3:
    if st.button("Swin Transformer"):
        st.write("Submitting the image to Swin Transformer...")

        # Add logic to send the image to Model 3 for processing

# Placeholder for response or results
st.write("Results from the selected model will be displayed here.")