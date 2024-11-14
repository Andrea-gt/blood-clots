import tempfile
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

from utils.simple_preprocessing import preprocess
from utils.patch_preprocessing import patch_preprocessing, plot_patches

# Initialize the session state for selected models if it doesn't already exist
if 'selected_models' not in st.session_state:
    st.session_state.selected_models = set()

# Function to toggle model selection
def toggle_model(model_name, selected):
    if selected:
        st.session_state.selected_models.add(model_name)
    else:
        st.session_state.selected_models.remove(model_name)

st.title("Ischemic Stroke Blood Clot Classification")

# File uploader for image submission
uploaded_image = st.file_uploader("Upload an image (TIFF format)", type=["tiff"])
    
# Buttons to select the AI model in a row
st.write("Choose an AI model:")
col1, col2, col3 = st.columns([1, 1, 3])

model_dict = {
    'ENB0': {
        'model': None,
        'preprocess': patch_preprocessing,
        'patches': 15,
        'name': "EfficientNet B0"
    },
    'ENB1': {
        'model': None,
        'preprocess': preprocess,
        'resolution': 512,
        'name': "EfficientNet B1"
    },
    'Swin': {
        'model': None,
        'preprocess': preprocess,
        'resolution': 384,
        'name': "Swin Transformer"
    },
}

# Set default checkbox values based on selected_models in session state
with col1:
    enb0_selected = 'ENB0' in st.session_state.selected_models
    enb0_checkbox = st.checkbox("EfficientNet B0", value=enb0_selected)
    if enb0_checkbox != enb0_selected:  # Only call toggle_model if the state changes
        toggle_model('ENB0', enb0_checkbox)

with col2:
    enb1_selected = 'ENB1' in st.session_state.selected_models
    enb1_checkbox = st.checkbox("EfficientNet B1", value=enb1_selected)
    if enb1_checkbox != enb1_selected:  # Only call toggle_model if the state changes
        toggle_model('ENB1', enb1_checkbox)

with col3:
    swin_selected = 'Swin' in st.session_state.selected_models
    swin_checkbox = st.checkbox("Swin Transformer", value=swin_selected)
    if swin_checkbox != swin_selected:  # Only call toggle_model if the state changes
        toggle_model('Swin', swin_checkbox)

if st.button("Submit & Predict"):
    if st.session_state.selected_models:

        temp_path = None

        if uploaded_image is not None:
            # Create a temporary file to store the uploaded image
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tiff") as temp_file:
                temp_file.write(uploaded_image.read())
                temp_path = temp_file.name

        for model in st.session_state.selected_models:
            st.subheader(f'{model_dict[model]['name']} Preprocessing Results')
            parameter = None
            if model == 'ENB0':
                parameter = model_dict[model]['patches']
            else:
                parameter = model_dict[model]['resolution']
            result = model_dict[model]['preprocess'](temp_path, parameter)

            if model == 'ENB0':
                plot_patches(result)

            else:
                plt.figure(figsize=(5, 5))  # Create a figure with a specified size
                plt.imshow(result, cmap='gray')
                plt.axis('off')  # Turn off axis labels
                plt.show()  # Show the plot
                st.pyplot(plt)
                plt.close()



