import tempfile
import numpy as np
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F

from utils.simple_preprocessing import preprocess, predict
from utils.patch_preprocessing import patch_preprocessing, plot_patches
from utils.models.ENB0 import EfficientNetClassifier
from utils.models.ENB1 import getB1Model
from utils.models.Swin import getSwinModel

num_classes = 2

# Initialize the model and move it to the device
ENB0 = EfficientNetClassifier(num_classes)
ENB0.load_state_dict(torch.load('models/best_model.pth', map_location=torch.device('cpu')))
ENB0.eval()

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
        'model': ENB0,
        'preprocess': patch_preprocessing,
        'patches': 15,
        'name': "EfficientNet B0",
        'val_loss_path': 'utils/viz_data/ENB0/val_loss.npy',
        'train_loss_path': 'utils/viz_data/ENB0/train_loss.npy',
        'y_pred_path': 'utils/viz_data/ENB0/y_pred.npy',
        'y_true_path': 'utils/viz_data/ENB0/y_true.npy'
    },
    'ENB1': {
        'model': getB1Model('models/efficientNet_classifier.pth'),
        'preprocess': preprocess,
        'resolution': 512,
        'name': "EfficientNet B1",
        'val_loss_path': 'utils/viz_data/ENB1/all_val_lossesNB1.npy',
        'train_loss_path': 'utils/viz_data/ENB1/all_train_lossesNB1.npy',
        'y_pred_path': 'utils/viz_data/ENB1/y_predNB1.npy',
        'y_true_path': 'utils/viz_data/y_trueSimple.npy'
    },
    'Swin': {
        'model': getSwinModel('models/swin_binary_classifier.pth'),
        'preprocess': preprocess,
        'resolution': 384,
        'name': "Swin Transformer",
        'val_loss_path': 'utils/viz_data/Swin/all_val_lossesSwin.npy',
        'train_loss_path': 'utils/viz_data/Swin/all_train_lossesSwin.npy',
        'y_pred_path': 'utils/viz_data/Swin/y_predSwin.npy',
        'y_true_path': 'utils/viz_data/y_trueSimple.npy'
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
        processed_models = []
        show = True

        temp_path = None

        if uploaded_image is not None:
            # Create a temporary file to store the uploaded image
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tiff") as temp_file:
                temp_file.write(uploaded_image.read())
                temp_path = temp_file.name

        for model in st.session_state.selected_models:
            if model != 'ENB0':
                if 'Simple' in processed_models:
                    show = False
                processed_models.append('Simple')
            else:
                processed_models.append('Patch')
            
            if show:
                st.header(f'{processed_models[-1]} Preprocessing Results')

            parameter = None
            if model == 'ENB0':
                parameter = model_dict[model]['patches']
            else:
                parameter = model_dict[model]['resolution']
            result = model_dict[model]['preprocess'](temp_path, parameter)

            if model == 'ENB0':
                plot_patches(result)

            elif show:
                result_img = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
                plt.figure(figsize=(5, 5))  # Create a figure with a specified size
                plt.imshow(result_img, cmap='gray')
                plt.axis('off')  # Turn off axis labels
                plt.show()  # Show the plot
                st.pyplot(plt)
                plt.close()

            processed_models.append(model)

            st.header(f'{model_dict[model]['name']} Prediction Results')

            if model_dict[model]['name'] == 'EfficientNet B0':

                # Disable gradient computation for inference
                with torch.no_grad():
                    # Get the model's raw output (logits)
                    result = result.to(torch.float32)
                    output = model_dict[model]['model'](result)

                    # Apply softmax to get probabilities
                    probabilities = F.softmax(output, dim=1)

                    # Convert probabilities to predicted labels (1 or 0)
                    predicted_labels = torch.argmax(probabilities, dim=1).tolist()

                    # Calculate the most voted label and its percentage
                    label_counts = torch.bincount(torch.tensor(predicted_labels))
                    most_voted_label = label_counts.argmax().item()
                    most_voted_percentage = (label_counts[most_voted_label] / len(predicted_labels)) * 100

                    # Print the most voted label and its percentage
                    st.subheader(f"Predicted Class: {'LAA' if most_voted_label == 1 else 'CE'} with {most_voted_percentage:.2f}% of the votes")

                    # Convert patches to a format suitable for display
                    plt.figure(figsize=(8, 5))  # Adjust figure size as needed
                    for i in range(15):
                        plt.subplot(3, 5, i + 1)  # Create a 3x5 grid for 15 patches
                        patch = result[i].cpu().numpy().transpose(1, 2, 0)  # Convert tensor to numpy and adjust dimensions

                        # Ensure pixel values are in the range [0, 1] for display
                        patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-5)

                        # If the image has only one channel (grayscale), handle it separately
                        if patch.shape[2] == 1:
                            patch = patch.squeeze(-1)  # Remove the last dimension for grayscale images
                            plt.imshow(patch, cmap='gray')
                        else:
                            plt.imshow(patch)

                        # Set the title with the predicted label
                        plt.title(f'Label: {'LAA' if predicted_labels[i] == 1 else 'CE'}')
                        plt.axis('off')  # Hide the axis

                    plt.tight_layout()
                    plt.show()
                    st.pyplot(plt)
                    plt.close()

                # Set model to train mode for saliency computation
                model_dict[model]['model'].eval()
    
                # Ensure gradients are enabled for the entire batch (if not already set)
                result.requires_grad_()
                
                saliency_maps = []
                predicted_labels = []

                for i in range(result.size(0)):
                    patch = result[i:i+1].clone().detach().requires_grad_(True)  # Explicitly set requires_grad=True for each patch
                    
                    output = model_dict[model]['model'](patch)
                    predicted_class = torch.argmax(output, dim=1).item()
                    predicted_labels.append(predicted_class)
                    
                    model_dict[model]['model'].zero_grad()
                    output[0, predicted_class].backward()
                    
                    # Check if grad is available for saliency computation
                    if patch.grad is not None:
                        saliency = patch.grad.abs()
                        saliency, _ = saliency.max(dim=1)
                        saliency_maps.append(saliency.squeeze().cpu().numpy())
                    else:
                        # Handle case if grad is None (to avoid errors)
                        saliency_maps.append(None)

                # Plot the saliency maps, handling cases where grad might be None
                plt.figure(figsize=(8, 5))
                for i in range(15):
                    plt.subplot(3, 5, i + 1)
                    if saliency_maps[i] is not None:
                        plt.imshow(saliency_maps[i], cmap='hot')
                        plt.title(f'Label: {"LAA" if predicted_labels[i] == 1 else "CE"}')
                    else:
                        plt.text(0.5, 0.5, "No Grad", ha='center', va='center')
                    
                    plt.axis('off')

                plt.tight_layout()
                plt.show()
                st.pyplot(plt)
                plt.close()

            else:
                probabilities, saliency = predict(model_dict[model]['model'], result)

                st.columns(3)[1].subheader(f"{probabilities[0]:.2f}% CE")
                st.columns(3)[1].subheader(f"{probabilities[1]:.2f}% LAA")

                st.subheader("Saliency Map")
                plt.figure(figsize=(5, 5))
                plt.imshow(saliency, cmap='hot')
                plt.axis('off')

                plt.show()
                st.pyplot(plt)
                plt.close()

elif st.button("View Model Performance"):
    if st.session_state.selected_models:
        for model in st.session_state.selected_models:

            st.header(f'{model_dict[model]['name']} Performance Overview')

            val_loss = np.load(model_dict[model]['val_loss_path'])
            train_loss = np.load(model_dict[model]['train_loss_path'])

            fig = go.Figure()
            epochs = np.arange(1, len(train_loss) + 1)

            fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode='lines+markers', name='Train Loss'))
            fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines+markers', name='Validation Loss'))

            # Customize the layout
            fig.update_layout(
                title='Model Training and Validation Loss',
                xaxis_title='Epochs',
                yaxis_title='Loss',
                hovermode='x unified',
                template='plotly_dark'  # You can change this to 'plotly', 'ggplot2', etc.
            )

            # Display the interactive plot in Streamlit
            st.plotly_chart(fig)

            y_true = np.load(model_dict[model]['y_true_path'])
            y_pred = np.load(model_dict[model]['y_pred_path'])

            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)

            # Create an interactive heatmap with Plotly
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted CE', 'Predicted LAA'],
                y=['Actual CE', 'Actual LAA'],
                hoverongaps=True,
                colorscale='Blues'
            ))

            # Update layout for better clarity
            fig.update_layout(
                title='Confusion Matrix',
                xaxis_title='Predicted Label',
                yaxis_title='True Label',
                template='plotly_dark'
            )

            # Display the interactive plot in Streamlit
            st.plotly_chart(fig, use_container_width=True)


