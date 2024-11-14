import torch
import numpy as np
from openslide import OpenSlide
import cv2
from torchvision import transforms

def preprocess(image_path, resolution):
    slide = OpenSlide(image_path)
    region = (1000, 1000)    
    size = (5000, 5000)
    
    # Read the region from the slide
    image = slide.read_region(region, 0, size)
    
    # Convert the PIL image to a NumPy array
    image = np.array(image)
    
    # Convert from BGRA to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

    # Convert to PyTorch tensor and normalize
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Apply transformations
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    return image

# Prediction and saliency function
def predict(model, image):
    # Class probabilities
    with torch.no_grad():
        output = model(image)
    
    # Apply sigmoid to get probability for class 1 (positive class)
    class_1 = torch.sigmoid(output).item() * 100  # Convert to percentage
    class_0 = 100 - class_1

    # Saliency Map
    image.requires_grad = True
    output = model(image)
    predicted_class = output.argmax(dim=1)

    # Zero gradients and backward pass
    model.zero_grad()
    output[0, predicted_class].backward()

    # Compute saliency map
    saliency, _ = torch.max(image.grad.data.abs(), dim=1)
    saliency = saliency.squeeze().cpu().numpy()

    return (class_0, class_1), saliency