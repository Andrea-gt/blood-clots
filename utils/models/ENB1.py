import torch
import torch.nn as nn
from torchvision import models

def getB1Model(load_path):
    # Define the EfficientNet model
    model = models.efficientnet_b1(pretrained=True)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),               # Dropout layer with 30% probability
        nn.Linear(num_features, 1)       # Final layer for binary classification
    )
    model.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))

    return model