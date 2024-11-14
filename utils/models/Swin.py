import torch
import timm
import torch.nn as nn

class SwinBinaryClassifier(nn.Module):
    def __init__(self, base_model, num_features, dropout_rate=0.5):
        super(SwinBinaryClassifier, self).__init__()
        self.base_model = base_model
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Reduce both spatial dimensions to 1x1
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer with specified rate
        self.classifier = nn.Linear(num_features, 1)  # Binary classifier

    def forward(self, x):
        x = self.base_model.forward_features(x)  # Features before the classification head

        # Flatten spatial dimensions and apply pooling
        x = x.permute(0, 3, 1, 2)  # Change to [batch_size, 1536, 12, 12]
        x = self.global_pool(x)  # Apply pooling to reduce to [batch_size, 1536, 1, 1]
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, 1536]

        # Apply dropout and classification layer
        x = self.dropout(x)  # Apply dropout before the final layer
        x = self.classifier(x)  # Final output
        return x

def getSwinModel(load_path):

    # Initialize the Swin model
    base_model = timm.create_model('swin_large_patch4_window12_384', pretrained=True)
    num_features = base_model.head.in_features  # Retrieve the in_features before replacing the head
    base_model.head = nn.Identity()  # Remove the classification head
    model = SwinBinaryClassifier(base_model, num_features, dropout_rate=0.5)

    # Load the state dictionary
    model.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))

    return model