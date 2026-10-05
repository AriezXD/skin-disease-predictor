import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

from src.preprocessing import get_transform


CLASSES = [
    "acne_vulgaris",
    "allergic_contact_dermatitis",
    "basal_cell_carcinoma",
    "folliculitis",
    "lichen_planus",
    "lupus_erythematosus",
    "neutrophilic_dermatoses",
    "photodermatoses",
    "psoriasis",
    "sarcoidosis",
    "scabies",
    "scleroderma",
    "squamous_cell_carcinoma"
]


class SkinDiseasePredictor:

    def __init__(self, model_path):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Recreate the exact EfficientNet-B0 architecture
        self.model = efficientnet_b0(weights=None)

        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(
                self.model.classifier[1].in_features,
                len(CLASSES)
            )
        )

        # Load trained weights
        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=self.device
            )
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = get_transform()

    def predict(self, image):

        image_tensor = self.transform(image)

        image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():

            outputs = self.model(image_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

        confidence, predicted_idx = torch.max(
            probabilities,
            dim=1
        )

        predicted_idx = predicted_idx.item()
        confidence = confidence.item()

        predicted_class = CLASSES[predicted_idx]

        return {
            "disease": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities[0].cpu().numpy()
        }