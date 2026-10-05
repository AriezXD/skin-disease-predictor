from PIL import Image
import numpy as np
import torch

from src.predictor import SkinDiseasePredictor, CLASSES
from src.gradcam import GradCAM


MODEL_PATH = "model/best_accuracy_model.pth"
IMAGE_PATH = "test_image.jpg"


# Load predictor
predictor = SkinDiseasePredictor(MODEL_PATH)

model = predictor.model
transform = predictor.transform
device = predictor.device


# Load image
image = Image.open(IMAGE_PATH).convert("RGB")

# Convert image to numpy
image_np = np.array(image)


# Preprocess
image_tensor = transform(image).unsqueeze(0)
image_tensor = image_tensor.to(device)


# Get prediction
with torch.no_grad():
    output = model(image_tensor)
    probabilities = torch.softmax(output, dim=1)

predicted_idx = torch.argmax(probabilities, dim=1).item()
confidence = probabilities[0, predicted_idx].item()

# Convert predicted index to disease name
predicted_disease = CLASSES[predicted_idx]

print("Predicted disease:", predicted_disease)
print("Confidence:", f"{confidence * 100:.2f}%")


# Grad-CAM
gradcam = GradCAM(model)

cam = gradcam.generate(
    image_tensor,
    predicted_idx
)


# Create Grad-CAM overlay
overlay = gradcam.overlay(
    image_np,
    cam
)


# Save result
output_path = "gradcam_result.jpg"

Image.fromarray(overlay).save(output_path)

print("Grad-CAM saved to:", output_path)