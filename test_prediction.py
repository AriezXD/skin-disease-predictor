print("TEST FILE STARTED")
from PIL import Image

from src.predictor import SkinDiseasePredictor


MODEL_PATH = "model/best_accuracy_model.pth"


predictor = SkinDiseasePredictor(MODEL_PATH)

print("Model loaded successfully!")
print("Device:", predictor.device)
print("Classes:", predictor.model.classifier)


# Change this to an actual skin image on your computer
IMAGE_PATH = "test_image.jpg"

image = Image.open(IMAGE_PATH).convert("RGB")

result = predictor.predict(image)

print("\nPrediction:")
print("Disease:", result["disease"])
print("Confidence:", f"{result['confidence'] * 100:.2f}%")