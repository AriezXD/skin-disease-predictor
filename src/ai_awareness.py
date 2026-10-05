import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors
from pydantic import BaseModel
from typing import List


load_dotenv()

client = genai.Client()


class SeekHelp(BaseModel):
    recommended: bool
    urgency: str
    reasons: List[str]


class DiseaseAwareness(BaseModel):
    what_is_it: str
    possible_factors: List[str]
    what_to_do_next: List[str]
    seek_help: SeekHelp
    important_note: str


def get_disease_awareness(disease, confidence):

    prompt = f"""
You are an educational health-information assistant inside a
skin-disease image classification project.

The machine-learning classifier predicted:

Disease: {disease}
Model score: {confidence * 100:.2f}%

Provide general educational information about this predicted disease.

IMPORTANT RULES:

- Do not change or override the predicted disease.
- Do not diagnose the user.
- Do not say the user definitely has the disease.
- The model score is NOT the probability that the user has the disease.
- Explain what the disease generally is.
- Explain possible contributing factors or common associations.
- Give general awareness-oriented next steps.
- Do not prescribe medication.
- Do not recommend specific treatments, products, doses, or medical procedures.
- Focus on monitoring symptoms, avoiding obvious irritation, documenting changes,
  and considering professional evaluation when appropriate.
- Explain when professional medical evaluation may be appropriate.
- Keep the explanation understandable to a general audience.
- Make it clear that an image-based prediction cannot confirm a diagnosis.
"""

    for attempt in range(3):

        try:

            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": DiseaseAwareness,
                },
            )

            return DiseaseAwareness.model_validate_json(
                response.text
            )

        except errors.ServerError as e:

            print(
                f"Gemini server unavailable. "
                f"Retry {attempt + 1}/3..."
            )

            if attempt < 2:
                time.sleep(3)
            else:
                raise e