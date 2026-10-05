import streamlit as st
from PIL import Image
import numpy as np

from src.predictor import SkinDiseasePredictor
from src.gradcam import GradCAM
from src.ai_awareness import get_disease_awareness


# =========================
# Configuration
# =========================

MODEL_PATH = "model/best_accuracy_model.pth"


st.set_page_config(
    page_title="Skin Disease Awareness AI",
    page_icon="🩺",
    layout="wide"
)


# =========================
# Load Model
# =========================

@st.cache_resource
def load_models():

    predictor = SkinDiseasePredictor(MODEL_PATH)

    gradcam = GradCAM(
        predictor.model
    )

    return predictor, gradcam


predictor, gradcam = load_models()


# =========================
# Header
# =========================

st.title("🩺 Skin Disease Awareness AI")

st.write(
    "An explainable AI system that combines image classification, "
    "Grad-CAM visualization, and AI-generated educational information."
)

st.warning(
    "Educational/research prototype only. "
    "The model prediction is not a medical diagnosis."
)


# =========================
# Image Upload
# =========================

uploaded_file = st.file_uploader(
    "Upload a skin image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")

    st.image(
        image,
        width=400
    )


    # =========================
    # Prediction
    # =========================

    with st.spinner("Analyzing image..."):

        result = predictor.predict(image)

    disease = result["disease"]
    confidence = result["confidence"]

    st.divider()

    st.subheader("🧠 Model Prediction")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Predicted Condition",
            disease.replace("_", " ").title()
        )

    with col2:

        st.metric(
            "Model Score",
            f"{confidence * 100:.2f}%"
        )


    # =========================
    # Low Confidence Warning
    # =========================

    if confidence < 0.50:

        st.warning(
            "The model has low confidence in this prediction. "
            "Treat the result as highly uncertain."
        )

    elif confidence < 0.75:

        st.info(
            "The model has moderate confidence. "
            "The prediction should not be interpreted as a diagnosis."
        )


    # =========================
    # Grad-CAM
    # =========================

    st.divider()

    st.subheader("🔍 Model Explanation")

    image_tensor = (
        predictor.transform(image)
        .unsqueeze(0)
        .to(predictor.device)
    )

    predicted_idx = int(
        np.argmax(result["probabilities"])
    )

    cam = gradcam.generate(
        image_tensor,
        predicted_idx
    )

    image_np = np.array(image)

    overlay = gradcam.overlay(
        image_np,
        cam
    )

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Original Image",
            width="stretch"
        )

    with col2:

        st.image(
            overlay,
            caption="Grad-CAM Explanation",
            width="stretch"
        )

    st.caption(
        "Grad-CAM highlights image regions that had greater influence "
        "on the model's prediction. It does not prove that those regions "
        "represent the disease."
    )


    # =========================
    # Gemini Awareness
    # =========================

    st.divider()

    st.subheader("🤖 AI-Generated Disease Awareness")

    with st.spinner(
        "Generating educational information..."
    ):

        awareness = get_disease_awareness(
            disease=disease,
            confidence=confidence
        )


    # =========================
    # What is it?
    # =========================

    st.markdown("### 🧠 What is it?")

    st.write(
        awareness.what_is_it
    )


    # =========================
    # Possible Factors
    # =========================

    st.markdown(
        "### 🔎 Possible Contributing Factors"
    )

    for factor in awareness.possible_factors:

        st.markdown(
            f"- {factor}"
        )


    # =========================
    # What to do next
    # =========================

    st.markdown(
        "### ➡️ What to do next"
    )

    for step in awareness.what_to_do_next:

        st.markdown(
            f"- {step}"
        )


    # =========================
    # Seek Help
    # =========================

    st.markdown(
        "### 🩺 When to Seek Professional Help"
    )

    st.write(
        f"**Urgency:** {awareness.seek_help.urgency}"
    )

    for reason in awareness.seek_help.reasons:

        st.markdown(
            f"- {reason}"
        )


    # =========================
    # Important Note
    # =========================

    st.info(
        awareness.important_note
    )