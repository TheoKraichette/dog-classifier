import streamlit as st
import requests
from PIL import Image
import os
import time

# En Docker le backend est joignable par son nom de service, pas par localhost.
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Dog Breed Classifier", page_icon="🐶")

st.title("🐶 Dog Breed Classifier")
st.write("Upload a picture of a dog and get the Top 3 predicted breeds!")

# Upload d'image
uploaded_file = st.file_uploader("📂 Choose a dog image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

    if st.button("Predict 🧠"):
        # Barre de chargement
        with st.spinner('Prediction in progress...'):
            file_bytes = uploaded_file.getvalue()
            files = {"file": (uploaded_file.name, file_bytes, "image/jpeg")}

            try:
                response = requests.post(f"{BACKEND_URL}/predict", files=files, timeout=120)

                if response.status_code == 200:
                    response_json = response.json()

                    if "predictions" in response_json:
                        preds = response_json["predictions"]
                        st.success("✅ Prediction successful!")

                        # Affichage des prédictions
                        for idx, pred in enumerate(preds):
                            st.write(f"**{idx + 1}. {pred['class']}** — {pred['score'] * 100:.2f}%")

                        # Optionnel : petit graphique
                        st.subheader("📊 Prediction Scores")
                        labels = [p["class"] for p in preds]
                        scores = [p["score"] for p in preds]
                        st.bar_chart(data=scores)

                    elif "error" in response_json:
                        st.error(f"❌ Server error: {response_json['error']}")
                    else:
                        st.error("❌ Unexpected server response.")
                else:
                    st.error(f"❌ Prediction failed! Error code {response.status_code}")

            except requests.exceptions.ConnectionError:
                st.error(f"❌ Could not connect to the FastAPI backend. Make sure it's running on {BACKEND_URL}")
