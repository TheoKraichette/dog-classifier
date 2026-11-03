from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import json
import io
import os

MODEL_PATH = os.getenv("MODEL_PATH", "models/final_transfer_model.h5")
CLASS_INDICES_PATH = os.getenv("CLASS_INDICES_PATH", "models/class_indices.json")

app = FastAPI()

# --- Charger les classes ---
with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)

# Inverser le dictionnaire pour avoir {index: classe}
index_to_class = {v: k for k, v in class_indices.items()}

# --- Modèle chargé à la première prédiction ---
# Le .h5 n'est pas versionné : on ne le charge pas à l'import, sinon l'API ne
# démarre pas du tout quand il est absent.
_model = None


def get_model():
    global _model
    if _model is None:
        from tensorflow.keras.models import load_model
        _model = load_model(MODEL_PATH)
    return _model


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "model_available": os.path.exists(MODEL_PATH),
        "model_loaded": _model is not None,
        "num_classes": len(index_to_class),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not os.path.exists(MODEL_PATH):
        return {
            "error": (
                f"Modèle introuvable : {MODEL_PATH}. "
                "Entraîner le modèle (scripts/train.py) puis déposer le fichier "
                "dans models/ — le dossier est monté dans le conteneur."
            )
        }

    try:
        model = get_model()

        # Charger l'image
        contents = await file.read()  # lire le contenu du fichier envoyé
        image = load_img(io.BytesIO(contents), target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0) / 255.0

        # Faire une prédiction
        predictions = model.predict(image)[0]
        top_indices = predictions.argsort()[-3:][::-1]

        # Associer l'indice au label de classe
        top_classes = [{"class": index_to_class[i], "score": float(predictions[i])} for i in top_indices]

        return {"predictions": top_classes}

    except Exception as e:
        return {"error": str(e)}
