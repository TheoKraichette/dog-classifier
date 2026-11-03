# 🐶 Dog Breed Classification with Deep Learning

Projet de classification de races de chiens utilisant un CNN from scratch et du Transfer Learning.

## 👨‍💻 Collaborateurs

- **Bouba** (Lead Developer & Data Preprocessing)
- **Théo** (Model Architect & Experimentation)
- **Hicham** (DevOps & Web Integration)

---

## 📂 Structure du projet

```
├── data/
│   ├── processed/          # Images redimensionnées
│   └── raw/                # Images brutes originales (.mat inclus)
├── models/                 # Modèles entraînés (.h5 CNN et Transfer Learning)
├── notebooks/              # Notebooks Jupyter (EDA, Modèles)
│   ├── cnn_scratch.ipynb
│   ├── transfer_learning.ipynb
│   ├── eda_images.ipynb
│   └── eda_classes.ipynb
├── presentation/           # Slides de présentation
├── scripts/                # Scripts Python (processing, training, augmentation)
│   ├── augmentation_data.py
│   ├── download_data.py
│   ├── infer.py
│   ├── models.py
│   ├── preprocess.py
│   ├── test_app.py
│   └── train.py
├── webapp/
│   ├── backend/
│   │   └── app.py          # FastAPI backend
│   └── frontend/
│       └── frontend_app.py # Streamlit frontend
├── requirements.txt        # Dépendances principales
└── README.md               # Ce fichier
```

---

## ⚙️ Installation

1. **Cloner le dépôt :**
```bash
git clone https://github.com/Calamia92/dog-classifier.git
cd dog-classifier
```

2. **Créer et activer un environnement virtuel :**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate
```

3. **Installer les dépendances :**
```bash
pip install -r requirements.txt -r dev-requirements.txt
```

---

## 🧠 Deep Learning & Traitement d'images

Technos principales utilisées :
- TensorFlow / Keras
- NumPy / Pandas
- Matplotlib
- Pillow (PIL)
- Albumentations
- OpenCV
- FastAPI
- Streamlit

---

## 🚀 Usage

### 1. Prétraitement des images
```bash
python scripts/preprocess.py --input_dir data/raw --output_dir data/processed --image_size 224
```

---

### 2. Entraînement du modèle

**a. CNN from scratch**
```bash
python scripts/train.py --model scratch --batch_size 32 --epochs 10
```

**b. Transfer Learning (VGG16)**
```bash
python scripts/train.py --model transfer --batch_size 32 --epochs 10
```

---

## 🐳 Lancement avec Docker (recommandé)

Aucun outillage Python n'est requis sur le poste, seulement Docker.

```bash
docker compose up -d --build
```

| Service   | URL                     | Détail                              |
|-----------|-------------------------|-------------------------------------|
| Frontend  | http://localhost:8501   | Interface Streamlit                 |
| Backend   | http://localhost:8001   | API FastAPI (port 8000 en interne)  |
| Health    | http://localhost:8001/health | État du service et du modèle   |
| Swagger   | http://localhost:8001/docs   | Documentation interactive      |

Les ports hôtes sont surchargeables (utile si un autre projet occupe déjà le port) :

```bash
BACKEND_PORT=9000 FRONTEND_PORT=9501 docker compose up -d
```

Arrêt : `docker compose down`. Logs : `docker compose logs -f backend`.

### ⚠️ Modèle entraîné requis pour les prédictions

Les poids `models/final_transfer_model.h5` ne sont **pas versionnés** (voir
`.gitignore`). L'API démarre malgré leur absence, mais `/predict` renvoie une
erreur explicite tant que le fichier manque — `/health` expose
`"model_available": false`.

Pour activer les prédictions : entraîner le modèle (voir *Entraînement* plus
bas) puis déposer le `.h5` dans `models/`. Le dossier est monté en volume,
un simple `docker compose restart backend` suffit — pas de rebuild.

---

## 🌐 Lancement de l'application Web (sans Docker)

### 1. Lancer le Backend FastAPI
Depuis la racine :
```bash
uvicorn webapp.backend.app:app --reload
```
Backend disponible sur : http://localhost:8000

### 2. Lancer le Frontend Streamlit
Depuis la racine :
```bash
streamlit run webapp/frontend/frontend_app.py
```
Frontend disponible sur : http://localhost:8501

---

## 📚 API Documentation

- **GET /** → Test serveur (Hello World)
- **GET /health** → État du service, disponibilité du modèle, nombre de classes
- **POST /predict** → Envoyer une image et obtenir les 3 meilleures prédictions

Exemple réponse JSON :
```json
{
  "predictions": [
    {"class": "Labrador_retriever", "score": 0.85},
    {"class": "Golden_retriever", "score": 0.10},
    {"class": "Beagle", "score": 0.05}
  ]
}
```

---

## ✅ Tests

Pour lancer les tests unitaires :
```bash
pytest scripts/test_app.py
```

---

## 📦 Dépendances principales (`requirements.txt`)

```text
numpy
pandas
tensorflow
keras
scikit-learn
matplotlib
Pillow
opencv-python
albumentations
fastapi
uvicorn
streamlit
python-multipart
```

---

## 📝 Licence

Projet sous licence **MIT**.