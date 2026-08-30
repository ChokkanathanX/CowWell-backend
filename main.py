from __future__ import annotations
import json
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from functools import lru_cache
from typing import Any
import math
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,make_scorer, f1_score
from sklearn.model_selection import train_test_split,StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path
from PIL import Image,ImageOps
import io
from fastapi import UploadFile, File
import tensorflow as tf
import tf_keras as keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VITALS = ["Temperature_C", "Pulse_Rate_bpm", "Weight_Change_Pct"]
MODEL_PATH = Path(__file__).parent / "models" / "keras_model.h5"

image_model = keras.models.load_model(MODEL_PATH)

IMAGE_CLASSES = [
    "Healthy",
    "LSD",
    "FMD",
    "NOT A COW",
]
IMG_SIZE = (224, 224)
DATASETS = {
    "en": os.path.join(BASE_DIR, "Training_english.csv"),
    "ta": os.path.join(BASE_DIR, "Training_tamil.csv"),
}
DISEASE_INFO = {
    "en": os.path.join(BASE_DIR, "disease_info.json"),
    "ta": os.path.join(BASE_DIR, "disease_info_tamil.json"),
}

# NOTE: MODEL_METRICS hardcoded dict has been removed.
# Accuracies are now always computed from a real train/test split inside
# load_models() below, so they reflect the actual current dataset instead
# of a stale, hand-typed number.

VET_CSV = os.path.join(
    BASE_DIR,
    "data",
    "tn_veterinary_hospitals.csv"
)
vet_df = pd.read_csv(VET_CSV).fillna("")

app = FastAPI(title="CowWell Cattle Disease API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def search_overpass_vets(lat, lon, radius):
    query = f"""
    [out:json][timeout:15];

    node["amenity"="veterinary"]
      (around:{radius},{lat},{lon});

    out tags;
    """
    try:
        response = requests.get(
            "https://overpass-api.de/api/interpreter",
            params={"data": query},
            timeout=30,
            headers={
                "User-Agent": "CowWell/1.0",
                "Accept": "application/json",
            },
        )

        print("OVERPASS STATUS:", response.status_code)

        response.raise_for_status()

        data = response.json()

    except Exception as e:
        print("OVERPASS ERROR:", e)
        raise HTTPException(
            status_code=502,
            detail=f"OpenStreetMap lookup failed: {e}",
        )

    results = []

    for element in data.get("elements", []):
        tags = element.get("tags", {})
        center = element.get("center", {})

        if element.get("type") == "node":
            vlat = element.get("lat")
            vlon = element.get("lon")
        else:
            vlat = center.get("lat")
            vlon = center.get("lon")

        if vlat is None or vlon is None:
            continue

        results.append({
            "name": (
                tags.get("name")
                or tags.get("operator")
                or "Veterinary facility"
            ),
            "lat": float(vlat),
            "lon": float(vlon),
            "phone": (
                tags.get("phone")
                or tags.get("contact:phone")
            ),
            "website": (
                tags.get("website")
                or tags.get("contact:website")
            ),
            "address": (
                tags.get("addr:full")
                or tags.get("addr:street")
                or ""
            ),
            "source": "OpenStreetMap",
        })

    return results


def distance_km(lat1, lon1, lat2, lon2):
    R = 6371.0

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = (
        math.sin(dp / 2) ** 2
        + math.cos(p1)
        * math.cos(p2)
        * math.sin(dl / 2) ** 2
    )

    return R * 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a),
    )


def _train_and_evaluate(lang: str):
    """Train models using 5-Fold Stratified CV with Macro F1."""

    if lang not in DATASETS:
        raise ValueError("Unsupported language")

    df = pd.read_csv(DATASETS[lang])

    if "prognosis" not in df.columns:
        raise ValueError("Dataset has no prognosis column")

    # Remove exact duplicate samples
    df = df.drop_duplicates().reset_index(drop=True)

    X = df.drop(columns=["prognosis"])
    y = df["prognosis"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    definitions = {
        "Random Forest": RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Naive Bayes": GaussianNB(),

        "SVM": make_pipeline(
            StandardScaler(),
            SVC(
                probability=True,
                random_state=42
            )
        ),
    }

    models = {}
    accuracies = {}

    # Macro F1 used internally for honest evaluation
    f1_scorer = make_scorer(
        f1_score,
        average="macro"
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    for name, clf in definitions.items():

        # 5-Fold Cross-Validation
        scores = cross_val_score(
            clf,
            X,
            y_enc,
            cv=cv,
            scoring=f1_scorer
        )

        # Mean Macro F1, presented to frontend as percentage
        mean_f1 = scores.mean()

        accuracies[name] = round(
            float(mean_f1 * 100),
            2
        )

        # Train final production model
        clf.fit(X, y_enc)
        models[name] = clf

    symptoms = [
        c for c in X.columns
        if c not in VITALS
    ]

    return (
        models,
        accuracies,
        le,
        symptoms,
        list(X.columns)
    )

@lru_cache(maxsize=2)
def _load_models_cached(lang: str):
    # Cached per-language so /predict stays fast: retraining Random Forest +
    # SVM from scratch on every single prediction request would make the
    # endpoint far too slow to be usable. The cache is invalidated whenever
    # the process restarts, and can be forced to refresh via /retrain below,
    # so accuracies always reflect a real evaluation rather than a constant.
    return _train_and_evaluate(lang)


def load_models(lang: str):
    return _load_models_cached(lang)


@lru_cache(maxsize=2)
def load_info(lang: str):
    with open(DISEASE_INFO[lang], "r", encoding="utf-8") as f:
        return json.load(f)


class PredictRequest(BaseModel):
    language: str = Field(default="en", pattern="^(en|ta)$")
    symptoms: list[str] = Field(default_factory=list)
    temperature_c: float = 38.5
    pulse_rate_bpm: float = 72
    weight_change_pct: float = 0


@app.get("/health")
def health():
    return {"status": "ok", "service": "CowWell"}


@app.post("/retrain")
def retrain(language: str = Query("en", pattern="^(en|ta)$")):
    """Force a fresh train/evaluate pass for a language, bypassing the cache.

    Call this after the underlying CSV changes (or whenever you want a
    guaranteed up-to-date accuracy number) instead of relying on a
    hardcoded figure.
    """
    _load_models_cached.cache_clear()
    _, accuracies, _, _, _ = load_models(language)
    return {"language": language, "model_accuracies": accuracies}


@app.get("/metadata")
def metadata(language: str = Query("en", pattern="^(en|ta)$")):
    try:
        _, accuracies, _, symptoms, _ = load_models(language)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    info = load_info(language)
    diseases = []
    for key, item in info.items():
        diseases.append({
            "key": key,
            "display_name": item.get("display_name", key),
            "severity": item.get("severity", "unknown"),
            "contagious": bool(item.get("contagious", False)),
        })
    return {
        "language": language,
        "symptoms": symptoms,
        "model_accuracies": accuracies,
        "diseases": diseases,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        models, accuracies, le, symptom_columns, feature_columns = load_models(req.language)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    selected = set(req.symptoms)
    row = {}
    vitals = {
        "Temperature_C": req.temperature_c,
        "Pulse_Rate_bpm": req.pulse_rate_bpm,
        "Weight_Change_Pct": req.weight_change_pct,
    }
    for col in feature_columns:
        row[col] = vitals[col] if col in vitals else int(col in selected)

    input_df = pd.DataFrame([row], columns=feature_columns)
    per_model = {}
    prob_stack = []

    for name, clf in models.items():
        probs = clf.predict_proba(input_df)[0]
        prob_stack.append(probs)
        idx = np.argsort(probs)[-3:][::-1]
        per_model[name] = [
            {
                "disease_key": le.inverse_transform([int(i)])[0],
                "probability": float(probs[i] * 100),
            }
            for i in idx
        ]

    info = load_info(req.language)

    def canonical_key(label: str) -> str:
        if label in info:
            return label
        for key, item in info.items():
            if item.get("display_name") == label:
                return key
        return label

    ensemble = np.mean(np.vstack(prob_stack), axis=0)
    top_idx = np.argsort(ensemble)[-3:][::-1]
    top3 = []
    for i in top_idx:
        label = le.inverse_transform([int(i)])[0]
        key = canonical_key(label)
        top3.append({
            "disease_key": key,
            "disease_name": info.get(key, {}).get("display_name", label),
            "probability": float(ensemble[i] * 100),
        })
    disease_key = top3[0]["disease_key"]
    disease = info.get(disease_key, {})

    # Keep every model's result localized while retaining a stable image key.
    for model_results in per_model.values():
        for item in model_results:
            key = canonical_key(item["disease_key"])
            item["disease_name"] = info.get(key, {}).get("display_name", item["disease_key"])
            item["disease_key"] = key

    return {
        "language": req.language,
        "predicted_disease_key": disease_key,
        "predicted_disease": disease.get("display_name", disease_key),
        "confidence": top3[0]["probability"],
        "top3": top3,
        "model_predictions": per_model,
        "model_accuracies": accuracies,
        "disease_info": disease,
        "selected_symptoms": req.symptoms,
        "vitals": {
            "temperature_c": req.temperature_c,
            "pulse_rate_bpm": req.pulse_rate_bpm,
            "weight_change_pct": req.weight_change_pct,
        },
    }


@app.get("/nearby-vets")
def nearby_vets(lat: float, lon: float, radius: int = 25000):
    csv_results = []

    for _, row in vet_df.iterrows():

        if pd.isna(row["latitude"]) or pd.isna(row["longitude"]):
            continue

        distance = distance_km(
            lat,
            lon,
            float(row["latitude"]),
            float(row["longitude"]),
        )

        if distance * 1000 <= radius:

            csv_results.append({
                "name": str(row["name"]),
                "address": str(row["address"]),
                "phone": str(row["phone"]) if not pd.isna(row["phone"]) else "",
                "lat": float(row["latitude"]),
                "lon": float(row["longitude"]),
                "distance_km": round(distance, 2),
                "source": "Tamil Nadu Veterinary Dataset",
            })

    # Sort nearest first
    csv_results.sort(
        key=lambda x: x["distance_km"]
    )

    return {
        "lat": lat,
        "lon": lon,
        "radius_m": radius,
        "count": len(csv_results),
        "facilities": csv_results[:20],
    }
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):

    try:
        contents = await file.read()

        image = Image.open(io.BytesIO(contents)).convert("RGB")

        image = ImageOps.fit(
            image,
            (224, 224),
            Image.Resampling.LANCZOS,
        )

        image_array = np.asarray(image).astype(np.float32)

        image_array = (image_array / 127.5) - 1

        image_array = np.expand_dims(image_array, axis=0)

        prediction = image_model.predict(
            image_array,
            verbose=0,
        )[0]

        class_index = int(np.argmax(prediction))
        confidence = float(prediction[class_index])

        prediction_name = IMAGE_CLASSES[class_index]

        # Reject uncertain predictions
        if confidence < 0.70:
            return {
                "prediction": "Uncertain",
                "confidence": round(confidence, 4),
                "message": "Please provide a clear image of a cow."
            }

        # Reject images classified as Not a Cow
        if prediction_name == "NOT A COW":
            return {
                "prediction": "Invalid Image",
                "confidence": round(confidence, 4),
                "message": "Please upload an image of a cow."
            }

        return {
            "prediction": prediction_name,
            "confidence": round(confidence, 4),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Image prediction failed: {str(e)}",
        )
