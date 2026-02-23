import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier

DATASET_DIR = Path("dataset")
MODEL_DIR = Path("model")
METRICS_PATH = MODEL_DIR / "metrics.json"
BEST_MODEL_PATH = MODEL_DIR / "best_model.joblib"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
RF_MODEL_PATH = MODEL_DIR / "random_forest_model.joblib"
XGB_MODEL_PATH = MODEL_DIR / "xgboost_model.joblib"

FALLBACK_EXTENSION = ".h5"

RANDOM_STATE = 42

SLANG_MAP = {
    "gk": "tidak",
    "ga": "tidak",
    "nggak": "tidak",
    "ngga": "tidak",
    "yg": "yang",
    "dr": "dari",
    "dgn": "dengan",
    "bgt": "banget",
    "aja": "saja",
    "lu": "kamu",
    "loe": "kamu",
    "gw": "saya",
    "gue": "saya",
    "wkwk": "",
    "wkwkwk": "",
    "bg": "",
}

CUSTOM_STOPWORDS = {
    "wkwk",
    "wkwkwk",
    "wkwkwkwk",
    "bg",
    "bro",
    "bang",
    "kak",
    "nih",
    "dong",
    "sih",
}

TEXT_CANDIDATES = [
    "text",
    "comment",
    "komentar",
    "tweet",
    "content",
    "sentence",
]

LABEL_CANDIDATES = [
    "label",
    "class",
    "target",
    "hate_speech",
    "is_hate",
]


@dataclass
class TrainedArtifacts:
    vectorizer: TfidfVectorizer
    model: Any
    algorithm: str


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_slang(text: str) -> str:
    tokens = text.split()
    normalized = [SLANG_MAP.get(token, token) for token in tokens]
    return " ".join(token for token in normalized if token)


def _clean_text(text: str) -> str:
    lowered = text.lower()
    no_url = re.sub(r"https?://\S+|www\.\S+", " ", lowered)
    no_mention = re.sub(r"@[A-Za-z0-9_]+", " ", no_url)
    no_digits = re.sub(r"\d+", " ", no_mention)
    no_punct = re.sub(r"[^\w\s]", " ", no_digits)
    no_emoji = re.sub(r"[\U00010000-\U0010ffff]", " ", no_punct)
    return _normalize_spaces(no_emoji)


def preprocess_series(texts: pd.Series) -> pd.Series:
    stemmer = StemmerFactory().create_stemmer()
    stopword_factory = StopWordRemoverFactory()
    stopwords = set(stopword_factory.get_stop_words()) | CUSTOM_STOPWORDS

    def _preprocess(text: Any) -> str:
        raw = "" if pd.isna(text) else str(text)
        cleaned = _clean_text(raw)
        normalized = _normalize_slang(cleaned)
        tokens = [token for token in normalized.split() if token and token not in stopwords]
        stemmed = [stemmer.stem(token) for token in tokens]
        return " ".join(stemmed)

    return texts.apply(_preprocess)


def _find_column(columns: list[str], candidates: list[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]

    for column in columns:
        col_lower = column.lower()
        if any(candidate in col_lower for candidate in candidates):
            return column

    return None


def _to_binary_label(labels: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(labels):
        numeric = pd.to_numeric(labels, errors="coerce").fillna(0)
        return (numeric > 0).astype(int)

    positive_tokens = {"1", "true", "yes", "hate", "hate_speech", "hs", "toxic"}

    def _map(value: Any) -> int:
        token = str(value).strip().lower()
        return 1 if token in positive_tokens else 0

    return labels.apply(_map)


def load_dataset_csvs(dataset_dir: Path = DATASET_DIR) -> tuple[pd.Series, pd.Series]:
    csv_files = sorted(dataset_dir.glob("*.csv"))
    parquet_files = sorted(dataset_dir.glob("*.parquet"))
    json_files = sorted(dataset_dir.glob("*.json"))
    jsonl_files = sorted(dataset_dir.glob("*.jsonl"))

    all_files = csv_files + parquet_files + json_files + jsonl_files
    if not all_files:
        raise FileNotFoundError("No dataset files found in dataset/ directory.")

    frames: list[pd.DataFrame] = []
    for path in all_files:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            frames.append(pd.read_csv(path))
        elif suffix == ".parquet":
            frames.append(pd.read_parquet(path))
        elif suffix == ".json":
            frames.append(pd.read_json(path))
        elif suffix == ".jsonl":
            frames.append(pd.read_json(path, lines=True))

    dataframe = pd.concat(frames, ignore_index=True)

    text_col = _find_column(list(dataframe.columns), TEXT_CANDIDATES)
    label_col = _find_column(list(dataframe.columns), LABEL_CANDIDATES)

    if text_col is None or label_col is None:
        raise ValueError(
            "Could not infer text/label columns. Please ensure dataset has text/comment and label/class columns."
        )

    text_series = dataframe[text_col].fillna("").astype(str)
    label_series = _to_binary_label(dataframe[label_col].fillna(0))

    return text_series, label_series


def evaluate_model(model: Any, x_valid, y_valid) -> dict[str, Any]:
    predictions = model.predict(x_valid)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_valid)[:, 1]
    else:
        probabilities = predictions

    try:
        roc_auc = float(roc_auc_score(y_valid, probabilities))
    except ValueError:
        roc_auc = 0.0

    return {
        "precision": float(precision_score(y_valid, predictions, zero_division=0)),
        "recall": float(recall_score(y_valid, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_valid, predictions, zero_division=0)),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_valid, predictions).tolist(),
    }


def _build_models(scale_pos_weight: float, fast: bool) -> dict[str, Any]:
    rf_estimators = 60 if fast else 100
    rf_max_depth = 18 if fast else None

    rf = RandomForestClassifier(
        n_estimators=rf_estimators,
        max_depth=rf_max_depth,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    xgb_estimators = 120 if fast else 200
    xgb_max_depth = 5 if fast else 6
    xgb_subsample = 0.8 if fast else 0.9
    xgb_colsample = 0.8 if fast else 0.9

    xgb = XGBClassifier(
        learning_rate=0.1,
        max_depth=xgb_max_depth,
        n_estimators=xgb_estimators,
        subsample=xgb_subsample,
        colsample_bytree=xgb_colsample,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,
    )

    return {"random_forest": rf, "xgboost": xgb}


def _tune_if_needed(model_name: str, model: Any, tune: bool):
    if not tune:
        return model

    if model_name == "random_forest":
        grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 15],
            "min_samples_split": [2, 5],
        }
    else:
        grid = {
            "n_estimators": [100, 200],
            "max_depth": [4, 6],
            "learning_rate": [0.05, 0.1],
        }

    search = GridSearchCV(model, grid, scoring="f1", cv=3, n_jobs=-1)
    return search


def _prepare_training_data(max_rows: int | None = None) -> tuple[
    pd.Series,
    pd.Series,
    Any,
    Any,
    Any,
    Any,
    TfidfVectorizer,
]:
    texts, labels = load_dataset_csvs()
    if labels.nunique() < 2:
        raise ValueError("Dataset must contain at least two classes for training.")

    processed_texts = preprocess_series(texts)

    if max_rows is not None and max_rows > 0 and len(processed_texts) > max_rows:
        sampled = pd.DataFrame({"text": processed_texts, "label": labels})
        sampled = sampled.sample(n=max_rows, random_state=RANDOM_STATE, replace=False)
        processed_texts = sampled["text"].reset_index(drop=True)
        labels = sampled["label"].reset_index(drop=True)

    x_train_text, x_valid_text, y_train, y_valid = train_test_split(
        processed_texts,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), norm="l2")
    x_train = vectorizer.fit_transform(x_train_text)
    x_valid = vectorizer.transform(x_valid_text)

    return processed_texts, labels, x_train, x_valid, y_train, y_valid, vectorizer


def _load_metrics_store() -> dict[str, Any]:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    return {
        "best_algorithm": None,
        "dataset_size": 0,
        "train_size": 0,
        "validation_size": 0,
        "metrics": {},
        "artifacts": {},
    }


def _persist_metrics(store: dict[str, Any]) -> None:
    METRICS_PATH.write_text(json.dumps(store, indent=2), encoding="utf-8")


def _resolve_artifact_path(primary_path: Path) -> Path | None:
    if primary_path.exists():
        return primary_path

    fallback_path = primary_path.with_suffix(FALLBACK_EXTENSION)
    if fallback_path.exists():
        return fallback_path

    return None


def train_single(model_name: str, tune: bool = False, fast: bool = False) -> dict[str, Any]:
    if model_name not in {"random_forest", "xgboost"}:
        raise ValueError("Model name must be random_forest or xgboost.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    max_rows = 20000 if fast else None
    processed_texts, labels, x_train, x_valid, y_train, y_valid, vectorizer = _prepare_training_data(
        max_rows=max_rows
    )

    positive_count = int(np.sum(y_train == 1))
    negative_count = int(np.sum(y_train == 0))
    scale_pos_weight = (negative_count / max(positive_count, 1)) if positive_count > 0 else 1.0

    models = _build_models(scale_pos_weight=scale_pos_weight, fast=fast)
    estimator = _tune_if_needed(model_name, models[model_name], tune=tune)
    estimator.fit(x_train, y_train)

    trained_model = estimator.best_estimator_ if hasattr(estimator, "best_estimator_") else estimator
    metrics = evaluate_model(trained_model, x_valid, y_valid)

    joblib.dump(vectorizer, VECTORIZER_PATH)
    if model_name == "random_forest":
        joblib.dump(trained_model, RF_MODEL_PATH)
    else:
        joblib.dump(trained_model, XGB_MODEL_PATH)

    store = _load_metrics_store()
    store["dataset_size"] = int(len(processed_texts))
    store["train_size"] = int(x_train.shape[0])
    store["validation_size"] = int(x_valid.shape[0])
    store.setdefault("metrics", {})[model_name] = metrics
    store.setdefault("artifacts", {})["vectorizer"] = str(VECTORIZER_PATH)
    store["artifacts"][model_name] = str(RF_MODEL_PATH if model_name == "random_forest" else XGB_MODEL_PATH)

    best_algorithm = store.get("best_algorithm")
    if best_algorithm and best_algorithm in store["metrics"]:
        current_best = store["metrics"][best_algorithm]["f1_score"]
    else:
        current_best = -1.0

    if metrics["f1_score"] >= current_best:
        best_algorithm = model_name
        best_artifacts = TrainedArtifacts(
            vectorizer=vectorizer,
            model=trained_model,
            algorithm=best_algorithm,
        )
        joblib.dump(best_artifacts, BEST_MODEL_PATH)
        store["artifacts"]["best_model"] = str(BEST_MODEL_PATH)

    store["best_algorithm"] = best_algorithm
    _persist_metrics(store)

    return {
        "best_algorithm": best_algorithm,
        "dataset_size": store["dataset_size"],
        "train_size": store["train_size"],
        "validation_size": store["validation_size"],
        "metrics": store["metrics"],
        "artifacts": store["artifacts"],
    }


def train_and_compare(tune: bool = False, fast: bool = False) -> dict[str, Any]:
    train_single("random_forest", tune=tune, fast=fast)
    return train_single("xgboost", tune=tune, fast=fast)


def load_best_model() -> TrainedArtifacts:
    model_path = _resolve_artifact_path(BEST_MODEL_PATH)
    if model_path is None:
        raise FileNotFoundError("Best model not found. Train the model first.")
    return joblib.load(model_path)


def load_model_by_name(model_name: str) -> TrainedArtifacts:
    vectorizer_path = _resolve_artifact_path(VECTORIZER_PATH)
    if vectorizer_path is None:
        raise FileNotFoundError("Vectorizer not found. Train the model first.")

    vectorizer = joblib.load(vectorizer_path)

    if model_name == "random_forest":
        model_path = _resolve_artifact_path(RF_MODEL_PATH)
    elif model_name == "xgboost":
        model_path = _resolve_artifact_path(XGB_MODEL_PATH)
    else:
        raise ValueError("Model name must be random_forest or xgboost.")

    if model_path is None:
        raise FileNotFoundError(f"Model {model_name} not found. Train the model first.")

    model = joblib.load(model_path)
    return TrainedArtifacts(vectorizer=vectorizer, model=model, algorithm=model_name)


def predict_text(text: str, algorithm: str = "best") -> dict[str, Any]:
    if algorithm == "best":
        artifacts = load_best_model()
    else:
        artifacts = load_model_by_name(algorithm)
    processed = preprocess_series(pd.Series([text]))
    features = artifacts.vectorizer.transform(processed)

    prediction = int(artifacts.model.predict(features)[0])
    if hasattr(artifacts.model, "predict_proba"):
        confidence = float(artifacts.model.predict_proba(features)[0][prediction])
    else:
        confidence = 1.0

    label = "hate_speech" if prediction == 1 else "non_hate_speech"
    return {
        "label": label,
        "confidence": confidence,
        "algorithm": artifacts.algorithm,
    }


def get_saved_metrics() -> dict[str, Any]:
    if not METRICS_PATH.exists():
        raise FileNotFoundError("Metrics not found. Train the model first.")

    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def get_available_models() -> list[str]:
    available: list[str] = []

    vectorizer_path = _resolve_artifact_path(VECTORIZER_PATH)

    if vectorizer_path and _resolve_artifact_path(RF_MODEL_PATH):
        available.append("random_forest")

    if vectorizer_path and _resolve_artifact_path(XGB_MODEL_PATH):
        available.append("xgboost")

    if _resolve_artifact_path(BEST_MODEL_PATH):
        available.insert(0, "best")

    return available
