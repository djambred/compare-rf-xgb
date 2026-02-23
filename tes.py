"""
=============================================================
Script Training → Simpan Model ke Format .h5
Berdasarkan pipeline: ml_pipeline.py (Michael)

Output folder model/:
  model/random_forest_model.h5
  model/xgboost_model.h5
  model/tfidf_vectorizer.h5
  model/best_model.h5
  model/metrics.json

Cara pakai:
  python train_to_h5.py                    # train RF + XGBoost, simpan .h5
  python train_to_h5.py --model rf         # train RF saja
  python train_to_h5.py --model xgb        # train XGBoost saja
  python train_to_h5.py --tune             # dengan GridSearchCV
  python train_to_h5.py --fast             # dataset max 20000 baris
=============================================================
"""

import argparse
import io
import json
import pickle
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split

warnings.filterwarnings("ignore")

# ── Coba import opsional ──────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    XGBOOST_OK = True
except ImportError:
    XGBOOST_OK = False
    print("⚠  XGBoost tidak terinstall. Jalankan: pip install xgboost")

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    SASTRAWI_OK = True
except ImportError:
    SASTRAWI_OK = False
    print("⚠  PySastrawi tidak terinstall. Jalankan: pip install PySastrawi")
    print("   Stemming & stopword Sastrawi dilewati.\n")

# =============================================================================
# KONFIGURASI PATH
# =============================================================================
DATASET_DIR       = Path("dataset")
MODEL_DIR         = Path("model")
METRICS_PATH      = MODEL_DIR / "metrics.json"

# ── Path file .h5 ─────────────────────────────────────────────────────────────
BEST_MODEL_PATH   = MODEL_DIR / "best_model.h5"
VECTORIZER_PATH   = MODEL_DIR / "tfidf_vectorizer.h5"
RF_MODEL_PATH     = MODEL_DIR / "random_forest_model.h5"
XGB_MODEL_PATH    = MODEL_DIR / "xgboost_model.h5"

RANDOM_STATE = 42

# =============================================================================
# KONFIGURASI PREPROCESSING
# =============================================================================
SLANG_MAP = {
    "gk": "tidak", "ga": "tidak", "nggak": "tidak", "ngga": "tidak",
    "yg": "yang", "dr": "dari", "dgn": "dengan", "bgt": "banget",
    "aja": "saja", "lu": "kamu", "loe": "kamu", "gw": "saya", "gue": "saya",
    "wkwk": "", "wkwkwk": "", "bg": "",
    # Tambahan normalisasi leetspeak kata kasar
    "b4bi": "babi", "b4b1": "babi", "anj1ng": "anjing",
    "4njing": "anjing", "s1al": "sial", "g0blok": "goblok",
    "t0lol": "tolol", "brengs3k": "brengsek",
}

CUSTOM_STOPWORDS = {
    "wkwk", "wkwkwk", "wkwkwkwk", "bg", "bro", "bang",
    "kak", "nih", "dong", "sih",
}

TEXT_CANDIDATES  = ["text", "comment", "komentar", "tweet", "content", "sentence"]
LABEL_CANDIDATES = ["label", "class", "target", "hate_speech", "is_hate", "labels"]


# =============================================================================
# DATACLASS
# =============================================================================
@dataclass
class TrainedArtifacts:
    vectorizer: TfidfVectorizer
    model: Any
    algorithm: str


# =============================================================================
# SIMPAN & LOAD .h5
# =============================================================================
def save_h5(obj: Any, path: Path) -> None:
    """
    Simpan objek Python (model sklearn/xgboost, vectorizer, dll)
    ke file berekstensi .h5 menggunakan joblib.

    Mengapa joblib? joblib adalah serializer standar scikit-learn,
    lebih efisien dari pickle untuk array numpy besar (bobot model RF).
    Ekstensi .h5 valid karena hanyalah label — format internalnya adalah
    joblib compressed binary, bukan HDF5 Keras, karena RF & XGBoost
    adalah model klasik (bukan neural network).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path, compress=3)
    size_kb = path.stat().st_size / 1024
    print(f"   💾 Disimpan → {path}  ({size_kb:.1f} KB)")


def load_h5(path: Path) -> Any:
    """Load objek dari file .h5 (joblib format)"""
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    return joblib.load(path)


# =============================================================================
# PREPROCESSING
# =============================================================================
def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def _normalize_slang(text: str) -> str:
    tokens = text.split()
    normalized = [SLANG_MAP.get(token, token) for token in tokens]
    return " ".join(t for t in normalized if t)

def _clean_text(text: str) -> str:
    lowered   = text.lower()
    no_url    = re.sub(r"https?://\S+|www\.\S+", " ", lowered)
    no_mention= re.sub(r"@[A-Za-z0-9_]+", " ", no_url)
    no_digits = re.sub(r"\d+", " ", no_mention)
    no_punct  = re.sub(r"[^\w\s]", " ", no_digits)
    no_emoji  = re.sub(r"[\U00010000-\U0010ffff]", " ", no_punct)
    return _normalize_spaces(no_emoji)

def preprocess_series(texts: pd.Series) -> pd.Series:
    """Pipeline preprocessing lengkap: clean → slang norm → stopword → stem"""
    # Stopwords
    if SASTRAWI_OK:
        stopwords = set(StopWordRemoverFactory().get_stop_words()) | CUSTOM_STOPWORDS
        stemmer   = StemmerFactory().create_stemmer()
    else:
        stopwords = CUSTOM_STOPWORDS | {
            "yang","di","ke","dari","dan","atau","ini","itu","dengan",
            "untuk","pada","adalah","dalam","akan","juga","sudah","saya",
            "kamu","ada","tidak","ya","si","lebih","serta","karena",
            "jadi","oleh","bisa","harus","sangat","saja",
        }
        stemmer = None

    def _preprocess(text: Any) -> str:
        raw      = "" if pd.isna(text) else str(text)
        cleaned  = _clean_text(raw)
        normed   = _normalize_slang(cleaned)
        tokens   = [t for t in normed.split() if t and t not in stopwords]
        if stemmer:
            tokens = [stemmer.stem(t) for t in tokens]
        return " ".join(tokens)

    return texts.apply(_preprocess)


# =============================================================================
# LOAD DATASET
# =============================================================================
def _find_column(columns: list, candidates: list) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    for col in columns:
        if any(cand in col.lower() for cand in candidates):
            return col
    return None

def _to_binary_label(labels: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(labels):
        return (pd.to_numeric(labels, errors="coerce").fillna(0) > 0).astype(int)
    positive_tokens = {"1","true","yes","hate","hate_speech","hs","toxic"}
    return labels.apply(lambda v: 1 if str(v).strip().lower() in positive_tokens else 0)

def load_dataset(dataset_dir: Path = DATASET_DIR) -> tuple[pd.Series, pd.Series]:
    """Baca semua file CSV/Parquet/JSON/JSONL dari folder dataset/"""
    all_files = (
        sorted(dataset_dir.glob("*.csv"))   +
        sorted(dataset_dir.glob("*.parquet")) +
        sorted(dataset_dir.glob("*.json"))  +
        sorted(dataset_dir.glob("*.jsonl"))
    )
    if not all_files:
        raise FileNotFoundError(
            f"Tidak ada file dataset di '{dataset_dir}/'. "
            "Taruh file CSV dataset di folder dataset/"
        )

    frames = []
    for p in all_files:
        suffix = p.suffix.lower()
        if suffix == ".csv":      frames.append(pd.read_csv(p))
        elif suffix == ".parquet":frames.append(pd.read_parquet(p))
        elif suffix == ".json":   frames.append(pd.read_json(p))
        elif suffix == ".jsonl":  frames.append(pd.read_json(p, lines=True))
        print(f"   📂 Loaded: {p.name} ({len(frames[-1])} baris)")

    df = pd.concat(frames, ignore_index=True)
    print(f"   Total: {len(df)} baris setelah merge\n")

    text_col  = _find_column(list(df.columns), TEXT_CANDIDATES)
    label_col = _find_column(list(df.columns), LABEL_CANDIDATES)

    if text_col is None or label_col is None:
        raise ValueError(
            f"Kolom teks/label tidak ditemukan. Kolom tersedia: {list(df.columns)}\n"
            f"Pastikan ada kolom: {TEXT_CANDIDATES} dan {LABEL_CANDIDATES}"
        )

    print(f"   ✅ Kolom teks  : '{text_col}'")
    print(f"   ✅ Kolom label : '{label_col}'")

    texts  = df[text_col].fillna("").astype(str)
    labels = _to_binary_label(df[label_col].fillna(0))
    print(f"   Distribusi label: {labels.value_counts().to_dict()}\n")
    return texts, labels


# =============================================================================
# EVALUASI
# =============================================================================
def evaluate_model(model, x_valid, y_valid) -> dict:
    preds = model.predict(x_valid)
    probas = model.predict_proba(x_valid)[:, 1] if hasattr(model, "predict_proba") else preds
    try:
        roc_auc = float(roc_auc_score(y_valid, probas))
    except ValueError:
        roc_auc = 0.0

    return {
        "precision"       : round(float(precision_score(y_valid, preds, zero_division=0)), 4),
        "recall"          : round(float(recall_score(y_valid, preds, zero_division=0)), 4),
        "f1_score"        : round(float(f1_score(y_valid, preds, zero_division=0)), 4),
        "roc_auc"         : round(roc_auc, 4),
        "confusion_matrix": confusion_matrix(y_valid, preds).tolist(),
    }


# =============================================================================
# BUILD MODEL
# =============================================================================
def _build_models(scale_pos_weight: float, fast: bool) -> dict:
    rf = RandomForestClassifier(
        n_estimators  = 60 if fast else 100,
        max_depth     = 18 if fast else None,
        class_weight  = "balanced",
        random_state  = RANDOM_STATE,
        n_jobs        = -1,
    )

    models = {"random_forest": rf}

    if XGBOOST_OK:
        xgb = XGBClassifier(
            learning_rate   = 0.1,
            max_depth       = 5 if fast else 6,
            n_estimators    = 120 if fast else 200,
            subsample       = 0.8 if fast else 0.9,
            colsample_bytree= 0.8 if fast else 0.9,
            objective       = "binary:logistic",
            eval_metric     = "logloss",
            random_state    = RANDOM_STATE,
            scale_pos_weight= scale_pos_weight,
            verbosity       = 0,
        )
        models["xgboost"] = xgb

    return models

def _tune_if_needed(model_name: str, model: Any, tune: bool):
    if not tune:
        return model
    grids = {
        "random_forest": {
            "n_estimators"   : [100, 200],
            "max_depth"      : [None, 15],
            "min_samples_split": [2, 5],
        },
        "xgboost": {
            "n_estimators": [100, 200],
            "max_depth"   : [4, 6],
            "learning_rate": [0.05, 0.1],
        },
    }
    print(f"   🔍 GridSearchCV aktif untuk {model_name}...")
    return GridSearchCV(model, grids[model_name], scoring="f1", cv=3, n_jobs=-1)


# =============================================================================
# TRAINING UTAMA
# =============================================================================
def _prepare_data(max_rows: int | None = None):
    print("=" * 60)
    print("STEP 1: LOAD DATASET")
    print("=" * 60)
    texts, labels = load_dataset()

    if labels.nunique() < 2:
        raise ValueError("Dataset harus memiliki minimal 2 kelas (0 dan 1).")

    print("=" * 60)
    print("STEP 2: PREPROCESSING")
    print("=" * 60)
    print("Menjalankan: cleaning → slang norm → stopword → stemming ...")
    processed = preprocess_series(texts)
    print("✅ Preprocessing selesai\n")

    if max_rows and len(processed) > max_rows:
        sample_df = pd.DataFrame({"text": processed, "label": labels})
        sample_df = sample_df.sample(n=max_rows, random_state=RANDOM_STATE)
        processed = sample_df["text"].reset_index(drop=True)
        labels    = sample_df["label"].reset_index(drop=True)
        print(f"⚡ Fast mode: dataset dipotong menjadi {max_rows} baris\n")

    print("=" * 60)
    print("STEP 3: TF-IDF VECTORIZATION")
    print("=" * 60)
    x_train_txt, x_valid_txt, y_train, y_valid = train_test_split(
        processed, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )
    print(f"Data latih : {len(x_train_txt)} komentar")
    print(f"Data uji   : {len(x_valid_txt)} komentar")

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), norm="l2")
    x_train = vectorizer.fit_transform(x_train_txt)
    x_valid = vectorizer.transform(x_valid_txt)
    print(f"Dimensi fitur TF-IDF: {x_train.shape[1]} fitur\n")

    return processed, labels, x_train, x_valid, y_train, y_valid, vectorizer


def _load_metrics_store() -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {"best_algorithm": None, "dataset_size": 0, "train_size": 0,
            "validation_size": 0, "metrics": {}, "artifacts": {}}

def _save_metrics(store: dict) -> None:
    METRICS_PATH.write_text(json.dumps(store, indent=2), encoding="utf-8")


def train_single(model_name: str, tune: bool = False, fast: bool = False) -> dict:
    if model_name not in {"random_forest", "xgboost"}:
        raise ValueError("model_name harus 'random_forest' atau 'xgboost'")
    if model_name == "xgboost" and not XGBOOST_OK:
        raise ImportError("XGBoost tidak terinstall. Jalankan: pip install xgboost")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    processed, labels, x_train, x_valid, y_train, y_valid, vectorizer = \
        _prepare_data(max_rows=20000 if fast else None)

    pos_count = int((y_train == 1).sum())
    neg_count = int((y_train == 0).sum())
    scale_pos_weight = (neg_count / max(pos_count, 1))

    print("=" * 60)
    print(f"STEP 4: TRAINING — {model_name.upper().replace('_', ' ')}")
    print("=" * 60)
    print(f"   Distribusi train: Non-Hate={neg_count}, Hate={pos_count}")
    print(f"   scale_pos_weight = {scale_pos_weight:.2f}")

    models    = _build_models(scale_pos_weight, fast)
    estimator = _tune_if_needed(model_name, models[model_name], tune)
    estimator.fit(x_train, y_train)

    trained = estimator.best_estimator_ if hasattr(estimator, "best_estimator_") else estimator
    metrics = evaluate_model(trained, x_valid, y_valid)

    print("\n" + "=" * 60)
    print("STEP 5: EVALUASI")
    print("=" * 60)
    print(f"   Precision       : {metrics['precision']:.4f}")
    print(f"   Recall          : {metrics['recall']:.4f}")
    print(f"   F1-Score        : {metrics['f1_score']:.4f}  ← metrik utama")
    print(f"   ROC-AUC         : {metrics['roc_auc']:.4f}")
    cm = metrics["confusion_matrix"]
    print(f"   Confusion Matrix:")
    print(f"     TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"     FN={cm[1][0]}  TP={cm[1][1]}")
    print(f"     → Hate terdeteksi (TP) : {cm[1][1]}")
    print(f"     → Hate lolos (FN)      : {cm[1][0]}")

    print("\n" + "=" * 60)
    print("STEP 6: SIMPAN MODEL KE .h5")
    print("=" * 60)

    # Simpan vectorizer
    save_h5(vectorizer, VECTORIZER_PATH)

    # Simpan model individual
    model_path = RF_MODEL_PATH if model_name == "random_forest" else XGB_MODEL_PATH
    save_h5(trained, model_path)

    # Update metrics store
    store = _load_metrics_store()
    store.update({
        "dataset_size"   : int(len(processed)),
        "train_size"     : int(x_train.shape[0]),
        "validation_size": int(x_valid.shape[0]),
    })
    store.setdefault("metrics", {})[model_name]   = metrics
    store.setdefault("artifacts", {})["vectorizer"] = str(VECTORIZER_PATH)
    store["artifacts"][model_name] = str(model_path)

    # Update best model
    best_algo    = store.get("best_algorithm")
    current_best = store["metrics"].get(best_algo, {}).get("f1_score", -1.0) if best_algo else -1.0

    if metrics["f1_score"] >= current_best:
        best_algo = model_name
        artifacts = {"vectorizer": vectorizer, "model": trained, "algorithm": best_algo}
        save_h5(artifacts, BEST_MODEL_PATH)
        store["artifacts"]["best_model"] = str(BEST_MODEL_PATH)

    store["best_algorithm"] = best_algo
    _save_metrics(store)

    print(f"\n✅ Semua file tersimpan di folder: {MODEL_DIR}/")
    print(f"🏆 Model terbaik saat ini: {store['best_algorithm']} "
          f"(F1={store['metrics'][store['best_algorithm']]['f1_score']:.4f})")

    return {**store, "metrics": store["metrics"]}


def train_and_compare(tune: bool = False, fast: bool = False) -> dict:
    """Train RF dan XGBoost, komparasi, simpan semua ke .h5"""
    print("\n🚀 TRAINING RANDOM FOREST + XGBOOST (KOMPARASI)\n")
    train_single("random_forest", tune=tune, fast=fast)
    print()
    result = train_single("xgboost", tune=tune, fast=fast)

    print("\n" + "=" * 60)
    print("📊 TABEL KOMPARASI AKHIR")
    print("=" * 60)
    metrics = result.get("metrics", {})
    print(f"{'Model':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'ROC-AUC':>10}")
    print("-" * 60)
    for name, m in metrics.items():
        print(f"{name:<20} {m['precision']:>10.4f} {m['recall']:>10.4f} "
              f"{m['f1_score']:>10.4f} {m['roc_auc']:>10.4f}")
    print(f"\n🏆 Model terbaik (F1-Score): {result['best_algorithm'].upper()}")
    return result


# =============================================================================
# LOAD & PREDIKSI
# =============================================================================
def load_best_model() -> TrainedArtifacts:
    d = load_h5(BEST_MODEL_PATH)
    if isinstance(d, dict):
        return TrainedArtifacts(vectorizer=d["vectorizer"], model=d["model"], algorithm=d["algorithm"])
    return d  # backward compat jika sudah TrainedArtifacts

def load_model_by_name(model_name: str) -> TrainedArtifacts:
    vectorizer = load_h5(VECTORIZER_PATH)
    path = RF_MODEL_PATH if model_name == "random_forest" else XGB_MODEL_PATH
    model = load_h5(path)
    return TrainedArtifacts(vectorizer=vectorizer, model=model, algorithm=model_name)

def predict_text(text: str, algorithm: str = "best") -> dict:
    """Prediksi satu komentar. algorithm: 'best', 'random_forest', 'xgboost'"""
    artifacts = load_best_model() if algorithm == "best" else load_model_by_name(algorithm)
    processed = preprocess_series(pd.Series([text]))
    features  = artifacts.vectorizer.transform(processed)
    pred      = int(artifacts.model.predict(features)[0])
    confidence= float(artifacts.model.predict_proba(features)[0][pred]) \
                if hasattr(artifacts.model, "predict_proba") else 1.0
    return {
        "label"     : "hate_speech" if pred == 1 else "non_hate_speech",
        "confidence": round(confidence, 4),
        "algorithm" : artifacts.algorithm,
    }

def get_saved_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))

def get_available_models() -> list[str]:
    available = []
    if VECTORIZER_PATH.exists() and RF_MODEL_PATH.exists():
        available.append("random_forest")
    if VECTORIZER_PATH.exists() and XGB_MODEL_PATH.exists():
        available.append("xgboost")
    if BEST_MODEL_PATH.exists():
        available.insert(0, "best")
    return available


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RF & XGBoost → simpan ke model/*.h5"
    )
    parser.add_argument(
        "--model", choices=["rf", "xgb", "both"], default="both",
        help="Model yang ditraining: rf | xgb | both (default: both)"
    )
    parser.add_argument("--tune", action="store_true", help="Aktifkan GridSearchCV")
    parser.add_argument("--fast", action="store_true", help="Mode cepat (max 20.000 data)")
    args = parser.parse_args()

    if args.model == "both":
        train_and_compare(tune=args.tune, fast=args.fast)
    elif args.model == "rf":
        train_single("random_forest", tune=args.tune, fast=args.fast)
    elif args.model == "xgb":
        train_single("xgboost", tune=args.tune, fast=args.fast)
