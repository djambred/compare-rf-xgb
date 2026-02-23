import os
import pandas as pd
import requests
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="MCH Hate Speech Demo", layout="centered")
st.title("Deteksi Ujaran Kebencian")

with st.sidebar:
    st.subheader("Backend")
    st.write(BACKEND_URL)

if st.button("Cek Health Backend"):
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=10)
        response.raise_for_status()
        st.success(f"Backend sehat: {response.json()}")
    except Exception as err:
        st.error(f"Gagal koneksi backend: {err}")

st.divider()
st.subheader("Training Model")

tune = st.checkbox("Aktifkan tuning (GridSearchCV sederhana)", value=False)
fast_mode = st.checkbox("Mode cepat (sampling & parameter ringan)", value=True)

col_rf, col_xgb = st.columns(2)
with col_rf:
    if st.button("Train Random Forest"):
        try:
            response = requests.post(
                f"{BACKEND_URL}/train/rf",
                json={"tune": tune, "fast": fast_mode},
                timeout=1800,
            )
            response.raise_for_status()
            train_result = response.json()
            st.success("Training Random Forest selesai")
            st.write(f"Best model saat ini: **{train_result['best_algorithm']}**")
        except Exception as err:
            st.error(f"Training RF gagal: {err}")

with col_xgb:
    if st.button("Train XGBoost"):
        try:
            response = requests.post(
                f"{BACKEND_URL}/train/xgboost",
                json={"tune": tune, "fast": fast_mode},
                timeout=1800,
            )
            response.raise_for_status()
            train_result = response.json()
            st.success("Training XGBoost selesai")
            st.write(f"Best model saat ini: **{train_result['best_algorithm']}**")
        except Exception as err:
            st.error(f"Training XGBoost gagal: {err}")

if st.button("Lihat Metrics Terakhir"):
    try:
        response = requests.get(f"{BACKEND_URL}/metrics", timeout=20)
        response.raise_for_status()
        metrics_payload = response.json()
        st.write(f"Best algorithm: **{metrics_payload['best_algorithm']}**")

        rows = []
        for model_name, values in metrics_payload["metrics"].items():
            rows.append(
                {
                    "model": model_name,
                    "precision": values["precision"],
                    "recall": values["recall"],
                    "f1_score": values["f1_score"],
                    "roc_auc": values["roc_auc"],
                    "confusion_matrix": str(values["confusion_matrix"]),
                }
            )

        dataframe = pd.DataFrame(rows)
        st.dataframe(dataframe, use_container_width=True)
        st.bar_chart(dataframe.set_index("model")[["f1_score", "precision", "recall", "roc_auc"]])
    except Exception as err:
        st.error(f"Ambil metrics gagal: {err}")

st.divider()
st.subheader("Prediksi")

available_models: list[str] = []
try:
    response = requests.get(f"{BACKEND_URL}/models", timeout=10)
    response.raise_for_status()
    available_models = response.json().get("available", [])
except Exception:
    available_models = []

if not available_models:
    st.warning("Belum ada model yang terlatih. Jalankan training terlebih dahulu.")

prediction_model = st.selectbox(
    "Pilih model",
    options=available_models if available_models else ["best"],
    index=0,
    disabled=not bool(available_models),
)
text = st.text_area("Masukkan komentar", placeholder="Contoh: kamu bodoh")

if st.button("Prediksi", disabled=not bool(available_models)):
    if not text.strip():
        st.warning("Teks tidak boleh kosong.")
    else:
        try:
            response = requests.post(
                f"{BACKEND_URL}/predict",
                json={"text": text, "algorithm": prediction_model},
                timeout=20,
            )
            response.raise_for_status()
            result = response.json()
            st.success("Prediksi berhasil")
            st.write(f"Label: **{result['label']}**")
            st.write(f"Model: **{result['algorithm']}**")
            st.write(f"Confidence: **{result['confidence']:.2f}**")
        except Exception as err:
            st.error(f"Prediksi gagal: {err}")

st.divider()
st.subheader("Uji dengan URL YouTube")
scrape_model = st.selectbox(
    "Model untuk klasifikasi hasil scraping",
    options=available_models if available_models else ["best"],
    index=0,
    disabled=not bool(available_models),
)

youtube_url = st.text_input(
    "URL YouTube",
    placeholder="https://www.youtube.com/watch?v=...",
)

scrape_mode = st.selectbox(
    "Mode scraping",
    options=["auto", "live_chat", "comments"],
    index=0,
)

max_items = st.number_input("Jumlah komentar", min_value=10, max_value=500, value=100, step=10)
predict_after_scrape = st.checkbox("Langsung klasifikasi hasil scraping", value=True)

if st.button("Scrape dari URL", disabled=not bool(available_models) and predict_after_scrape):
    if not youtube_url.strip():
        st.warning("URL YouTube wajib diisi.")
    else:
        try:
            response = requests.post(
                f"{BACKEND_URL}/scrape",
                json={
                    "url": youtube_url,
                    "mode": scrape_mode,
                    "max_items": int(max_items),
                    "predict": predict_after_scrape,
                    "algorithm": scrape_model,
                },
                timeout=240,
            )
            response.raise_for_status()
            payload = response.json()

            st.success(f"Scraping selesai dari sumber: {payload['source']}")
            st.write(f"Total komentar: **{payload['summary']['total_comments']}**")
            st.write(f"Hate speech: **{payload['summary']['hate_speech_count']}**")
            st.write(f"Non hate speech: **{payload['summary']['non_hate_speech_count']}**")

            if payload.get("prediction_error"):
                st.warning(f"Prediksi tidak dijalankan: {payload['prediction_error']}")

            result_df = pd.DataFrame(payload["items"])
            if not result_df.empty:
                st.dataframe(result_df, use_container_width=True)
            else:
                st.info("Tidak ada komentar yang berhasil diambil dari URL ini.")
        except Exception as err:
            st.error(f"Scraping gagal: {err}")
