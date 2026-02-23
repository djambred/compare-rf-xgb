import os
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Configure page
st.set_page_config(
    page_title="Deteksi Ujaran Kebencian",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com",
        "Report a bug": None,
        "About": "Komparasi RF vs XGBoost untuk deteksi ujaran kebencian bahasa Indonesia"
    }
)

# Custom CSS untuk better styling
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #0e1117;
    }
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .status-ok {
        color: #09ab3b;
        font-weight: bold;
    }
    .status-error {
        color: #ff2b2b;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ===== SIDEBAR NAVIGATION =====
st.sidebar.markdown("# 📊 MCH Detection")
st.sidebar.markdown("---")

# Health check di sidebar
st.sidebar.subheader("🔗 Backend Status")
health_col = st.sidebar.columns([1, 1])
try:
    response = requests.get(f"{BACKEND_URL}/health", timeout=5)
    response.raise_for_status()
    with health_col[0]:
        st.sidebar.markdown('<p class="status-ok">✓ Connected</p>', unsafe_allow_html=True)
    with health_col[1]:
        st.sidebar.write(BACKEND_URL.replace("http://", "").split(":")[0])
except Exception:
    st.sidebar.markdown('<p class="status-error">✗ Disconnected</p>', unsafe_allow_html=True)

st.sidebar.markdown("---")

# Navigation menu
menu = st.sidebar.radio(
    "📌 **Navigasi Menu**",
    ["🏠 Home", "🎯 Training", "📈 Metrics", "🔮 Prediksi", "🎬 YouTube"],
    index=0
)

# Get available models
available_models: list[str] = []
try:
    response = requests.get(f"{BACKEND_URL}/models", timeout=10)
    response.raise_for_status()
    available_models = response.json().get("available", [])
except Exception:
    pass

# ===== HOME PAGE =====
if menu == "🏠 Home":
    st.markdown("# 🎯 Deteksi Ujaran Kebencian Indonesia")
    st.markdown("**Komparasi Random Forest vs XGBoost untuk Deteksi Ujaran Kebencian**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Status Backend", "Online" if available_models else "Offline")
    with col2:
        st.metric("Model Tersedia", len(available_models))
    with col3:
        st.metric("Version", "1.0.0")
    
    st.markdown("---")
    st.markdown("""
    ### 📋 Fitur Aplikasi
    
    ✨ **Training Model**
    - Latih Random Forest atau XGBoost secara terpisah
    - Opsi tuning dan fast mode untuk efisiensi
    
    📊 **Analisis Metrics**
    - Lihat performa model (Precision, Recall, F1, ROC-AUC)
    - Confusion matrix dalam bentuk heatmap
    - Perbandingan visual antar model
    
    🔮 **Prediksi Real-time**
    - Input teks dalam bahasa Indonesia
    - Deteksi ujaran kebencian dengan confidence score
    
    🎬 **Uji YouTube**
    - Scrape komentar dari video YouTube
    - Klasifikasi otomatis hasil scraping
    - Tampilkan statistik hate speech vs non-hate
    """)

# ===== TRAINING PAGE =====
elif menu == "🎯 Training":
    st.markdown("# 🎯 Training Model")
    
    st.markdown("### ⚙️ Konfigurasi Training")
    col1, col2 = st.columns(2)
    
    with col1:
        tune = st.checkbox("🔍 Aktifkan Tuning", value=False, help="GridSearchCV untuk hyperparameter")
    with col2:
        fast_mode = st.checkbox("⚡ Mode Cepat", value=True, help="Sampling & parameter ringan")
    
    st.markdown("---")
    st.markdown("### 🚀 Mulai Training")
    
    col_rf, col_xgb = st.columns(2)
    
    with col_rf:
        st.markdown("#### 🌳 Random Forest")
        if st.button("▶️ Train Random Forest", key="btn_rf", use_container_width=True):
            with st.spinner("🔄 Training Random Forest... ini butuh waktu"):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/train/rf",
                        json={"tune": tune, "fast": fast_mode},
                        timeout=1800,
                    )
                    response.raise_for_status()
                    train_result = response.json()
                    st.success("✅ Training RF selesai!")
                    st.info(f"Best model saat ini: **{train_result['best_algorithm']}**")
                    st.json(train_result)
                except Exception as err:
                    st.error(f"❌ Training RF gagal: {err}")
    
    with col_xgb:
        st.markdown("#### 🚀 XGBoost")
        if st.button("▶️ Train XGBoost", key="btn_xgb", use_container_width=True):
            with st.spinner("🔄 Training XGBoost... ini butuh waktu"):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/train/xgboost",
                        json={"tune": tune, "fast": fast_mode},
                        timeout=1800,
                    )
                    response.raise_for_status()
                    train_result = response.json()
                    st.success("✅ Training XGBoost selesai!")
                    st.info(f"Best model saat ini: **{train_result['best_algorithm']}**")
                    st.json(train_result)
                except Exception as err:
                    st.error(f"❌ Training XGBoost gagal: {err}")

# ===== METRICS PAGE =====
elif menu == "📈 Metrics":
    st.markdown("# 📈 Analisis Performa Model")
    
    if st.button("🔄 Refresh Metrics", use_container_width=True):
        st.rerun()
    
    st.markdown("---")
    
    try:
        response = requests.get(f"{BACKEND_URL}/metrics", timeout=20)
        response.raise_for_status()
        metrics_payload = response.json()
        
        # Best model info
        st.markdown(f"### 🏆 Best Model: **{metrics_payload['best_algorithm'].upper()}**")
        
        rows = []
        for model_name, values in metrics_payload["metrics"].items():
            rows.append({
                "model": model_name,
                "precision": values["precision"],
                "recall": values["recall"],
                "f1_score": values["f1_score"],
                "roc_auc": values["roc_auc"],
                "confusion_matrix": values["confusion_matrix"],
            })
        
        dataframe = pd.DataFrame(rows)
        
        # Metrics table
        st.markdown("### 📊 Ringkasan Metrics")
        display_df = dataframe.drop(columns=["confusion_matrix"]).copy()
        st.dataframe(display_df.style.format({
            "precision": "{:.4f}",
            "recall": "{:.4f}",
            "f1_score": "{:.4f}",
            "roc_auc": "{:.4f}",
        }), use_container_width=True)
        
        # Confusion matrices
        st.markdown("---")
        st.markdown("### 🔍 Confusion Matrix per Model")
        cols_cm = st.columns(len(dataframe))
        for idx, (col, row) in enumerate(zip(cols_cm, dataframe.itertuples())):
            with col:
                model_name = row.model
                cm = np.array(row.confusion_matrix)
                
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    cbar=False,
                    xticklabels=["Non-Hate", "Hate"],
                    yticklabels=["Non-Hate", "Hate"],
                    ax=ax,
                    annot_kws={"fontsize": 12, "weight": "bold"}
                )
                ax.set_title(f"{model_name}", fontweight="bold", fontsize=12)
                ax.set_ylabel("Actual", fontsize=10)
                ax.set_xlabel("Predicted", fontsize=10)
                st.pyplot(fig, use_container_width=True)
        
        # Visual comparisons
        st.markdown("---")
        st.markdown("### 📊 Visual Komparasi")
        
        tab1, tab2, tab3 = st.tabs(["📈 F1-Score", "🎯 Precision vs Recall", "🏅 ROC-AUC"])
        
        with tab1:
            st.caption("F1-Score Ranking (lebih tinggi lebih baik)")
            f1_sorted = dataframe.set_index("model")[["f1_score"]].sort_values("f1_score", ascending=False)
            st.bar_chart(f1_sorted)
        
        with tab2:
            st.caption("Precision vs Recall Comparison")
            comparison_df = dataframe.set_index("model")[["precision", "recall"]]
            st.bar_chart(comparison_df)
        
        with tab3:
            st.caption("ROC-AUC Score")
            roc_sorted = dataframe.set_index("model")[["roc_auc"]].sort_values("roc_auc", ascending=False)
            st.bar_chart(roc_sorted)
        
        # Explanations
        st.markdown("---")
        st.markdown("### 📚 Penjelasan Teknis")
        
        with st.expander("📖 Definisi Metrics", expanded=True):
            st.markdown("""
            **Precision** = TP / (TP + FP)
            - Dari semua prediksi "Hate", berapa yang benar?
            - Precision tinggi = model "hati-hati", rendah false positive
            
            **Recall** = TP / (TP + FN)
            - Dari semua "Hate" sebenarnya, berapa yang terdeteksi?
            - Recall tinggi = model "menangkap" sebagian besar kasus
            
            **F1-Score** = 2 × (Precision × Recall) / (Precision + Recall)
            - Rata-rata harmonik Precision & Recall
            - Cocok untuk data tidak seimbang
            
            **ROC-AUC** = Area under Receiver Operating Characteristic
            - Kemampuan model di berbagai threshold
            - 0.5 = random, 1.0 = sempurna
            """)
        
        with st.expander("🔍 Interpretasi Confusion Matrix"):
            st.markdown("""
            - **TN (True Negative)**: Non-hate → diprediksi non-hate ✓
            - **FP (False Positive)**: Non-hate → diprediksi hate ✗ (salah tuduh)
            - **FN (False Negative)**: Hate → diprediksi non-hate ✗ (terlewat)
            - **TP (True Positive)**: Hate → diprediksi hate ✓
            """)
        
        with st.expander("💡 Trade-off & Best Practice"):
            st.markdown("""
            **Skenario Moderation (Platform Social Media)**
            - Prioritas: Recall tinggi (tangkap sebanyak mungkin)
            - Alasan: False negative (ujaran lolos) lebih berbahaya
            
            **Skenario Verifikasi User**
            - Prioritas: Precision tinggi (hindari salah tuduh)
            - Alasan: False positive (label salah) merugikan user
            
            **Skenario Balanced**
            - Prioritas: F1-Score tinggi
            - Alasan: Tidak ada priority jelas antar error type
            """)
        
    except Exception as err:
        st.error(f"❌ Gagal ambil metrics: {err}")

# ===== PREDICTION PAGE =====
elif menu == "🔮 Prediksi":
    st.markdown("# 🔮 Prediksi Real-time")
    
    if not available_models:
        st.warning("⚠️ Belum ada model yang terlatih. Jalankan training terlebih dahulu.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            text = st.text_area(
                "📝 Masukkan teks untuk diklasifikasi",
                placeholder="Contoh: kamu ini idiot banget deh...",
                height=150
            )
        
        with col2:
            prediction_model = st.selectbox(
                "🤖 Pilih Model",
                options=available_models,
                index=0,
            )
        
        if st.button("🔍 PREDIKSI", use_container_width=True, type="primary"):
            if not text.strip():
                st.warning("⚠️ Teks tidak boleh kosong.")
            else:
                with st.spinner("🔄 Prediksi sedang diproses..."):
                    try:
                        response = requests.post(
                            f"{BACKEND_URL}/predict",
                            json={"text": text, "algorithm": prediction_model},
                            timeout=20,
                        )
                        response.raise_for_status()
                        result = response.json()
                        
                        # Result display
                        st.success("✅ Prediksi berhasil!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            label_color = "🔴" if result['label'] == "HATE" else "🟢"
                            st.metric("Label", f"{label_color} {result['label']}")
                        with col2:
                            st.metric("Model", result['algorithm'].replace("_", " ").title())
                        with col3:
                            confidence = result['confidence']
                            st.metric("Confidence", f"{confidence*100:.1f}%")
                        
                        # Detailed info
                        st.markdown("---")
                        st.markdown("### 📋 Teks yang dianalisis:")
                        st.info(f'"{text}"')
                        
                    except Exception as err:
                        st.error(f"❌ Prediksi gagal: {err}")

# ===== YOUTUBE PAGE =====
elif menu == "🎬 YouTube":
    st.markdown("# 🎬 Scraping & Klasifikasi YouTube")
    
    if not available_models:
        st.warning("⚠️ Belum ada model yang terlatih. Jalankan training terlebih dahulu.")
    else:
        st.markdown("### 📤 Konfigurasi Scraping")
        
        col1, col2 = st.columns(2)
        with col1:
            youtube_url = st.text_input(
                "🔗 URL YouTube",
                placeholder="https://www.youtube.com/watch?v=...",
            )
        with col2:
            scrape_model = st.selectbox(
                "🤖 Model Klasifikasi",
                options=available_models,
                index=0,
            )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            scrape_mode = st.selectbox(
                "📌 Mode Scraping",
                options=["auto", "live_chat", "comments"],
                index=0,
            )
        with col2:
            max_items = st.number_input(
                "📊 Jumlah Data",
                min_value=10,
                max_value=500,
                value=100,
                step=10
            )
        with col3:
            predict_after_scrape = st.checkbox("⚡ Klasifikasi Otomatis", value=True)
        
        if st.button("🚀 SCRAPE", use_container_width=True, type="primary"):
            if not youtube_url.strip():
                st.warning("⚠️ URL YouTube harus diisi.")
            else:
                with st.spinner("🔄 Sedang scraping & klasifikasi... tunggu yaa"):
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
                        
                        st.success("✅ Scraping selesai!")
                        
                        # Summary statistics
                        st.markdown("### 📊 Statistik Hasil")
                        col1, col2, col3 = st.columns(3)
                        
                        total = payload['summary']['total_comments']
                        hate_count = payload['summary']['hate_speech_count']
                        non_hate_count = payload['summary']['non_hate_speech_count']
                        
                        with col1:
                            st.metric("Total Komentar", total)
                        with col2:
                            st.metric("🔴 Hate Speech", hate_count, f"{hate_count/total*100:.1f}%")
                        with col3:
                            st.metric("🟢 Non-Hate", non_hate_count, f"{non_hate_count/total*100:.1f}%")
                        
                        # Pie chart
                        st.markdown("---")
                        fig, ax = plt.subplots(figsize=(6, 4))
                        sizes = [hate_count, non_hate_count]
                        labels = [f"Hate ({hate_count})", f"Non-Hate ({non_hate_count})"]
                        colors = ["#ff6b6b", "#51cf66"]
                        ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
                        ax.set_title("Distribusi Klasifikasi", fontweight="bold", fontsize=12)
                        st.pyplot(fig, use_container_width=True)
                        
                        # Results table
                        st.markdown("---")
                        st.markdown("### 📋 Detail Komentar")
                        
                        if payload.get("prediction_error"):
                            st.warning(f"⚠️ Prediksi tidak dijalankan: {payload['prediction_error']}")
                        
                        result_df = pd.DataFrame(payload["items"])
                        if not result_df.empty:
                            # Add styling
                            st.dataframe(
                                result_df.head(50),
                                use_container_width=True,
                                height=400
                            )
                            
                            if len(result_df) > 50:
                                st.info(f"📌 Menampilkan 50 dari {len(result_df)} komentar")
                        else:
                            st.info("ℹ️ Tidak ada komentar yang berhasil diambil dari URL ini.")
                        
                    except Exception as err:
                        st.error(f"❌ Scraping gagal: {err}")
