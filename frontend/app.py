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
    ["🏠 Home", "🎯 Training", "📈 Metrics", "🔮 Prediksi", "🎬 YouTube", "📚 Dokumentasi"],
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

# ===== DOCUMENTATION PAGE =====
elif menu == "📚 Dokumentasi":
    st.markdown("# 📚 Dokumentasi & Teori Model")
    
    # Diagram Proses
    st.markdown("## 📊 Diagram Proses Training")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        ```
        ┌─────────────────────────────────────┐
        │     INPUT TEXT (BAHASA INDONESIA)   │
        └────────────────┬────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────────┐
        │        TEXT PREPROCESSING           │
        │   • URL/mention/digit removal       │
        │   • Slang normalization             │
        │   • Lowercase conversion            │
        │   • Stopword removal (Sastrawi)     │
        │   • Stemming (Sastrawi)             │
        └────────────────┬────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────────┐
        │       FEATURE EXTRACTION            │
        │   TF-IDF Vectorization              │
        │   • max_features=5000               │
        │   • ngram_range=(1,2)               │
        │   • norm='l2'                       │
        └────────────────┬────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────────┐
        │    MODEL TRAINING & EVALUATION      │
        │   • Random Forest                   │
        │   • XGBoost                         │
        └────────────────┬────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────────┐
        │    METRICS & COMPARISON             │
        │   • Precision, Recall, F1           │
        │   • ROC-AUC, Confusion Matrix       │
        └─────────────────────────────────────┘
        ```
        """)
    
    with col2:
        st.markdown("""
        ### 📋 Alur Data Training
        
        1️⃣ **Data Loading**
           - Dataset: Indonesian Hate Speech Superset (HuggingFace)
           - Split: 80% train, 20% validation
        
        2️⃣ **Preprocessing**
           - Cleaning & normalization
           - Stemming dengan Sastrawi
           - Stopword removal
        
        3️⃣ **Vectorization**
           - Konversi teks → angka (TF-IDF)
           - Feature dimensionality: 5000
        
        4️⃣ **Model Training**
           - RF: 100 trees (fast=60)
           - XGB: 200 rounds (fast=120)
        
        5️⃣ **Evaluation**
           - Hitung metrics & simpan artifacts
        """)
    
    st.markdown("---")
    
    # Rumusan Matematis
    st.markdown("## 🧮 Rumusan Perhitungan Model")
    
    # Random Forest section
    with st.expander("🌳 **RANDOM FOREST**", expanded=True):
        st.markdown("""
        ### Definisi & Konsep
        
        Random Forest adalah ensemble learning method yang mengkombinasikan multiple decision trees 
        untuk classification problem.
        
        #### Algoritma Dasar
        
        1. **Bootstrap Sampling**: Membuat B subset dari N data points (sampling with replacement)
           ```
           Subset_i = Random sample dari data dengan replacement
           ```
        
        2. **Build Decision Trees**: Untuk setiap subset, build tree dengan random feature selection
           ```
           Untuk setiap node:
               - Cari feature terbaik dari m random features (m = √total_features)
               - Split node berdasarkan information gain terbaik
               - Ulangi sampai leaf nodes
           ```
        
        3. **Feature Importance**: Seberapa banyak setiap feature mengurangi impurity
           ```
           Importance(feature) = Σ(decrease in impurity) / number of trees
           ```
        
        4. **Aggregation untuk Classification**:
           ```
           ŷ = Mode(y₁, y₂, ..., y_B)  [majority voting]
           Confidence = (votes for class) / B
           ```
        
        #### Gini Impurity (Information Gain Criterion)
        
        ```
        Gini(t) = 1 - Σ(p_i)²
        
        Dimana:
        - t = node
        - p_i = proporsi class i di node t
        
        Node split (gain):
        Gain = Gini(parent) - Σ(n_child/n_parent × Gini(child))
        ```
        
        #### Hyperparameter Kami
        
        | Parameter | Normal Mode | Fast Mode | Penjelasan |
        |-----------|------------|-----------|-----------|
        | n_estimators | 100 | 60 | Jumlah trees (lebih banyak = lebih akurat tapi lambat) |
        | max_depth | None | 15 | Kedalaman max tree (pembatasan mencegah overfitting) |
        | min_samples_split | 2 | 5 | Min samples untuk split node |
        | min_samples_leaf | 1 | 2 | Min samples di leaf node |
        | max_features | 'sqrt' | 'sqrt' | Jumlah features untuk pertimbangan per split |
        
        """)
    
    # XGBoost section
    with st.expander("🚀 **XGBOOST**", expanded=True):
        st.markdown("""
        ### Definisi & Konsep
        
        XGBoost (eXtreme Gradient Boosting) adalah gradient boosting framework yang menggunakan
        sequential ensemble dengan optimization tree regularization.
        
        #### Algoritma Dasar (Boosting)
        
        Sequential building untuk meminimalkan residuals dari prediksi sebelumnya:
        
        ```
        ŷ₀ = average(y)  # Initial prediction
        
        For iteration m = 1 to M:
            residuals = y - ŷ_{m-1}
            tree_m = fit tree untuk residuals
            ŷ_m = ŷ_{m-1} + learning_rate × tree_m
        
        Final prediction:
        ŷ_final = ŷ_0 + Σ(learning_rate × tree_m)
        ```
        
        #### Loss Function Optimization
        
        ```
        L(y, ŷ) = Σ l(y_i, ŷ_i) + Σ Ω(tree_k)
        
        Dimana:
        - l = log loss untuk classification
        - Ω = regularization term (mencegah overfitting)
        
        l(y_i, ŷ_i) = -[y_i × log(σ(ŷ_i)) + (1-y_i) × log(1-σ(ŷ_i))]
        σ(x) = sigmoid function = 1/(1+e^(-x))
        
        Regularization:
        Ω(tree_k) = γ×T_k + λ×Σ(w_j²)
        - γ = complexity penalty per leaf
        - λ = L2 regularization untuk leaf weights
        - T_k = number of leaves
        ```
        
        #### Gain & Split Selection (dengan Regularization)
        
        ```
        Gain = [G_L² / (H_L + λ)] + [G_R² / (H_R + λ)] - [G² / (H + λ)] - γ
        
        Dimana:
        - G = gradient sum (first derivative)
        - H = hessian sum (second derivative)
        - L, R = left & right split
        
        Score = Gain / (0.5 × depth_penalty)  # Ada penalty untuk depth
        ```
        
        #### Hyperparameter Kami
        
        | Parameter | Normal Mode | Fast Mode | Penjelasan |
        |-----------|------------|-----------|-----------|
        | n_estimators | 200 | 120 | Jumlah boosting rounds |
        | max_depth | 6 | 4 | Kedalaman tree (XGB biasa lebih shallow) |
        | learning_rate | 0.1 | 0.1 | Shrinkage factor untuk setiap update |
        | subsample | 0.8 | 0.6 | % data untuk train setiap tree |
        | colsample_bytree | 0.8 | 0.6 | % features untuk setiap tree |
        | min_child_weight | 1 | 5 | Min sum of instance weight di child |
        | gamma | 0 | 1 | Min loss reduction untuk split |
        | lambda | 1 | 2 | L2 regularization |
        | alpha | 0 | 0 | L1 regularization |
        | scale_pos_weight | auto | auto | Balance weight untuk imbalanced data |
        
        #### Keunggulan XGBoost vs Random Forest
        
        | Aspek | Random Forest | XGBoost |
        |-------|---------------|---------|
        | Voting | Majority voting | Weighted sum |
        | Sequential | Tidak bergantung | Bergantung pada error sebelumnya |
        | Tuning | Sederhana | Complex tapi powerful |
        | Speed | Cepat | Lebih lambat (boosting) |
        | Overfitting | Jarang | Perlu monitoring |
        | Interpretability | Good | Hard |
        
        """)
    
    st.markdown("---")
    
    # Evaluation Metrics
    st.markdown("## 📊 Evaluation Metrics (Rumusan)")
    
    st.markdown("""
    ### 1. Confusion Matrix Components
    
    ```
                   Predicted Negative    Predicted Positive
    Actual Negative       TN                    FP
    Actual Positive       FN                    TP
    ```
    
    ### 2. Metrics Calculation
    
    #### Accuracy (Keakuratan overall)
    ```
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    Range: 0 to 1
    Interpretation: Proporsi prediksi yang benar
    Kekurangan: Kurang reliabel untuk imbalanced data
    ```
    
    #### Precision (Presisi/Positive Predictive Value)
    ```
    Precision = TP / (TP + FP)
    
    Range: 0 to 1
    Interpretation: Dari semua prediksi POSITIVE, berapa yang benar?
    Use case: Saat false positive mahal (e.g., false alarm)
    ```
    
    #### Recall / Sensitivity / True Positive Rate
    ```
    Recall = TP / (TP + FN)
    
    Range: 0 to 1
    Interpretation: Dari semua POSITIVE sebenarnya, berapa yang terdeteksi?
    Use case: Saat false negative mahal (e.g., miss hate speech)
    ```
    
    #### F1-Score (Harmonic Mean)
    ```
    F1 = 2 × (Precision × Recall) / (Precision + Recall)
    
    Range: 0 to 1
    Interpretation: Balance antara Precision & Recall
    Best for: Imbalanced dataset (prioritas di sini)
    ```
    
    #### Specificity (True Negative Rate)
    ```
    Specificity = TN / (TN + FP)
    
    Range: 0 to 1
    Interpretation: Kemampuan mendeteksi NEGATIVE dengan benar
    ```
    
    #### ROC-AUC (Receiver Operating Characteristic - Area Under Curve)
    ```
    ROC Curve: Plot TPR vs FPR pada berbagai threshold
    
    TPR = Recall = TP / (TP + FN)
    FPR = False Positive Rate = FP / (TN + FP)
    
    AUC = Area di bawah ROC curve
    Range: 0 to 1 (0.5 = random classifier, 1.0 = perfect)
    ```
    
    ### 3. Praktik Interpretasi untuk Hate Speech Detection
    
    #### Skenario: Content Moderation (Platform Social Media)
    ```
    Priority: Recall tinggi (menangkap sebagian besar hate speech)
    Alasan: False negative (hate lolos) lebih berbahaya daripada 
            false positive (label salah)
    
    Trade-off: Precision mungkin rendah (lebih banyak false alarm)
    Decision: Gunakan model dengan F1 & Recall tinggi
    ```
    
    #### Skenario: User Verification (Cegah Abuse)
    ```
    Priority: Precision tinggi (hindari salah tuduh)
    Alasan: False positive (tuduh user salah) merugikan innocent user
    
    Trade-off: Recall mungkin rendah (beberapa hate terlewat)
    Decision: Gunakan model dengan Precision tinggi
    ```
    
    #### Skenario: Balanced Deployment
    ```
    Priority: F1-Score tinggi (seimbang precision & recall)
    Alasan: Tidak clear mana error type yang lebih mahal
    
    Trade-off: Kompromi antara dua metrik
    Decision: Gunakan model dengan F1 tertinggi (default di app)
    ```
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Referensi")
    st.markdown("""
    - Breiman, L. (2001). "Random Forests". Machine Learning, 45(1), 5-32.
    - Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System". KDD '16.
    - Indonesian Hate Speech Superset: https://huggingface.co/datasets/manueltonneau/indonesian-hate-speech-superset
    """)

