# Komparasi Model Machine Learning Random Forest dan XGBoost 

---

# BUSINESS REQUIREMENTS DOCUMENT (BRD)  
**Proyek**: Implementasi dan Evaluasi Model Deteksi Ujaran Kebencian pada Komentar Live Streaming  

---

## 1. PENDAHULUAN  
### 1.1 Latar Belakang  
Penelitian ini bertujuan membandingkan performa algoritma Random Forest dan XGBoost dalam mengklasifikasikan ujaran kebencian pada komentar *live streaming* berbahasa Indonesia. BRD ini disusun untuk memandu pelaksanaan teknis agar sesuai dengan metodologi yang telah dirancang, meminimalkan risiko deviasi, serta memastikan hasil yang valid dan reprodusibel.

### 1.2 Tujuan BRD  
- Mendefinisikan secara rinci kebutuhan teknis dan non-teknis dalam pelaksanaan penelitian.  
- Menjembatani konsep teoritis (proposal) dengan implementasi praktik (coding, eksperimen).  
- Menjadi acuan evaluasi kemajuan dan keberhasilan proyek.  

---

## 2. RUANG LINGKUP PROYEK  
### 2.1 In-Scope  
- Pengumpulan data latih dari dataset publik *Indonesian Hate Speech Superset* (HuggingFace).  
- Scraping data uji dari *live chat replay* YouTube (minimal 3 saluran berbeda, total 1000 komentar).  
- Pelabelan manual data uji oleh minimal 2 annotator dengan panduan yang jelas.  
- Pra-pemrosesan teks: cleaning, normalisasi slang, stopword removal, stemming.  
- Ekstraksi fitur menggunakan TF-IDF dengan normalisasi L2.  
- Implementasi model Random Forest dan XGBoost dengan *hyperparameter tuning* dasar.  
- Evaluasi menggunakan F1-Score, Precision, Recall, ROC-AUC, dan *confusion matrix*.  
- Analisis perbandingan dan rekomendasi model terbaik.  

### 2.2 Out-of-Scope  
- Pengembangan antarmuka pengguna (UI) atau plugin moderasi.  
- Penggunaan model deep learning (LSTM, Transformer).  
- Deteksi multi-kelas atau jenis ujaran kebencian spesifik.  
- Analisis sentimen di luar klasifikasi biner.  

---

## 3. STAKEHOLDERS  
| Stakeholder | Peran |
|-------------|-------|
| Mahasiswa (Peneliti) | Pelaksana utama: pengumpulan data, coding, analisis |
| Dosen Pembimbing | Pemberi arahan metodologi, validasi hasil |
| Annotator (2 orang) | Pelabel manual data uji (teman sejawat atau asisten) |
| Penguji Skripsi | Evaluator akhir kelayakan penelitian |

---

## 4. KEBUTUHAN FUNGSIONAL (FUNCTIONAL REQUIREMENTS)  
| ID | Deskripsi Kebutuhan | Prioritas |
|----|----------------------|-----------|
| F-01 | Sistem mampu mengunduh dataset *Indonesian Hate Speech Superset* dari HuggingFace secara otomatis (via API atau manual) | Tinggi |
| F-02 | Sistem mampu melakukan *scraping* komentar dari *live chat replay* YouTube menggunakan pustaka `pytchat` atau `youtube-comment-downloader` | Tinggi |
| F-03 | Sistem menyediakan antarmuka sederhana (misal: spreadsheet) untuk pelabelan manual oleh annotator, dilengkapi panduan dan contoh | Tinggi |
| F-04 | Sistem mampu membersihkan teks: hapus mention, URL, angka, tanda baca, dan *emoticon* | Tinggi |
| F-05 | Sistem mampu menormalisasi kata tidak baku menggunakan kamus slang (misal: dari github atau buatan sendiri) | Tinggi |
| F-06 | Sistem mampu menghapus *stopword* menggunakan daftar yang telah disesuaikan (tambah kata khas *live chat* seperti "wkwk", "bg") | Tinggi |
| F-07 | Sistem mampu melakukan *stemming* bahasa Indonesia menggunakan library Sastrawi | Tinggi |
| F-08 | Sistem mampu mengubah teks menjadi vektor TF-IDF dengan parameter: `max_features=5000`, `ngram_range=(1,2)`, `norm='l2'` | Tinggi |
| F-09 | Sistem mampu membagi data latih menjadi *training* dan *validation* (80:20) secara *stratified* | Tinggi |
| F-10 | Sistem mampu melatih model Random Forest dengan parameter dasar: `n_estimators=100`, `max_depth=None`, `class_weight='balanced'` | Tinggi |
| F-11 | Sistem mampu melatih model XGBoost dengan parameter dasar: `learning_rate=0.1`, `max_depth=6`, `scale_pos_weight` (dihitung dari rasio kelas) | Tinggi |
| F-12 | Sistem mampu melakukan *hyperparameter tuning* sederhana (GridSearchCV) untuk kedua model pada ruang parameter terbatas | Sedang |
| F-13 | Sistem mampu mengevaluasi model menggunakan metrik: *Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix* | Tinggi |
| F-14 | Sistem mampu menyajikan perbandingan performa dalam bentuk tabel dan grafik batang | Sedang |
| F-15 | Sistem mampu menyimpan model terlatih dan vektorizer untuk penggunaan ulang | Rendah |

---

## 5. KEBUTUHAN NON-FUNGSIONAL (NON-FUNCTIONAL REQUIREMENTS)  
| ID | Deskripsi Kebutuhan | Target |
|----|----------------------|--------|
| NF-01 | Waktu pelatihan model tidak melebihi 30 menit di komputer lokal (spesifikasi minimal: 8GB RAM, CPU 4 core) | ≤ 30 menit |
| NF-02 | Kode harus ditulis dengan dokumentasi yang jelas (comment) dan mengikuti PEP 8 | - |
| NF-03 | Semua proses harus dapat direproduksi ulang oleh pihak lain (gunakan *seed* random) | - |
| NF-04 | Data uji harus mewakili variasi bahasa dari minimal 3 streamer berbeda | - |
| NF-05 | Tingkat kesepakatan antar annotator (Cohen’s Kappa) minimal 0,7 | ≥ 0.7 |
| NF-06 | Hasil evaluasi harus dilaporkan dengan interval kepercayaan (misal: dari cross-validation) | - |

---

## 6. ASUMSI  
- Dataset latih dari HuggingFace memiliki label yang valid dan seimbang atau tidak terlalu timpang.  
- Library yang digunakan (pytchat, Sastrawi, scikit-learn, xgboost) kompatibel dengan lingkungan pengembangan.  
- Annotator memiliki pemahaman yang cukup tentang ujaran kebencian dalam konteks Indonesia.  
- Tidak ada perubahan besar pada API YouTube selama proses scraping.  

---

## 7. KENDALA (CONSTRAINTS)  
- **Waktu**: Penyelesaian seluruh tahap maksimal 3 bulan.  
- **Sumber Daya**: Hanya menggunakan komputer pribadi tanpa GPU.  
- **Data**: Tidak semua video menyediakan *live chat replay*, sehingga pemilihan video dibatasi.  
- **Etika**: Data komentar yang di-scraping tidak akan dipublikasikan mentah, hanya digunakan untuk penelitian.  

---

## 8. TAHAPAN PELAKSANAAN (MILESTONES)  
| Tahap | Aktivitas | Output | Estimasi Waktu |
|-------|-----------|--------|----------------|
| 1 | Persiapan lingkungan dan instalasi library | Environment siap pakai | 1 hari |
| 2 | Pengumpulan data latih (HuggingFace) | File CSV data latih | 1 hari |
| 3 | Scraping data uji (YouTube) & seleksi video | File CSV data uji mentah | 3 hari |
| 4 | Pelabelan data uji oleh annotator | File CSV data uji berlabel | 5 hari |
| 5 | Pra-pemrosesan teks (bersih, normalisasi, stem) | Data bersih siap vektorisasi | 2 hari |
| 6 | Ekstraksi fitur TF-IDF | Matriks fitur dan vektorizer | 1 hari |
| 7 | Pelatihan model Random Forest & evaluasi awal | Model RF + metrik | 1 hari |
| 8 | Pelatihan model XGBoost & evaluasi awal | Model XGB + metrik | 1 hari |
| 9 | Hyperparameter tuning (GridSearchCV) | Model terbaik masing-masing | 2 hari |
| 10 | Evaluasi komparatif dan visualisasi | Tabel, grafik, analisis | 2 hari |
| 11 | Penyusunan laporan akhir (BAB IV & V) | Draft skripsi | 5 hari |
| **Total** | | | **~23 hari kerja** |

---

## 9. METRIK KEBERHASILAN PROYEK  
1. **F1-Score** model terbaik pada data uji ≥ 0,80.  
2. **Cohen’s Kappa** antar annotator ≥ 0,7.  
3. Semua kebutuhan fungsional prioritas tinggi terimplementasi.  
4. Dokumentasi kode dan laporan selesai tepat waktu.  

---

## 10. RISIKO DAN MITIGASI  
| Risiko | Dampak | Mitigasi |
|--------|--------|----------|
| Data uji tidak cukup karena fitur *replay chat* tidak tersedia | Gagal menguji model | Cari saluran dengan banyak VOD; gunakan data dari beberapa video berbeda |
| Ketidaksepakatan annotator tinggi (Kappa rendah) | Label tidak konsisten | Adakan diskusi awal, buat panduan rinci, lakukan dua putaran labeling |
| Performa model buruk (F1 < 0,7) | Kesimpulan tidak kuat | Perluas fitur (n-gram), coba teknik penanganan imbalance (SMOTE), atau perbaiki preprocessing |
| Waktu scraping lama karena batasan YouTube | Terlambat | Gunakan proxy atau batasi jumlah komentar per video; scraping bertahap |

---

## 11. Menjalankan Aplikasi (Docker Compose)

Struktur aplikasi:
- Backend API: `FastAPI` di port `8000`
- Frontend UI: `Streamlit` di port `8501`
- Downloader dataset Hugging Face: `scripts/download_dataset.py`

### 1) Build service
```bash
docker compose build
```

### 2) Download dataset ke folder `dataset/` (sekali di awal)
```bash
docker compose --profile init run --rm dataset_downloader
```

Jika file dataset sudah ada, downloader akan otomatis skip.

Jika butuh autentikasi Hugging Face, gunakan token (bukan email/password):

```bash
export HF_TOKEN=hf_xxx
docker compose --profile init run --rm dataset_downloader
```

### 3) Jalankan aplikasi
```bash
docker compose up app
```

### 4) Training & evaluasi model (RF vs XGBoost)
- Buka Streamlit di `http://localhost:8501`
- Klik **Train Random Forest** untuk melatih RF
- Klik **Train XGBoost** untuk melatih XGBoost
- (Opsional) centang tuning untuk `GridSearchCV` sederhana
- (Opsional) aktifkan **Mode cepat** untuk mempercepat training (sampling dataset dan parameter lebih ringan)
- Klik **Lihat Metrics Terakhir** untuk melihat tabel metrik dan grafik perbandingan

Preprocessing yang diterapkan otomatis sebelum training/prediksi:
- Cleaning teks (hapus mention, URL, angka, tanda baca, emoticon)
- Normalisasi slang sederhana
- Stopword removal (Sastrawi + tambahan kata live chat)
- Stemming bahasa Indonesia (Sastrawi)
- TF-IDF: `max_features=5000`, `ngram_range=(1,2)`, `norm='l2'`

Model yang dilatih:
- Random Forest: `n_estimators=100`, `max_depth=None`, `class_weight='balanced'`
- XGBoost: `learning_rate=0.1`, `max_depth=6`, `scale_pos_weight` otomatis dari rasio kelas

Artefak hasil training:
- Vectorizer: `model/tfidf_vectorizer.joblib`
- Random Forest: `model/random_forest_model.joblib`
- XGBoost: `model/xgboost_model.joblib`
- Model terbaik: `model/best_model.joblib`
- Metrics komparasi: `model/metrics.json`

### 5) Uji dengan URL YouTube (scraping otomatis)
- Buka Streamlit di `http://localhost:8501`
- Isi URL YouTube pada bagian **Uji dengan URL YouTube**
- Pilih mode:
	- `auto` (coba live chat dulu, fallback komentar biasa)
	- `live_chat` (khusus live chat replay jika tersedia)
	- `comments` (komentar YouTube biasa)
- Klik **Scrape dari URL**

Pada bagian prediksi, kamu bisa pilih model secara spesifik: `best`, `random_forest`, atau `xgboost`.
Prediksi hanya tersedia untuk model yang sudah terlatih dan tersimpan di folder `model/`.

Backend endpoint untuk scraping:
- `POST /scrape`
- Request body:
	- `url`: link YouTube
	- `mode`: `auto | live_chat | comments`
	- `max_items`: maksimal komentar (10-500)
	- `predict`: `true/false` untuk langsung klasifikasi

### Akses aplikasi
- Streamlit: http://localhost:8501
- FastAPI health check: http://localhost:8000/health

Catatan: `dataset_downloader` mengambil dataset `manueltonneau/indonesian-hate-speech-superset` dan menyimpan file CSV ke folder lokal `dataset/` melalui volume mapping. Untuk unduh ulang paksa, jalankan:

```bash
docker compose --profile init run --rm -e FORCE_DOWNLOAD=true dataset_downloader
```