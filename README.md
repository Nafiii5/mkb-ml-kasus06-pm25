# Kasus 06 · Prakiraan PM2.5 Bandung (+3 Jam)

Sistem peringatan dini konsentrasi PM2.5 **tiga jam ke depan** untuk 5 stasiun pemantau kualitas udara di Kota Bandung (`Dago`, `Soekarno-Hatta`, `Cibeureum`, `Antapani`, dan `Ujungberung`) bagi warga rentan (penderita asma) dan petugas dinas kesehatan lingkungan.

- **Tautan Aplikasi Web:** *(Tempel URL Streamlit Cloud kamu di sini setelah Langkah 3)*

## Temuan Audit Data (`data/raw.csv`)
1. **Standardisasi Nama Stasiun:** Menyatukan 16 variasi penulisan (huruf kapital, spasi berlebih, singkatan `SOETTA`) menjadi 5 nama stasiun baku.
2. **Koreksi Zona Waktu UTC $\rightarrow$ WIB:** Stasiun `Soekarno-Hatta` mengalami pergeseran pencatatan ke UTC pada periode 1 Juli–30 September 2025 (puncak suhu harian bergeser 7 jam). Diperbaiki dengan menambahkan `+7 jam` sebelum menghapus duplikat.
3. **Pembersihan Duplikat & Nilai Sentinel:** Menghapus baris duplikat log, mengubah nilai galat `-999.0` dan `0.0` menjadi `NaN`, serta melengkapi *grid* waktu per jam utuh (`87.600` baris).

## Hasil Evaluasi (Data Uji: Mei–Juni 2026)

| Model | MAE ($\mu\text{g/m}^3$) | Recall Tidak Sehat (`>55`) | Precision Tidak Sehat (`>55`) | F1-Score (`>55`) |
| :--- | :---: | :---: | :---: | :---: |
| **Persistence Baseline** | 6.513 | 29.88% | 30.25% | 0.3006 |
| **HistGradientBoosting (Final)** | **4.594** | **28.63%** | **65.71%** | **0.3988** |

## Struktur Repositori
- `data/` — Dataset mentah (`raw.csv`) dan hasil pembersihan (`clean.csv`)
- `notebooks/01_eda_dan_pembersihan.ipynb` — Analisis eksplorasi data dan visualisasi anomali
- `src/features.py` — Pembersihan data, rekayasa fitur bebas kebocoran waktu, dan kategorisasi risiko
- `src/train.py` — Pelatihan model *baseline* vs kandidat dan evaluasi *test set*
- `artifacts/` — Model terlatih (`model.joblib`) dan metadata evaluasi (`metadata.json`)
- `app/streamlit_app.py` — Aplikasi web interaktif
- `docs/` — Grafik EDA dan draf laporan proyek

## Cara Menjalankan di Komputer Lokal
```bash
pip install -r requirements.txt
python -m src.train
streamlit run app/streamlit_app.py