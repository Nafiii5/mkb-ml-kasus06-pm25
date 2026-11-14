import json
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.features import FEATURE_COLS, STATIONS, build_features, categorize_pm25

st.set_page_config(page_title="Prakiraan PM2.5 Bandung (+3 Jam)", page_icon="🌫️", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(ROOT_DIR / "artifacts" / "model.joblib")
    with open(ROOT_DIR / "artifacts" / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


model, metadata = load_artifacts()

st.title("🌫️ Sistem Peringatan Dini Kualitas Udara Bandung (PM2.5 +3 Jam)")
st.caption("Dinas Kesehatan Lingkungan Kota Bandung · Prakiraan konsentrasi PM2.5 tiga jam ke depan berbasis data sensor per jam.")

with st.sidebar:
    st.header("Parameter Sensor Saat Ini")
    stasiun = st.selectbox("Pilih Stasiun Pemantau", STATIONS, index=1)
    tanggal = st.date_input("Tanggal Observasi", value=datetime(2026, 6, 15))
    jam = st.slider("Jam Observasi (WIB)", min_value=0, max_value=23, value=15)

    pm25_now = st.number_input("PM2.5 Saat Ini (µg/m³)", min_value=-100.0, max_value=600.0, value=48.0, step=0.5)
    suhu = st.number_input("Suhu Udara (°C)", min_value=-10.0, max_value=50.0, value=26.5, step=0.1)
    kelembaban = st.number_input("Kelembapan Udara (%)", min_value=0.0, max_value=150.0, value=76.0, step=1.0)
    angin_kec = st.number_input("Kecepatan Angin (m/s)", min_value=0.0, max_value=50.0, value=1.2, step=0.1)
    angin_arah = st.slider("Arah Angin (derajat)", min_value=0, max_value=360, value=140)
    hujan = st.number_input("Curah Hujan (mm)", min_value=0.0, max_value=200.0, value=0.0, step=0.1)

    with st.expander("Riwayat PM2.5 Sebelumnya (Opsional)"):
        pm25_1h = st.number_input("PM2.5 1 Jam Lalu (µg/m³)", min_value=0.0, max_value=500.0, value=float(max(pm25_now - 4.0, 1.0)))
        pm25_3h = st.number_input("PM2.5 3 Jam Lalu (µg/m³)", min_value=0.0, max_value=500.0, value=float(max(pm25_now - 10.0, 1.0)))
        pm25_24h = st.number_input("PM2.5 Kemarin Jam Sama (µg/m³)", min_value=0.0, max_value=500.0, value=float(max(pm25_now, 1.0)))

# Validasi masukan tidak wajar
errors = []
if pm25_now <= 0 or pm25_now > 500:
    errors.append("Nilai PM2.5 harus berada pada rentang wajar (0.1 – 500 µg/m³).")
if suhu < 10 or suhu > 42:
    errors.append("Suhu udara Bandung harus berada pada rentang 10°C – 42°C.")
if kelembaban < 10 or kelembaban > 100:
    errors.append("Kelembapan udara harus berada pada rentang 10% – 100%.")

if errors:
    for err in errors:
        st.error(f"⚠️ Masukan tidak valid: {err}")
else:
    dt_now = pd.Timestamp(year=tanggal.year, month=tanggal.month, day=tanggal.day, hour=jam)
    times = pd.date_range(dt_now - pd.Timedelta(hours=24), dt_now, freq="h")

    rows = []
    for t in times:
        if t == dt_now:
            pm_val = pm25_now
        elif t == dt_now - pd.Timedelta(hours=1):
            pm_val = pm25_1h
        elif t <= dt_now - pd.Timedelta(hours=24):
            pm_val = pm25_24h
        else:
            pm_val = pm25_3h

        rows.append({
            "waktu": t,
            "stasiun": stasiun,
            "pm25_ugm3": pm_val,
            "suhu_C": suhu,
            "kelembaban_pct": kelembaban,
            "kecepatan_angin_ms": angin_kec,
            "arah_angin_deg": angin_arah,
            "curah_hujan_mm": hujan,
        })

    feat_df = build_features(pd.DataFrame(rows), is_training=False)
    pred_3h = float(np.clip(model.predict(feat_df.iloc[[-1]][FEATURE_COLS])[0], 0, None))
    info = categorize_pm25(pred_3h)
    target_dt = dt_now + pd.Timedelta(hours=3)

    col1, col2, col3 = st.columns(3)
    col1.metric("PM2.5 Saat Ini", f"{pm25_now:.1f} µg/m³", f"Pukul {dt_now.strftime('%H:%M')} WIB")
    col2.metric(
        f"Prakiraan Pukul {target_dt.strftime('%H:%M')} WIB",
        f"{pred_3h:.1f} µg/m³",
        f"{pred_3h - pm25_now:+.1f} µg/m³ vs sekarang",
        delta_color="inverse",
    )
    col3.metric("Status Udara +3 Jam", info["kategori"], "Ambang Tidak Sehat: > 55 µg/m³")

    if info["kategori"] == "Tidak Sehat":
        st.error(f"🚨 **{info['kategori'].upper()}** — {info['saran']}")
    elif info["kategori"] == "Sedang":
        st.warning(f"⚠️ **{info['kategori'].upper()}** — {info['saran']}")
    else:
        st.success(f"✅ **{info['kategori'].upper()}** — {info['saran']}")

    st.divider()
    best_name = metadata["selected_model"]
    test_m = metadata["test_metrics"][best_name]
    base_m = metadata["test_metrics"]["Persistence_Baseline"]
    st.subheader("Performa Model pada Data Uji (Mei–Juni 2026)")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE Model Final", f"{test_m['mae_ugm3']} µg/m³", f"{test_m['mae_ugm3'] - base_m['mae_ugm3']:+.2f} vs Baseline ({base_m['mae_ugm3']})")
    m2.metric("Recall Peringatan Tidak Sehat (>55)", f"{test_m['unhealthy_recall_pct']}%", f"vs Baseline {base_m['unhealthy_recall_pct']}%")
    m3.metric("Precision Peringatan Tidak Sehat", f"{test_m['unhealthy_precision_pct']}%", f"vs Baseline {base_m['unhealthy_precision_pct']}%")