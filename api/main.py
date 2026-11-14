import json
from datetime import datetime
from typing import Optional
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features import FEATURE_COLS, STATIONS, build_features, categorize_pm25

app = FastAPI(
    title="API Prakiraan PM2.5 Bandung (+3 Jam)",
    description="Layanan peringatan dini kualitas udara 3 jam ke depan untuk 5 stasiun pemantau di Kota Bandung.",
    version="1.0.0",
)

model = joblib.load("artifacts/model.joblib")
with open("artifacts/metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)


class PredictionRequest(BaseModel):
    stasiun: str = Field(..., example="Soekarno-Hatta")
    waktu: str = Field(..., example="2026-06-15 15:00", description="Format YYYY-MM-DD HH:MM (WIB)")
    pm25_ugm3: float = Field(..., gt=0, le=500, example=48.5)
    suhu_C: float = Field(..., ge=5, le=45, example=27.2)
    kelembaban_pct: float = Field(..., ge=0, le=100, example=74.0)
    kecepatan_angin_ms: float = Field(..., ge=0, le=50, example=1.4)
    arah_angin_deg: float = Field(..., ge=0, le=360, example=135.0)
    curah_hujan_mm: float = Field(..., ge=0, le=300, example=0.0)
    pm25_1h_lalu: Optional[float] = Field(None, gt=0, le=500)
    pm25_3h_lalu: Optional[float] = Field(None, gt=0, le=500)
    pm25_24h_lalu: Optional[float] = Field(None, gt=0, le=500)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": metadata["selected_model"],
        "test_mae_ugm3": metadata["test_metrics"][metadata["selected_model"]]["mae_ugm3"],
    }


@app.post("/predict")
def predict_pm25(req: PredictionRequest):
    if req.stasiun not in STATIONS:
        raise HTTPException(status_code=400, detail=f"Stasiun harus salah satu dari: {STATIONS}")

    try:
        dt_now = pd.to_datetime(req.waktu)
    except Exception:
        raise HTTPException(status_code=400, detail="Format waktu tidak valid. Gunakan YYYY-MM-DD HH:MM.")

    # Susun histori mini (25 titik waktu) agar build_features() menghitung lag & rolling secara identik
    p1 = req.pm25_1h_lalu if req.pm25_1h_lalu is not None else req.pm25_ugm3
    p3 = req.pm25_3h_lalu if req.pm25_3h_lalu is not None else req.pm25_ugm3
    p24 = req.pm25_24h_lalu if req.pm25_24h_lalu is not None else req.pm25_ugm3

    times = pd.date_range(dt_now - pd.Timedelta(hours=24), dt_now, freq="h")
    rows = []
    for t in times:
        if t == dt_now:
            pm_val = req.pm25_ugm3
        elif t == dt_now - pd.Timedelta(hours=1):
            pm_val = p1
        elif t <= dt_now - pd.Timedelta(hours=24):
            pm_val = p24
        else:
            pm_val = p3

        rows.append({
            "waktu": t,
            "stasiun": req.stasiun,
            "pm25_ugm3": pm_val,
            "suhu_C": req.suhu_C,
            "kelembaban_pct": req.kelembaban_pct,
            "kecepatan_angin_ms": req.kecepatan_angin_ms,
            "arah_angin_deg": req.arah_angin_deg,
            "curah_hujan_mm": req.curah_hujan_mm,
        })

    hist_df = pd.DataFrame(rows)
    feat_df = build_features(hist_df, is_training=False)
    latest_row = feat_df.iloc[[-1]][FEATURE_COLS]

    pred_3h = float(np.clip(model.predict(latest_row)[0], 0, None))
    info = categorize_pm25(pred_3h)
    target_time = (dt_now + pd.Timedelta(hours=3)).strftime("%Y-%m-%d %H:%M WIB")

    return {
        "stasiun": req.stasiun,
        "waktu_input": dt_now.strftime("%Y-%m-%d %H:%M WIB"),
        "waktu_prakiraan": target_time,
        "pm25_saat_ini_ugm3": round(req.pm25_ugm3, 1),
        "prakiraan_pm25_3h_ugm3": round(pred_3h, 1),
        "selisih_ugm3": round(pred_3h - req.pm25_ugm3, 1),
        "kategori": info["kategori"],
        "saran": info["saran"],
    }