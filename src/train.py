import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, mean_absolute_error, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLS, build_features, clean_raw_dataframe

UNHEALTHY_THRESHOLD = 55.0


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    true_alert = y_true > UNHEALTHY_THRESHOLD
    pred_alert = y_pred > UNHEALTHY_THRESHOLD

    recall = float(recall_score(true_alert, pred_alert, zero_division=0))
    precision = float(precision_score(true_alert, pred_alert, zero_division=0))
    f1 = float(f1_score(true_alert, pred_alert, zero_division=0))

    return {
        "mae_ugm3": round(mae, 3),
        "unhealthy_recall_pct": round(recall * 100, 2),
        "unhealthy_precision_pct": round(precision * 100, 2),
        "unhealthy_f1": round(f1, 4),
        "true_unhealthy_hours": int(true_alert.sum()),
        "caught_unhealthy_hours": int((true_alert & pred_alert).sum()),
    }


def main():
    Path("artifacts").mkdir(exist_ok=True)
    raw_df = pd.read_csv("data/raw.csv")

    print("1. Membersihkan data mentah...")
    clean_df = clean_raw_dataframe(raw_df)
    clean_df.to_csv("data/clean.csv", index=False)

    print("2. Membangun fitur deret waktu...")
    feat_df = build_features(clean_df, is_training=True)

    # Buang baris dengan target asli NaN atau PM2.5 saat ini NaN
    model_df = feat_df.dropna(subset=["pm25_ugm3", "target_pm25_3h"]).copy()

    # Pembagian berbasis waktu (Time-based split)
    train_mask = model_df["waktu"] < "2026-03-01 00:00:00"
    val_mask = (model_df["waktu"] >= "2026-03-01 00:00:00") & (model_df["waktu"] < "2026-05-01 00:00:00")
    test_mask = model_df["waktu"] >= "2026-05-01 00:00:00"

    train_df = model_df[train_mask]
    val_df = model_df[val_mask]
    test_df = model_df[test_mask]

    X_train, y_train = train_df[FEATURE_COLS], train_df["target_pm25_3h"].values
    X_val, y_val = val_df[FEATURE_COLS], val_df["target_pm25_3h"].values
    X_test, y_test = test_df[FEATURE_COLS], test_df["target_pm25_3h"].values

    print(f"Jumlah sampel -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 3. Evaluasi Baseline Persistence pada Validation Set
    val_baseline_pred = val_df["pm25_ugm3"].values
    val_baseline_metrics = evaluate_predictions(y_val, val_baseline_pred)
    print("\n[Validation] Baseline Persistence:", val_baseline_metrics)

    # 4. Kandidat Model
    candidates = {
        "Ridge_Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            loss="absolute_error",  # Dioptimalkan langsung untuk metrik utama MAE!
            learning_rate=0.05,
            max_iter=400,
            max_leaf_nodes=31,
            min_samples_leaf=30,
            random_state=42,
        ),
    }

    best_name = None
    best_model = None
    best_val_mae = float("inf")
    val_results = {"Persistence_Baseline": val_baseline_metrics}

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = np.clip(model.predict(X_val), 0, None)
        metrics = evaluate_predictions(y_val, preds)
        val_results[name] = metrics
        print(f"[Validation] {name}:", metrics)

        if metrics["mae_ugm3"] < best_val_mae:
            best_val_mae = metrics["mae_ugm3"]
            best_name = name
            best_model = model

    # 5. Latih ulang model terbaik pada gabungan Train + Validation sebelum ujian akhir Test Set
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    best_model.fit(train_val_df[FEATURE_COLS], train_val_df["target_pm25_3h"].values)

    # 6. Buka Test Set SATU KALI di akhir
    test_baseline_metrics = evaluate_predictions(y_test, test_df["pm25_ugm3"].values)
    test_final_preds = np.clip(best_model.predict(X_test), 0, None)
    test_final_metrics = evaluate_predictions(y_test, test_final_preds)

    print("\n=== HASIL UJIAN AKHIR (TEST SET: MEI–JUNI 2026) ===")
    print("Baseline Persistence:", test_baseline_metrics)
    print(f"Model Final ({best_name}):", test_final_metrics)

    # Simpan artifact
    joblib.dump(best_model, "artifacts/model.joblib")
    metadata = {
        "kasus": "Kasus 06 - Prakiraan PM2.5 Bandung (+3 Jam)",
        "selected_model": best_name,
        "feature_columns": FEATURE_COLS,
        "unhealthy_threshold_ugm3": UNHEALTHY_THRESHOLD,
        "split_dates": {
            "train": "< 2026-03-01",
            "validation": "2026-03-01 s.d. 2026-04-30",
            "test": ">= 2026-05-01",
        },
        "validation_metrics": val_results,
        "test_metrics": {
            "Persistence_Baseline": test_baseline_metrics,
            best_name: test_final_metrics,
        },
    }
    with open("artifacts/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nArtifact berhasil disimpan ke artifacts/model.joblib dan artifacts/metadata.json")


if __name__ == "__main__":
    main()