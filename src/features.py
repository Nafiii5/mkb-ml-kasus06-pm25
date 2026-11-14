import numpy as np
import pandas as pd

STATIONS = ["Dago", "Soekarno-Hatta", "Cibeureum", "Antapani", "Ujungberung"]

STATION_MAP = {
    "dago": "Dago",
    "cibeureum": "Cibeureum",
    "soekarno hatta": "Soekarno-Hatta",
    "soetta": "Soekarno-Hatta",
    "ujungberung": "Ujungberung",
    "ujung berung": "Ujungberung",
    "antapani": "Antapani",
}


def normalize_station_name(name: str) -> str:
    cleaned = " ".join(str(name).strip().lower().replace("-", " ").split())
    if cleaned not in STATION_MAP:
        raise ValueError(f"Nama stasiun tidak dikenali: {name}")
    return STATION_MAP[cleaned]


def clean_raw_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Membersihkan data mentah: nama stasiun, koreksi UTC->WIB, duplikat, sentinel, & grid waktu."""
    df = df_raw.copy()
    df["stasiun"] = df["stasiun"].apply(normalize_station_name)
    df["waktu"] = pd.to_datetime(df["waktu"])

    # Koreksi bug zona waktu UTC (+7 jam) pada Soekarno-Hatta (1 Juli - 30 Sept 2025 WIB)
    # Baris UTC dimulai dari '2025-06-30 17:00' (= '2025-07-01 00:00 WIB')
    # sampai '2025-09-30 16:00' (= '2025-09-30 23:00 WIB').
    mask_sh = df["stasiun"] == "Soekarno-Hatta"
    sh_indices = df[mask_sh].index

    # Identifikasi titik transisi dari 2025-06-30 23:00 WIB kembali ke 2025-06-30 17:00 UTC
    sh_df = df.loc[sh_indices].copy()
    time_jump = sh_df["waktu"].diff() < pd.Timedelta(hours=-2)
    if time_jump.any():
        start_utc_idx = time_jump[time_jump].index[0]
        end_utc_mask = (df.index >= start_utc_idx) & mask_sh & (df["waktu"] <= pd.Timestamp("2025-09-30 16:00:00"))
        df.loc[end_utc_mask, "waktu"] = df.loc[end_utc_mask, "waktu"] + pd.Timedelta(hours=7)

    # Hapus baris duplikat identik per (stasiun, waktu)
    df = df.sort_values(["stasiun", "waktu"]).drop_duplicates(subset=["stasiun", "waktu"], keep="first")

    # Bersihkan nilai sentinel / mustahil secara fisik
    df.loc[(df["pm25_ugm3"] <= 0) | (df["pm25_ugm3"] > 500), "pm25_ugm3"] = np.nan
    df.loc[(df["suhu_C"] < 5) | (df["suhu_C"] > 45), "suhu_C"] = np.nan
    df.loc[(df["kelembaban_pct"] < 0) | (df["kelembaban_pct"] > 100), "kelembaban_pct"] = np.nan
    df.loc[(df["kecepatan_angin_ms"] < 0) | (df["kecepatan_angin_ms"] > 50), "kecepatan_angin_ms"] = np.nan
    df.loc[(df["arah_angin_deg"] < 0) | (df["arah_angin_deg"] > 360), "arah_angin_deg"] = np.nan
    df.loc[df["curah_hujan_mm"] < 0, "curah_hujan_mm"] = np.nan

    # Reindex ke grid waktu per jam utuh agar shift() presisi secara kronologis
    full_time = pd.date_range(df["waktu"].min(), df["waktu"].max(), freq="h")
    reindexed_list = []
    for st in STATIONS:
        sub = df[df["stasiun"] == st].set_index("waktu").reindex(full_time)
        sub.index.name = "waktu"
        sub["stasiun"] = st
        reindexed_list.append(sub.reset_index())

    clean_df = pd.concat(reindexed_list, ignore_index=True)
    return clean_df


def build_features(df_clean: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Membentuk fitur berbasis informasi <= t tanpa kebocoran masa depan.
    Jika is_training=True, juga membentuk kolom target 'target_pm25_3h'.
    """
    df = df_clean.sort_values(["stasiun", "waktu"]).copy()

    # Simpan nilai PM2.5 mentah (sebelum forward-fill) untuk target yang jujur
    df["pm25_raw_for_target"] = df["pm25_ugm3"]

    # Imputasi kausal (hanya melihat ke belakang maksimal 3 jam, tidak mengintip masa depan)
    sensor_cols = [
        "pm25_ugm3",
        "suhu_C",
        "kelembaban_pct",
        "kecepatan_angin_ms",
        "arah_angin_deg",
        "curah_hujan_mm",
    ]
    for col in sensor_cols:
        df[col] = df.groupby("stasiun")[col].ffill(limit=3)

    # Fitur waktu siklikal
    hour = df["waktu"].dt.hour
    dow = df["waktu"].dt.dayofweek
    month = df["waktu"].dt.month
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    df["is_weekend"] = (dow >= 5).astype(int)

    # Dekomposisi vektor angin
    rad = np.deg2rad(df["arah_angin_deg"])
    df["wind_u"] = df["kecepatan_angin_ms"] * np.cos(rad)
    df["wind_v"] = df["kecepatan_angin_ms"] * np.sin(rad)

    # Fitur lag & rolling per stasiun (hanya masa lalu <= t)
    g = df.groupby("stasiun")
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"pm25_lag_{lag}h"] = g["pm25_ugm3"].shift(lag)

    df["pm25_diff_1h"] = df["pm25_ugm3"] - df["pm25_lag_1h"]
    df["pm25_diff_3h"] = df["pm25_ugm3"] - df["pm25_lag_3h"]
    df["pm25_diff_24h"] = df["pm25_ugm3"] - df["pm25_lag_24h"]

    df["pm25_roll_mean_3h"] = g["pm25_ugm3"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["pm25_roll_std_3h"] = g["pm25_ugm3"].transform(lambda s: s.rolling(3, min_periods=2).std()).fillna(0)
    df["pm25_roll_mean_6h"] = g["pm25_ugm3"].transform(lambda s: s.rolling(6, min_periods=1).mean())
    df["pm25_roll_max_6h"] = g["pm25_ugm3"].transform(lambda s: s.rolling(6, min_periods=1).max())

    df["suhu_diff_3h"] = df["suhu_C"] - g["suhu_C"].shift(3)
    df["kelembaban_diff_3h"] = df["kelembaban_pct"] - g["kelembaban_pct"].shift(3)
    df["hujan_roll_3h"] = g["curah_hujan_mm"].transform(lambda s: s.rolling(3, min_periods=1).sum())

    # Fitur agregasi kota Bandung pada jam t (spasial antarstasiun)
    city_stats = (
        df.groupby("waktu")["pm25_ugm3"]
        .agg(city_pm25_mean="mean", city_pm25_max="max")
        .reset_index()
    )
    df = df.merge(city_stats, on="waktu", how="left")
    df["pm25_rel_city"] = df["pm25_ugm3"] - df["city_pm25_mean"]

    # One-hot encoding stasiun secara deterministik agar konsisten saat inferensi API
    for st in STATIONS:
        df[f"stasiun_{st}"] = (df["stasiun"] == st).astype(int)

    if is_training:
        df["target_pm25_3h"] = df.groupby("stasiun")["pm25_raw_for_target"].shift(-3)

    df = df.drop(columns=["pm25_raw_for_target"])
    return df


FEATURE_COLS = [
    "pm25_ugm3",
    "suhu_C",
    "kelembaban_pct",
    "kecepatan_angin_ms",
    "curah_hujan_mm",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "wind_u",
    "wind_v",
    "pm25_lag_1h",
    "pm25_lag_2h",
    "pm25_lag_3h",
    "pm25_lag_6h",
    "pm25_lag_12h",
    "pm25_lag_24h",
    "pm25_diff_1h",
    "pm25_diff_3h",
    "pm25_diff_24h",
    "pm25_roll_mean_3h",
    "pm25_roll_std_3h",
    "pm25_roll_mean_6h",
    "pm25_roll_max_6h",
    "suhu_diff_3h",
    "kelembaban_diff_3h",
    "hujan_roll_3h",
    "city_pm25_mean",
    "city_pm25_max",
    "pm25_rel_city",
] + [f"stasiun_{st}" for st in STATIONS]


def categorize_pm25(val: float) -> dict:
    """Kategori kualitas udara sesuai brief (> 55 µg/m³ = Tidak Sehat)."""
    if val <= 15.0:
        return {
            "kategori": "Baik",
            "kode_warna": "green",
            "saran": "Kualitas udara bersih. Aktivitas luar ruangan aman dilakukan seperti biasa.",
        }
    elif val <= 55.0:
        return {
            "kategori": "Sedang",
            "kode_warna": "yellow",
            "saran": "Kualitas udara masih dapat diterima, namun kelompok sensitif sebaiknya membatasi aktivitas fisik berat di luar ruangan.",
        }
    else:
        return {
            "kategori": "Tidak Sehat",
            "kode_warna": "red",
            "saran": "Peringatan dini: PM2.5 diprakirakan melebihi 55 µg/m³. Warga—terutama penderita asma, anak-anak, dan lansia—disarankan memakai masker N95/KN95 dan mengurangi aktivitas luar ruangan.",
        }