# =============================================================================
# TESIS: Prediksi pH Sistem Akuaponik Menggunakan Model Hybrid LSTM-GRU
# Author  : Ardi
# Program : Magister Data Science - Institut Teknologi dan Bisnis STIKOM Bali
# Python  : 3.12 | TensorFlow/Keras | Spyder 6.1.2
# =============================================================================
# STRUKTUR FILE:
#   1. Import & Konfigurasi Global
#   2. Fungsi Preprocessing Data
#   3. Fungsi Feature Engineering
#   4. Fungsi Pembangunan Model (LSTM, GRU, Hybrid LSTM-GRU)
#   5. Fungsi Training & Hyperparameter Tuning
#   6. Fungsi Evaluasi & Metrik
#   7. Fungsi Visualisasi
#   8. Main Pipeline (Eksekusi Utama)
# =============================================================================

# =============================================================================
# BAGIAN 1: IMPORT & KONFIGURASI GLOBAL
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # Untuk Spyder: ganti ke 'Qt5Agg' jika ingin inline
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Scikit-learn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# TensorFlow / Keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Suppress TF info logs
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, LSTM, GRU, Dense, Dropout,
    LayerNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint,
    ReduceLROnPlateau, CSVLogger
)
from tensorflow.keras.regularizers import l2

warnings.filterwarnings('ignore')

# -------------------------------------------------------------------
# SEED – Reproducibility
# -------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -------------------------------------------------------------------
# PATH KONFIGURASI
# -------------------------------------------------------------------
BASE_DIR    = Path(__file__).parent          # folder yang sama dengan script
DATA_PATH   = BASE_DIR / "sources" / "aquaponic_letuce_dataset_.csv"
OUTPUT_DIR  = BASE_DIR / "outputs"
MODEL_DIR   = OUTPUT_DIR / "models"
PLOT_DIR    = OUTPUT_DIR / "plots"
LOG_DIR     = OUTPUT_DIR / "logs"

for d in [OUTPUT_DIR, MODEL_DIR, PLOT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# HYPERPARAMETER GLOBAL (sesuai proposal BAB IV)
# -------------------------------------------------------------------
CFG = {
    # Data
    "target_col"    : "water_pH",
    "feature_cols"  : ["water_pH", "TDS", "water_temp"],
    "test_size"     : 0.20,          # 80-20 split
    "val_size"      : 0.10,          # 10% dari train untuk validasi

    # Sliding window
    "timesteps"     : 12,            # panjang jendela waktu (lag)
    "n_features"    : 3,             # pH, TDS, suhu

    # Arsitektur model
    "lstm_units_1"  : 64,
    "lstm_units_2"  : 32,
    "gru_units_1"   : 64,
    "gru_units_2"   : 32,
    "dense_units"   : 16,
    "dropout_rate"  : 0.2,

    # Training
    "batch_size"    : 64,
    "patience"      : 15,            # EarlyStopping
    "reduce_lr_pat" : 7,

    # Hyperparameter Tuning Skenario (BAB IV.4)
    # -- SET 1 PERCOBAAN DULU UNTUK TESTING --
    "epoch_variants"  : [10],
    "lr_variants"     : [0.001],
    "unit_variants"   : [(64, 32)],

    # Smoothing
    "ma_window"     : 3,

    # Grafik
    "fig_dpi"       : 150,
}

print("=" * 65)
print("  THESIS: Hybrid LSTM-GRU untuk Prediksi pH Akuaponik")
print("  TensorFlow version :", tf.__version__)
print("=" * 65)


# =============================================================================
# BAGIAN 2: FUNGSI PREPROCESSING DATA
# =============================================================================

def load_and_inspect(path: Path) -> pd.DataFrame:
    """
    Memuat CSV, auto-detect kolom timestamp, dan cetak ringkasan awal.
    Sesuai BAB IV.3.1 - Persiapan Data.
    """
    print("\n[1/8] MEMUAT DATASET ...")
    if not path.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {path}\n"
            f"Pastikan CSV ada di: {path.parent}"
        )

    df = pd.read_csv(path)
    print(f"      Ukuran dataset awal  : {df.shape[0]:,} baris x {df.shape[1]} kolom")
    print(f"      Kolom               : {list(df.columns)}")

    # --- Auto-detect & parse kolom timestamp ---
    ts_candidates = [c for c in df.columns
                     if any(k in c.lower() for k in ["time", "date", "stamp", "ts"])]
    if ts_candidates:
        ts_col = ts_candidates[0]
        df[ts_col] = pd.to_datetime(df[ts_col], infer_datetime_format=True, errors='coerce')
        df = df.set_index(ts_col).sort_index()
        print(f"      Kolom timestamp      : '{ts_col}'")
        print(f"      Rentang data         : {df.index.min()} s/d {df.index.max()}")
    else:
        print("      [PERINGATAN] Kolom timestamp tidak ditemukan. Menggunakan indeks integer.")

    # --- Auto-rename kolom ke nama standar jika berbeda ---
    rename_map = {}
    for col in df.columns:
        cl = col.lower().replace(" ", "_")
        if "ph"   in cl and col != "water_pH":  rename_map[col] = "water_pH"
        if "tds"  in cl and col != "TDS":        rename_map[col] = "TDS"
        if "temp" in cl and col != "water_temp": rename_map[col] = "water_temp"
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
        print(f"      Kolom di-rename      : {rename_map}")

    return df


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Statistik deskriptif untuk EDA."""
    stats = df[CFG["feature_cols"]].describe().T
    stats["skewness"] = df[CFG["feature_cols"]].skew()
    stats["kurtosis"] = df[CFG["feature_cols"]].kurt()
    print("\n      Statistik Deskriptif:")
    print(stats.to_string())
    return stats


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Penanganan nilai hilang:
    - Interpolasi linier (BAB IV.3.2.1)
    - Forward fill untuk sisa NaN
    Sesuai persamaan interpolasi linier di BAB III.4.1.
    """
    missing_before = df[CFG["feature_cols"]].isnull().sum()
    if missing_before.sum() > 0:
        print(f"\n      Missing values sebelum: {missing_before.to_dict()}")
        df[CFG["feature_cols"]] = (
            df[CFG["feature_cols"]]
            .interpolate(method='linear', limit_direction='both')
            .ffill()
            .bfill()
        )
        missing_after = df[CFG["feature_cols"]].isnull().sum()
        print(f"      Missing values setelah: {missing_after.to_dict()}")
    else:
        print("      Tidak ada missing values ditemukan.")
    return df


def remove_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deteksi dan penanganan outlier menggunakan metode IQR.
    Sesuai BAB IV.3.2 - Outlier Detection.
    """
    df_clean = df.copy()
    total_removed = 0
    for col in CFG["feature_cols"]:
        Q1  = df_clean[col].quantile(0.25)
        Q3  = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 3.0 * IQR    # batas longgar agar data sensor tidak terlalu banyak dibuang
        upper = Q3 + 3.0 * IQR
        mask  = (df_clean[col] < lower) | (df_clean[col] > upper)
        n_out = mask.sum()
        if n_out > 0:
            df_clean.loc[mask, col] = np.nan   # ganti dulu jadi NaN
            total_removed += n_out
    if total_removed:
        print(f"      Outlier ditemukan    : {total_removed:,} titik → diisi interpolasi")
        df_clean = handle_missing_values(df_clean)
    else:
        print("      Tidak ada outlier signifikan.")
    return df_clean


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Hapus duplikat (BAB IV.3.1 - Duplicate Removal)."""
    n_dup = df.duplicated().sum()
    if n_dup:
        df = df[~df.duplicated()]
        print(f"      Duplikat dihapus     : {n_dup:,} baris")
    return df


def apply_moving_average(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Smoothing dengan moving average untuk meredam noise sensor.
    Sesuai BAB IV.3.2.2 & persamaan MA(t).
    """
    for col in CFG["feature_cols"]:
        df[col] = (df[col]
                   .rolling(window=window, min_periods=1, center=True)
                   .mean())
    print(f"      Moving Average smoothing diterapkan (window={window})")
    return df


def normalize_features(df: pd.DataFrame):
    """
    Min-Max Normalization ke [0, 1].
    Sesuai BAB III.4.2 & BAB IV.3.2.3.
    Mengembalikan (df_scaled, scaler_dict).
    """
    scalers = {}
    df_scaled = df.copy()
    for col in CFG["feature_cols"]:
        sc = MinMaxScaler(feature_range=(0, 1))
        df_scaled[col] = sc.fit_transform(df_scaled[[col]])
        scalers[col] = sc
    print("      Min-Max Normalization diterapkan untuk semua fitur.")
    return df_scaled, scalers


def full_preprocessing(path: Path):
    """
    Pipeline preprocessing lengkap — memanggil semua langkah berurutan.
    Mengembalikan (df_raw, df_processed, scalers).
    """
    print("\n" + "=" * 65)
    print("  TAHAP PREPROCESSING DATA  (BAB IV.3)")
    print("=" * 65)

    df_raw      = load_and_inspect(path)
    stats       = descriptive_stats(df_raw)
    df          = remove_duplicates(df_raw.copy())
    df          = handle_missing_values(df)
    df          = remove_outliers_iqr(df)
    df          = apply_moving_average(df, window=CFG["ma_window"])
    df_scaled, scalers = normalize_features(df)

    print(f"\n      Dataset final        : {df_scaled.shape[0]:,} baris")
    return df_raw, df_scaled, scalers, stats


# =============================================================================
# BAGIAN 3: FEATURE ENGINEERING & PEMBENTUKAN DATASET
# =============================================================================

def pearson_correlation_analysis(df_raw: pd.DataFrame):
    """
    Analisis korelasi Pearson antara pH, TDS, dan suhu.
    Sesuai BAB III.4.3 & BAB II.5.3 - Optimasi Fitur Multivariat.
    """
    print("\n[2/8] ANALISIS KORELASI PEARSON ...")
    corr_matrix = df_raw[CFG["feature_cols"]].corr(method='pearson')
    print("      Matriks Korelasi Pearson:")
    print(corr_matrix.to_string())
    return corr_matrix


def create_sliding_window(data: np.ndarray, timesteps: int):
    """
    Transformasi time series ke format supervised learning
    menggunakan metode jendela geser (sliding window).
    Sesuai BAB III.3 - persamaan X_t dan Y_t.

    Input  : data shape (N, n_features)
    Output : X shape (N-timesteps, timesteps, n_features)
             y shape (N-timesteps,)  ← hanya kolom pH (indeks 0)
    """
    X, y = [], []
    ph_idx = CFG["feature_cols"].index(CFG["target_col"])   # indeks kolom pH
    for i in range(timesteps, len(data)):
        X.append(data[i - timesteps: i, :])       # semua fitur dalam window
        y.append(data[i, ph_idx])                 # target = pH berikutnya
    return np.array(X), np.array(y)


def split_dataset(df_scaled: pd.DataFrame):
    """
    Pembagian dataset: Train 80% | Test 20% (sequential, tidak diacak).
    Sesuai BAB IV.3.2.4.
    Mengembalikan (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    print("\n[3/8] PEMBAGIAN DATASET ...")
    data = df_scaled[CFG["feature_cols"]].values

    # Split sekuensial
    n         = len(data)
    n_test    = int(n * CFG["test_size"])
    n_train   = n - n_test

    train_raw = data[:n_train]
    test_raw  = data[n_train - CFG["timesteps"]:]   # include overlap untuk window

    X_train_full, y_train_full = create_sliding_window(train_raw, CFG["timesteps"])
    X_test,       y_test       = create_sliding_window(test_raw,  CFG["timesteps"])

    # Pisahkan validation dari training
    n_val    = int(len(X_train_full) * CFG["val_size"])
    X_val    = X_train_full[-n_val:]
    y_val    = y_train_full[-n_val:]
    X_train  = X_train_full[:-n_val]
    y_train  = y_train_full[:-n_val]

    print(f"      X_train : {X_train.shape}   y_train : {y_train.shape}")
    print(f"      X_val   : {X_val.shape}   y_val   : {y_val.shape}")
    print(f"      X_test  : {X_test.shape}   y_test  : {y_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# BAGIAN 4: ARSITEKTUR MODEL  (BAB IV.3.3)
# =============================================================================

def build_lstm_model(units_1=64, units_2=32, dense_units=16,
                     dropout=0.2, lr=0.001) -> Model:
    """
    Model LSTM Baseline.
    Sesuai Tabel BAB IV.3.3.1
    """
    inp  = Input(shape=(CFG["timesteps"], CFG["n_features"]), name="input")
    x    = LSTM(units_1, return_sequences=True,
                activation='tanh', name="lstm_1")(inp)
    x    = Dropout(dropout, name="dropout_1")(x)
    x    = LSTM(units_2, return_sequences=False,
                activation='tanh', name="lstm_2")(x)
    x    = Dropout(dropout, name="dropout_2")(x)
    x    = Dense(dense_units, activation='relu', name="dense_hidden")(x)
    out  = Dense(1, activation='linear', name="output")(x)

    model = Model(inputs=inp, outputs=out, name="LSTM_Model")
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse',
                  metrics=['mae'])
    return model


def build_gru_model(units_1=64, units_2=32, dense_units=16,
                    dropout=0.2, lr=0.001) -> Model:
    """
    Model GRU Baseline.
    Sesuai Tabel BAB IV.3.3.2
    """
    inp  = Input(shape=(CFG["timesteps"], CFG["n_features"]), name="input")
    x    = GRU(units_1, return_sequences=True,
               activation='tanh', name="gru_1")(inp)
    x    = Dropout(dropout, name="dropout_1")(x)
    x    = GRU(units_2, return_sequences=False,
               activation='tanh', name="gru_2")(x)
    x    = Dropout(dropout, name="dropout_2")(x)
    x    = Dense(dense_units, activation='relu', name="dense_hidden")(x)
    out  = Dense(1, activation='linear', name="output")(x)

    model = Model(inputs=inp, outputs=out, name="GRU_Model")
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse',
                  metrics=['mae'])
    return model


def build_hybrid_lstm_gru_model(units_1=64, units_2=32, dense_units=16,
                                dropout=0.2, lr=0.001) -> Model:
    """
    Model Hybrid LSTM-GRU (MODEL USULAN).
    Layer 1 : LSTM – menangkap dependensi jangka panjang
    Layer 2 : GRU  – efisiensi komputasi pola jangka pendek
    Sesuai Tabel BAB IV.3.3.3 & BAB III.5.4 (Arsitektur Sekuensial).
    """
    inp  = Input(shape=(CFG["timesteps"], CFG["n_features"]), name="input")
    # LSTM Layer: return_sequences=True → meneruskan seluruh sekuens ke GRU
    x    = LSTM(units_1, return_sequences=True,
                activation='tanh', name="lstm_1")(inp)
    x    = Dropout(dropout, name="dropout_1")(x)
    # GRU Layer: return_sequences=False → hanya output terakhir
    x    = GRU(units_2, return_sequences=False,
               activation='tanh', name="gru_2")(x)
    x    = Dropout(dropout, name="dropout_2")(x)
    x    = Dense(dense_units, activation='relu', name="dense_hidden")(x)
    out  = Dense(1, activation='linear', name="output")(x)

    model = Model(inputs=inp, outputs=out, name="Hybrid_LSTM_GRU_Model")
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse',
                  metrics=['mae'])
    return model


def get_callbacks(model_name: str, fold_id: str = "") -> list:
    """Callbacks standar untuk semua model."""
    suffix = f"_{fold_id}" if fold_id else ""
    return [
        EarlyStopping(monitor='val_loss', patience=CFG["patience"],
                      restore_best_weights=True, verbose=0),
        ModelCheckpoint(
            filepath=str(MODEL_DIR / f"{model_name}{suffix}_best.keras"),
            monitor='val_loss', save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=CFG["reduce_lr_pat"], min_lr=1e-6, verbose=0),
        CSVLogger(str(LOG_DIR / f"{model_name}{suffix}_log.csv"), append=False),
    ]


# =============================================================================
# BAGIAN 5: TRAINING & HYPERPARAMETER TUNING  (BAB IV.3.4)
# =============================================================================

def train_model(model, model_name, X_train, y_train, X_val, y_val,
                epochs=100, batch_size=64, fold_id=""):
    """
    Melatih satu model dan mengembalikan (model, history).
    """
    print(f"      Training {model_name} | epochs={epochs} | "
          f"batch={batch_size} ...", end=" ", flush=True)
    t0 = datetime.now()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks(model_name, fold_id),
        verbose=0,
        shuffle=False,     # time series — jangan diacak
    )
    elapsed = (datetime.now() - t0).total_seconds()
    best_ep = np.argmin(history.history['val_loss']) + 1
    best_vl = min(history.history['val_loss'])
    print(f"Selesai {elapsed:.1f}s | best epoch={best_ep} | "
          f"val_loss={best_vl:.6f}")
    return model, history


def hyperparameter_tuning(X_train, y_train, X_val, y_val):
    """
    Hyperparameter tuning sesuai BAB IV.3.4 - Model LSTM-GRU:
      - Variasi epoch      : [50, 75, 100, 150]
      - Variasi LR         : [0.0005, 0.001, 0.002]
      - Variasi hidden units: [(32,16), (64,32), (128,64)]
    Mengembalikan konfigurasi terbaik berdasarkan val_loss terendah.
    """
    print("\n[4/8] HYPERPARAMETER TUNING (Hybrid LSTM-GRU) ...")
    print("      Skenario yang diuji:")
    print(f"        Epochs      : {CFG['epoch_variants']}")
    print(f"        LR          : {CFG['lr_variants']}")
    print(f"        Hidden Units: {CFG['unit_variants']}")

    results = []
    trial   = 0

    total_trials = (len(CFG["epoch_variants"]) *
                    len(CFG["lr_variants"]) *
                    len(CFG["unit_variants"]))
    print(f"      Total percobaan     : {total_trials}\n")

    for ep in CFG["epoch_variants"]:
        for lr in CFG["lr_variants"]:
            for (u1, u2) in CFG["unit_variants"]:
                trial += 1
                m = build_hybrid_lstm_gru_model(
                    units_1=u1, units_2=u2,
                    dense_units=CFG["dense_units"],
                    dropout=CFG["dropout_rate"],
                    lr=lr
                )
                fold_id = f"tune_ep{ep}_lr{str(lr).replace('.','')}_u{u1}x{u2}"
                m, hist = train_model(
                    m, "HybridLSTMGRU", X_train, y_train, X_val, y_val,
                    epochs=ep, batch_size=CFG["batch_size"], fold_id=fold_id
                )
                best_val_loss = min(hist.history['val_loss'])
                actual_epochs = len(hist.history['val_loss'])
                results.append({
                    "trial"       : trial,
                    "epochs"      : ep,
                    "lr"          : lr,
                    "units_1"     : u1,
                    "units_2"     : u2,
                    "val_loss"    : best_val_loss,
                    "actual_ep"   : actual_epochs,
                    "model"       : m,
                    "history"     : hist,
                })
                tf.keras.backend.clear_session()    # bebaskan memori GPU/CPU

    # Pilih berdasarkan val_loss minimum
    df_res = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ["model", "history"]}
        for r in results
    ])
    df_res.to_csv(OUTPUT_DIR / "tuning_results.csv", index=False)
    print("\n      Tabel Hasil Tuning (Top 5):")
    print(df_res.nsmallest(5, "val_loss").to_string(index=False))

    best_idx  = df_res["val_loss"].idxmin()
    best_cfg  = results[best_idx]
    print(f"\n      ✓ Konfigurasi Terbaik:")
    print(f"        Epochs   : {best_cfg['epochs']}")
    print(f"        LR       : {best_cfg['lr']}")
    print(f"        Units    : ({best_cfg['units_1']}, {best_cfg['units_2']})")
    print(f"        Val Loss : {best_cfg['val_loss']:.6f}")

    return best_cfg, df_res


# =============================================================================
# BAGIAN 6: EVALUASI & METRIK  (BAB III.6 & BAB IV.3.5)
# =============================================================================

def inverse_transform_predictions(y_pred_scaled, y_true_scaled, scalers):
    """
    Mengubah prediksi yang ternormalisasi kembali ke skala pH asli.
    """
    sc = scalers[CFG["target_col"]]
    y_pred = sc.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = sc.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
    return y_pred, y_true


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    model_name: str = "") -> dict:
    """
    Menghitung RMSE, MAE, R² sesuai BAB III.6 & BAB IV.3.5.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)

    metrics = {"Model": model_name, "MSE": mse, "RMSE": rmse,
               "MAE": mae, "R2": r2}
    print(f"      {model_name:30s} | RMSE={rmse:.6f} | "
          f"MAE={mae:.6f} | R²={r2:.6f}")
    return metrics


def evaluate_all_models(models_dict: dict, X_test, y_test, scalers) -> pd.DataFrame:
    """
    Evaluasi semua model secara serentak dan kembalikan tabel perbandingan.
    """
    print("\n[6/8] EVALUASI KOMPARATIF MODEL ...")
    print(f"      {'Model':30s} | {'RMSE':>10} | {'MAE':>10} | {'R²':>8}")
    print("      " + "-" * 68)

    all_metrics  = []
    all_preds    = {}

    for name, model in models_dict.items():
        y_pred_sc = model.predict(X_test, verbose=0).flatten()
        y_pred, y_true = inverse_transform_predictions(y_pred_sc, y_test, scalers)
        m = compute_metrics(y_true, y_pred, name)
        all_metrics.append(m)
        all_preds[name] = {"y_pred": y_pred, "y_true": y_true}

    df_metrics = pd.DataFrame(all_metrics).set_index("Model")
    df_metrics.to_csv(OUTPUT_DIR / "evaluation_metrics.csv")
    print(f"\n      Tabel tersimpan: {OUTPUT_DIR}/evaluation_metrics.csv")
    return df_metrics, all_preds


# =============================================================================
# BAGIAN 7: VISUALISASI
# =============================================================================

def plot_time_series_overview(df_raw: pd.DataFrame, stats: pd.DataFrame):
    """Gambar 1: Overview time series mentah (EDA)."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    labels = ['pH Air', 'TDS (ppm)', 'Suhu Air (°C)']

    for i, (col, color, label) in enumerate(
            zip(CFG["feature_cols"], colors, labels)):
        if col in df_raw.columns:
            axes[i].plot(df_raw.index, df_raw[col],
                         color=color, linewidth=0.5, alpha=0.8)
            axes[i].set_ylabel(label, fontsize=10)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_title(
                f"{label} | Mean={df_raw[col].mean():.3f} | "
                f"Std={df_raw[col].std():.3f}", fontsize=9)

    axes[-1].set_xlabel("Waktu", fontsize=10)
    fig.suptitle(
        "Overview Time Series Parameter Kualitas Air Akuaponik\n"
        "(Januari – Maret 2023)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save_fig(fig, "01_time_series_overview")


def plot_correlation_heatmap(corr_matrix: pd.DataFrame):
    """Gambar 2: Heatmap korelasi Pearson."""
    fig, ax = plt.subplots(figsize=(7, 5))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, annot=True, fmt=".4f", cmap='coolwarm',
                center=0, square=True, linewidths=0.5,
                annot_kws={"size": 12}, ax=ax)
    ax.set_title("Matriks Korelasi Pearson\n(pH, TDS, Suhu)",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    _save_fig(fig, "02_pearson_correlation")


def plot_distribution(df_raw: pd.DataFrame):
    """Gambar 3: Distribusi masing-masing fitur."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    labels = ['pH Air', 'TDS (ppm)', 'Suhu Air (°C)']

    for ax, col, color, label in zip(
            axes, CFG["feature_cols"], colors, labels):
        if col in df_raw.columns:
            ax.hist(df_raw[col].dropna(), bins=50,
                    color=color, edgecolor='white', alpha=0.8)
            ax.axvline(df_raw[col].mean(), color='red',
                       linestyle='--', linewidth=1.5, label='Mean')
            ax.set_title(label, fontsize=11, fontweight='bold')
            ax.set_xlabel("Nilai", fontsize=9)
            ax.set_ylabel("Frekuensi", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Distribusi Parameter Kualitas Air",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save_fig(fig, "03_distribution")


def plot_training_history(histories: dict):
    """Gambar 4: Kurva Training & Validation Loss ketiga model."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = {'train': '#2196F3', 'val': '#F44336'}

    for ax, (name, hist) in zip(axes, histories.items()):
        tl = hist.history['loss']
        vl = hist.history['val_loss']
        ep = range(1, len(tl) + 1)
        ax.plot(ep, tl, color=colors['train'], linewidth=1.5, label='Training Loss')
        ax.plot(ep, vl, color=colors['val'],   linewidth=1.5,
                linestyle='--', label='Validation Loss')
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Loss (MSE)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    fig.suptitle("Kurva Training vs Validation Loss",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save_fig(fig, "04_training_history")


def plot_prediction_comparison(all_preds: dict, n_samples: int = 500):
    """Gambar 5: Perbandingan prediksi vs aktual untuk tiga model."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    colors = ['#2196F3', '#4CAF50', '#9C27B0']

    for ax, (name, preds), color in zip(axes, all_preds.items(), colors):
        y_true = preds["y_true"][:n_samples]
        y_pred = preds["y_pred"][:n_samples]
        x      = range(len(y_true))
        ax.plot(x, y_true,  color='gray',  linewidth=1.0,
                alpha=0.7, label='Aktual pH')
        ax.plot(x, y_pred, color=color, linewidth=1.2,
                alpha=0.9, label=f'Prediksi {name}')
        ax.fill_between(x, y_true, y_pred,
                        alpha=0.15, color=color)
        ax.set_ylabel("pH", fontsize=9)
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Indeks Data Test", fontsize=10)
    fig.suptitle("Perbandingan Prediksi vs Aktual pH\n"
                 f"(n={n_samples} sampel pertama dari test set)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save_fig(fig, "05_prediction_comparison")


def plot_scatter_actual_vs_pred(all_preds: dict):
    """Gambar 6: Scatter plot aktual vs prediksi (ideal = diagonal)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['#2196F3', '#4CAF50', '#9C27B0']

    for ax, (name, preds), color in zip(axes, all_preds.items(), colors):
        y_true = preds["y_true"]
        y_pred = preds["y_pred"]
        ax.scatter(y_true, y_pred, alpha=0.15, s=5,
                   color=color, rasterized=True)

        # Garis ideal y = x
        lim = [min(y_true.min(), y_pred.min()) - 0.1,
               max(y_true.max(), y_pred.max()) + 0.1]
        ax.plot(lim, lim, 'r--', linewidth=1.5, label='Ideal (y = ŷ)')
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("pH Aktual", fontsize=9)
        ax.set_ylabel("pH Prediksi", fontsize=9)
        r2 = r2_score(y_true, y_pred)
        ax.set_title(f"{name}\nR² = {r2:.4f}", fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Scatter Plot: Aktual vs Prediksi pH",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save_fig(fig, "06_scatter_actual_vs_pred")


def plot_metrics_comparison(df_metrics: pd.DataFrame):
    """Gambar 7: Bar chart perbandingan metrik ketiga model."""
    metrics_to_plot = ['RMSE', 'MAE', 'R2']
    colors_bar = ['#F44336', '#FF9800', '#4CAF50']

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, metric, color in zip(axes, metrics_to_plot, colors_bar):
        vals   = df_metrics[metric]
        models = vals.index.tolist()
        bars   = ax.bar(models, vals, color=color, edgecolor='white',
                        alpha=0.85, zorder=3)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_ylabel(metric, fontsize=9)
        ax.grid(axis='y', alpha=0.4, zorder=0)
        ax.tick_params(axis='x', rotation=10, labelsize=8)

        # Annotasi nilai di atas bar
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2.,
                    bar.get_height() + max(vals) * 0.02,
                    f'{val:.4f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

        # Tandai model terbaik
        best_idx = (vals.idxmin() if metric != 'R2'
                    else vals.idxmax())
        best_pos = models.index(best_idx)
        axes[axes.tolist().index(ax)].patches[best_pos].set_edgecolor('black')
        axes[axes.tolist().index(ax)].patches[best_pos].set_linewidth(2.5)

    fig.suptitle("Perbandingan Metrik Evaluasi Model\n"
                 "(garis tebal = model terbaik)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save_fig(fig, "07_metrics_comparison")


def plot_error_distribution(all_preds: dict):
    """Gambar 8: Distribusi error residual."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['#2196F3', '#4CAF50', '#9C27B0']

    for ax, (name, preds), color in zip(axes, all_preds.items(), colors):
        residuals = preds["y_true"] - preds["y_pred"]
        ax.hist(residuals, bins=60, color=color, edgecolor='white', alpha=0.8)
        ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Error')
        ax.axvline(residuals.mean(), color='orange', linestyle='-.', linewidth=1.5,
                   label=f'Mean={residuals.mean():.4f}')
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.set_xlabel("Residual (Aktual - Prediksi)", fontsize=9)
        ax.set_ylabel("Frekuensi", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Distribusi Error Residual",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save_fig(fig, "08_error_distribution")


def plot_tuning_heatmap(df_tuning: pd.DataFrame):
    """Gambar 9: Heatmap hasil hyperparameter tuning."""
    # Pivot per LR dan units kombinasi (avg across epochs)
    df_tuning["units"] = (df_tuning["units_1"].astype(str) + "x"
                          + df_tuning["units_2"].astype(str))
    pivot = df_tuning.groupby(["lr", "units"])["val_loss"].min().unstack()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".6f", cmap="YlOrRd_r",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Min Val Loss"})
    ax.set_title("Heatmap Hyperparameter Tuning\n"
                 "(Min Validation Loss per Kombinasi LR × Hidden Units)",
                 fontsize=11, fontweight='bold')
    ax.set_xlabel("Hidden Units (Layer1 x Layer2)", fontsize=10)
    ax.set_ylabel("Learning Rate", fontsize=10)
    plt.tight_layout()
    _save_fig(fig, "09_tuning_heatmap")


def _save_fig(fig, name: str):
    """Helper menyimpan gambar."""
    path = PLOT_DIR / f"{name}.png"
    fig.savefig(path, dpi=CFG["fig_dpi"], bbox_inches='tight')
    plt.close(fig)
    print(f"      Gambar tersimpan  : plots/{name}.png")


# =============================================================================
# BAGIAN 8: MAIN PIPELINE
# =============================================================================

def main():
    start_time = datetime.now()
    print("\n" + "=" * 65)
    print("  MEMULAI PIPELINE PENELITIAN")
    print(f"  Waktu mulai: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # ------------------------------------------------------------------
    # STEP 1-2: Preprocessing & EDA
    # ------------------------------------------------------------------
    df_raw, df_scaled, scalers, stats = full_preprocessing(DATA_PATH)

    # ------------------------------------------------------------------
    # STEP 2: Korelasi Pearson & Visualisasi EDA
    # ------------------------------------------------------------------
    corr_matrix = pearson_correlation_analysis(df_raw)

    print("\n[2/8] MEMBUAT VISUALISASI EDA ...")
    plot_time_series_overview(df_raw, stats)
    plot_correlation_heatmap(corr_matrix)
    plot_distribution(df_raw)

    # ------------------------------------------------------------------
    # STEP 3: Pembentukan Dataset
    # ------------------------------------------------------------------
    (X_train, X_val, X_test,
     y_train, y_val, y_test) = split_dataset(df_scaled)

    # ------------------------------------------------------------------
    # STEP 4: Hyperparameter Tuning (Hybrid LSTM-GRU)
    # ------------------------------------------------------------------
    best_cfg, df_tuning = hyperparameter_tuning(
        X_train, y_train, X_val, y_val
    )
    plot_tuning_heatmap(df_tuning)

    # ------------------------------------------------------------------
    # STEP 5: Training Model Final (konfigurasi terbaik)
    # ------------------------------------------------------------------
    print("\n[5/8] TRAINING MODEL FINAL ...")
    best_ep = best_cfg["epochs"]
    best_lr = best_cfg["lr"]
    best_u1 = best_cfg["units_1"]
    best_u2 = best_cfg["units_2"]

    model_lstm, hist_lstm = train_model(
        build_lstm_model(best_u1, best_u2, lr=best_lr),
        "LSTM", X_train, y_train, X_val, y_val,
        epochs=best_ep, batch_size=CFG["batch_size"]
    )

    model_gru, hist_gru = train_model(
        build_gru_model(best_u1, best_u2, lr=best_lr),
        "GRU", X_train, y_train, X_val, y_val,
        epochs=best_ep, batch_size=CFG["batch_size"]
    )

    model_hybrid, hist_hybrid = train_model(
        build_hybrid_lstm_gru_model(best_u1, best_u2, lr=best_lr),
        "HybridLSTMGRU", X_train, y_train, X_val, y_val,
        epochs=best_ep, batch_size=CFG["batch_size"]
    )

    # Simpan model final
    model_lstm.save(MODEL_DIR / "final_LSTM.keras")
    model_gru.save(MODEL_DIR  / "final_GRU.keras")
    model_hybrid.save(MODEL_DIR / "final_HybridLSTMGRU.keras")
    print("      Model final tersimpan di outputs/models/")

    # Ringkasan arsitektur
    print("\n      Ringkasan Arsitektur:")
    for name, model in [("LSTM", model_lstm),
                        ("GRU", model_gru),
                        ("Hybrid LSTM-GRU", model_hybrid)]:
        print(f"\n      -- {name} --")
        model.summary(print_fn=lambda x: print(f"         {x}"))

    # Plot learning curves
    histories = {
        "LSTM"           : hist_lstm,
        "GRU"            : hist_gru,
        "Hybrid LSTM-GRU": hist_hybrid,
    }
    plot_training_history(histories)

    # ------------------------------------------------------------------
    # STEP 6: Evaluasi Komparatif
    # ------------------------------------------------------------------
    models_dict = {
        "LSTM"           : model_lstm,
        "GRU"            : model_gru,
        "Hybrid LSTM-GRU": model_hybrid,
    }
    df_metrics, all_preds = evaluate_all_models(
        models_dict, X_test, y_test, scalers
    )

    # ------------------------------------------------------------------
    # STEP 7: Visualisasi Hasil
    # ------------------------------------------------------------------
    print("\n[7/8] MEMBUAT VISUALISASI HASIL ...")
    plot_prediction_comparison(all_preds)
    plot_scatter_actual_vs_pred(all_preds)
    plot_metrics_comparison(df_metrics)
    plot_error_distribution(all_preds)

    # ------------------------------------------------------------------
    # STEP 8: Laporan Akhir
    # ------------------------------------------------------------------
    elapsed = (datetime.now() - start_time).total_seconds()
    print("\n" + "=" * 65)
    print("  RINGKASAN HASIL AKHIR")
    print("=" * 65)
    print(df_metrics.to_string())

    best_model_name = df_metrics["RMSE"].idxmin()
    print(f"\n  ✓ Model Terbaik (RMSE terendah) : {best_model_name}")
    print(f"    RMSE : {df_metrics.loc[best_model_name, 'RMSE']:.6f}")
    print(f"    MAE  : {df_metrics.loc[best_model_name, 'MAE']:.6f}")
    print(f"    R²   : {df_metrics.loc[best_model_name, 'R2']:.6f}")
    print(f"\n  Total waktu eksekusi : {elapsed/60:.1f} menit")
    print(f"  Output tersimpan di  : {OUTPUT_DIR.resolve()}")
    print("\n  File Output:")
    print("    📊 outputs/plots/          → 9 grafik visualisasi")
    print("    🤖 outputs/models/         → 3 model (.keras)")
    print("    📋 outputs/evaluation_metrics.csv")
    print("    📋 outputs/tuning_results.csv")
    print("    📝 outputs/logs/           → training logs CSV")
    print("=" * 65)

    return df_metrics, all_preds, histories


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    df_metrics, all_preds, histories = main()