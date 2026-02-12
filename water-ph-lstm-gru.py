import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR    = Path(__file__).parent
DATA_PATH   = BASE_DIR / "sources" / "aquaponic_letuce_dataset_.csv"
OUTPUT_DIR  = BASE_DIR / "outputs"
MODEL_DIR   = OUTPUT_DIR / "models"
PLOT_DIR    = OUTPUT_DIR / "plots"
LOG_DIR     = OUTPUT_DIR / "logs"

for d in [OUTPUT_DIR, MODEL_DIR, PLOT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CFG = {
    "target_col"    : "water_pH",
    "feature_cols"  : ["water_pH", "TDS", "water_temp"],
    "test_size"     : 0.20,
    "val_size"      : 0.10,

    "timesteps"     : 12,
    "n_features"    : 3,

    "lstm_units_1"  : 64,
    "lstm_units_2"  : 32,
    "gru_units_1"   : 64,
    "gru_units_2"   : 32,
    "dense_units"   : 16,
    "dropout_rate"  : 0.2,

    "batch_size"    : 64,
    "patience"      : 15,
    "reduce_lr_pat" : 7,

    "epoch_variants"  : [10],
    "lr_variants"     : [0.001],
    "unit_variants"   : [(64, 32)],

    "ma_window"     : 3,

    "fig_dpi"       : 150,
}

print("=" * 65)
print("  THESIS: Hybrid LSTM-GRU untuk Prediksi pH Akuaponik")
print("  TensorFlow version :", tf.__version__)
print("=" * 65)


def load_and_inspect(path: Path) -> pd.DataFrame:
    print("\n[1/8] MEMUAT DATASET ...")
    if not path.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {path}\n"
            f"Pastikan CSV ada di: {path.parent}"
        )

    df = pd.read_csv(path)
    print(f"      Ukuran dataset awal  : {df.shape[0]:,} baris x {df.shape[1]} kolom")
    print(f"      Kolom               : {list(df.columns)}")

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
    stats = df[CFG["feature_cols"]].describe().T
    stats["skewness"] = df[CFG["feature_cols"]].skew()
    stats["kurtosis"] = df[CFG["feature_cols"]].kurt()
    print("\n      Statistik Deskriptif:")
    print(stats.to_string())
    return stats


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
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
    df_clean = df.copy()
    total_removed = 0
    for col in CFG["feature_cols"]:
        Q1  = df_clean[col].quantile(0.25)
        Q3  = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 3.0 * IQR
        upper = Q3 + 3.0 * IQR
        mask  = (df_clean[col] < lower) | (df_clean[col] > upper)
        n_out = mask.sum()
        if n_out > 0:
            df_clean.loc[mask, col] = np.nan
            total_removed += n_out
    if total_removed:
        print(f"      Outlier ditemukan    : {total_removed:,} titik → diisi interpolasi")
        df_clean = handle_missing_values(df_clean)
    else:
        print("      Tidak ada outlier signifikan.")
    return df_clean


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    n_dup = df.duplicated().sum()
    if n_dup:
        df = df[~df.duplicated()]
        print(f"      Duplikat dihapus     : {n_dup:,} baris")
    return df


def apply_moving_average(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    for col in CFG["feature_cols"]:
        df[col] = (df[col]
                   .rolling(window=window, min_periods=1, center=True)
                   .mean())
    print(f"      Moving Average smoothing diterapkan (window={window})")
    return df


def normalize_features(df: pd.DataFrame):
    scalers = {}
    df_scaled = df.copy()
    for col in CFG["feature_cols"]:
        sc = MinMaxScaler(feature_range=(0, 1))
        df_scaled[col] = sc.fit_transform(df_scaled[[col]])
        scalers[col] = sc
    print("      Min-Max Normalization diterapkan untuk semua fitur.")
    return df_scaled, scalers


def full_preprocessing(path: Path):
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


def pearson_correlation_analysis(df_raw: pd.DataFrame):
    print("\n[2/8] ANALISIS KORELASI PEARSON ...")
    corr_matrix = df_raw[CFG["feature_cols"]].corr(method='pearson')
    print("      Matriks Korelasi Pearson:")
    print(corr_matrix.to_string())
    return corr_matrix


def create_sliding_window(data: np.ndarray, timesteps: int):
    X, y = [], []
    ph_idx = CFG["feature_cols"].index(CFG["target_col"])
    for i in range(timesteps, len(data)):
        X.append(data[i - timesteps: i, :])
        y.append(data[i, ph_idx])
    return np.array(X), np.array(y)


def split_dataset(df_scaled: pd.DataFrame):
    print("\n[3/8] PEMBAGIAN DATASET ...")
    data = df_scaled[CFG["feature_cols"]].values

    n         = len(data)
    n_test    = int(n * CFG["test_size"])
    n_train   = n - n_test

    train_raw = data[:n_train]
    test_raw  = data[n_train - CFG["timesteps"]:]

    X_train_full, y_train_full = create_sliding_window(train_raw, CFG["timesteps"])
    X_test,       y_test       = create_sliding_window(test_raw,  CFG["timesteps"])

    n_val    = int(len(X_train_full) * CFG["val_size"])
    X_val    = X_train_full[-n_val:]
    y_val    = y_train_full[-n_val:]
    X_train  = X_train_full[:-n_val]
    y_train  = y_train_full[:-n_val]

    print(f"      X_train : {X_train.shape}   y_train : {y_train.shape}")
    print(f"      X_val   : {X_val.shape}   y_val   : {y_val.shape}")
    print(f"      X_test  : {X_test.shape}   y_test  : {y_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_lstm_model(units_1=64, units_2=32, dense_units=16,
                     dropout=0.2, lr=0.001) -> Model:
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
    inp  = Input(shape=(CFG["timesteps"], CFG["n_features"]), name="input")
    x    = LSTM(units_1, return_sequences=True,
                activation='tanh', name="lstm_1")(inp)
    x    = Dropout(dropout, name="dropout_1")(x)
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


def train_model(model, model_name, X_train, y_train, X_val, y_val,
                epochs=100, batch_size=64, fold_id=""):
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
        shuffle=False,
    )
    elapsed = (datetime.now() - t0).total_seconds()
    best_ep = np.argmin(history.history['val_loss']) + 1
    best_vl = min(history.history['val_loss'])
    print(f"Selesai {elapsed:.1f}s | best epoch={best_ep} | "
          f"val_loss={best_vl:.6f}")
    return model, history


def hyperparameter_tuning(X_train, y_train, X_val, y_val):
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
                tf.keras.backend.clear_session()

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


def inverse_transform_predictions(y_pred_scaled, y_true_scaled, scalers):
    sc = scalers[CFG["target_col"]]
    y_pred = sc.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = sc.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
    return y_pred, y_true


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    model_name: str = "") -> dict:
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


def plot_time_series_overview(df_raw: pd.DataFrame, stats: pd.DataFrame):
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
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['#2196F3', '#4CAF50', '#9C27B0']

    for ax, (name, preds), color in zip(axes, all_preds.items(), colors):
        y_true = preds["y_true"]
        y_pred = preds["y_pred"]
        ax.scatter(y_true, y_pred, alpha=0.15, s=5,
                   color=color, rasterized=True)

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

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2.,
                    bar.get_height() + max(vals) * 0.02,
                    f'{val:.4f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

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
    path = PLOT_DIR / f"{name}.png"
    fig.savefig(path, dpi=CFG["fig_dpi"], bbox_inches='tight')
    plt.close(fig)
    print(f"      Gambar tersimpan  : plots/{name}.png")


def main():
    start_time = datetime.now()
    print("\n" + "=" * 65)
    print("  MEMULAI PIPELINE PENELITIAN")
    print(f"  Waktu mulai: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    df_raw, df_scaled, scalers, stats = full_preprocessing(DATA_PATH)

    corr_matrix = pearson_correlation_analysis(df_raw)

    print("\n[2/8] MEMBUAT VISUALISASI EDA ...")
    plot_time_series_overview(df_raw, stats)
    plot_correlation_heatmap(corr_matrix)
    plot_distribution(df_raw)

    (X_train, X_val, X_test,
     y_train, y_val, y_test) = split_dataset(df_scaled)

    best_cfg, df_tuning = hyperparameter_tuning(
        X_train, y_train, X_val, y_val
    )
    plot_tuning_heatmap(df_tuning)

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

    model_lstm.save(MODEL_DIR / "final_LSTM.keras")
    model_gru.save(MODEL_DIR  / "final_GRU.keras")
    model_hybrid.save(MODEL_DIR / "final_HybridLSTMGRU.keras")
    print("      Model final tersimpan di outputs/models/")

    print("\n      Ringkasan Arsitektur:")
    for name, model in [("LSTM", model_lstm),
                        ("GRU", model_gru),
                        ("Hybrid LSTM-GRU", model_hybrid)]:
        print(f"\n      -- {name} --")
        model.summary(print_fn=lambda x: print(f"         {x}"))

    histories = {
        "LSTM"           : hist_lstm,
        "GRU"            : hist_gru,
        "Hybrid LSTM-GRU": hist_hybrid,
    }
    plot_training_history(histories)

    models_dict = {
        "LSTM"           : model_lstm,
        "GRU"            : model_gru,
        "Hybrid LSTM-GRU": model_hybrid,
    }
    df_metrics, all_preds = evaluate_all_models(
        models_dict, X_test, y_test, scalers
    )

    print("\n[7/8] MEMBUAT VISUALISASI HASIL ...")
    plot_prediction_comparison(all_preds)
    plot_scatter_actual_vs_pred(all_preds)
    plot_metrics_comparison(df_metrics)
    plot_error_distribution(all_preds)

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


if __name__ == "__main__":
    df_metrics, all_preds, histories = main()
