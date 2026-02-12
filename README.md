# Water pH Prediction for Aquaponic Systems Using Hybrid LSTM-GRU

Predicting water pH levels in aquaponic lettuce cultivation systems using deep learning. This project compares three recurrent neural network architectures LSTM, GRU, and a proposed Hybrid LSTM-GRU model to determine the most effective approach for time-series pH forecasting.

## Model Architectures

All three models share the same input shape `(12 timesteps, 3 features)` and predict a single pH value.

| Model | Layer 1 | Layer 2 | Dense | Output |
|---|---|---|---|---|
| **LSTM** | LSTM(64) + Dropout(0.2) | LSTM(32) + Dropout(0.2) | Dense(16, ReLU) | Dense(1, Linear) |
| **GRU** | GRU(64) + Dropout(0.2) | GRU(32) + Dropout(0.2) | Dense(16, ReLU) | Dense(1, Linear) |
| **Hybrid LSTM-GRU** | LSTM(64) + Dropout(0.2) | GRU(32) + Dropout(0.2) | Dense(16, ReLU) | Dense(1, Linear) |

The hybrid model combines LSTM (capturing long-term dependencies) with GRU (efficient short-term pattern recognition) in a sequential architecture.

## Dataset

Sensor readings from an aquaponic lettuce system collected between January–March 2023 (`sources/aquaponic_letuce_dataset_.csv`).

| Feature | Description |
|---|---|
| `water_pH` | Water pH level (target variable) |
| `TDS` | Total Dissolved Solids (ppm) |
| `water_temp` | Water temperature (°C) |

## Pipeline

The end-to-end pipeline runs in 8 steps:

1. **Data Loading** — CSV ingestion with auto timestamp detection
2. **Preprocessing** — Duplicate removal, linear interpolation, IQR outlier detection (3x IQR), moving average smoothing (window=3), Min-Max normalization
3. **Feature Engineering** — Pearson correlation analysis, sliding window transformation (12 steps), sequential train/val/test split (80/10/10, no shuffle)
4. **Hyperparameter Tuning** — Grid search over epochs, learning rate, and hidden units for the Hybrid model
5. **Model Training** — All three models trained with EarlyStopping (patience=15), ReduceLROnPlateau, and ModelCheckpoint callbacks
6. **Evaluation** — Inverse-transformed predictions evaluated with RMSE, MAE, and R²
7. **Visualization** — 9 plots covering EDA, training curves, predictions, and error analysis
8. **Summary Report** — Best model selection based on lowest RMSE

## Getting Started

### Prerequisites

- Python 3.12+
- TensorFlow/Keras
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn

### Installation

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn
```

### Running

```bash
# Run the full pipeline
python water-ph-lstm-gru.py
```

Or open and run the Jupyter notebook:

```bash
jupyter notebook water-ph-lstm-gru.ipynb
```

### Outputs

Results are saved to `outputs/`:

```
outputs/
├── models/          # Trained Keras models (.keras)
├── plots/           # 9 PNG visualizations (150 DPI)
├── logs/            # Per-epoch training history (CSV)
├── evaluation_metrics.csv
└── tuning_results.csv
```

## Evaluation Metrics

| Metric | Description |
|---|---|
| **RMSE** | Root Mean Squared Error — primary model selection criterion |
| **MAE** | Mean Absolute Error |
| **R²** | Coefficient of Determination |

## Visualizations

| # | Plot | Description |
|---|---|---|
| 01 | Time Series Overview | Raw pH, TDS, and temperature trends |
| 02 | Pearson Correlation | Feature correlation heatmap |
| 03 | Distribution | Feature histograms |
| 04 | Training History | Training vs validation loss curves |
| 05 | Prediction Comparison | Actual vs predicted pH (line plot) |
| 06 | Scatter Plot | Actual vs predicted with R² |
| 07 | Metrics Comparison | Bar chart of RMSE, MAE, R² across models |
| 08 | Error Distribution | Residual analysis |
| 09 | Tuning Heatmap | Hyperparameter search results |

## License

This project is licensed under the GNU General Public License v2.0 — see [LICENSE](LICENSE) for details.
