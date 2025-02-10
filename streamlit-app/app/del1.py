import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import duckdb
from scipy.stats import skew, kurtosis

# Apply Seaborn theme
sns.set_style("darkgrid")

# Load pre-trained models
pca = joblib.load('models/pca_model.pkl')
clf = joblib.load('models/camp_model.pkl')


# Function to load and preprocess data using DuckDB
def load_and_preprocess_data(file_path):
    con = duckdb.connect()
    con.execute(f"CREATE TABLE data AS SELECT * FROM read_csv_auto('{file_path}')")
    df = con.execute("SELECT * FROM data").fetchdf()
    vals = [row for _, row in df.iloc[:, 1:].iterrows()]
    X = np.array([
        [s.mean(), s.std(), s.min(), s.max(), s.median(), skew(s), kurtosis(s), np.diff(s).mean()]
        for s in vals
    ])
    con.close()
    return df, X, vals


# Polynomial function for curve fitting
def poly(x, params):
    return sum([params[i] * x ** i for i in range(len(params))])


# Streamlit App UI
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🔋 Battery Cycle Life Prediction</h1>", unsafe_allow_html=True)
st.markdown("#### 📊 Predicting Battery Life using PCA & ML Models", unsafe_allow_html=True)

# Sidebar for file selection
st.sidebar.markdown("### 📂 Data Selection")
data_file = st.sidebar.file_uploader("Upload Battery Data CSV", type=['csv'])

if data_file:
    with st.spinner("Loading data..."):
        df_original, _, _ = load_and_preprocess_data(data_file)
        _, X, vals = load_and_preprocess_data(data_file)

        # Apply PCA and ML model
        X_pca = pca.transform(np.vstack(vals))
        X_combined = np.hstack([X, X_pca])
        y_pred = clf.predict(X_combined)

        # User selection for Cell Number
        cell_number = st.sidebar.slider("Select Cell Number", 0, len(y_pred) - 1, 0)

        # Prediction
        if st.sidebar.button("🔍 Predict"):
            preds = y_pred[cell_number]
            preds_coeff = preds[1:round(preds[0]) + 2]
            y_true1 = np.abs(df_original.iloc[cell_number][1:].dropna())
            x = np.linspace(0, len(y_true1), len(y_true1))
            y_pred1 = np.abs(poly(x, preds_coeff))

            # Plot styling
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x, y_true1, label="Actual Data", color="#007acc", linewidth=2)
            ax.plot(x, y_pred1[:len(y_true1)], label="Predicted Data", color="#FF4B4B", linestyle='--', linewidth=2)
            ax.fill_between(x, y_true1, y_pred1[:len(y_true1)], color='gray', alpha=0.2)
            ax.set_xlabel('Cycle Number', fontsize=12, fontweight='bold')
            ax.set_ylabel('Discharge Capacity', fontsize=12, fontweight='bold')
            ax.set_title(f'Battery Performance Prediction (Cell {cell_number})', fontsize=14, fontweight='bold', color="#333333")
            ax.legend(fontsize=12)
            ax.grid(alpha=0.3)

            # Show plot in Streamlit
            st.pyplot(fig)
