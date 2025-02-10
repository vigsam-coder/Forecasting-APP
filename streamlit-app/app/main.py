import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import duckdb
from scipy.stats import skew, kurtosis
import seaborn as sns

pca = joblib.load('models/pca_model.pkl')
clf = joblib.load('models/camp_model.pkl')

sns.set_style("darkgrid")

def load_and_preprocess_data(file_path):
    con = duckdb.connect()

    con.execute(f"""
        CREATE TABLE data AS SELECT * FROM read_csv_auto('{file_path}')
    """)

    query = "SELECT * FROM data"
    df = con.execute(query).fetchdf()

    vals = [row for _, row in df.iloc[:, 1:].iterrows()]

    X = np.array([
        [s.mean(), s.std(), s.min(), s.max(), s.median(), skew(s), kurtosis(s), np.diff(s).mean()]
        for s in vals
    ])

    con.close()  # Close DuckDB connection
    return df, X, vals


# Define polynomial function for curve fitting
def poly(x, params):
    return sum([params[i] * x ** i for i in range(len(params))])

st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🔋 CAMP Model Inference From Database</h1>", unsafe_allow_html=True)
st.markdown("#### 📊 Forecasting Battery Cycle Life", unsafe_allow_html=True)



# Load data directly from database CSV files
_, X, vals = load_and_preprocess_data('database/smooth_data.csv')
df_original, _, _ = load_and_preprocess_data('database/original_data.csv')

# Apply PCA and ML model
X_pca = pca.transform(np.vstack(vals))
X_combined = np.hstack([X, X_pca])
y_pred = clf.predict(X_combined)

# User input for index selection
cell_number = st.number_input("Enter Cell Number", min_value=0, max_value=len(y_pred) - 1, step=1)

if st.button("Predict"):
    # Get prediction for user-selected cell number
    preds = y_pred[cell_number]
    preds_coeff = preds[1:round(preds[0]) + 2]

    # Extract true values
    y_true1 = np.abs(df_original.iloc[cell_number][1:].dropna())

    # Define x-axis
    x = np.linspace(0, len(y_true1), len(y_true1))

    # Calculate predicted values
    y_pred1 = np.abs(poly(x, preds_coeff))

    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y_true1, label="True Values", color='blue')
    ax.plot(x, y_pred1[:len(y_true1)], label="Predicted Values", color='red', linestyle='--')
    ax.set_xlabel('Cycle Number')
    ax.set_ylabel('Discharge Capacity')
    ax.set_title(f'True vs Predicted Curve (Cell {cell_number})')
    ax.legend()
    ax.grid(True)

    # Show the plot in Streamlit
    st.pyplot(fig)
