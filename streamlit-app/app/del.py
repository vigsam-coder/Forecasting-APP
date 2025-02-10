import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import duckdb
from scipy.stats import skew, kurtosis

file_path = 'database/original_data.csv'

# Function to load and preprocess data using DuckDB
def load_and_preprocess_data(file_path):
    # Connect to DuckDB (in-memory)
    con = duckdb.connect()

    # Load CSV into DuckDB
    con.execute(f"""
        CREATE TABLE data AS SELECT * FROM read_csv_auto('{file_path}')
    """)

    # Fetch data as a dataframe
    query = "SELECT * FROM data"
    df = con.execute(query).fetchdf()

    # Extract features for ML model
    vals = [row for _, row in df.iloc[:, 1:].iterrows()]

    X = np.array([
        [s.mean(), s.std(), s.min(), s.max(), s.median(), skew(s), kurtosis(s), np.diff(s).mean()]
        for s in vals
    ])

    con.close()  # Close DuckDB connection
    return df, X, vals

df_original, X, vals = load_and_preprocess_data('database/smooth_data.csv')

pca = joblib.load('models/pca_model.pkl')
clf = joblib.load('models/camp_model.pkl')

def poly(x, params):
    return sum([params[i] * x ** i for i in range(len(params))])

X_pca = pca.transform(np.vstack(vals))
X_combined = np.hstack([X, X_pca])
y_pred = clf.predict(X_combined)

cell_number = 10

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
plt.imshow(y_pred1.reshape((len(y_true1), 1)), aspect='auto', cmap='hot')
plt.colorbar()
