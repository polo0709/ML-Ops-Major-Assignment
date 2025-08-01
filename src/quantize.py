import joblib
import numpy as np
import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load trained model
model = joblib.load("model.joblib")
coef = model.coef_
intercept = model.intercept_

# Save unquantized model parameters
unquant_file = "unquant_params.joblib"
joblib.dump({"coef": coef, "intercept": intercept}, unquant_file)

# Evaluate R² score with original model
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)
y_pred = model.predict(X_test)
original_r2 = r2_score(y_test, y_pred)

# Measure original model size
original_size = os.path.getsize(unquant_file) / 1024  # in KB

# -------------------- Quantization: float16 --------------------
quant_coef = coef.astype(np.float16)
quant_intercept = np.float16(intercept)

quant_file = "quant_params.joblib"
joblib.dump({"coef": quant_coef, "intercept": quant_intercept}, quant_file)

# Measure quantized model size
quant_size = os.path.getsize(quant_file) / 1024  # in KB

# -------------------- Dequantization & Inference --------------------
# Cast back to float64 for inference
dequant_coef = quant_coef.astype(np.float64)
dequant_intercept = float(quant_intercept)

# Manual inference
y_pred_quant = np.dot(X_test, dequant_coef) + dequant_intercept
quant_r2 = r2_score(y_test, y_pred_quant)

# -------------------- Reporting --------------------
print(" Quantization Summary (float16):")
print(f" R² Score (Original):     {original_r2:.4f}")
print(f" R² Score (Quantized):    {quant_r2:.4f}")
print(f" Model Size (Original):   {original_size:.2f} KB")
print(f" Model Size (Quantized):  {quant_size:.2f} KB")
print(f" Size Reduction:          {(100 * (original_size - quant_size) / original_size):.2f}%")
