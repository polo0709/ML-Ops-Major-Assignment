import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from utils import load_data, split_data, evaluate_model
from sklearn.metrics import r2_score, mean_squared_error


def quantize_coefficients_uint8(arr):
    """Min-max quantization of each coefficient using uint8 (asymmetric)."""
    q = np.zeros_like(arr, dtype=np.uint8)
    scales = np.zeros_like(arr, dtype=np.float32)
    mins = np.zeros_like(arr, dtype=np.float32)

    for i, val in enumerate(arr):
        min_val = val  # Scalar, since it's a single coefficient
        max_val = val
        if abs(val) < 1e-8:
            scale = 1.0  # avoid divide-by-zero
        else:
            scale = (max_val - min_val) / 255.0

        q[i] = 0  # constant value, since max == min
        scales[i] = scale
        mins[i] = min_val
    return q, scales, mins


def dequantize_coefficients_uint8(q, scales, mins):
    """Dequantize from uint8."""
    return q.astype(np.float32) * scales + mins


def memory_size(arr):
    return arr.nbytes / 1024


def main():
    # Load model
    model: LinearRegression = joblib.load("model.joblib")
    coef = model.coef_.astype(np.float32)
    intercept = np.array([model.intercept_], dtype=np.float32)

    # Save original params
    joblib.dump({"coef": coef, "intercept": intercept}, "unquant_params.joblib")

    # Quantize coefficients
    q_coef, scales_coef, mins_coef = quantize_coefficients_uint8(coef)
    q_intercept, scales_intercept, mins_intercept = quantize_coefficients_uint8(intercept)

    joblib.dump({
        "q_coef": q_coef,
        "scales_coef": scales_coef,
        "mins_coef": mins_coef,
        "q_intercept": q_intercept,
        "scales_intercept": scales_intercept,
        "mins_intercept": mins_intercept,
    }, "quant_params.joblib")

    # Memory
    orig_mem = memory_size(coef) + memory_size(intercept)
    quant_mem = memory_size(q_coef) + memory_size(q_intercept)
    print("\n✅ Quantization Summary (uint8 per-weight min-max):")
    print(f"Model Size (Original):   {orig_mem:.2f} KB")
    print(f"Model Size (Quantized):  {quant_mem:.2f} KB")
    print(f"Size Reduction:          {(100 * (orig_mem - quant_mem) / orig_mem):.2f}%")

    # Evaluate
    X, y = load_data()
    _, X_test, _, y_test = split_data(X, y)
    r2_orig, mse_orig = evaluate_model(model, X_test, y_test)

    deq_coef = dequantize_coefficients_uint8(q_coef, scales_coef, mins_coef)
    deq_intercept = dequantize_coefficients_uint8(q_intercept, scales_intercept, mins_intercept)[0]
    preds = X_test @ deq_coef + deq_intercept
    r2_q = r2_score(y_test, preds)
    mse_q = mean_squared_error(y_test, preds)

    print("\n📊 Evaluation Metrics:")
    print(f"Original R²:     {r2_orig:.4f}, MSE: {mse_orig:.4f}")
    print(f"Quantized R²:    {r2_q:.4f}, MSE: {mse_q:.4f}")
    print("\nSample quantized coefficients:", q_coef[:5])
    print("Sample predictions:", preds[:5])


if __name__ == "__main__":
    main()
