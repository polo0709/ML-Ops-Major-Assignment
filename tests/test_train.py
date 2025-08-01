import os
import joblib
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Global fixture to load model and test data only once
@pytest.fixture(scope="module")
def model_and_data():
    assert os.path.exists("model.joblib"), "model.joblib not found. Run train.py first."
    model = joblib.load("model.joblib")
    data = fetch_california_housing()
    _, X_test, _, y_test = train_test_split(data.data, data.target)
    return model, X_test, y_test

# Test 1: Ensure model file was created
def test_model_file_exists():
    assert os.path.exists("model.joblib"), "model.joblib was not saved."

# Test 2: Check if model is an instance of LinearRegression
def test_model_type(model_and_data):
    model, _, _ = model_and_data
    assert isinstance(model, LinearRegression), "Model is not a LinearRegression instance."

# Test 3: Ensure the model was trained (i.e., it has coefficients)
def test_model_is_trained(model_and_data):
    model, _, _ = model_and_data
    assert hasattr(model, "coef_"), "Trained model does not have coef_ attribute."

# Test 4: Validate that model performs above threshold R² score
def test_r2_score_threshold(model_and_data):
    model, X_test, y_test = model_and_data
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    assert r2 > 0.5, f"R² score too low: {r2:.4f}"
