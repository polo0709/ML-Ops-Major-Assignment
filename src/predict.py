# Import required libraries
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Step 1: Load the trained model
model = joblib.load("model.joblib")

# Step 2: Load the California Housing dataset
data = fetch_california_housing()

# Step 3: Split data (only need test set for inference)
_, X_test, _, y_test = train_test_split(data.data, data.target)

# Step 4: Predict using the loaded model on a few test samples
y_pred = model.predict(X_test[:5])

# Step 5: Print the predictions
print("Predictions:", y_pred)
