import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
import os

# Get the folder where train_model.py is located
current_folder = os.path.dirname(os.path.abspath(__file__))

# Full path to the CSV file
csv_path = os.path.join(current_folder, "Crop_recommendation.csv")

print("CSV Path:", csv_path)   # This is for checking

data = pd.read_csv(csv_path)

# Features and Target
X = data.drop("label", axis=1)
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
prediction = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, prediction)
print("Model Accuracy:", accuracy)

# Save model
joblib.dump(model, "crop_model.pkl")

print("Model saved as crop_model.pkl")