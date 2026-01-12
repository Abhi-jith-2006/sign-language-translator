import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

# Load data
df = pd.read_csv("data/gestures.csv")

X = df.drop("label", axis=1)
y = df["label"]

# Train / test split (basic sanity check)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")

# Save model
os.makedirs("model", exist_ok=True)
with open("model/gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved to model/gesture_model.pkl")
