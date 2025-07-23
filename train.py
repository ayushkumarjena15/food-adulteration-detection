import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
path = "cleaned_food_adulteration_data.csv"
data = pd.read_csv(path)

# Convert severity to numeric for regression
severity_map = {"Minor": 1, "Moderate": 2, "Severe": 3}
data["severity_numeric"] = data["severity"].map(severity_map)

# Use only these 3 columns as features
X = data[["product_name", "brand", "adulterant"]]
X = pd.get_dummies(X)

# Targets
y_class = data["health_risk"]
y_reg = data["severity_numeric"]

# Encode classification target
label_encoder = LabelEncoder()
y_class_encoded = label_encoder.fit_transform(y_class)

# Split dataset
X_train, X_test, y_class_train, y_class_test = train_test_split(
    X, y_class_encoded, test_size=0.2, random_state=42)
_, _, y_reg_train, y_reg_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42)

# Train models
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_class_train)

reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, y_reg_train)

# Save models
joblib.dump(clf, "model_class.pkl")
joblib.dump(reg, "model_reg.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(list(X.columns), "model_columns.pkl")

print("✅ Models trained with product, brand & adulterant only!")
