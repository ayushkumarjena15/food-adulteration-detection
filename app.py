from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load dataset and models
data = pd.read_csv("cleaned_food_adulteration_data.csv")
clf = joblib.load("model_class.pkl")
reg = joblib.load("model_reg.pkl")
label_encoder = joblib.load("label_encoder.pkl")
model_columns = joblib.load("model_columns.pkl")

products = sorted(data["product_name"].unique())
brands = sorted(data["brand"].unique())
adulterants = sorted(data["adulterant"].unique())

@app.route('/')
def home():
    return render_template("index.html",
                           products=products,
                           brands=brands,
                           adulterants=adulterants)

@app.route('/predict', methods=["POST"])
def predict():
    product = request.form.get("product_name")
    brand = request.form.get("brand")
    adulterant = request.form.get("adulterant")

    # Prepare input for ML
    input_df = pd.DataFrame([[product, brand, adulterant]],
                            columns=["product_name", "brand", "adulterant"])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Predictions
    class_pred_num = clf.predict(input_df)[0]
    class_pred = label_encoder.inverse_transform([class_pred_num])[0]
    severity_pred = reg.predict(input_df)[0]

    # Lookup action taken
    row = data[(data["product_name"] == product) &
               (data["brand"] == brand) &
               (data["adulterant"] == adulterant)]
    action_taken = "Not found"
    if not row.empty:
        action_taken = row["action_taken"].values[0]

    return render_template("index.html",
                           products=products,
                           brands=brands,
                           adulterants=adulterants,
                           result=True,
                           product=product,
                           brand=brand,
                           adulterant=adulterant,
                           health_risk=class_pred,
                           severity=round(severity_pred, 2),
                           action_taken=action_taken)

if __name__ == "__main__":
    app.run(debug=True)
