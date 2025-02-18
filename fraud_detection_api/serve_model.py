from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained models
model_fraud_detection = joblib.load('Fraud__Detection_RandomForest_best_model.pkl')
model_credit_card_fraud = joblib.load('Credit_Card_Fraud_RandomForest_best_model.pkl')

# API endpoint for fraud detection model
@app.route('/predict_fraud', methods=['POST'])
def predict_fraud():
    data = request.get_json()
    print("Received data for fraud detection:", data)  # Debug statement
    if not data or 'features' not in data:
        return jsonify({"error": "Invalid input: 'features' key is missing"}), 400
    prediction = model_fraud_detection.predict([data['features']])
    return jsonify({"prediction": prediction[0]})

# API endpoint for credit card fraud detection model
@app.route('/predict_credit_card_fraud', methods=['POST'])
def predict_credit_card_fraud():
    data = request.get_json()
    print("Received data for credit card fraud detection:", data)  # Debug statement
    if not data or 'features' not in data:
        return jsonify({"error": "Invalid input: 'features' key is missing"}), 400
    prediction = model_credit_card_fraud.predict([data['features']])
    return jsonify({"prediction": prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)