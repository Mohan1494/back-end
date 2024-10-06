from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import re

# Load the vectorizer and the model
try:
    vectorizer = joblib.load('vectorizer.pkl')
    model = joblib.load('model.pkl')
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def preprocess_text(text):
    """Preprocess text by removing unwanted characters."""
    cleaned_text = re.sub('[^அ-ஹஂா-ு-ூெ-ைொ-்]', ' ', text)
    return cleaned_text

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/analyze', methods=['POST'])
def analyze():
    # Get data from the request
    data = request.json
    text = data.get('text')

    if not text:
        return jsonify({'error': 'No input provided'}), 400

    # Preprocess the input
    preprocessed_text = preprocess_text(text)

    # Transform input using the vectorizer
    try:
        transformed_input = vectorizer.transform([preprocessed_text])
    except Exception as e:
        print(f"Error transforming input: {e}")
        return jsonify({'error': 'Transformation failed'}), 500

    # Predict sentiment using the model
    try:
        prediction = model.predict(transformed_input)[0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Prediction failed'}), 500

    # Assuming the model returns classes like 0 (negative), 1 (neutral), 2 (positive)
    sentiment_mapping = {0: 'Not Favorable', 1: 'Neutral', 2: 'Favorable'}
    sentiment = sentiment_mapping.get(prediction, 'Unknown')

    # Return prediction
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
