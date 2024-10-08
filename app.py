from flask import Flask, request, jsonify
import joblib
import os
import re
from flask_cors import CORS

# Load the vectorizer and the model
vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('model.pkl')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def preprocess_text(text):
    """Preprocess text by removing unwanted characters."""
    cleaned_text = re.sub('[^அ-ஹஂா-ு-ூெ-ைொ-்]', ' ', text)
    return cleaned_text

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get data from the request
        data = request.json
        text = data.get('text')

        if not text:
            return jsonify({'error': 'No input provided'}), 400

        # Preprocess the input
        preprocessed_text = preprocess_text(text)

        # Transform input using the vectorizer
        transformed_input = vectorizer.transform([preprocessed_text])

        # Predict sentiment using the model
        prediction = model.predict(transformed_input)

        # Assuming the model returns classes like 0 (negative), 1 (neutral), 2 (positive)
        sentiment_mapping = {0: 'Not Favorable', 1: 'Neutral', 2: 'Favorable'}
        sentiment = sentiment_mapping.get(prediction[0], 'Unknown')

        # Return prediction
        return jsonify({'sentiment': sentiment})

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
