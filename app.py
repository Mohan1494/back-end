from flask import Flask, request, jsonify
import joblib
import os

# Load the vectorizer and the model
vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('model.pkl')

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    # Get data from the request
    data = request.json
    text = data.get('text')

    if not text:
        return jsonify({'error': 'No input provided'}), 400

    # Preprocess the input using the vectorizer
    try:
        transformed_input = vectorizer.transform([text])
    except Exception as e:
        return jsonify({'error': f'Error in preprocessing input: {str(e)}'}), 500

    # Predict sentiment using the model
    try:
        prediction = model.predict(transformed_input)
    except Exception as e:
        return jsonify({'error': f'Error in predicting sentiment: {str(e)}'}), 500

    # Assuming the model returns classes like 0 (negative), 1 (neutral), 2 (positive)
    sentiment_classes = {0: 'negative', 1: 'neutral', 2: 'positive'}
    sentiment = sentiment_classes.get(prediction[0], 'unknown')

    # Return prediction
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
