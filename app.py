import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the model
MODEL_PATH = 'model (5).pkl'
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# HTML layout template string (No separate HTML file needed)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>XGBoost Model Predictor</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f4f7f6;
            margin: 0;
            padding: 40px 20px;
            display: flex;
            justify-content: center;
        }
        .container {
            background: #ffffff;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            max-width: 500px;
            width: 100%;
        }
        h2 {
            text-align: center;
            color: #333;
            margin-bottom: 20px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #555;
        }
        input[type="number"] {
            width: 100%;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
            box-sizing: border-box;
        }
        button {
            width: 100%;
            padding: 12px;
            background-color: #0070f3;
            border: none;
            color: white;
            font-size: 16px;
            font-weight: bold;
            border-radius: 4px;
            cursor: pointer;
            transition: background 0.3s;
        }
        button:hover {
            background-color: #0051cb;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 4px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
        }
        .success { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    </style>
</head>
<body>

<div class="container">
    <h2>Model Prediction Dashboard</h2>
    <form method="POST" action="/predict">
        {% for i in range(1, 8) %}
        <div class="form-group">
            <label for="feature_{{ i }}">Feature {{ i }}:</label>
            <input type="number" step="any" name="feature_{{ i }}" id="feature_{{ i }}" required>
        </div>
        {% endfor %}
        
        <button type="submit">Predict</button>
    </form>

    {% if prediction_text %}
    <div class="result success">
        {{ prediction_text }}
    </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract inputs from form for the 7 features
        input_features = [float(request.form[f'feature_{i}']) for i in range(1, 8)]
        
        # Convert to numpy array and reshape for prediction
        final_features = [np.array(input_features)]
        prediction = model.predict(final_features)
        
        # Format output
        output = prediction[0]
        prediction_text = f'Predicted Class: {output}'
        
    except Exception as e:
        prediction_text = f'Error in prediction: {str(e)}'

    return render_template_string(HTML_TEMPLATE, prediction_text=prediction_text)

# Essential for Vercel to treat this file correctly as a handler
if __name__ == "__main__":
    app.run(debug=True)
