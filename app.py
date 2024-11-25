from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load the trained model (ensure the model.pkl file is in the same directory)
with open('best_catboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML file

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve input data from the form
        data = request.form

        # Convert form inputs to model-compatible format
        features = [    
            float(data['input1']),  # Age
            float(data['input2']),  # Sex
            float(data['input3']),  # Chest Pain Type
            float(data['input4']),  # Resting Blood Pressure
            float(data['input5']),  # Cholesterol
            float(data['input6']),  # Fasting Blood Pressure
            float(data['input7']),  # Resting ECG
            float(data['input8']),  # Maximum Heart Rate
            float(data['input9']),  # Exercise-Induced Angina
            float(data['input10']), # Old Peak
            float(data['input11'])  # ST Slope
        ]

        # Make a prediction
        prediction = model.predict([features])  # Ensure input shape matches your model
        result = "Positive for Heart Disease" if prediction[0] == 1 else "Negative for Heart Disease"

        # Return the result as JSON
        return jsonify({'prediction':result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
