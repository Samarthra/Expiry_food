from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('best_model.pkl')

@app.route('/')
def home():
    return "Welcome to the model prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    
        # Get the input data from the request
        data = request.get_json()
        
        # Ensure input data is provided
        

        # Convert input data to a DataFrame
        input_data = pd.DataFrame(data,index=[0])
        
        # Make prediction using the loaded model
        predicted_days_until_spoilage = model.predict(input_data)
        # Return the prediction as a JSON response
        response= jsonify({'predicted_days_until_spoilage': predicted_days_until_spoilage[0]})
    
   
        return response

if __name__ == '__main__':
    app.run(debug=True)
