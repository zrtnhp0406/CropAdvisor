import os
import pickle
import numpy as np
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from models.crop_predictor import predict_crop

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key-for-development")

# Load the pre-trained model
model_path = os.path.join(os.path.dirname(__file__), 'models/crop_recommendation_model.pkl')
try:
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logging.info("Model loaded successfully")
    else:
        model = None
        logging.warning(f"Model file not found at {model_path}. Place your pre-trained model here.")
except Exception as e:
    model = None
    logging.error(f"Error loading model: {str(e)}")

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction_result = None
    model_predictions = None
    
    if request.method == 'POST':
        try:
            # Get form data
            n = float(request.form['nitrogen'])
            p = float(request.form['phosphorus'])
            k = float(request.form['potassium'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])
            
            # Store input values in session for display
            session['input_data'] = {
                'nitrogen': n,
                'phosphorus': p,
                'potassium': k,
                'temperature': temperature,
                'humidity': humidity,
                'ph': ph,
                'rainfall': rainfall
            }
            
            # Make prediction using the ensemble
            prediction_result = predict_crop(n, p, k, temperature, humidity, ph, rainfall)
            
            # Get individual model predictions if available
            model_dir = os.path.join(os.path.dirname(__file__), 'models')
            model_predictions = {}
            model_names = ['KNN', 'Random Forest', 'SVM', 'Logistic Regression', 'Decision Tree']
            file_names = ['knn_model.pkl', 'random_forest_model.pkl', 'svm_model.pkl', 
                          'logistic_regression_model.pkl', 'decision_tree_model.pkl']
            
            input_data = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
            
            for idx, file_name in enumerate(file_names):
                model_path = os.path.join(model_dir, file_name)
                if os.path.exists(model_path):
                    try:
                        with open(model_path, 'rb') as f:
                            model_instance = pickle.load(f)
                        pred = model_instance.predict(input_data)[0]
                        model_predictions[model_names[idx]] = pred
                    except Exception as model_err:
                        logging.error(f"Error with model {file_name}: {str(model_err)}")
            
            flash(f'Recommended crop: {prediction_result}', 'success')
                
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            flash(f'Error making prediction: {str(e)}', 'danger')
    
    return render_template('predict.html', prediction=prediction_result, 
                          model_predictions=model_predictions,
                          input_data=session.get('input_data', None))

@app.route('/crops')
def crops():
    return render_template('crops.html')

@app.route('/crop/<crop_name>')
def crop_detail(crop_name):
    return render_template('crop_detail.html', crop_name=crop_name)

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('index.html', error="Page not found"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('index.html', error="Internal server error"), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
