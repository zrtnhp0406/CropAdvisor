import os
import pickle
import numpy as np
import logging
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def create_models():
    """
    Creates five different machine learning models for crop prediction:
    - KNN (K-Nearest Neighbors)
    - Random Forest
    - SVM (Support Vector Machine)
    - Logistic Regression
    - Decision Tree
    
    Returns:
    - Dictionary of initialized models
    """
    models = {
        'knn': KNeighborsClassifier(n_neighbors=5),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'svm': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True),
        'logistic_regression': LogisticRegression(max_iter=1000, multi_class='multinomial'),
        'decision_tree': DecisionTreeClassifier(random_state=42)
    }
    return models

def predict_crop(n, p, k, temperature, humidity, ph, rainfall):
    """
    Advanced prediction function that uses multiple models to recommend crops
    based on soil and environmental parameters.
    
    Uses a voting ensemble of 5 models:
    - KNN
    - Random Forest
    - SVM
    - Logistic Regression
    - Decision Tree
    
    The final prediction is based on the most frequent prediction among all models.
    If models can't be loaded, falls back to a rule-based system.
    
    Parameters:
    - n: Nitrogen content in soil
    - p: Phosphorus content in soil
    - k: Potassium content in soil
    - temperature: Temperature in Celsius
    - humidity: Humidity percentage
    - ph: pH value of soil
    - rainfall: Rainfall in mm
    
    Returns:
    - String containing the recommended crop
    """
    try:
        # Try to load pre-trained models
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        input_data = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
        
        # Check if we have pre-trained models
        models_exist = all(os.path.exists(os.path.join(model_dir, f"{model_name}_model.pkl")) 
                           for model_name in ['knn', 'random_forest', 'svm', 'logistic_regression', 'decision_tree'])
        
        if models_exist:
            # Load all models and make predictions
            predictions = []
            for model_name in ['knn', 'random_forest', 'svm', 'logistic_regression', 'decision_tree']:
                model_path = os.path.join(model_dir, f"{model_name}_model.pkl")
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                pred = model.predict(input_data)[0]
                predictions.append(pred)
                logging.info(f"{model_name.upper()} prediction: {pred}")
            
            # Get the most common prediction (voting)
            prediction_counts = Counter(predictions)
            most_common_prediction = prediction_counts.most_common(1)[0][0]
            
            logging.info(f"Final ensemble prediction: {most_common_prediction}")
            return most_common_prediction
        
        # Try to load single pre-trained model as fallback
        model_path = os.path.join(model_dir, 'crop_recommendation_model.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Use the model for prediction
            prediction = model.predict(input_data)[0]
            logging.info(f"Using single model prediction: {prediction}")
            return prediction
        
        # If models not found, use basic rule-based system
        logging.warning("Using fallback prediction rules")
        
        # Simple rule-based system (this is simplified and not as accurate as a trained model)
        # These rules are just examples and should be replaced with actual agricultural knowledge
        if ph < 5.5:
            if rainfall > 200 and temperature > 25:
                return "rice"
            else:
                return "blueberry"
        elif ph < 6.5:
            if k > 40 and rainfall < 100:
                return "maize"
            else:
                return "wheat"
        elif ph < 7.5:
            if n > 40 and p > 40:
                return "cotton"
            else:
                return "sunflower"
        else:  # pH >= 7.5
            if rainfall > 200:
                return "sugarcane"
            else:
                return "chickpea"
                
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        # If all else fails, return a general recommendation
        return "Unable to make prediction. Please check your input values."
