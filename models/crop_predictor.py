import os
import pickle
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# This is a fallback prediction function that will be used if the model fails to load
def predict_crop(n, p, k, temperature, humidity, ph, rainfall):
    """
    Fallback prediction function that uses simple rules to recommend crops
    when the pre-trained model is not available.
    
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
        # Try to load model
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models/crop_recommendation_model.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Use the model for prediction
            input_data = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
            prediction = model.predict(input_data)[0]
            return prediction
        
        # If model not found, use basic rule-based system
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
