import numpy as np
import random # Used for generating random sequence when the user chooses "Auto-Generate"
import os
import json
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from typing import List, Optional

# --- Configuration ---
LOOK_BACK = 100 # Sequence length required by the LSTM model
MODEL_PATH = r'C:\Users\Chandramouli bandaru\OneDrive\Desktop\StockPricePrediction\stockPrice (1).keras'

# --- Global Variables ---
model = None

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Helper Function to Load Model ---
def load_model_asset():
    """Load the trained Keras model upon startup."""
    global model
    try:
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Error: Model file '{MODEL_PATH}' not found.")
            # We don't raise an error to let the server start, but it won't predict
            model = None
            return

        # Load the Keras Model (LSTM/GRU)
        model = load_model(MODEL_PATH)
        print("✅ Keras Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None

# --- Route to Render the Input Form ---
@app.route('/', methods=['GET'])
def index():
    """Renders the HTML form (templates/index.html)."""
    # Ensure your HTML file is in a folder named 'templates'
    return render_template('index.html')

# --- Prediction API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Please ensure 'lstm_model.h5' exists."}), 500

    try:
        data = request.get_json(force=True)
        
        # Get inputs
        input_sequence: List[float] = data.get('sequence', [])
        min_limit: Optional[float] = float(data.get('min_limit'))
        max_limit: Optional[float] = float(data.get('max_limit'))

        # Input Validation for Limits
        if min_limit is None or max_limit is None or min_limit >= max_limit:
            return jsonify({"error": "Invalid Min/Max Limits. Ensure Min is less than Max."}), 400
        
        # --- CORE LOGIC: Handle Auto-Generation vs. Manual Input ---
        
        if not input_sequence:
            # SCENARIO 1: Auto-Generate 100 values
            if min_limit >= max_limit: # Should have been caught above, but safety check
                 return jsonify({"error": "Cannot auto-generate: Min Limit must be less than Max Limit."}), 400
                 
            # Generate 100 random values within the user-defined range
            input_sequence = [random.uniform(min_limit, max_limit) for _ in range(LOOK_BACK)]
            
        elif len(input_sequence) != LOOK_BACK:
            # SCENARIO 2: Manual input provided but was the wrong length
            return jsonify({
                "error": f"Manual input must contain exactly {LOOK_BACK} values. Received {len(input_sequence)}."
            }), 400

        # --- Prediction Pipeline ---

        # 1. MinMaxScaler: Dynamic creation based on user limits
        scaler = MinMaxScaler(feature_range=(0, 1))
        # Fit the scaler using the user-provided min/max bounds
        dummy_data = np.array([[min_limit], [max_limit]])
        scaler.fit(dummy_data) 
        
        # 2. Scale the input sequence (from raw price to 0-1)
        input_array = np.array(input_sequence).astype('float32').reshape(-1, 1)
        scaled_input = scaler.transform(input_array)
        
        # 3. Reshape for LSTM: (1, 100, 1) -> (num_samples, time_steps, features)
        reshaped_array = scaled_input.reshape(1, LOOK_BACK, 1)

        # 4. Predict
        # We use verbose=0 to suppress output in the server log
        scaled_prediction = model.predict(reshaped_array, verbose=0)[0]

        # 5. Inverse Transform (from 0-1 back to price)
        # Note: Scaler requires a 2D array for inverse_transform
        final_prediction = scaler.inverse_transform(scaled_prediction.reshape(-1, 1))[0][0]

        # 6. Return the result
        return jsonify({
            "status": "success",
            "predicted_actual_price": float(final_prediction),
            "data_range_used": f"Min: {min_limit}, Max: {max_limit}"
        })

    except Exception as e:
        # Catch any unexpected server errors
        print(f"Prediction Error: {e}")
        return jsonify({"error": f"An unexpected server error occurred. Details: {str(e)}"}), 500

# --- Main Run Block ---
if __name__ == '__main__':
    # Load model once when the application starts
    load_model_asset() 
    
    # Run the application (debug=True is great for development)
    app.run(host='0.0.0.0', port=5000, debug=True)