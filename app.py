import numpy as np
import random 
import os
import json
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from typing import List, Optional

LOOK_BACK = 100 
MODEL_PATH = r'C:\Users\Chandramouli bandaru\OneDrive\Desktop\StockPricePrediction\stockPrice (1).keras'

model = None

app = Flask(__name__)

def load_model_asset():
    """Load the trained Keras model upon startup."""
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Error: Model file '{MODEL_PATH}' not found.")
            model = None
            return

        model = load_model(MODEL_PATH)
        print("✅ Keras Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None

@app.route('/', methods=['GET'])
def index():
    """Renders the HTML form (templates/index.html)."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Please ensure 'lstm_model.h5' exists."}), 500

    try:
        data = request.get_json(force=True)
        
        input_sequence: List[float] = data.get('sequence', [])
        min_limit: Optional[float] = float(data.get('min_limit'))
        max_limit: Optional[float] = float(data.get('max_limit'))

        if min_limit is None or max_limit is None or min_limit >= max_limit:
            return jsonify({"error": "Invalid Min/Max Limits. Ensure Min is less than Max."}), 400
        
        
        if not input_sequence:
            if min_limit >= max_limit:
                 return jsonify({"error": "Cannot auto-generate: Min Limit must be less than Max Limit."}), 400
                 
            input_sequence = [random.uniform(min_limit, max_limit) for _ in range(LOOK_BACK)]
            
        elif len(input_sequence) != LOOK_BACK:
            return jsonify({
                "error": f"Manual input must contain exactly {LOOK_BACK} values. Received {len(input_sequence)}."
            }), 400


        scaler = MinMaxScaler(feature_range=(0, 1))
        dummy_data = np.array([[min_limit], [max_limit]])
        scaler.fit(dummy_data) 
        
        input_array = np.array(input_sequence).astype('float32').reshape(-1, 1)
        scaled_input = scaler.transform(input_array)
        
        reshaped_array = scaled_input.reshape(1, LOOK_BACK, 1)

        scaled_prediction = model.predict(reshaped_array, verbose=0)[0]

        final_prediction = scaler.inverse_transform(scaled_prediction.reshape(-1, 1))[0][0]

        return jsonify({
            "status": "success",
            "predicted_actual_price": float(final_prediction),
            "data_range_used": f"Min: {min_limit}, Max: {max_limit}"
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": f"An unexpected server error occurred. Details: {str(e)}"}), 500

if __name__ == '__main__':
    load_model_asset() 
    
    app.run(host='0.0.0.0', port=5000, debug=True)
