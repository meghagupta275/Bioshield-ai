import joblib
import os
import numpy as np

def test_model():
    # Check the model path
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'tap_anomaly_model.joblib'))
    print(f"Model path: {model_path}")
    print(f"Model exists: {os.path.exists(model_path)}")
    
    if os.path.exists(model_path):
        try:
            # Load the model
            model = joblib.load(model_path)
            print(f"Model loaded successfully: {type(model)}")
            
            # Check model attributes
            if hasattr(model, 'n_features_in_'):
                print(f"Model expects {model.n_features_in_} features")
            
            # Test with 5 features (what we're sending)
            test_features = [0.5, 0.3, 0.7, 0.1, 2.0]  # 5 features
            print(f"Testing with {len(test_features)} features: {test_features}")
            
            try:
                X = np.array(test_features).reshape(1, -1)
                prediction = model.predict(X)
                print(f"Prediction successful: {prediction}")
            except Exception as e:
                print(f"Prediction failed: {e}")
                
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print("Model file not found!")

if __name__ == "__main__":
    test_model() 