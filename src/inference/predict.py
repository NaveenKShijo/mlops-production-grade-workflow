import joblib
import os
import pandas as pd 


def load_artifacts(model_dir: str):
    """
    Load both the mode and scaler that were saved during training
    """
    model = joblib.load(os.path.join(model_dir, "model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    return model, scaler

def predict(model, scaler, input_data: pd.DataFrame):    
    """
    input_data: a DataFrame with the same columns as X_train (after encoding).
    The scaler transforms it the same way training data was scaled.
    """
    input_scaled = scaler.transform(input_data)
    return model.predict(input_scaled).tolist()

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.getenv("MODEL_DIR", os.path.join(BASE_DIR, "models"))
    model, scaler = load_artifacts(model_dir)

    sample = pd.DataFrame([{"age": 25, "bmi": 27.5, "children": 0,
        "sex_male": 1, "smoker_yes": 0,
        "region_northwest": 0, "region_southeast": 1, "region_southwest": 0}])
    result = predict(model, scaler, sample)
    print(result)