import numpy as np
from src.training.train import preprocess, train_model, evaluate_model

def test_preprocess(dummy_insurance_data):
    # Act
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess(dummy_insurance_data)
    
    # Assert
    assert X_train_scaled is not None
    assert len(y_train) + len(y_test) == len(dummy_insurance_data)
    
    # Check if categorical columns were one-hot encoded (we expect more columns now)
    original_cols = len(dummy_insurance_data.columns) - 1 # excluding target
    assert X_train_scaled.shape[1] > original_cols

def test_train_model():
    # Arrange: Create dummy scaled data
    X_train = np.random.rand(10, 5)
    y_train = np.random.rand(10)
    
    # Act
    model = train_model(X_train, y_train)
    
    # Assert
    assert model is not None
    assert hasattr(model, "predict")  # It should have a predict method

def test_evaluate_model():
    # Arrange: Mock a model that always predicts exactly the y_test values (perfect score)
    class PerfectMockModel:
        def predict(self, X):
            return np.array([10.0, 20.0, 30.0])
            
    X_test = np.zeros((3, 5))
    y_test = np.array([10.0, 20.0, 30.0])
    
    # Act
    metrics = evaluate_model(PerfectMockModel(), X_test, y_test)
    
    # Assert
    assert "mae" in metrics
    assert "r2_score" in metrics
    assert metrics["mae"] == 0.0      # Perfect prediction means 0 error
    assert metrics["r2_score"] == 1.0 # Perfect prediction means 1.0 R2
