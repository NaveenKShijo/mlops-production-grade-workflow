# In Pytest, conftest.py is a special file used to share fixtures (setup code or dummy data) across multiple test files without needing to explicitly import them.

# For example, both your training tests and inference tests might need a dummy Pandas DataFrame to test preprocessing and predictions. Instead of creating that DataFrame in every test file, you create it once in conftest.py, and Pytest automatically makes it available to any test function that asks for it.


import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer

@pytest.fixture
def dummy_insurance_data():
    """Creates a small mock dataframe that mimics the real insurance.csv"""
    data = {
        'age': [19, 18, 28, 33, 32],
        'sex': ['female', 'male', 'male', 'male', 'male'],
        'bmi': [27.9, 33.77, 33.0, 22.705, 28.88],
        'children': [0, 1, 3, 0, 0],
        'smoker': ['yes', 'no', 'no', 'no', 'no'],
        'region': ['southwest', 'southeast', 'southeast', 'northwest', 'northwest'],
        'charges': [16884.924, 1725.5523, 4449.462, 21984.47061, 3866.8552]
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_trained_model():
    """Creates a dummy trained model and scaler for testing inference"""
    model = LinearRegression()
    # Mocking that the model was already trained on 8 features
    model.coef_ = np.array([1.5, 2.0, 0.5, 3.1, -1.2, 0.8, -0.4, 0.1])
    model.intercept_ = 10.0
    
    # Mocking a fitted column transformer
    continuous_cols = ['age', 'bmi', 'children']
    ct = ColumnTransformer([('scale', MinMaxScaler(), continuous_cols)], remainder='passthrough')
    # Fit it on some random data to initialize it
    dummy_x = pd.DataFrame(np.random.rand(5, 3), columns=continuous_cols)
    ct.fit(dummy_x)
    
    return model, ct
