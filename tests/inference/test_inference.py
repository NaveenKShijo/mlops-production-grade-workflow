import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

# We must patch load_artifacts BEFORE importing the FastAPI app, 
# because main.py calls load_artifacts at the global level during import.
@pytest.fixture
def client(mock_trained_model):
    with patch("src.inference.main.load_artifacts", return_value=mock_trained_model):
        from src.inference.main import app
        return TestClient(app)

def test_ping_endpoint(client):
    # Act
    response = client.get("/ping")
    
    # Assert
    assert response.status_code == 200
    assert response.json() == {"status": "Healthy"}

def test_invocations_endpoint(client):
    # Arrange: This is the raw JSON payload structure SageMaker will send
    payload = {
        "age": 25, 
        "bmi": 27.5, 
        "children": 0,
        "sex_male": 1, 
        "smoker_yes": 0,
        "region_northwest": 0, 
        "region_southeast": 1, 
        "region_southwest": 0
    }
    
    # Act
    response = client.post("/invocations", json=payload)
    
    # Assert
    assert response.status_code == 200
    # It should return a list of predictions
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert isinstance(data[0], float)
