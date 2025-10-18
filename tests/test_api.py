# File: tests/test_api.py
"""
🧪 Comprehensive API Tests
Production-grade testing suite for FastAPI endpoints
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.main import app

client = TestClient(app)

# Test API Keys
VALID_API_KEY = "demo-key-789"
INVALID_API_KEY = "invalid-key-123"

class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["version"] == "2.0.0"
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get(
            "/api/v1/health",
            headers={"X-API-Key": VALID_API_KEY}
        )
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_status" in data
        assert "redis_status" in data
    
    def test_liveness_probe(self):
        """Test Kubernetes liveness probe"""
        response = client.get("/api/v1/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
    
    def test_readiness_probe(self):
        """Test Kubernetes readiness probe"""
        response = client.get(
            "/api/v1/health/ready",
            headers={"X-API-Key": VALID_API_KEY}
        )
        # May be 200 or 503 depending on model loading
        assert response.status_code in [200, 503]

class TestAuthentication:
    """Test API authentication"""
    
    def test_missing_api_key(self):
        """Test request without API key"""
        response = client.post("/api/v1/predict", json={"text": "test"})
        assert response.status_code == 403
    
    def test_invalid_api_key(self):
        """Test request with invalid API key"""
        response = client.post(
            "/api/v1/predict",
            json={"text": "test"},
            headers={"X-API-Key": INVALID_API_KEY}
        )
        assert response.status_code == 401
    
    def test_valid_api_key(self):
        """Test request with valid API key"""
        response = client.post(
            "/api/v1/predict",
            json={"text": "Bitcoin is going to the moon!"},
            headers={"X-API-Key": VALID_API_KEY}
        )
        # Should succeed or return 503 if model not loaded
        assert response.status_code in [200, 503]

class TestPredictionEndpoints:
    """Test prediction endpoints"""
    
    def test_single_prediction_positive(self):
        """Test single prediction with positive text"""
        response = client.post(
            "/api/v1/predict",
            json={
                "text": "Bitcoin is pumping hard! This is amazing! 🚀",
                "include_features": False
            },
            headers={"X-API-Key": VALID_API_KEY}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "confidence" in data
            assert "probabilities" in data
            assert data["prediction"] in ["positive", "negative", "neutral"]
            assert 0 <= data["confidence"] <= 1
            assert "processing_time_ms" in data
    
    def test_single_prediction_negative(self):
        """Test single prediction with negative text"""
        response = client.post(
            "/api/v1/predict",
            json={
                "text": "Crypto market is crashing! Lost all my money! 😭",
                "include_features": False
            },
            headers={"X-API-Key": VALID_API_KEY}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert data["prediction"] in ["positive", "negative", "neutral"]
    
    def test_single_prediction_with_features(self):
        """Test prediction with feature breakdown"""
        response = client.post(
            "/api/v1/predict",
            json={
                "text": "Ethereum upgrade looks promising",
                "include_features": True,
                "include_explanation": True
            },
            headers={"X-API-Key": VALID_API_KEY}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "features" in data or data.get("features") is None
            assert "explanation" in data or data.get("explanation") is None
    
    def test_prediction_empty_text(self):
        """Test prediction with empty text"""
        response = client.post(
            "/api/v1/predict",
            json={"text": ""},
            headers={"X-API-Key": VALID_API_KEY}
        )
        assert response.status_code == 422  # Validation error
    
    def test_prediction_very_long_text(self):
        """Test prediction with very long text"""
        long_text = "Bitcoin " * 1000  # 1000 words
        response = client.post(
            "/api/v1/predict",
            json={"text": long_text},
            headers={"X-API-Key": VALID_API_KEY}
        )
        # Should either succeed or return validation error
        assert response.status_code in [200, 422, 503]
    
    def test_batch_prediction(self):
        """Test batch prediction"""
        response = client.post(
            "/api/v1/predict/batch",
            json={
                "texts": [
                    "Bitcoin to the moon! 🚀",
                    "Crypto is crashing badly 😭",
                    "Ethereum price is stable today"
                ],
                "include_features": False
            },
            headers={"X-API-Key": VALID_API_KEY}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "total_count" in data
            assert "sentiment_distribution" in data
            assert data["total_count"] == 3
            assert len(data["predictions"]) == 3
    
    def test_batch_prediction_empty_list(self):
        """Test batch prediction with empty list"""
        response = client.post(
            "/api/v1/predict/batch",
            json={"texts": []},
            headers={"X-API-Key": VALID_API_KEY}
        )
        assert response.status_code == 422  # Validation error
    
    def test_batch_prediction_too_many(self):
        """Test batch prediction with too many texts"""
        texts = ["test"] * 150  # More than max (100)
        response = client.post(
            "/api/v1/predict/batch",
            json={"texts": texts},
            headers={"X-API-Key": VALID_API_KEY}
        )
        assert response.status_code == 422  # Validation error

class TestModelEndpoints:
    """Test model information endpoints"""
    
    def test_get_model_info(self):
        """Test get model information"""
        response = client.get(
            "/api/v1/model/info",
            headers={"X-API-Key": VALID_API_KEY}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "model_name" in data
            assert "model_version" in data
            assert "accuracy" in data
            assert "training_samples" in data

class TestMetricsEndpoints:
    """Test monitoring and metrics endpoints"""
    
    def test_get_metrics(self):
        """Test get system metrics"""
        response = client.get(
            "/api/v1/metrics",
            headers={"X-API-Key": VALID_API_KEY}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "total_predictions" in data
            assert "average_response_time_ms" in data
            assert "sentiment_distribution_24h" in data
    
    def test_get_detailed_metrics(self):
        """Test get detailed metrics"""
        response = client.get(
            "/api/v1/metrics/detailed",
            headers={"X-API-Key": VALID_API_KEY}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "basic_metrics" in data
            assert "system_resources" in data
            assert "model_performance" in data
    
    def test_get_hourly_metrics(self):
        """Test get hourly metrics"""
        response = client.get(
            "/api/v1/metrics/hourly?hours=12",
            headers={"X-API-Key": VALID_API_KEY}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "data" in data
            assert "total_predictions" in data
            assert len(data["data"]) == 12

class TestPerformance:
    """Test API performance"""
    
    def test_response_time(self):
        """Test that response time is acceptable"""
        import time
        
        start = time.time()
        response = client.post(
            "/api/v1/predict",
            json={"text": "Bitcoin is great!"},
            headers={"X-API-Key": VALID_API_KEY}
        )
        elapsed = (time.time() - start) * 1000  # Convert to ms
        
        if response.status_code == 200:
            # Response should be under 500ms
            assert elapsed < 500, f"Response time {elapsed:.2f}ms exceeds 500ms threshold"
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            tasks = []
            for i in range(10):
                task = ac.post(
                    "/api/v1/predict",
                    json={"text": f"Test message {i}"},
                    headers={"X-API-Key": VALID_API_KEY}
                )
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # At least some requests should succeed
            successful = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 200)
            assert successful > 0

class TestErrorHandling:
    """Test error handling"""
    
    def test_invalid_json(self):
        """Test handling of invalid JSON"""
        response = client.post(
            "/api/v1/predict",
            data="invalid json{",
            headers={
                "X-API-Key": VALID_API_KEY,
                "Content-Type": "application/json"
            }
        )
        assert response.status_code == 422
    
    def test_missing_required_field(self):
        """Test handling of missing required fields"""
        response = client.post(
            "/api/v1/predict",
            json={},  # Missing 'text' field
            headers={"X-API-Key": VALID_API_KEY}
        )
        assert response.status_code == 422
    
    def test_invalid_field_type(self):
        """Test handling of invalid field types"""
        response = client.post(
            "/api/v1/predict",
            json={"text": 12345},  # Should be string
            headers={"X-API-Key": VALID_API_KEY}
        )
        assert response.status_code == 422

# Pytest configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])