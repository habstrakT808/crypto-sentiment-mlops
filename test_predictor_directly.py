#!/usr/bin/env python
"""Test predictor directly"""
import asyncio
from src.serving.predictor import SentimentPredictor

async def test():
    print("Loading predictor...")
    predictor = SentimentPredictor()
    await predictor.load_model()
    
    print(f"Model loaded: {predictor.is_loaded}")
    print(f"Model metadata: {predictor.model_metadata}")
    
    print("\nTesting prediction...")
    result = await predictor.predict_single("Bitcoin is going to the moon!")
    
    print("\nResult:")
    for key, value in result.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    asyncio.run(test())

