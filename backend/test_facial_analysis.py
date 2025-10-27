#!/usr/bin/env python3
"""
Test script for real facial analysis system
"""

import requests
import base64
import cv2
import numpy as np
import io
from PIL import Image

def create_test_image():
    """Create a test image for facial analysis"""
    # Create a simple test image with a face-like pattern
    img = np.ones((200, 200, 3), dtype=np.uint8) * 128
    
    # Add face features
    # Eyes
    cv2.circle(img, (80, 80), 10, (0, 0, 0), -1)
    cv2.circle(img, (120, 80), 10, (0, 0, 0), -1)
    
    # Nose
    cv2.line(img, (100, 100), (100, 130), (0, 0, 0), 2)
    
    # Mouth (smile)
    cv2.ellipse(img, (100, 150), (20, 10), 0, 0, 180, (0, 0, 0), 2)
    
    # Convert to base64
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return img_base64

def test_facial_analysis():
    """Test the facial analysis endpoint"""
    print("🧪 Testing Real Facial Analysis System")
    print("=" * 50)
    
    # Test health endpoint
    print("1. Testing health endpoint...")
    try:
        response = requests.get("http://localhost:8000/api/facial-analysis/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ Health check passed")
            print(f"   📊 Analyzer type: {health_data.get('analyzer_type', 'Unknown')}")
            print(f"   🤖 Model loaded: {health_data.get('real_analyzer', {}).get('model_loaded', False)}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False
    
    # Test emotions endpoint
    print("\n2. Testing emotions endpoint...")
    try:
        response = requests.get("http://localhost:8000/api/facial-analysis/emotions")
        if response.status_code == 200:
            emotions_data = response.json()
            print(f"   ✅ Emotions endpoint working")
            print(f"   😊 Supported emotions: {emotions_data.get('emotions', [])}")
        else:
            print(f"   ❌ Emotions endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Emotions endpoint error: {e}")
    
    # Test facial analysis with test image
    print("\n3. Testing facial analysis with test image...")
    try:
        # Create test image
        test_image = create_test_image()
        
        # Prepare request data
        data = {
            "image": test_image
        }
        
        # Send request
        response = requests.post("http://localhost:8000/api/facial-analysis/", json=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Facial analysis successful")
            print(f"   😊 Detected emotion: {result.get('emotion', 'Unknown')}")
            print(f"   📊 Confidence: {result.get('confidence', 0):.2f}")
            print(f"   👤 Face detected: {result.get('faceDetected', False)}")
            print(f"   🔍 Detection method: {result.get('detection_method', 'Unknown')}")
            print(f"   🤖 Emotion method: {result.get('emotion_method', 'Unknown')}")
        else:
            print(f"   ❌ Facial analysis failed: {response.status_code}")
            print(f"   📝 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Facial analysis error: {e}")
        return False
    
    print("\n🎉 All tests passed! Real facial analysis system is working.")
    return True

if __name__ == "__main__":
    success = test_facial_analysis()
    exit(0 if success else 1)

