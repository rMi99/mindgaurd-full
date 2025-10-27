#!/usr/bin/env python3
"""
Facial Data and Models Download Script for MindGuard
Downloads all required facial datasets and models (FER, DeepFace, Dlib, etc.)
"""

import os
import urllib.request
import zipfile
import bz2
import shutil
from pathlib import Path

# Configuration
DATA_DIR = "backend/data"
MODELS_DIR = os.path.join(DATA_DIR, "models")
CASCADES_DIR = os.path.join(DATA_DIR, "cascades")

# Create directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CASCADES_DIR, exist_ok=True)

print("🎯 MindGuard Facial Data Downloader")
print("=" * 50)

# Models to download
models = {
    "shape_predictor_68_face_landmarks.dat": {
        "url": "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
        "path": os.path.join(MODELS_DIR, "shape_predictor_68_face_landmarks.dat"),
        "compressed": True
    },
    "haarcascade_frontalface_default.xml": {
        "url": "https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml",
        "path": os.path.join(CASCADES_DIR, "haarcascade_frontalface_default.xml"),
        "compressed": False
    },
    "haarcascade_profileface.xml": {
        "url": "https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_profileface.xml",
        "path": os.path.join(CASCADES_DIR, "haarcascade_profileface.xml"),
        "compressed": False
    },
    "haarcascade_eye.xml": {
        "url": "https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_eye.xml",
        "path": os.path.join(CASCADES_DIR, "haarcascade_eye.xml"),
        "compressed": False
    }
}

def download_file(url, target_path, compressed=False):
    """Download a file from URL to target path"""
    if os.path.exists(target_path):
        print(f"✔️ {os.path.basename(target_path)} already exists.")
        return True
    
    try:
        print(f"⬇️ Downloading {os.path.basename(target_path)}...")
        
        # Download to temporary file first
        temp_path = target_path + ".tmp"
        urllib.request.urlretrieve(url, temp_path)
        
        # Handle compressed files
        if compressed and url.endswith('.bz2'):
            print(f"📦 Extracting {os.path.basename(target_path)}...")
            with bz2.BZ2File(temp_path, 'rb') as source:
                with open(target_path, 'wb') as target:
                    shutil.copyfileobj(source, target)
            os.remove(temp_path)
        else:
            # Move temp file to final location
            shutil.move(temp_path, target_path)
        
        print(f"✅ {os.path.basename(target_path)} downloaded successfully.")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download {os.path.basename(target_path)}: {str(e)}")
        # Clean up temp file if it exists
        temp_path = target_path + ".tmp"
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

def download_fer_dataset():
    """Download FER dataset if not present"""
    fer_dir = os.path.join(DATA_DIR, "fer2013")
    if os.path.exists(fer_dir):
        print("✔️ FER2013 dataset already exists.")
        return True
    
    print("📊 FER2013 dataset not found. Please download manually from:")
    print("   https://www.kaggle.com/datasets/msambare/fer2013")
    print("   Extract to: backend/data/fer2013/")
    return False

def download_additional_models():
    """Download additional models that might be needed"""
    additional_models = {
        "mmod_human_face_detector.dat": {
            "url": "http://dlib.net/files/mmod_human_face_detector.dat.bz2",
            "path": os.path.join(MODELS_DIR, "mmod_human_face_detector.dat"),
            "compressed": True
        }
    }
    
    print("\n🔧 Downloading additional models...")
    for name, config in additional_models.items():
        download_file(config["url"], config["path"], config["compressed"])

def main():
    """Main download function"""
    print("🚀 Starting facial data and models download...")
    
    success_count = 0
    total_count = len(models)
    
    # Download main models
    for name, config in models.items():
        if download_file(config["url"], config["path"], config["compressed"]):
            success_count += 1
    
    # Download additional models
    download_additional_models()
    
    # Check for FER dataset
    download_fer_dataset()
    
    print("\n" + "=" * 50)
    print(f"📊 Download Summary:")
    print(f"   ✅ Successfully downloaded: {success_count}/{total_count} models")
    print(f"   📁 Data directory: {os.path.abspath(DATA_DIR)}")
    print(f"   🎯 Models directory: {os.path.abspath(MODELS_DIR)}")
    print(f"   🔍 Cascades directory: {os.path.abspath(CASCADES_DIR)}")
    
    if success_count == total_count:
        print("🎉 All facial data and models ready!")
        return True
    else:
        print("⚠️ Some downloads failed. Check the logs above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)



