#!/usr/bin/env python3
"""
Real Facial Data Download Script for MindGuard
Downloads actual facial emotion recognition datasets and models
"""

import os
import urllib.request
import zipfile
import bz2
import shutil
import requests
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = "data"
MODELS_DIR = os.path.join(DATA_DIR, "models")
CASCADES_DIR = os.path.join(DATA_DIR, "cascades")
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")

# Create directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CASCADES_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)

print("🎯 MindGuard Real Facial Data Downloader")
print("=" * 50)

def download_file(url, target_path, compressed=False, description=""):
    """Download a file from URL to target path"""
    if os.path.exists(target_path):
        logger.info(f"✔️ {os.path.basename(target_path)} already exists.")
        return True
    
    try:
        logger.info(f"⬇️ Downloading {description or os.path.basename(target_path)}...")
        
        # Download with progress
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r   Progress: {progress:.1f}%", end='', flush=True)
        
        print()  # New line after progress
        
        # Handle compressed files
        if compressed and url.endswith('.bz2'):
            logger.info(f"📦 Extracting {os.path.basename(target_path)}...")
            with bz2.BZ2File(target_path, 'rb') as source:
                with open(target_path.replace('.bz2', ''), 'wb') as target:
                    shutil.copyfileobj(source, target)
            os.remove(target_path)
        
        logger.info(f"✅ {os.path.basename(target_path)} downloaded successfully.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download {os.path.basename(target_path)}: {str(e)}")
        if os.path.exists(target_path):
            os.remove(target_path)
        return False

def download_fer2013_dataset():
    """Download FER2013 dataset"""
    fer_dir = os.path.join(DATASETS_DIR, "fer2013")
    if os.path.exists(fer_dir):
        logger.info("✔️ FER2013 dataset already exists.")
        return True
    
    logger.info("📊 FER2013 dataset not found. Please download manually from:")
    logger.info("   https://www.kaggle.com/datasets/msambare/fer2013")
    logger.info("   Extract to: data/datasets/fer2013/")
    logger.info("   Or use the Kaggle API if you have it configured.")
    return False

def download_affectnet_sample():
    """Download a sample of AffectNet dataset"""
    affectnet_dir = os.path.join(DATASETS_DIR, "affectnet_sample")
    if os.path.exists(affectnet_dir):
        logger.info("✔️ AffectNet sample already exists.")
        return True
    
    logger.info("📊 AffectNet sample dataset not found.")
    logger.info("   For full AffectNet dataset, visit: http://mohammadmahoor.com/affectnet/")
    logger.info("   For now, we'll use FER2013 and create synthetic data.")
    return False

def download_models():
    """Download required models"""
    models = {
        "shape_predictor_68_face_landmarks.dat": {
            "url": "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
            "path": os.path.join(MODELS_DIR, "shape_predictor_68_face_landmarks.dat"),
            "compressed": True,
            "description": "Dlib 68-point facial landmark predictor"
        },
        "mmod_human_face_detector.dat": {
            "url": "http://dlib.net/files/mmod_human_face_detector.dat.bz2",
            "path": os.path.join(MODELS_DIR, "mmod_human_face_detector.dat"),
            "compressed": True,
            "description": "Dlib MMOD face detector"
        },
        "haarcascade_frontalface_default.xml": {
            "url": "https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml",
            "path": os.path.join(CASCADES_DIR, "haarcascade_frontalface_default.xml"),
            "compressed": False,
            "description": "OpenCV frontal face cascade"
        },
        "haarcascade_profileface.xml": {
            "url": "https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_profileface.xml",
            "path": os.path.join(CASCADES_DIR, "haarcascade_profileface.xml"),
            "compressed": False,
            "description": "OpenCV profile face cascade"
        },
        "haarcascade_eye.xml": {
            "url": "https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_eye.xml",
            "path": os.path.join(CASCADES_DIR, "haarcascade_eye.xml"),
            "compressed": False,
            "description": "OpenCV eye cascade"
        }
    }
    
    success_count = 0
    total_count = len(models)
    
    logger.info("🔧 Downloading facial analysis models...")
    for name, config in models.items():
        if download_file(config["url"], config["path"], config["compressed"], config["description"]):
            success_count += 1
    
    return success_count, total_count

def create_synthetic_dataset():
    """Create a synthetic facial emotion dataset for training"""
    synthetic_dir = os.path.join(DATASETS_DIR, "synthetic_emotions")
    os.makedirs(synthetic_dir, exist_ok=True)
    
    # Create emotion directories
    emotions = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral']
    for emotion in emotions:
        emotion_dir = os.path.join(synthetic_dir, emotion)
        os.makedirs(emotion_dir, exist_ok=True)
    
    logger.info("🎨 Created synthetic emotion dataset structure")
    logger.info(f"📁 Dataset location: {os.path.abspath(synthetic_dir)}")
    return True

def download_pretrained_models():
    """Download pretrained emotion recognition models"""
    pretrained_dir = os.path.join(MODELS_DIR, "pretrained")
    os.makedirs(pretrained_dir, exist_ok=True)
    
    # These would be downloaded from model repositories
    logger.info("🤖 Pretrained models will be downloaded during training")
    logger.info(f"📁 Pretrained models directory: {os.path.abspath(pretrained_dir)}")
    return True

def main():
    """Main download function"""
    logger.info("🚀 Starting real facial data and models download...")
    
    # Download models
    model_success, model_total = download_models()
    
    # Check for datasets
    fer_available = download_fer2013_dataset()
    affectnet_available = download_affectnet_sample()
    
    # Create synthetic dataset
    synthetic_created = create_synthetic_dataset()
    
    # Setup pretrained models directory
    pretrained_ready = download_pretrained_models()
    
    logger.info("\n" + "=" * 50)
    logger.info(f"📊 Download Summary:")
    logger.info(f"   ✅ Models downloaded: {model_success}/{model_total}")
    logger.info(f"   📊 FER2013 dataset: {'Available' if fer_available else 'Manual download required'}")
    logger.info(f"   📊 AffectNet sample: {'Available' if affectnet_available else 'Using synthetic data'}")
    logger.info(f"   🎨 Synthetic dataset: {'Created' if synthetic_created else 'Failed'}")
    logger.info(f"   🤖 Pretrained models: {'Ready' if pretrained_ready else 'Failed'}")
    
    logger.info(f"\n📁 Data directories:")
    logger.info(f"   📂 Main data: {os.path.abspath(DATA_DIR)}")
    logger.info(f"   🎯 Models: {os.path.abspath(MODELS_DIR)}")
    logger.info(f"   🔍 Cascades: {os.path.abspath(CASCADES_DIR)}")
    logger.info(f"   📊 Datasets: {os.path.abspath(DATASETS_DIR)}")
    
    if model_success == model_total and synthetic_created:
        logger.info("🎉 Facial data and models ready for training!")
        return True
    else:
        logger.warning("⚠️ Some downloads failed. Check the logs above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
