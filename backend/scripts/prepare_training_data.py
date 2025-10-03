import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import cv2
import librosa
import soundfile as sf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_SAMPLES_PER_CLASS = 50  # Generate 50 samples per emotion class

def create_dummy_facial_data():
    logger.info("Generating facial training data...")
    facial_dir = "data/facial"
    os.makedirs(facial_dir, exist_ok=True)
    
    # Generate dummy facial expression images
    for emotion in EMOTIONS:
        emotion_dir = os.path.join(facial_dir, emotion)
        os.makedirs(emotion_dir, exist_ok=True)
        
        for i in range(NUM_SAMPLES_PER_CLASS):
            # Create a random 48x48 grayscale image
            img = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
            # Add some structure to make it look more like a face
            cv2.circle(img, (24, 24), 20, 255, 2)  # Head outline
            cv2.circle(img, (16, 16), 3, 255, -1)  # Left eye
            cv2.circle(img, (32, 16), 3, 255, -1)  # Right eye
            
            img_path = os.path.join(emotion_dir, f"{emotion}_{i}.png")
            cv2.imwrite(img_path, img)
    
    logger.info(f"Generated {NUM_SAMPLES_PER_CLASS * len(EMOTIONS)} facial images")

def create_dummy_audio_data():
    logger.info("Generating audio training data...")
    audio_dir = "data/audio"
    os.makedirs(audio_dir, exist_ok=True)
    
    # Generate dummy audio samples
    for emotion in EMOTIONS:
        emotion_dir = os.path.join(audio_dir, emotion)
        os.makedirs(emotion_dir, exist_ok=True)
        
        for i in range(NUM_SAMPLES_PER_CLASS):
            # Create a random audio signal
            duration = 3.0  # seconds
            sr = 22050  # sample rate
            t = np.linspace(0, duration, int(sr * duration))
            signal = np.sin(2 * np.pi * 440 * t) * np.random.random()
            
            audio_path = os.path.join(emotion_dir, f"{emotion}_{i}.wav")
            sf.write(audio_path, signal, sr)
    
    logger.info(f"Generated {NUM_SAMPLES_PER_CLASS * len(EMOTIONS)} audio samples")

def create_dummy_multimodal_data():
    logger.info("Generating multimodal training data...")
    multimodal_dir = "data/multimodal"
    os.makedirs(multimodal_dir, exist_ok=True)
    
    # Create a DataFrame to store multimodal features
    data = []
    
    for emotion in EMOTIONS:
        for i in range(NUM_SAMPLES_PER_CLASS):
            # Generate dummy features
            facial_features = np.random.random(128)  # Face embedding
            audio_features = np.random.random(128)   # Audio embedding
            text_features = np.random.random(128)    # Text embedding
            
            data.append({
                'emotion': emotion,
                'facial_features': facial_features.tolist(),
                'audio_features': audio_features.tolist(),
                'text_features': text_features.tolist()
            })
    
    # Convert to DataFrame and shuffle
    df = pd.DataFrame(data)
    df = shuffle(df, random_state=42)
    
    # Save to CSV
    df.to_csv(os.path.join(multimodal_dir, "multimodal_features.csv"), index=False)
    logger.info(f"Generated {len(df)} multimodal samples")

def main():
    logger.info("Starting data preparation...")
    create_dummy_facial_data()
    create_dummy_audio_data()
    create_dummy_multimodal_data()
    logger.info("✅ Data preparation completed!")

if __name__ == "__main__":
    main()