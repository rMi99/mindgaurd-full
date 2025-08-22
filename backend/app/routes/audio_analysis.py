from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import base64
import io
import os
import tempfile
import logging

import numpy as np
import librosa

# Whisper can be heavy; load lazily
_whisper_model = None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/audio-analysis", tags=["audio-analysis"])


class AudioChunk(BaseModel):
    audio: str  # base64 data URL or raw base64
    contentType: Optional[str] = None  # e.g., "audio/webm", "audio/wav"
    sampleRate: Optional[int] = None


class AudioEmotionResult(BaseModel):
    transcript: Optional[str] = None
    language: Optional[str] = None
    confidence: Optional[float] = None
    metrics: Dict[str, float]
    emotions: Dict[str, float]


def _decode_audio_to_wav_bytes(audio_b64: str) -> bytes:
    try:
        if "," in audio_b64:
            audio_b64 = audio_b64.split(",", 1)[1]
        return base64.b64decode(audio_b64)
    except Exception as e:
        logger.error(f"Failed to decode audio bytes: {e}")
        raise HTTPException(status_code=400, detail="Invalid audio data")


def _extract_audio_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    # Basic speech prosody features
    rms = librosa.feature.rms(y=y).mean()
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()

    # Pitch estimation (can be noisy)
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=400, sr=sr)
        pitch = np.nanmean(f0)
        voicing = np.nanmean(voiced_probs)
    except Exception:
        pitch = float("nan")
        voicing = 0.0

    metrics = {
        "rms": float(rms),
        "zcr": float(zcr),
        "mfcc1": float(mfcc[0]) if len(mfcc) > 0 else 0.0,
        "spectral_centroid": float(spectral_centroid),
        "spectral_bandwidth": float(spectral_bandwidth),
        "pitch_hz": float(pitch) if np.isfinite(pitch) else 0.0,
        "voicing": float(voicing),
    }
    return metrics


def _heuristic_emotion_from_features(metrics: Dict[str, float]) -> Dict[str, float]:
    # Simple arousal/valence heuristics
    arousal = 0.0
    arousal += np.clip(metrics.get("rms", 0) * 10, 0, 1)
    arousal += np.clip(metrics.get("zcr", 0) * 2, 0, 1)
    arousal += np.clip(metrics.get("spectral_centroid", 0) / 4000, 0, 1)
    arousal = float(np.clip(arousal / 3, 0, 1))

    valence = 0.5
    valence += np.clip(metrics.get("mfcc1", 0) / 100, -0.5, 0.5)
    valence = float(np.clip(valence, 0, 1))

    # Map to emotions (toy mapping)
    emotions = {
        "happy": max(0.0, min(1.0, 0.6 * valence + 0.4 * arousal)),
        "sad": max(0.0, min(1.0, 1.0 - valence)),
        "angry": max(0.0, min(1.0, arousal * (1.0 - valence * 0.5))),
        "neutral": max(0.0, min(1.0, 1.0 - abs(valence - 0.5) - abs(arousal - 0.5) + 0.2)),
        "fear": max(0.0, min(1.0, arousal * (1.0 - valence))),
        "confusion": max(0.0, min(1.0, 0.5 - (valence - 0.5) + (0.5 - abs(arousal - 0.5)) * 0.3)),
        "stress": max(0.0, min(1.0, arousal)),
    }

    # Normalize
    total = sum(emotions.values()) or 1.0
    emotions = {k: float(v) / total for k, v in emotions.items()}
    return emotions


def _load_whisper():
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper  # type: ignore
            _whisper_model = whisper.load_model("base")
            logger.info("Whisper model loaded")
        except Exception as e:
            logger.warning(f"Whisper load failed: {e}")
            _whisper_model = False  # sentinel for disabled
    return _whisper_model


@router.post("/", response_model=AudioEmotionResult)
async def analyze_audio(chunk: AudioChunk) -> AudioEmotionResult:
    try:
        raw_bytes = _decode_audio_to_wav_bytes(chunk.audio)

        # Save to temp file to let librosa/audioread handle decoding of various formats
        suffix = ".webm"
        if chunk.contentType and "wav" in chunk.contentType:
            suffix = ".wav"
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        try:
            with open(tmp_path, "wb") as f:
                f.write(raw_bytes)

            # Load audio
            y, sr = librosa.load(tmp_path, sr=16000, mono=True)
            if y.size == 0:
                raise HTTPException(status_code=400, detail="Empty audio")

            metrics = _extract_audio_features(y, sr)
            emotions = _heuristic_emotion_from_features(metrics)

            # Transcribe if whisper available
            transcript = None
            language = None
            confidence = None

            model = _load_whisper()
            if model not in (None, False):
                try:
                    result = model.transcribe(tmp_path, fp16=False, language=None)
                    transcript = result.get("text")
                    language = result.get("language")
                    confidence = float(result.get("segments", [{}])[0].get("avg_logprob", -5.0)) if result.get("segments") else None
                except Exception as e:
                    logger.warning(f"Whisper transcription failed: {e}")

            return AudioEmotionResult(
                transcript=transcript,
                language=language,
                confidence=confidence,
                metrics=metrics,
                emotions=emotions,
            )
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze audio")


@router.get("/health")
async def health_check():
    model = _load_whisper()
    return {
        "status": "healthy",
        "whisper": bool(model not in (None, False)),
        "message": "Audio analysis service running",
    }


