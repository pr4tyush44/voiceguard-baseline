"""Shared, lightweight audio features for the VoiceGuard baseline.

This is intentionally a small classical model pipeline.  It is a baseline for
anti-spoofing experiments, not proof that a caller is or is not a real person.
"""

from __future__ import annotations

import numpy as np
import librosa
import webrtcvad

SAMPLE_RATE = 16_000
WINDOW_SECONDS = 4.0
N_MFCC = 20


def prepare_audio(
    audio: np.ndarray, sample_rate: float, *, target_sample_rate: int = SAMPLE_RATE,
    window_seconds: float = WINDOW_SECONDS,
) -> np.ndarray:
    """Convert audio to mono, resample, and return one fixed-length window.

    Padding is used only when a clip is short.  Longer clips use a centered
    window; the streaming application repeatedly analyzes its most-recent
    window, so no audio is written to disk.
    """
    y = np.asarray(audio, dtype=np.float32)
    if y.ndim == 2:
        # librosa returns (channels, samples), whereas sounddevice returns
        # (frames, channels).  Average the smaller dimension as channels.
        channel_axis = 0 if y.shape[0] < y.shape[1] else 1
        y = y.mean(axis=channel_axis)
    if y.ndim != 1:
        raise ValueError(f"Expected mono or stereo audio, got shape {y.shape}.")
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    if not len(y):
        raise ValueError("Audio contains no samples.")
    if int(round(sample_rate)) != target_sample_rate:
        y = librosa.resample(y, orig_sr=sample_rate, target_sr=target_sample_rate)

    wanted = int(target_sample_rate * window_seconds)
    if len(y) < wanted:
        return np.pad(y, (0, wanted - len(y)))
    start = (len(y) - wanted) // 2
    return y[start : start + wanted]


def feature_vector(audio: np.ndarray, sample_rate: float) -> np.ndarray:
    """Return a compact 86-value feature vector for one audio window.

    Each time-frequency descriptor contributes its mean and standard deviation.
    This preserves a simple, fast inference path while avoiding the original
    notebook's unused and heavyweight Torch/LFCC dependency.
    """
    y = prepare_audio(audio, sample_rate)
    if np.max(np.abs(y)) < 1e-6:
        raise ValueError("Audio is effectively silent.")

    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=512, hop_length=160)
    delta = librosa.feature.delta(mfcc)
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=512, hop_length=160)
    centroid = librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE, n_fft=512, hop_length=160)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=SAMPLE_RATE, roll_percent=0.85, n_fft=512, hop_length=160)

    descriptors = (mfcc, delta, flatness, centroid, rolloff)
    values = [np.concatenate((d.mean(axis=1), d.std(axis=1))) for d in descriptors]
    vector = np.concatenate(values).astype(np.float32)
    if not np.isfinite(vector).all():
        raise ValueError("Feature extraction produced non-finite values.")
    return vector


def speech_fraction(audio: np.ndarray, sample_rate: float, *, aggressiveness: int = 2) -> float:
    """Estimate how much of a window contains speech-like audio.

    This is a WebRTC voice-activity gate, not a human-identity check. It keeps
    music, silence, and very weak signals from being reported as either genuine
    or spoofed speech by a classifier trained only on spoken ASVspoof clips.
    """
    if aggressiveness not in range(4):
        raise ValueError("VAD aggressiveness must be between 0 and 3.")
    y = prepare_audio(audio, sample_rate)
    pcm = np.clip(y, -1.0, 1.0)
    pcm = (pcm * np.iinfo(np.int16).max).astype("<i2")
    frame_samples = int(SAMPLE_RATE * 0.03)  # WebRTC VAD accepts 10, 20, or 30 ms frames.
    vad = webrtcvad.Vad(aggressiveness)
    decisions = [
        vad.is_speech(pcm[offset : offset + frame_samples].tobytes(), SAMPLE_RATE)
        for offset in range(0, len(pcm) - frame_samples + 1, frame_samples)
    ]
    return float(np.mean(decisions)) if decisions else 0.0


def feature_from_path(path: str) -> np.ndarray:
    audio, sample_rate = librosa.load(path, sr=None, mono=False)
    return feature_vector(audio, sample_rate)
