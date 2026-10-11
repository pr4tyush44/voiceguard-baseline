"""Run VoiceGuard's small anti-spoofing baseline on a live audio input.

The program does not record audio.  It holds a rolling buffer in memory and
prints a conservative risk signal every second after the first four seconds.
Use an audio device that contains the call audio (for example an authorised
virtual audio cable or a platform-provided audio stream).  It cannot capture
arbitrary mobile, cellular, or encrypted-app calls by itself.
"""

from __future__ import annotations

import argparse
import collections
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import sounddevice as sd

from voice_features import feature_vector, speech_fraction


def list_devices() -> None:
    print(sd.query_devices())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("voiceguard_baseline.joblib"))
    parser.add_argument("--device", help="Input-device index or name. Omit to use the OS default input.")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between classifications.")
    parser.add_argument("--min-speech-fraction", type=float, default=0.25, help="Do not classify windows with less speech-like audio.")
    args = parser.parse_args()
    if args.list_devices:
        list_devices()
        return
    if args.interval <= 0:
        parser.error("--interval must be positive.")
    if not 0 <= args.min_speech_fraction <= 1:
        parser.error("--min-speech-fraction must be between 0 and 1.")
    if not args.model.is_file():
        parser.error(f"Model does not exist: {args.model}. Train it first.")

    artifact = joblib.load(args.model)
    required = {"model", "operating_threshold", "sample_rate", "window_seconds"}
    if not required.issubset(artifact):
        parser.error("This is not a VoiceGuard baseline model artifact.")
    model = artifact["model"]
    threshold = float(artifact["operating_threshold"])
    target_rate = int(artifact["sample_rate"])
    window_seconds = float(artifact["window_seconds"])

    device_info = sd.query_devices(args.device, "input")
    input_rate = float(device_info["default_samplerate"])
    max_samples = int(input_rate * window_seconds)
    samples: collections.deque[np.ndarray] = collections.deque()
    sample_count = 0
    sample_lock = threading.Lock()

    def on_audio(indata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags) -> None:
        nonlocal sample_count
        if status:
            print(f"Audio warning: {status}")
        chunk = np.asarray(indata, dtype=np.float32).copy()
        with sample_lock:
            samples.append(chunk)
            sample_count += len(chunk)
            while samples and sample_count - len(samples[0]) >= max_samples:
                sample_count -= len(samples.popleft())

    print(f"Listening to {device_info['name']!r} at {input_rate:.0f} Hz.")
    print("No audio is saved. Press Ctrl+C to stop. Scores are model margins, not fraud probabilities.")
    smoothed_score: float | None = None
    try:
        with sd.InputStream(device=args.device, channels=min(2, int(device_info["max_input_channels"])), samplerate=input_rate, callback=on_audio):
            while True:
                time.sleep(args.interval)
                with sample_lock:
                    current_count = sample_count
                    current = np.concatenate(tuple(samples), axis=0)[-max_samples:] if samples else None
                if current_count < max_samples:
                    print(f"Waiting for {window_seconds:.0f} seconds of audio …")
                    continue
                try:
                    voiced = speech_fraction(current, input_rate)
                    if voiced < args.min_speech_fraction:
                        print(f"{time.strftime('%H:%M:%S')}  NO-SPEECH/LOW-VOICE  speech={voiced:.0%}")
                        continue
                    vector = feature_vector(current, input_rate).reshape(1, -1)
                    raw_score = float(model.decision_function(vector)[0])
                except ValueError as exc:
                    print(f"No decision: {exc}")
                    continue
                smoothed_score = raw_score if smoothed_score is None else 0.65 * smoothed_score + 0.35 * raw_score
                label = "SPOOF-RISK" if smoothed_score >= threshold else "BONA-FIDE-LIKE"
                print(f"{time.strftime('%H:%M:%S')}  {label:16} speech={voiced:.0%}  margin={smoothed_score:+.3f}")
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
