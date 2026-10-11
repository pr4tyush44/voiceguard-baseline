from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16_000


def write_protocol_split(base: Path, split: str, folder: str, frequency_offset: float) -> None:
    audio_dir = base / folder / "flac"
    audio_dir.mkdir(parents=True)
    lines = []
    time_axis = np.arange(SAMPLE_RATE * 4) / SAMPLE_RATE
    for label, frequency in (("bonafide", 220.0 + frequency_offset), ("spoof", 1200.0 + frequency_offset)):
        for index in range(2):
            utterance = f"{split}_{label}_{index}"
            # Deliberately distinct synthetic audio for a pipeline-only smoke test.
            audio = 0.1 * np.sin(2 * np.pi * (frequency + index * 20) * time_axis)
            sf.write(audio_dir / f"{utterance}.flac", audio, SAMPLE_RATE)
            lines.append(f"speaker {utterance} - - {label}\n")
    protocol = base / "ASVspoof2019_LA_cm_protocols" / f"ASVspoof2019.LA.cm.{split}.{'trn' if split == 'train' else 'trl'}.txt"
    protocol.parent.mkdir(exist_ok=True)
    protocol.write_text("".join(lines), encoding="utf-8")


class TrainSmokeTest(unittest.TestCase):
    def test_training_writes_model_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir) / "LA"
            write_protocol_split(base, "train", "ASVspoof2019_LA_train", 0)
            write_protocol_split(base, "dev", "ASVspoof2019_LA_dev", 30)
            write_protocol_split(base, "eval", "ASVspoof2019_LA_eval", 60)
            model = Path(temp_dir) / "model.joblib"
            report = Path(temp_dir) / "report.json"
            subprocess.run([
                sys.executable, str(PROJECT_ROOT / "train_baseline.py"),
                "--dataset-root", str(base),
                "--max-train-per-class", "0",
                "--max-dev-per-class", "0",
                "--max-eval-per-class", "0",
                "--model-out", str(model),
                "--report-out", str(report),
            ], check=True, cwd=PROJECT_ROOT)
            result = json.loads(report.read_text(encoding="utf-8"))
            self.assertTrue(model.is_file())
            self.assertIn("dev", result)
            self.assertIn("eval", result)
            self.assertEqual(result["eval"]["samples"], 4)


if __name__ == "__main__":
    unittest.main()
