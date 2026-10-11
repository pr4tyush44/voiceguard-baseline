"""Train and evaluate a compact ASVspoof logical-access baseline.

Example:
    python train_baseline.py --dataset-root /path/to/ASVspoof2019/LA/LA \
        --model-out voiceguard_baseline.joblib --report-out training_report.json

The script uses protocol-defined train/dev/eval partitions.  It trains only on
train, selects the operating threshold only on dev, and touches eval once for a
final held-out estimate.  It does not make a real-world fraud-performance claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from voice_features import SAMPLE_RATE, WINDOW_SECONDS, feature_from_path

SEED = 42


def find_dataset_root(root: Path) -> Path:
    candidates = (root, root / "LA", root / "LA" / "LA")
    for candidate in candidates:
        if (candidate / "ASVspoof2019_LA_cm_protocols").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find ASVspoof2019_LA_cm_protocols below " + str(root)
    )


def read_protocol(base: Path, protocol_file: str, audio_folder: str) -> list[tuple[Path, int]]:
    protocol = base / "ASVspoof2019_LA_cm_protocols" / protocol_file
    audio_root = base / audio_folder / "flac"
    if not protocol.is_file():
        raise FileNotFoundError(f"Protocol file not found: {protocol}")

    records: list[tuple[Path, int]] = []
    for line_number, line in enumerate(protocol.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if len(fields) < 5:
            raise ValueError(f"Malformed protocol line {line_number} in {protocol.name}")
        label = fields[-1].lower()
        if label not in {"bonafide", "spoof"}:
            raise ValueError(
                f"{protocol.name} has no usable labels (found {label!r}). "
                "Do not evaluate on a hidden-label protocol."
            )
        records.append((audio_root / f"{fields[1]}.flac", int(label == "spoof")))
    return records


def balanced_subset(records: list[tuple[Path, int]], maximum_per_class: int | None, rng: np.random.Generator) -> list[tuple[Path, int]]:
    if maximum_per_class is None:
        return records
    groups = {label: [record for record in records if record[1] == label] for label in (0, 1)}
    if not groups[0] or not groups[1]:
        raise ValueError("A split must contain both bona-fide and spoofed clips.")
    chosen = []
    for label in (0, 1):
        count = min(len(groups[label]), maximum_per_class)
        indexes = rng.choice(len(groups[label]), size=count, replace=False)
        chosen.extend(groups[label][i] for i in indexes)
    rng.shuffle(chosen)
    return chosen


def build_matrix(records: Iterable[tuple[Path, int]], split_name: str) -> tuple[np.ndarray, np.ndarray]:
    xs, ys, failures = [], [], []
    records = list(records)
    for index, (path, label) in enumerate(records, start=1):
        try:
            xs.append(feature_from_path(str(path)))
            ys.append(label)
        except Exception as exc:  # Report unusable data instead of silently changing the split.
            failures.append(f"{path}: {exc}")
        if index % 100 == 0 or index == len(records):
            print(f"{split_name}: {index}/{len(records)} clips processed")
    if failures:
        preview = "\n".join(failures[:5])
        raise RuntimeError(f"{split_name} had {len(failures)} unreadable clips. First failures:\n{preview}")
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int8)


def eer_and_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    fnr = 1.0 - tpr
    point = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[point] + fnr[point]) / 2), float(thresholds[point])


def evaluate(name: str, model: Pipeline, x: np.ndarray, y: np.ndarray, threshold: float) -> dict[str, object]:
    scores = model.decision_function(x)
    eer, _ = eer_and_threshold(y, scores)
    predictions = (scores >= threshold).astype(np.int8)
    return {
        "samples": int(len(y)),
        "bonafide": int((y == 0).sum()),
        "spoof": int((y == 1).sum()),
        "eer": eer,
        "accuracy_at_dev_threshold": float(accuracy_score(y, predictions)),
        "confusion_matrix_rows_true_0_bonafide_1_spoof": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
    }


def parse_limit(value: int) -> int | None:
    return None if value == 0 else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, default=Path("voiceguard_baseline.joblib"))
    parser.add_argument("--report-out", type=Path, default=Path("training_report.json"))
    parser.add_argument("--max-train-per-class", type=int, default=5000, help="0 uses every available clip.")
    parser.add_argument("--max-dev-per-class", type=int, default=2000, help="0 uses every available clip.")
    parser.add_argument("--max-eval-per-class", type=int, default=2000, help="0 uses every available labelled clip.")
    args = parser.parse_args()
    for value in (args.max_train_per_class, args.max_dev_per_class, args.max_eval_per_class):
        if value < 0:
            parser.error("Maximum examples per class must be zero or positive.")

    base = find_dataset_root(args.dataset_root)
    rng = np.random.default_rng(SEED)
    specs = {
        "train": ("ASVspoof2019.LA.cm.train.trn.txt", "ASVspoof2019_LA_train", parse_limit(args.max_train_per_class)),
        "dev": ("ASVspoof2019.LA.cm.dev.trl.txt", "ASVspoof2019_LA_dev", parse_limit(args.max_dev_per_class)),
        "eval": ("ASVspoof2019.LA.cm.eval.trl.txt", "ASVspoof2019_LA_eval", parse_limit(args.max_eval_per_class)),
    }
    matrices = {}
    for name, (protocol, folder, limit) in specs.items():
        records = balanced_subset(read_protocol(base, protocol, folder), limit, rng)
        matrices[name] = build_matrix(records, name)

    x_train, y_train = matrices["train"]
    x_dev, y_dev = matrices["dev"]
    x_eval, y_eval = matrices["eval"]
    model = Pipeline([
        ("scale", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")),
    ])
    model.fit(x_train, y_train)

    dev_scores = model.decision_function(x_dev)
    dev_eer, operating_threshold = eer_and_threshold(y_dev, dev_scores)
    report = {
        "warning": "Benchmark results are not a claim of performance on live calls, cloned voices, or fraud.",
        "dataset_root": str(base),
        "random_seed": SEED,
        "feature_specification": "20 MFCC + MFCC delta + spectral flatness/centroid/rolloff; mean and standard deviation; 86 values",
        "sample_rate": SAMPLE_RATE,
        "window_seconds": WINDOW_SECONDS,
        "threshold_selected_on": "dev EER operating point",
        "dev_eer": dev_eer,
        "operating_threshold": operating_threshold,
        "dev": evaluate("dev", model, x_dev, y_dev, operating_threshold),
        "eval": evaluate("eval", model, x_eval, y_eval, operating_threshold),
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "operating_threshold": operating_threshold,
        "sample_rate": SAMPLE_RATE,
        "window_seconds": WINDOW_SECONDS,
        "feature_version": "mfcc-delta-spectral-v1",
    }, args.model_out)
    args.report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved model to {args.model_out}")


if __name__ == "__main__":
    main()
