# VoiceGuard: a small real-time audio anti-spoofing baseline

VoiceGuard is a compact, reproducible baseline for experiments in speech anti-spoofing. It is deliberately small enough to run locally on a laptop.

> **Not a fraud verdict or identity check.** The model distinguishes `bona-fide-like` speech from audio resembling the ASVspoof spoof class. It cannot prove a caller is human, identify a caller, or determine that fraud has occurred.

## Included

- `train_baseline.py` trains a small RBF-SVM on protocol-labelled ASVspoof 2019 Logical Access clips.
- `voice_features.py` extracts a fast 86-value MFCC/delta/spectral feature vector.
- `realtime_detector.py` keeps a four-second **in-memory** audio buffer, gates out silence/low-speech windows with WebRTC voice activity detection, and classifies once per second. It never saves raw audio.
- `tests/` covers audio-layout handling and a complete synthetic-data training smoke test.

## Important scope

The output is an **anti-spoofing risk signal**, not proof that a caller is a human, a fraudster, or a particular identity. It can only distinguish *bona-fide-like speech* from *speech resembling the ASVspoof spoof class*; it does not identify people and it deliberately makes no decision on low-speech windows. A benchmark-trained detector can fail on new speech generators, codecs, replayed audio, noise, languages, and real telephone channels. Do not automatically block a caller or make a fraud decision from this model alone.

Windows does not expose every telephone or VoIP call's audio to background applications. To analyze a call, choose an authorised input device that genuinely contains the call audio—such as an enterprise calling-platform integration or an approved virtual loopback device. Use only with the consent and legal authority required where the call occurs.

## Install

Use Python 3.10–3.13 in a virtual environment, then run:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
python -m unittest discover -s tests -t . -v
```

## Train and obtain evidence

Point the script at the extracted ASVspoof Logical Access dataset. The root may be the dataset folder, `LA`, or `LA/LA`.

```powershell
python train_baseline.py `
  --dataset-root "D:\datasets\ASVspoof2019\LA\LA" `
  --model-out voiceguard_baseline.joblib `
  --report-out training_report.json
```

By default it trains with at most 5,000 clips per class and tests with at most 2,000 clips per class. Set a maximum to `0` to use all clips. The model is fitted only to `train`; the decision threshold is selected on `dev`; `eval` is held out for the reported estimate. Do not quote a result until `training_report.json` has been generated and reviewed.

## Run live

First identify the input that contains the authorised call audio:

```powershell
python realtime_detector.py --list-devices
```

Then start the detector (replace the device name/index):

```powershell
python realtime_detector.py --model voiceguard_baseline.joblib --device "Call Audio Loopback"
```

It prints `BONA-FIDE-LIKE` or `SPOOF-RISK` after the first four seconds, updating every second. `margin` is a raw classifier score, **not a probability or percentage of fraud**.

## What changed from the notebook

- Uses one implementation rather than repeated, conflicting cells.
- Validates dataset paths, protocol format, audio failures, and hidden/absent labels.
- Does not use `SVC(probability=True)` or call its uncalibrated value a fake percentage.
- Selects a threshold on development data rather than hard-coding `0.5`.
- Calculates an explicit approximate EER and a held-out evaluation report.
- Removes the simulated phone-origin/contact score; it was unrelated to acoustic anti-spoofing and could be misleading.
- Avoids bundling Torch and LFCC solely for a small classical baseline.

## Before production

Evaluate by attack type and on new TTS/voice-clone engines, telephony codecs, replay attacks, noise, short clips, languages, accents, microphones, and all intended calling platforms. Then add calibrated scores, a human-review workflow, user-facing uncertainty language, monitoring for model drift, and a privacy/security review.

## Publish to GitHub

Do not commit the downloaded dataset, trained `.joblib` models, recordings, or `training_report.json`; `.gitignore` excludes them. From this folder:

```powershell
git init
git add .
git commit -m "Initial VoiceGuard anti-spoofing baseline"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/voiceguard-baseline.git
git push -u origin main
```
