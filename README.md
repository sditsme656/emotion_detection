# Real-Time Student Emotion Detection (PyTorch + OpenCV)

A multi-stage, real-time emotion monitoring pipeline that detects faces from a live webcam stream, classifies student emotion, and applies temporal smoothing to reduce prediction flicker.

## Pipeline Architecture

1. **Frame Capture (Threaded)**
   - Uses `cv2.VideoCapture` with a threaded queue (`ThreadedCamera`) so capture continues while inference runs.
   - The queue keeps the **most recent frame**, minimizing lag.

2. **Stage 1: Face Detection**
   - `detector.py` provides `FaceDetector`.
   - Primary backend: **MediaPipe Face Detection** (if available).
   - Fallback backend: OpenCV Haar cascade.

3. **Stage 2: Emotion Classification**
   - `model.py` defines `EmotionClassifier` with selectable backbones:
     - `resnet18`
     - `efficientnet_b0`
   - Designed for FER-style grayscale face inputs (`1x48x48`).

4. **Temporal Smoothing**
   - `utils.py` provides `RollingEmotionSmoother`.
   - Uses rolling average across a configurable window (`--window`, default `30` frames ≈ 2 seconds at ~15 FPS).

5. **UI Overlay**
   - `inference.py` overlays:
     - Detected `Emotion`
     - `Confidence` score
   - Display window title: `Student Emotion Monitor`.

## File Structure

- `model.py` — Emotion CNN architecture and checkpoint loader.
- `utils.py` — Face preprocessing + rolling-window temporal smoothing.
- `detector.py` — Face detection and ROI cropping.
- `inference.py` — Real-time webcam inference pipeline.
- `train.py` — FER2013-style training boilerplate.

## Dataset

You can train using:

- **FER2013** (Kaggle): https://www.kaggle.com/datasets/msambare/fer2013
- Optional alternative: **AffectNet**: http://mohammadmahoor.com/affectnet/

### Expected training layout

`train.py` expects a directory format compatible with `torchvision.datasets.ImageFolder`:

```text
data/
  train/
    angry/
    disgust/
    fear/
    happy/
    sad/
    surprise/
    neutral/
  val/
    angry/
    disgust/
    fear/
    happy/
    sad/
    surprise/
    neutral/
```

## Setup

Install dependencies:

```bash
pip install torch torchvision opencv-python pillow numpy
```

Optional (recommended for faster/more robust face detection):

```bash
pip install mediapipe
```

## Training Procedure

### Data augmentation

`train.py` includes:

- Grayscale conversion
- Resize to `48x48`
- Random horizontal flip
- Random rotation
- Tensor normalization to `[-1, 1]`

### Loss function

- `CrossEntropyLoss`

### Optimizer

- `Adam` (default learning rate: `1e-3`)

### Example training command

```bash
python train.py --data-dir ./data --epochs 20 --batch-size 64 --backbone resnet18
```

Best checkpoint is saved to:

```text
checkpoints/best_emotion_model.pt
```

## Real-time Inference

Run webcam inference:

```bash
python inference.py --checkpoint checkpoints/best_emotion_model.pt --backbone resnet18 --window 30
```

### Useful flags

- `--camera` webcam index (default `0`)
- `--checkpoint` trained model path
- `--backbone` `resnet18` or `efficientnet_b0`
- `--window` temporal smoothing window size
- `--face-conf` face detector minimum confidence

Press **`q`** to quit the live window.

## Notes on CPU/GPU

Both training and inference automatically choose device:

- CUDA GPU if available
- CPU otherwise

Inference uses `torch.no_grad()` for efficiency.
