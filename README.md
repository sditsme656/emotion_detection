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

## End-to-End: Download Dataset, Train, and Use

Follow this sequence if you are starting from scratch.

### 1) Install dependencies

```bash
pip install torch torchvision opencv-python pillow numpy
```

Optional face detector backend:

```bash
pip install mediapipe
```

### 2) Download and prepare dataset

Recommended dataset: **FER2013** from Kaggle:

- https://www.kaggle.com/datasets/msambare/fer2013

After downloading/extracting, organize images into this ImageFolder format:

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

Notes:

- Class folder names must match the 7 emotion labels above.
- If your download does not come pre-split into `train` and `val`, create a split yourself (e.g., 80/20 stratified split).

### 3) Train the emotion model

Example command:

```bash
python train.py --data-dir ./data --epochs 20 --batch-size 64 --backbone resnet18
```

Alternative backbone:

```bash
python train.py --data-dir ./data --epochs 20 --batch-size 64 --backbone efficientnet_b0
```

What the script does:

- Loads data from `data/train` and `data/val`
- Applies augmentation on train images
- Trains with CrossEntropy + Adam
- Saves the best validation checkpoint to:

```text
checkpoints/best_emotion_model.pt
```

### 4) Test/use the model in real time (webcam)

Run inference with your best checkpoint:

```bash
python inference.py --checkpoint checkpoints/best_emotion_model.pt --backbone resnet18 --window 30
```

If you trained with EfficientNet, use:

```bash
python inference.py --checkpoint checkpoints/best_emotion_model.pt --backbone efficientnet_b0 --window 30
```

Press `q` in the OpenCV window to quit.

### 5) Quick troubleshooting

- **No face boxes**: lower `--face-conf` (example: `--face-conf 0.3`) or improve lighting.
- **Very unstable labels**: increase smoothing window (example: `--window 45`).
- **Backbone mismatch error**: inference backbone must match the one used for training.
- **Slow performance**: reduce camera resolution and use GPU if available.

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
