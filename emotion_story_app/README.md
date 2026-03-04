# Emotion-Aware Storytelling (CPU MVP)

A real-time, CPU-friendly storytelling demo that adapts narrative flow from live facial emotion.

## Overview

This MVP captures webcam/phone video in the browser, detects one primary face, estimates emotion probabilities, smooths them over time with hysteresis, and advances a branching story graph on a slower story tick.

## 3-Worker Architecture

1. **Worker 1 (UI/Main Thread)**
   - `app/main_streamlit.py`
   - Handles Streamlit + WebRTC video rendering.
   - Sends latest frames to shared state and only reads derived results (no heavy inference).

2. **Worker 2 (Vision Thread)**
   - `app/workers/vision_worker.py`
   - Runs MediaPipe face detection and FER emotion inference on CPU.
   - Selects one primary face and applies temporal smoothing + hysteresis.

3. **Worker 3 (Story Thread)**
   - `app/workers/story_worker.py`
   - Progresses story graph at a slow periodic tick.
   - Branches only at decision nodes and pauses transitions if face is absent.

## Project Layout

```text
emotion_story_app/
  README.md
  requirements.txt
  .gitignore
  story_graphs/
    story_graph.json
  app/
    __init__.py
    main_streamlit.py
    main_opencv.py
    config.py
    state.py
    workers/
      __init__.py
      vision_worker.py
      story_worker.py
    vision/
      __init__.py
      face_detector.py
      primary_face.py
      emotion_model.py
      smoother.py
      roi.py
    story/
      __init__.py
      graph_schema.py
      graph_engine.py
      renderer.py
    utils/
      __init__.py
      timing.py
      logger.py
  tests/
    test_smoother.py
    test_story_graph.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run (Primary: Streamlit + WebRTC)

```bash
cd emotion_story_app
streamlit run app/main_streamlit.py
```

Then open the provided browser URL and grant camera permissions.

## Run (Fallback: OpenCV Window)

If `streamlit-webrtc` has environment/browser limitations:

```bash
cd emotion_story_app
python -m app.main_opencv
```

Press `q` to exit the OpenCV window.

## CPU Performance / Tuning Knobs

Tune these in `app/config.py`:

- `EMOTION_FPS` (lower = less CPU usage)
- `WINDOW_SECONDS` (larger = more stable, less responsive)
- `STABLE_FRAMES` (higher = stronger hysteresis)
- `STORY_TICK_SECONDS` (higher = slower story transitions)
- `FACE_DETECT_EVERY_N` (higher = fewer detections)

## Troubleshooting

### Webcam permissions
- Ensure browser camera permission is allowed.
- Close other apps that may lock camera access.

### streamlit-webrtc issues
- Try latest Chromium/Chrome.
- If WebRTC is blocked in your environment, use `python -m app.main_opencv` fallback.

### mediapipe install issues
- Upgrade pip/setuptools/wheel first.
- Ensure Python version is 3.10+.
- If installation fails on your platform, try creating a clean virtualenv.

## Story Graph Editing

Edit `story_graphs/story_graph.json`:

- `meta.start_node` must exist in `nodes`.
- Node `type` is `narration` or `decision`.
- Narration node requires `next`.
- Decision node requires `choices` and `fallback_to`.
- Optional `variants` map emotion labels to alternate text.

Supported emotion labels:
`angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, `neutral`.

## Tests

```bash
cd emotion_story_app
pytest -q
```
