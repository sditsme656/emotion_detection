from app.vision.smoother import EmotionSmoother


def test_hysteresis_requires_stable_frames_before_switch() -> None:
    smoother = EmotionSmoother(window_seconds=10.0, stable_frames=3, min_conf=0.55)

    for i in range(2):
        emotion, conf, _ = smoother.update({"happy": 0.9, "neutral": 0.1}, ts=float(i))
        assert emotion == "neutral"
        assert conf == 0.0

    emotion, conf, _ = smoother.update({"happy": 0.9, "neutral": 0.1}, ts=2.0)
    assert emotion == "happy"
    assert conf >= 0.55


def test_low_confidence_does_not_switch() -> None:
    smoother = EmotionSmoother(window_seconds=10.0, stable_frames=2, min_conf=0.8)

    for i in range(5):
        emotion, _, _ = smoother.update({"sad": 0.6, "neutral": 0.4}, ts=float(i))
        assert emotion == "neutral"
