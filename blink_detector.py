"""
blink_detector.py
------------------
Computer vision engine for BlinkTrack Analytics & Fatigue Monitor.

Uses MediaPipe's modern Tasks Vision API (`FaceLandmarker`) instead of the
deprecated `mp.solutions.face_mesh`. Handles:
  - Auto-downloading the `.task` model bundle if not present locally.
  - Extracting eye landmarks and computing the Eye Aspect Ratio (EAR).
  - Smoothing EAR over a rolling window to reduce frame noise.
  - A small state machine that turns EAR dips into discrete BlinkEvent
    records with precise start/end timestamps and duration.
"""

from __future__ import annotations

import os
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# --------------------------------------------------------------------------
# Landmark indices (MediaPipe FaceMesh / FaceLandmarker topology, 468/478 pts)
# --------------------------------------------------------------------------
# Order for each eye is: [p1, p2, p3, p4, p5, p6] matching the classic
# Soukupová & Čech EAR formulation, where p1-p4 is the horizontal axis and
# (p2,p6) / (p3,p5) are the two vertical pairs.
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")

# --------------------------------------------------------------------------
# Tunables
# --------------------------------------------------------------------------
EAR_SMOOTHING_WINDOW = 3          # frames to moving-average over
DEFAULT_EAR_THRESHOLD = 0.21      # below this counts as "eye closed"
MIN_BLINK_FRAMES = 1              # min consecutive closed frames to count
MAX_BLINK_DURATION = 0.6          # seconds; longer closures treated as "eyes closed", not a blink


@dataclass
class BlinkEvent:
    """A single completed blink."""
    start_time: float
    end_time: float
    duration: float = field(init=False)

    def __post_init__(self):
        self.duration = round(self.end_time - self.start_time, 3)

@dataclass
class BlinkDetectorConfig:
    model_path: Optional[str] = None
    ear_threshold: float = DEFAULT_EAR_THRESHOLD
    smoothing_window: int = EAR_SMOOTHING_WINDOW


def ensure_model_downloaded(progress_callback=None) -> str:
    """
    Make sure the FaceLandmarker .task model bundle exists locally,
    downloading it if necessary. Returns the local path to the model.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        return MODEL_PATH

    def _hook(block_num, block_size, total_size):
        if progress_callback and total_size > 0:
            downloaded = block_num * block_size
            pct = min(1.0, downloaded / total_size)
            progress_callback(pct)

    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=_hook)
    return MODEL_PATH


def _euclidean(p1, p2) -> float:
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def compute_ear(landmarks, eye_indices) -> float:
    """
    Compute the Eye Aspect Ratio for one eye given normalized landmark
    coordinates and the six indices for that eye, ordered [p1..p6].

    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    """
    pts = [(landmarks[i].x, landmarks[i].y) for i in eye_indices]
    p1, p2, p3, p4, p5, p6 = pts
    vertical_1 = _euclidean(p2, p6)
    vertical_2 = _euclidean(p3, p5)
    horizontal = _euclidean(p1, p4)
    if horizontal <= 1e-6:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


class BlinkDetector:
    """
    Wraps a MediaPipe FaceLandmarker in VIDEO mode and turns each frame
    into an EAR reading + blink state machine update.
    """

    def __init__(
        self,
        model_path: Optional[BlinkDetectorConfig] = None,
        ear_threshold: float = DEFAULT_EAR_THRESHOLD,
        smoothing_window: int = EAR_SMOOTHING_WINDOW,
        on_model_download_progress=None,
    ):
        if isinstance(model_path, BlinkDetectorConfig):
            config = model_path
            model_path = config.model_path
            ear_threshold = config.ear_threshold
            smoothing_window = config.smoothing_window
        self.config = BlinkDetectorConfig(model_path, ear_threshold, smoothing_window)
        self.ear_threshold = ear_threshold
        self._ear_history: deque[float] = deque(maxlen=smoothing_window)

        model_path = model_path or ensure_model_downloaded(on_model_download_progress)

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

        # Blink state machine
        self._eye_closed = False
        self._closed_since: Optional[float] = None
        self._frame_ts_ms = 0

        self.last_smoothed_ear: float = 1.0
        self.face_detected: bool = False

    def close(self):
        try:
            self._landmarker.close()
        except Exception:
            pass

    def process_frame(self, rgb_frame: np.ndarray, timestamp_s: Optional[float] = None):
        timestamp_s = timestamp_s if timestamp_s is not None else time.time()
        """
        Feed one RGB frame (numpy array, HxWx3) into the detector.

        Returns a dict:
            {
              "ear": float,                # smoothed EAR (0 if no face)
              "face_detected": bool,
              "is_blinking": bool,          # currently in a closed-eye state
              "blink_event": BlinkEvent | None,  # non-None the frame a blink completes
            }
        """
        timestamp_s = timestamp_s if timestamp_s is not None else time.time()
        # VIDEO mode requires strictly increasing millisecond timestamps.
        self._frame_ts_ms = max(self._frame_ts_ms + 1, int(timestamp_s * 1000))

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect(mp_image)

        blink_event: Optional[BlinkEvent] = None

        if not result.face_landmarks:
            self.face_detected = False
            # Don't spuriously close/open eyes when tracking is lost.
            return {
                "ear": self.last_smoothed_ear,
                "face_detected": False,
                "is_blinking": self._eye_closed,
                "blink_event": None,
            }

        self.face_detected = True
        landmarks = result.face_landmarks[0]

        left_ear = compute_ear(landmarks, LEFT_EYE_IDX)
        right_ear = compute_ear(landmarks, RIGHT_EYE_IDX)
        raw_ear = (left_ear + right_ear) / 2.0

        self._ear_history.append(raw_ear)
        smoothed_ear = float(np.mean(self._ear_history))
        self.last_smoothed_ear = smoothed_ear

        is_closed_now = smoothed_ear < self.ear_threshold

        if is_closed_now and not self._eye_closed:
            # Eye just closed
            self._eye_closed = True
            self._closed_since = timestamp_s
        elif not is_closed_now and self._eye_closed:
            # Eye just reopened -> a blink completed
            self._eye_closed = False
            if self._closed_since is not None:
                duration = timestamp_s - self._closed_since
                if 0 < duration <= MAX_BLINK_DURATION:
                    blink_event = BlinkEvent(start_time=self._closed_since, end_time=timestamp_s)
                # else: treat as a deliberate long eye closure, not a blink
            self._closed_since = None

        return {
            "ear": smoothed_ear,
            "face_detected": True,
            "is_blinking": self._eye_closed,
            "blink_event": blink_event,
        }

    def process(self, bgr_frame: np.ndarray, timestamp_s: Optional[float] = None):
        """Process an OpenCV BGR frame and return the app-facing result."""
        rgb_frame = bgr_frame[:, :, ::-1]
        result = self.process_frame(rgb_frame, timestamp_s)
        debug = {
            "face_found": result["face_detected"],
            "ear": result["ear"],
            "eye_closed": result["is_blinking"],
        }
        return result["blink_event"], debug
