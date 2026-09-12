import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from blink_detector import BlinkEvent


class BlinkStatsEngine:
    def __init__(self, session_start_time: float):
        self.session_start_time = session_start_time
        self.events: List[BlinkEvent] = []
        self.last_blink_time = session_start_time
        self.max_staring_streak = 0.0

    def register_blink(self, event: BlinkEvent):
        self.events.append(event)
        streak = event.start_time - self.last_blink_time
        if streak > self.max_staring_streak:
            self.max_staring_streak = streak
        self.last_blink_time = event.end_time

    def compute_metrics(self) -> Dict[str, Any]:
        now = time.time()
        elapsed_sec = max(1.0, now - self.session_start_time)
        elapsed_min = elapsed_sec / 60.0

        current_stare = now - self.last_blink_time
        max_streak = max(self.max_staring_streak, current_stare)

        total_blinks = len(self.events)
        bpm = total_blinks / elapsed_min

        if total_blinks > 0:
            durations = [e.duration for e in self.events]
            avg_duration = float(np.mean(durations))
            fastest_duration = float(np.min(durations))
            longest_duration = float(np.max(durations))

            ibis = []
            prev_time = self.session_start_time
            for e in self.events:
                ibis.append(e.start_time - prev_time)
                prev_time = e.end_time

            ibi_cv = float(np.std(ibis) / (np.mean(ibis) + 1e-6))
            rhythm_score = max(0.0, min(100.0, 100.0 * (1.0 - min(ibi_cv, 1.0))))

            dur_cv = float(np.std(durations) / (avg_duration + 1e-6))
            consistency_score = max(0.0, min(100.0, 100.0 * (1.0 - min(dur_cv, 1.0))))
        else:
            avg_duration = 0.0
            fastest_duration = 0.0
            longest_duration = 0.0
            consistency_score = 0.0
            rhythm_score = 0.0

        return {
            "total_blinks": total_blinks,
            "elapsed_seconds": elapsed_sec,
            "bpm": round(bpm, 2),
            "avg_duration": round(avg_duration, 3),
            "fastest_duration": round(fastest_duration, 3),
            "longest_duration": round(longest_duration, 3),
            "max_staring_streak": round(max_streak, 2),
            "consistency_score": round(consistency_score, 1),
            "rhythm_score": round(rhythm_score, 1),
        }

    def to_dataframe(self) -> pd.DataFrame:
        if not self.events:
            return pd.DataFrame(columns=["start_time", "end_time", "duration_sec"])
        return pd.DataFrame(
            [
                {
                    "start_time": round(e.start_time - self.session_start_time, 3),
                    "end_time": round(e.end_time - self.session_start_time, 3),
                    "duration_sec": round(e.duration, 3),
                }
                for e in self.events
            ]
        )

class SessionStats:
    def __init__(self, metrics: Dict[str, Any]):
        self.total_blinks = metrics["total_blinks"]
        self.blinks_per_minute = metrics["bpm"]
        self.avg_blink_duration = metrics["avg_duration"]
        self.fastest_blink = metrics["fastest_duration"]
        self.longest_blink = metrics["longest_duration"]
        self.longest_no_blink_streak = metrics["max_staring_streak"]
        self.blink_consistency = metrics["consistency_score"]
        self.blink_rhythm = metrics["rhythm_score"]

class BlinkTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self._engine = BlinkStatsEngine(time.time())

    def record(self, event: BlinkEvent):
        self._engine.register_blink(event)

    def compute(self) -> SessionStats:
        return SessionStats(self._engine.compute_metrics())