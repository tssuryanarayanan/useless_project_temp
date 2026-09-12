"""
performance_index.py
---------------------
Composite "Blink Performance Index" scoring, letter grading, and Plotly gauge visualization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import plotly.graph_objects as go

IDEAL_BPM_LOW, IDEAL_BPM_HIGH = 15.0, 20.0
IDEAL_DURATION_LOW, IDEAL_DURATION_HIGH = 0.12, 0.35

WEIGHTS = {
    "rate": 0.45,
    "consistency": 0.30,
    "duration": 0.25,
}

GRADE_BANDS = [
    (97, "A+"), (93, "A"), (90, "A-"),
    (87, "B+"), (83, "B"), (80, "B-"),
    (77, "C+"), (73, "C"), (70, "C-"),
    (65, "D"), (0, "F"),
]


def _band_score(value: float, low: float, high: float, falloff: float) -> float:
    if low <= value <= high:
        return 100.0
    if value < low:
        distance = low - value
    else:
        distance = value - high
    score = 100.0 * max(0.0, 1.0 - (distance / falloff))
    return score


def rate_score(bpm: float) -> float:
    return round(_band_score(bpm, IDEAL_BPM_LOW, IDEAL_BPM_HIGH, falloff=15.0), 1)


def duration_score(avg_duration_s: float) -> float:
    return round(_band_score(avg_duration_s, IDEAL_DURATION_LOW, IDEAL_DURATION_HIGH, falloff=0.25), 1)


def grade_for_score(score: float) -> str:
    for threshold, grade in GRADE_BANDS:
        if score >= threshold:
            return grade
    return "F"


@dataclass
class PerformanceResult:
    composite_score: float
    grade: str
    rate_score: float
    consistency_score: float
    duration_score: float

    @property
    def score(self) -> float:
        return self.composite_score


def compute_blink_performance_index(stats) -> PerformanceResult:
    return compute_performance_index(
        bpm=float(stats.blinks_per_minute),
        consistency_score_0_100=float(stats.blink_consistency),
        avg_duration_s=float(stats.avg_blink_duration),
    )


def compute_performance_index(
    bpm: float,
    consistency_score_0_100: float,
    avg_duration_s: float,
) -> PerformanceResult:
    r = rate_score(bpm)
    c = round(consistency_score_0_100, 1)
    d = duration_score(avg_duration_s)

    composite = (
        r * WEIGHTS["rate"]
        + c * WEIGHTS["consistency"]
        + d * WEIGHTS["duration"]
    )
    composite = round(composite, 1)

    return PerformanceResult(
        composite_score=composite,
        grade=grade_for_score(composite),
        rate_score=r,
        consistency_score=c,
        duration_score=d,
    )


def calculate_performance_index(metrics: Dict[str, Any]) -> Tuple[float, str]:
    """Helper wrapper for app.py passing a metrics dictionary."""
    res = compute_performance_index(
        bpm=float(metrics.get("bpm", 0.0)),
        consistency_score_0_100=float(metrics.get("consistency_score", 0.0)),
        avg_duration_s=float(metrics.get("avg_duration", 0.0)),
    )
    return res.composite_score, res.grade


def create_gauge_chart(score: float, grade: str) -> go.Figure:
    """Builds a responsive gauge indicator for Streamlit."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": f"Blink Performance Index (Grade: {grade})", "font": {"size": 20}},
            number={"suffix": "/100", "font": {"size": 28}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#7f8c8d"},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [0, 60], "color": "rgba(231, 76, 60, 0.4)"},
                    {"range": [60, 80], "color": "rgba(241, 196, 15, 0.4)"},
                    {"range": [80, 100], "color": "rgba(46, 204, 113, 0.4)"},
                ],
                "threshold": {
                    "line": {"color": "#2c3e50", "width": 4},
                    "thickness": 0.75,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=25, r=25, t=50, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#ecf0f1", "family": "Inter, Arial"},
    )
    return fig