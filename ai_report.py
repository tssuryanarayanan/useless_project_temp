"""
ai_report.py
------------
Diagnostic rule engine + Gemini-powered sports-analyst commentary.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

# -------------------------------------------------------------- Thresholds
LOW_BPM_THRESHOLD = 10.0
DROWSY_DURATION_THRESHOLD = 0.35
INCOMPLETE_DURATION_THRESHOLD = 0.12
PROLONGED_STARE_THRESHOLD = 15.0


@dataclass
class DiagnosticFlag:
    name: str
    severity: str  # "info" | "warning" | "critical"
    message: str
    recommendation: str


def run_diagnostics(summary: dict) -> List[DiagnosticFlag]:
    flags: List[DiagnosticFlag] = []

    bpm = summary.get("bpm", 0.0)
    avg_duration = summary.get("avg_duration", 0.0)
    longest_streak = summary.get("max_staring_streak", summary.get("longest_no_blink_streak", 0.0))

    if bpm < LOW_BPM_THRESHOLD:
        flags.append(
            DiagnosticFlag(
                name="Digital Eye Strain Risk",
                severity="warning",
                message=f"Blink rate is {bpm} BPM, below the {LOW_BPM_THRESHOLD:.0f} BPM safety floor.",
                recommendation="Follow the 20-20-20 rule: every 20 minutes, look at something 20 feet away for 20 seconds.",
            )
        )

    if avg_duration > DROWSY_DURATION_THRESHOLD:
        flags.append(
            DiagnosticFlag(
                name="Drowsiness Biomarker",
                severity="warning",
                message=f"Average blink duration is {avg_duration}s, longer than the {DROWSY_DURATION_THRESHOLD}s threshold.",
                recommendation="Slow eyelid closures suggest fatigue. Consider a short break or adjusting sleep.",
            )
        )

    if 0 < avg_duration < INCOMPLETE_DURATION_THRESHOLD:
        flags.append(
            DiagnosticFlag(
                name="Incomplete Blinking",
                severity="info",
                message=f"Average blink duration is {avg_duration}s, under the {INCOMPLETE_DURATION_THRESHOLD}s full-closure threshold.",
                recommendation="Partial blinks dry out the corneal tear film. Practice deliberate full blinks.",
            )
        )

    if longest_streak > PROLONGED_STARE_THRESHOLD:
        flags.append(
            DiagnosticFlag(
                name="Prolonged Stare Alert",
                severity="critical",
                message=f"Longest unblinking stretch was {longest_streak}s (limit: {PROLONGED_STARE_THRESHOLD:.0f}s).",
                recommendation="Extended staring accelerates tear evaporation. Set reminders to blink consciously.",
            )
        )

    if not flags:
        flags.append(
            DiagnosticFlag(
                name="All Clear",
                severity="info",
                message="No eye strain or fatigue risk factors detected.",
                recommendation="Great ocular cadence. Keep it up!",
            )
        )

    return flags


def generate_clinical_findings(metrics: Dict[str, Any]) -> List[str]:
    flags = run_diagnostics(metrics)
    icon_map = {"info": "ℹ️", "warning": "⚠️", "critical": "🔴"}
    return [
        f"{icon_map.get(f.severity, '•')} **{f.name}**: {f.message} *Recommendation:* {f.recommendation}"
        for f in flags
    ]


import random

def generate_gemini_commentary(
    metrics: Dict[str, Any], score: Any, grade: str, api_key: str = ""
) -> Optional[str]:
    """Generates varied, humorous commentary using dynamic personas via Gemini."""
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        flags = run_diagnostics(metrics)
        flag_summary = ", ".join(f"{f.name}: {f.message}" for f in flags)
        streak = metrics.get("max_staring_streak", metrics.get("longest_no_blink_streak", 0.0))

        # Rotate among funny personas on every run
        personas = [
            (
                "an unhinged, high-octane esports shoutcaster screaming about an intense tournament match",
                "Treat their blinking or staring like a high-stakes Grand Finals championship play. Call out their tear film HP bar and eye stamina stats."
            ),
            (
                "an aggressively disappointed Gordon Ramsay reviewing a culinary disaster",
                "Treat their eyes like an overcooked steak or raw chicken. Roast their staring streaks as if they served you dry desert sand."
            ),
            (
                "a hushed, whispering vintage golf commentator describing an agonizing putt",
                "Be deadpan, extremely quiet, and disappointed as if watching someone miss a 2-foot putt on the 18th hole."
            ),
            (
                "a dramatic nature documentary narrator (in the style of David Attenborough)",
                "Observe the human creature at its glowing display terminal struggling in its natural habitat against the harsh winds of dry office air."
            )
        ]

        persona_role, persona_style = random.choice(personas)

        prompt = f"""You are {persona_role}.
{persona_style}

Rules:
- Keep it under 110 words.
- Be punchy, witty, and roast the user based on the numbers below without being offensive.
- Weave the stats directly into your bit.

Telemetry:
- Letter Grade: {grade} (Score: {score}/100)
- Blink Cadence: {metrics.get('bpm', 0.0)} Blinks Per Minute (Target: 15-20 BPM)
- Average Closure Time: {metrics.get('avg_duration', 0.0)} seconds
- Peak Staring Lockdown: {streak} seconds unblinking
- Consistency Score: {metrics.get('consistency_score', 0.0)}%
- Health Warnings: {flag_summary}
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.95,  # Higher creativity for comedic punchlines
                max_output_tokens=300,
            ),
        )

        return response.text.strip() if response.text else None
    except Exception:
        return None


def generate_ai_commentary(
    metrics: Dict[str, Any], score: Any, grade: str, api_key: str = ""
) -> str:
    """Primary commentary generator with deterministic fallback."""
    commentary = generate_gemini_commentary(metrics, score, grade, api_key)
    if commentary:
        return commentary

    # Offline / No-Key Fallback:
    streak = metrics.get("max_staring_streak", metrics.get("longest_no_blink_streak", 0.0))
    return (
        f"**Color Commentary Breakdown (Grade: {grade} | {score}/100)**\n\n"
        f"Taking a look at the telemetry: the subject clocked {metrics.get('bpm', 0.0)} BPM with an average "
        f"closure duration of {metrics.get('avg_duration', 0.0)}s and a peak unblinking lockdown streak of {streak}s. "
        f"Corneal lubrication is getting tested out there on the court. Prioritize intentional micro-recoveries "
        f"to keep your ocular stats in the top tier."
    )

def generate_report(stats, performance, use_llm: bool = False) -> str:
    metrics = {
        "bpm": stats.blinks_per_minute,
        "avg_duration": stats.avg_blink_duration,
        "max_staring_streak": stats.longest_no_blink_streak,
        "consistency_score": stats.blink_consistency,
    }
    if use_llm:
        return generate_ai_commentary(metrics, performance.score, performance.grade)
    return generate_ai_commentary(metrics, performance.score, performance.grade, api_key="")