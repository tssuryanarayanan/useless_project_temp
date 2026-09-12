import io
import time
import platform
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from blink_detector import BlinkDetector
from blink_stats import BlinkStatsEngine
from performance_index import calculate_performance_index
from ai_report import generate_clinical_findings, generate_ai_commentary

st.set_page_config(
    page_title="BlinkTrack Analytics & Fatigue Monitor",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Clean Theme Styling: Clinical Green & White ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {
        --bg-main: #FFFFFF;
        --bg-card: #F8FAFC;
        --accent-emerald: #10B981;
        --accent-emerald-dark: #059669;
        --forest-green: #065F46;
        --alert-red: #EF4444;
        --text-slate: #1E293B;
        --text-muted: #64748B;
        --border-color: #E2E8F0;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: var(--bg-main);
        color: var(--text-slate);
    }

    /* Hide default sidebar completely */
    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
        visibility: hidden !important;
    }

    /* Hide Streamlit Top Chrome */
    #MainMenu, header, footer, [data-testid="stToolbar"], .stDeployButton {
        display: none !important;
        visibility: hidden !important;
        height: 0% !important;
    }

    /* Style native Streamlit bordered containers cleanly */
    [data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.06) !important;
        padding: 20px !important;
    }

    /* Primary Emerald Button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #10B981 0%, #059669 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        padding: 12px 28px !important;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.25) !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.5) !important;
        transform: translateY(-1px) !important;
    }

    /* Secondary and Download Buttons */
    div.stButton > button,
    div[data-testid="stDownloadButton"] > button {
        background-color: #FFFFFF !important;
        color: #065F46 !important;
        border: 1.5px solid #10B981 !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04) !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {
        background-color: #F0FDF4 !important;
        color: #047857 !important;
        border-color: #059669 !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15) !important;
        transform: translateY(-1px) !important;
    }

    /* Stop Action Button */
    button[key="btn_stop_live"] {
        background: linear-gradient(90deg, #EF4444 0%, #DC2626 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.25) !important;
    }
    button[key="btn_stop_live"]:hover {
        background: #B91C1C !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 16px rgba(239, 68, 68, 0.45) !important;
    }

    /* Monitoring Top Command Header */
    .monitoring-topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 14px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.06);
    }

    .status-pill-live {
        background: rgba(16, 185, 129, 0.12);
        color: #059669;
        border: 1px solid #10B981;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.8rem;
        letter-spacing: 0.8px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }

    /* Video HUD Overlay Badge */
    .glass-hud {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(8px);
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 8px 16px;
        margin-bottom: 8px;
    }

    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
    }
    div[data-testid="stMetricLabel"] p {
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        color: #64748B !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetricValue"] div {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        color: #065F46 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_green_logo(size=220):
    st.markdown(
        f"""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin-bottom: 12px;">
            <svg width="{size}" height="{int(size * 0.4)}" viewBox="0 0 400 160" fill="none" xmlns="http://www.w3.org/2000/svg" style="filter: drop-shadow(0 4px 14px rgba(16, 185, 129, 0.25));">
                <defs>
                    <linearGradient id="emeraldGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stop-color="#34D399" />
                        <stop offset="50%" stop-color="#10B981" />
                        <stop offset="100%" stop-color="#059669" />
                    </linearGradient>
                </defs>
                <path d="M 50 80 Q 200 -20 350 70 Q 280 40 200 40 Q 120 40 50 80 Z" fill="url(#emeraldGrad)" />
                <path d="M 70 85 Q 200 150 330 95 Q 250 125 180 115 Q 110 105 70 85 Z" fill="url(#emeraldGrad)" />
                <circle cx="200" cy="75" r="42" stroke="url(#emeraldGrad)" stroke-width="9" fill="none" />
                <circle cx="200" cy="75" r="26" fill="url(#emeraldGrad)" />
                <circle cx="209" cy="68" r="7" fill="#FFFFFF" opacity="0.95" />
            </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )


def create_donut_chart(score: float, grade: str) -> go.Figure:
    if score >= 70:
        accent_color = "#10B981"
        remaining_color = "rgba(16, 185, 129, 0.12)"
    elif score >= 40:
        accent_color = "#F59E0B"
        remaining_color = "rgba(245, 158, 11, 0.12)"
    else:
        accent_color = "#EF4444"
        remaining_color = "rgba(239, 68, 68, 0.12)"

    fig = go.Figure(
        data=[
            go.Pie(
                values=[score, max(0.0, 100.0 - score)],
                hole=0.74,
                sort=False,
                direction="clockwise",
                marker=dict(colors=[accent_color, remaining_color]),
                hoverinfo="none",
                textinfo="none",
            )
        ]
    )

    fig.update_layout(
        showlegend=False,
        height=190,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[
            dict(
                text=f"<b>{int(score)}</b><span style='font-size:16px;color:#64748B'>/100</span><br><span style='font-size:15px;color:{accent_color};font-weight:700;'>GRADE {grade}</span>",
                x=0.5,
                y=0.5,
                font=dict(size=26, color="#065F46", family="Inter"),
                showarrow=False,
            )
        ],
    )
    return fig


def draw_hud_overlay(
    frame: np.ndarray,
    ear: float,
    threshold: float,
    is_blinking: bool,
    face_detected: bool,
) -> np.ndarray:
    h, w, _ = frame.shape
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 46), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    if not face_detected:
        cv2.putText(
            frame,
            "ALERT: NO FACE DETECTED",
            (18, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 220),
            2,
            cv2.LINE_AA,
        )
        return frame

    ear_color = (16, 185, 129) if ear >= threshold else (0, 0, 220)
    cv2.putText(
        frame,
        f"EAR: {ear:.3f}",
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        ear_color,
        2,
        cv2.LINE_AA,
    )

    if is_blinking:
        cv2.putText(
            frame,
            "STATUS: BLINKING",
            (w - 215, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 100, 240),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame,
            "STATUS: EYES OPEN",
            (w - 215, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (16, 185, 129),
            2,
            cv2.LINE_AA,
        )

    return frame


def main():
    EAR_THRESHOLD = 0.21

    # --- Session State Management ---
    if "stage" not in st.session_state:
        st.session_state.stage = "landing"
    if "session_stats" not in st.session_state:
        st.session_state.session_stats = None
    if "final_report_ready" not in st.session_state:
        st.session_state.final_report_ready = False

    # Stored timer defaults
    if "cfg_duration_mins" not in st.session_state:
        st.session_state.cfg_duration_mins = 2
    if "cfg_duration_secs" not in st.session_state:
        st.session_state.cfg_duration_secs = 0

    # =========================================================================
    # STAGE 1: LANDING SCREEN
    # =========================================================================
    if st.session_state.stage == "landing":
        st.markdown("<div style='height: 12vh;'></div>", unsafe_allow_html=True)
        _, center_col, _ = st.columns([1, 2, 1])

        with center_col:
            with st.container(border=True):
                render_green_logo(size=240)
                st.markdown(
                    """
                    <div style='text-align: center;'>
                        <h1 style='color: #065F46; font-size: 2.3rem; font-weight: 800; margin-bottom: 6px; letter-spacing: -0.5px;'>
                            BlinkTrack Analytics
                        </h1>
                        <p style='color: #64748B; font-size: 1.05rem; margin-bottom: 32px; font-weight: 500;'>
                            Real-time ergonomic fatigue monitoring using computer vision.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("Start Blink Analysis", use_container_width=True, type="primary"):
                    st.session_state.stage = "configure"
                    st.rerun()

    # =========================================================================
    # STAGE 2: CONFIGURATION CARD
    # =========================================================================
    elif st.session_state.stage == "configure":
        st.markdown("<div style='height: 8vh;'></div>", unsafe_allow_html=True)
        _, center_col, _ = st.columns([1, 1.6, 1])

        with center_col:
            with st.container(border=True):
                render_green_logo(size=140)
                st.markdown(
                    """
                    <div style='text-align: center; margin-bottom: 20px;'>
                        <h2 style='color: #065F46; font-size: 1.65rem; font-weight: 800; margin-bottom: 4px;'>
                            Session Settings
                        </h2>
                        <p style='color: #64748B; font-size: 0.9rem;'>
                            Set your monitoring duration before initializing camera.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("<p style='font-size:0.85rem;font-weight:700;color:#065F46;margin-bottom:8px;'>⏱️ SESSION DURATION (MAX 60 MIN)</p>", unsafe_allow_html=True)
                col_m, col_s = st.columns(2)
                with col_m:
                    st.session_state.cfg_duration_mins = st.number_input(
                        "MINUTES", min_value=0, max_value=60, value=int(st.session_state.cfg_duration_mins), step=1
                    )
                with col_s:
                    st.session_state.cfg_duration_secs = st.number_input(
                        "SECONDS", min_value=0, max_value=59, value=int(st.session_state.cfg_duration_secs), step=5
                    )

                total_seconds = min(3600, (st.session_state.cfg_duration_mins * 60) + st.session_state.cfg_duration_secs)
                if total_seconds == 0:
                    total_seconds = 60

                st.markdown(
                    f"<div style='text-align:center;padding:10px;background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;font-size:0.92rem;color:#64748B;margin-top:10px;margin-bottom:24px;'>"
                    f"Configured Runtime: <b style='color:#065F46;font-size:1.05rem;'>{total_seconds // 60:02d}:{total_seconds % 60:02d}</b></div>",
                    unsafe_allow_html=True,
                )

                btn_col1, btn_col2 = st.columns([2, 1])
                with btn_col1:
                    if st.button("Begin Monitoring", use_container_width=True, type="primary"):
                        st.session_state.stage = "monitoring"
                        st.session_state.final_report_ready = False
                        st.rerun()
                with btn_col2:
                    if st.button("Back", use_container_width=True):
                        st.session_state.stage = "landing"
                        st.rerun()

    # =========================================================================
    # STAGE 3: LIVE MONITORING & RESULTS VIEW
    # =========================================================================
    elif st.session_state.stage == "monitoring":
        total_seconds = min(3600, (st.session_state.cfg_duration_mins * 60) + st.session_state.cfg_duration_secs)
        if total_seconds == 0:
            total_seconds = 60

        # Top Bar
        timer_placeholder = st.empty()
        timer_placeholder.markdown(
            f"""
            <div class="monitoring-topbar">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.4rem;">👁️</span>
                    <div>
                        <span style="font-weight: 800; color: #065F46; font-size: 1.1rem; display: block; line-height: 1.2;">BlinkTrack Live Console</span>
                        <span style="color: #64748B; font-size: 0.78rem; font-weight: 500;">Clinical Edge AI Ocular Telemetry</span>
                    </div>
                </div>
                <div>
                    <span class="status-pill-live">🔴 LIVE</span>
                </div>
                <div style="font-weight: 700; color: #065F46; font-size: 0.95rem;">
                    ⏱️ Time Remaining: {total_seconds // 60:02d}:{total_seconds % 60:02d}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        video_col, stats_col = st.columns([2, 1])

        with video_col:
            with st.container(border=True):
                video_hud_placeholder = st.empty()
                st_frame = st.empty()

        with stats_col:
            with st.container(border=True):
                st.markdown("<p style='font-size:0.82rem;font-weight:700;color:#64748B;margin:0 0 6px 0;text-transform:uppercase;'>Performance Index</p>", unsafe_allow_html=True)
                gauge_placeholder = st.empty()
                gauge_placeholder.plotly_chart(
                    create_donut_chart(100.0, "A+"),
                    key="monitoring_donut_init",
                    use_container_width=True,
                )

                st.markdown("<hr style='margin: 12px 0; border: none; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)

                r1_c1, r1_c2 = st.columns(2)
                kpi_bpm = r1_c1.empty()
                kpi_count = r1_c2.empty()

                r2_c1, r2_c2 = st.columns(2)
                kpi_dur = r2_c1.empty()
                kpi_streak = r2_c2.empty()

                kpi_bpm.metric("Blinks / Min", "0.0", help="Calculated rate of blinks per minute (Ideal: 15–20 BPM)")
                kpi_count.metric("Total Blinks", "0", help="Cumulative detected blinks")
                kpi_dur.metric("Avg Duration", "0.00s", help="Average eyelid full closure time in seconds")
                kpi_streak.metric("No-Blink Streak", "0.0s", help="Peak continuous open-eye duration without a full blink")

                st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
                stop_clicked = st.button("Stop Session", key="btn_stop_live", use_container_width=True)
                if stop_clicked:
                    st.session_state.stage = "configure"
                    st.rerun()

        # Camera & State Ingestion Loop
        if not st.session_state.final_report_ready:
            stats_engine = BlinkStatsEngine(session_start_time=time.time())
            st.session_state.session_stats = stats_engine

            detector = BlinkDetector(
                ear_threshold=EAR_THRESHOLD,
                smoothing_window=3,
            )

            backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
            cap = cv2.VideoCapture(0, backend)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            start_time = time.time()
            frame_counter = 0

            try:
                while st.session_state.stage == "monitoring":
                    elapsed = time.time() - start_time
                    remaining = max(0, total_seconds - elapsed)

                    mins, secs = divmod(int(remaining), 60)
                    timer_placeholder.markdown(
                        f"""
                        <div class="monitoring-topbar">
                            <div style="display: flex; align-items: center; gap: 10px;">
                                <span style="font-size: 1.4rem;">👁️</span>
                                <div>
                                    <span style="font-weight: 800; color: #065F46; font-size: 1.1rem; display: block; line-height: 1.2;">BlinkTrack Live Console</span>
                                    <span style="color: #64748B; font-size: 0.78rem; font-weight: 500;">Clinical Edge AI Ocular Telemetry</span>
                                </div>
                            </div>
                            <div>
                                <span class="status-pill-live">🔴 LIVE</span>
                            </div>
                            <div style="font-weight: 700; color: #065F46; font-size: 0.95rem;">
                                ⏱️ Time Remaining: {mins:02d}:{secs:02d}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if elapsed >= total_seconds:
                        st.balloons()
                        break

                    ret, frame = cap.read()
                    if not ret:
                        st.error("Hardware Camera Stream Offline or Disconnected.")
                        break

                    frame_counter += 1
                    display_frame = cv2.flip(frame, 1)
                    rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

                    detection_result = detector.process_frame(rgb_frame, timestamp_s=time.time())
                    smoothed_ear = detection_result["ear"]
                    face_detected = detection_result["face_detected"]
                    is_blinking = detection_result["is_blinking"]
                    completed_blink = detection_result["blink_event"]

                    if completed_blink:
                        stats_engine.register_blink(completed_blink)

                    status_color = "#EF4444" if is_blinking else "#10B981"
                    status_text = "EYES CLOSED" if is_blinking else "EYES OPEN"

                    video_hud_placeholder.markdown(
                        f"""
                        <div class="glass-hud">
                            <span style="font-weight:700;font-size:0.85rem;color:{status_color};">STATUS: {status_text}</span>
                            <span style="font-weight:600;font-size:0.85rem;color:#1E293B;">
                                EAR: <b style="color:{status_color};">{smoothed_ear:.3f}</b>
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    annotated_frame = draw_hud_overlay(
                        display_frame,
                        smoothed_ear,
                        EAR_THRESHOLD,
                        is_blinking,
                        face_detected,
                    )

                    st_frame.image(annotated_frame, channels="BGR", use_container_width=True)

                    if frame_counter % 8 == 0:
                        live_metrics = stats_engine.compute_metrics()
                        score, grade = calculate_performance_index(live_metrics)

                        gauge_placeholder.plotly_chart(
                            create_donut_chart(score, grade),
                            key=f"live_gauge_{frame_counter}",
                            use_container_width=True,
                        )
                        kpi_bpm.metric("Blinks / Min", f"{live_metrics['bpm']}")
                        kpi_count.metric("Total Blinks", f"{live_metrics['total_blinks']}")
                        kpi_dur.metric("Avg Duration", f"{live_metrics['avg_duration']}s")
                        kpi_streak.metric("No-Blink Streak", f"{live_metrics['max_staring_streak']}s")

            finally:
                cap.release()
                detector.close()
                st.session_state.final_report_ready = True
                video_hud_placeholder.empty()

        # Diagnostic Report Screen
        if st.session_state.final_report_ready and st.session_state.session_stats:
            st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
            st.markdown(
                """
                <h3 style='color: #065F46; font-size: 1.6rem; font-weight: 800; margin-bottom: 12px;'>
                    📋 Ergonomic Health & Diagnostic Scorecard
                </h3>
                """,
                unsafe_allow_html=True,
            )

            final_metrics = st.session_state.session_stats.compute_metrics()
            final_score, final_grade = calculate_performance_index(final_metrics)

            col_l, col_r = st.columns([1.1, 1.1])
            with col_l:
                with st.container(border=True):
                    st.markdown("<p style='font-size:0.85rem;font-weight:700;color:#64748B;'>SESSION PERFORMANCE INDEX</p>", unsafe_allow_html=True)
                    st.plotly_chart(
                        create_donut_chart(final_score, final_grade),
                        key="final_summary_report_donut",
                        use_container_width=True,
                    )
                    st.markdown(
                        f"""
                        <div style='background:#FFFFFF;border:1px solid #E2E8F0;border-radius:12px;padding:16px;font-size:0.88rem;color:#1E293B;line-height:1.75;'>
                        • <b>Composite Score:</b> Grade <span style='color:#10B981;'><b>{final_grade}</b></span> ({final_score}/100)<br>
                        • <b>Total Blinks Monitored:</b> {final_metrics['total_blinks']}<br>
                        • <b>Blink Cadence:</b> {final_metrics['bpm']} BPM (Ideal: 15–20)<br>
                        • <b>Blink Duration Range:</b> {final_metrics['fastest_duration']}s – {final_metrics['longest_duration']}s<br>
                        • <b>Longest Staring Lockdown:</b> {final_metrics['max_staring_streak']}s<br>
                        • <b>Rhythm Stability Index:</b> {final_metrics['rhythm_score']}%
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with col_r:
                with st.container(border=True):
                    st.markdown("<p style='font-size:0.85rem;font-weight:700;color:#64748B;'>CLINICAL DIAGNOSTICS & AI REVIEW</p>", unsafe_allow_html=True)
                    findings = generate_clinical_findings(final_metrics)
                    for f in findings:
                        st.markdown(f)

                    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
                    with st.spinner("Compiling ocular telemetry insights..."):
                        gemini_key = st.secrets.get("GEMINI_API_KEY", "")
                        commentary = generate_ai_commentary(
                            final_metrics, final_score, final_grade, api_key=gemini_key
                        )
                        st.success(commentary)

            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown("### 📥 Session Telemetry Export")
                st.caption("Export high-frequency blink timestamps and duration records for ergonomic compliance and clinical audits.")
                df_blinks = st.session_state.session_stats.to_dataframe()
                csv_data = df_blinks.to_csv(index=False).encode("utf-8")

                col_down, col_rst = st.columns([2, 1])
                with col_down:
                    st.download_button(
                        label="Download Blink Log (CSV)",
                        data=csv_data,
                        file_name="blinktrack_session_log.csv",
                        mime="text/csv",
                        key="btn_download_csv",
                    )
                with col_rst:
                    if st.button("New Session", use_container_width=True):
                        st.session_state.stage = "configure"
                        st.session_state.final_report_ready = False
                        st.rerun()


if __name__ == "__main__":
    main()