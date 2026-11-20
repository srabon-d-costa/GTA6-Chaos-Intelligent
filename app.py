import uuid

from src.dynamic_pipeline import analyze_video
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="VICE CITY AI | Chaos Control",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent

REPORTS = ROOT / "outputs" / "reports"
VIDEOS = ROOT / "outputs" / "videos"
HEATMAPS = ROOT / "outputs" / "heatmaps"

CHAOS_PATH = REPORTS / "chaos_scores.csv"
ACTIVITY_PATH = REPORTS / "activity_metrics.csv"
DETECTION_PATH = VIDEOS / "detection_results.csv"
PEAK_PATH = REPORTS / "peak_chaos_moments.csv"
SUMMARY_PATH = REPORTS / "chaos_summary.txt"

PROCESSED_VIDEO = VIDEOS / "Game_detected_web.mp4"
ORIGINAL_VIDEO = ROOT / "data" / "sample_videos" / "Game.mp4"

CROWD_HEATMAP = HEATMAPS / "crowd_heatmap.png"
VEHICLE_HEATMAP = HEATMAPS / "vehicle_heatmap.png"

# ==========================================
# 2. VICE CITY PREMIUM CSS
# ==========================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, rgba(188, 39, 130, 0.12), transparent 35%),
        radial-gradient(ellipse at 90% 10%, rgba(0, 220, 255, 0.10), transparent 35%),
        #080B14;
    color: #F5F5FF;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.block-container {
    padding-top: 1.8rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

h1, h2, h3 {
    color: #F8FAFF !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101020 0%, #090D19 100%);
    border-right: 1px solid rgba(0, 220, 255, 0.20);
}

[data-testid="stSidebar"] h1 {
    font-family: 'Orbitron', sans-serif;
    color: #FF4FA3 !important;
}

[data-testid="stSidebar"] label {
    color: #D8DDF0 !important;
}

[data-testid="stMetric"] {
    background: transparent;
}

.stButton button {
    border: 1px solid #00D9FF;
    border-radius: 10px;
    color: #00D9FF;
    background: rgba(0, 217, 255, 0.07);
}

.stButton button:hover {
    border-color: #FF4FA3;
    color: #FF4FA3;
}

div[data-testid="stDataFrame"] {
    border: 1px solid rgba(0, 217, 255, 0.18);
    border-radius: 12px;
}

hr {
    border-color: rgba(0, 217, 255, 0.18);
}

.hero {
    background:
        linear-gradient(110deg,
            rgba(31, 13, 44, 0.97),
            rgba(12, 29, 48, 0.96));
    border: 1px solid rgba(0, 217, 255, 0.40);
    border-radius: 20px;
    padding: 30px;
    margin-bottom: 25px;
    position: relative;
    overflow: hidden;
}

.hero h1 {
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(25px, 3vw, 39px);
    letter-spacing: 2px;
    background: linear-gradient(90deg, #FF4FA3, #FF9A67, #00D9FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}

.hero p {
    color: #BFC9E2;
    font-size: 14px;
}

.hero-tag {
    color: #00D9FF;
    font-family: 'Orbitron', sans-serif;
    font-size: 10px;
    letter-spacing: 3px;
}

.kpi-card {
    background: linear-gradient(145deg, #171C2C, #101522);
    border: 1px solid rgba(0, 217, 255, 0.22);
    border-radius: 16px;
    padding: 20px;
    min-height: 125px;
    box-shadow: 0 5px 22px rgba(0, 0, 0, 0.22);
}

.kpi-card:hover {
    border-color: #FF4FA3;
    transform: translateY(-2px);
    transition: 0.2s;
}

.kpi-label {
    color: #A6B1CA;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.4px;
}

.kpi-value {
    color: #FFFFFF;
    font-family: 'Orbitron', sans-serif;
    font-size: 27px;
    font-weight: 700;
    margin-top: 12px;
}

.kpi-icon {
    float: right;
    font-size: 23px;
}

.section-heading {
    font-family: 'Orbitron', sans-serif;
    color: #00D9FF;
    font-size: 16px;
    letter-spacing: 1px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(0, 217, 255, 0.20);
    margin-bottom: 17px;
}

.info-panel {
    background: #121827;
    border: 1px solid rgba(255, 79, 163, 0.25);
    border-radius: 15px;
    padding: 18px;
}

.status-pill {
    display: inline-block;
    padding: 6px 12px;
    background: rgba(0, 217, 255, 0.10);
    color: #00D9FF;
    border: 1px solid rgba(0, 217, 255, 0.35);
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}

.footer {
    text-align: center;
    color: #7D88A4;
    font-size: 11px;
    padding: 25px;
    letter-spacing: 1px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. DATA LOADING
# ==========================================

@st.cache_data
def load_csv(path):

    return pd.read_csv(path)


# Use the latest analyzed upload across the entire dashboard.
# If no upload exists in this session, use the original sample reports.

dynamic_results = st.session_state.get("dynamic_results")

if dynamic_results is not None:

    chaos = dynamic_results["chaos"]
    activity = dynamic_results["metrics"]
    detections = dynamic_results["detections"]
    active_peaks = dynamic_results["peaks"]

    active_crowd_heatmap = Path(dynamic_results["crowd_heatmap"])
    active_vehicle_heatmap = Path(dynamic_results["vehicle_heatmap"])

    active_summary = dynamic_results["summary"]
    active_video_name = dynamic_results.get(
        "video_name",
        "Latest uploaded video"
    )

    data_source_label = f"LATEST VIDEO: {active_video_name}"

else:

    required_files = [
        CHAOS_PATH,
        ACTIVITY_PATH,
        DETECTION_PATH
    ]

    missing = [
        str(p) for p in required_files
        if not p.exists()
    ]

    if missing:
        st.error("Analysis files are missing.")
        for file in missing:
            st.write(file)
        st.stop()

    chaos = load_csv(str(CHAOS_PATH))
    activity = load_csv(str(ACTIVITY_PATH))
    detections = load_csv(str(DETECTION_PATH))

    active_peaks = (
        load_csv(str(PEAK_PATH))
        if PEAK_PATH.exists()
        else pd.DataFrame()
    )

    active_crowd_heatmap = CROWD_HEATMAP
    active_vehicle_heatmap = VEHICLE_HEATMAP

    active_summary = (
        SUMMARY_PATH.read_text(encoding="utf-8")
        if SUMMARY_PATH.exists()
        else "No static summary report is available."
    )

    active_video_name = "Original sample analysis"
    data_source_label = "ORIGINAL SAMPLE VIDEO ANALYSIS"


if chaos.empty or activity.empty or detections.empty:
    st.error("The selected analysis contains no data.")
    st.stop()


# ==========================================
# 4. PREPARE STATISTICS
# ==========================================

average_chaos = chaos["chaos_score"].mean()
maximum_chaos = chaos["chaos_score"].max()

peak_row = chaos.loc[
    chaos["chaos_score"].idxmax()
]

peak_time = peak_row["timestamp_sec"]

total_detections = len(detections)

if "track_id" in detections.columns:

    tracked = detections.dropna(
        subset=["track_id"]
    ).drop_duplicates(
        subset=["class", "track_id"]
    )

else:

    tracked = pd.DataFrame()

unique_objects = len(tracked)

if not tracked.empty:

    object_counts = (
        tracked.groupby("class")
        .size()
        .reset_index(name="count")
    )

else:

    object_counts = pd.DataFrame(
        columns=["class", "count"]
    )

# ==========================================
# 5. PLOTLY THEME
# ==========================================

def style_chart(fig, height=350):

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter",
            color="#DCE5F5"
        ),
        height=height,
        margin=dict(l=20, r=20, t=55, b=25),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )

    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.07)"
    )

    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.07)"
    )

    return fig


# ==========================================
# 6. KPI CARD COMPONENT
# ==========================================

def kpi_card(icon, label, value):

    st.markdown(
        f"""
        <div class="kpi-card">
            <span class="kpi-icon">{icon}</span>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================
# 7. SIDEBAR NAVIGATION
# ==========================================

with st.sidebar:

    st.markdown(
        """
        <h1>🌴 VICE CITY AI</h1>
        <p style="color:#9DAAC3;font-size:12px;">
        CHAOS CONTROL CENTER
        </p>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Video Intelligence",
            "Object Analytics",
            "Motion Intelligence",
            "Heatmap Analysis",
            "Chaos Intelligence",
            "Reports"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.markdown(
        """
        <div class="status-pill">
        ● AI SYSTEM ONLINE
        </div>
        <br><br>
        <span style="color:#8995AD;font-size:11px;">
        YOLO11 · BYTETRACK · OPENCV<br>
        VICE CITY VISION v2.0
        </span>
        """,
        unsafe_allow_html=True
    )

# ==========================================
# 8. GLOBAL HERO HEADER
# ==========================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-tag">WELCOME TO THE AI CITY</div>
        <h1>GTA VI CHAOS INTELLIGENCE</h1>
        <p>
        Deep Learning meets the streets of Vice City.
        Detect. Track. Analyze. Understand.
        </p>
        <span class="status-pill">
        🌴 GTA VI INSPIRED AI ANALYTICS
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(f"ACTIVE DATA SOURCE: {data_source_label}")

# ==========================================
# 9. OVERVIEW PAGE
# ==========================================

if page == "Overview":

    st.markdown(
        '<div class="section-heading">SYSTEM OVERVIEW</div>',
        unsafe_allow_html=True
    )

    # KPI cards

    cols = st.columns(4)

    with cols[0]:
        kpi_card(
            "🔥",
            "AVERAGE CHAOS",
            f"{average_chaos:.2f}"
        )

    with cols[1]:
        kpi_card(
            "⚡",
            "PEAK CHAOS",
            f"{maximum_chaos:.2f}"
        )

    with cols[2]:
        kpi_card(
            "🎯",
            "TRACKED OBJECTS",
            f"{unique_objects}"
        )

    with cols[3]:
        kpi_card(
            "⏱️",
            "PEAK MOMENT",
            f"{peak_time:.2f}s"
        )

    st.write("")

    # Gauge and chaos timeline

    left, right = st.columns([1, 2])

    with left:

        st.markdown(
            '<div class="section-heading">CHAOS METER</div>',
            unsafe_allow_html=True
        )

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=average_chaos,
                number={
                    "suffix": "/100",
                    "font": {
                        "size": 33,
                        "color": "#FFFFFF"
                    }
                },
                title={
                    "text": "AVERAGE CHAOS",
                    "font": {
                        "size": 13,
                        "color": "#AEBBD4"
                    }
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickcolor": "#AEBBD4"
                    },
                    "bar": {
                        "color": "#FF4FA3",
                        "thickness": 0.28
                    },
                    "bgcolor": "#151C2B",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 25], "color": "#153B49"},
                        {"range": [25, 50], "color": "#174D63"},
                        {"range": [50, 75], "color": "#573454"},
                        {"range": [75, 100], "color": "#713044"}
                    ],
                    "threshold": {
                        "line": {
                            "color": "#00D9FF",
                            "width": 4
                        },
                        "thickness": 0.8,
                        "value": average_chaos
                    }
                }
            )
        )

        gauge.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=290,
            margin=dict(l=20, r=20, t=55, b=10)
        )

        st.plotly_chart(
            gauge,
            width="stretch"
        )

    with right:

        st.markdown(
            '<div class="section-heading">CHAOS TIMELINE</div>',
            unsafe_allow_html=True
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=chaos["timestamp_sec"],
                y=chaos["chaos_score"],
                mode="lines",
                name="Chaos Score",
                line=dict(
                    color="#00D9FF",
                    width=3
                ),
                fill="tozeroy",
                fillcolor="rgba(0,217,255,0.10)"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[peak_time],
                y=[maximum_chaos],
                mode="markers",
                name="Peak Chaos",
                marker=dict(
                    size=13,
                    color="#FF4FA3",
                    line=dict(
                        color="#FFFFFF",
                        width=2
                    )
                )
            )
        )

        fig.update_layout(
            title="Visual Activity Across Gameplay",
            xaxis_title="Video Time (seconds)",
            yaxis_title="Chaos Score",
            yaxis=dict(range=[0, 100])
        )

        st.plotly_chart(
            style_chart(fig, 340),
            width="stretch"
        )

    # Object distribution

    st.markdown(
        '<div class="section-heading">CITY ACTIVITY</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    with left:

        st.markdown("#### Object Distribution")

        if not object_counts.empty:

            fig = px.bar(
                object_counts,
                x="class",
                y="count",
                color="class",
                color_discrete_sequence=[
                    "#00D9FF",
                    "#FF4FA3",
                    "#FF9D5C",
                    "#A78BFA",
                    "#41E6A1",
                    "#F9D65C"
                ]
            )

            fig.update_layout(
                showlegend=False,
                xaxis_title="Object Class",
                yaxis_title="Unique Tracks"
            )

            st.plotly_chart(
                style_chart(fig),
                width="stretch"
            )

        else:

            st.info("No tracked object data.")

    with right:

        st.markdown("#### Activity Composition")

        fig = go.Figure(
            go.Pie(
                labels=[
                    "Motion",
                    "Crowd",
                    "Vehicles",
                    "Complexity"
                ],
                values=[35, 30, 25, 10],
                hole=0.62,
                marker=dict(
                    colors=[
                        "#00D9FF",
                        "#FF4FA3",
                        "#FF9D5C",
                        "#A78BFA"
                    ]
                )
            )
        )

        fig.update_layout(
            title="Chaos Score Weight Distribution",
            showlegend=True
        )

        st.plotly_chart(
            style_chart(fig),
            width="stretch"
        )

    # Peak moments

    st.markdown(
        '<div class="section-heading">TOP CHAOS MOMENTS</div>',
        unsafe_allow_html=True
    )

    if not active_peaks.empty:

        st.dataframe(
            active_peaks,
            width="stretch",
            hide_index=True
        )

    else:

        st.info("No peak-chaos moments are available for this analysis.")



# ==========================================
# DYNAMIC VIDEO INTELLIGENCE
# ==========================================

elif page == "Video Intelligence":

    st.markdown(
        '<div class="section-heading">'
        'DYNAMIC GAMEPLAY VIDEO INTELLIGENCE'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("### 🎬 AI VIDEO ANALYZER")

    st.caption(
        "Upload any GTA VI gameplay video and generate "
        "a completely new AI intelligence report."
    )

    uploaded_video = st.file_uploader(
        "Upload GTA VI gameplay",
        type=["mp4", "mov", "avi"],
        key="dynamic_video_upload"
    )

    # Clear stale results when a different video is selected

    if uploaded_video is not None:

        upload_signature = (
            uploaded_video.name,
            uploaded_video.size
        )

        if st.session_state.get("upload_signature") != upload_signature:

            st.session_state.pop("dynamic_results", None)

            st.session_state["upload_signature"] = upload_signature

        st.success(
            f"Selected video: {uploaded_video.name}"
        )

        st.video(uploaded_video)

        st.caption(
            f"File size: {uploaded_video.size / (1024 * 1024):.2f} MB"
        )

        st.divider()

        if st.button(
            "🚀 START AI ANALYSIS",
            width="stretch",
            type="primary"
        ):

            run_id = uuid.uuid4().hex[:8]

            run_dir = (
                ROOT / "outputs" / "dynamic_uploads" / run_id
            )

            run_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            suffix = Path(
                uploaded_video.name
            ).suffix.lower()

            if suffix not in [".mp4", ".mov", ".avi"]:
                st.error("Unsupported video format.")
                st.stop()

            input_path = run_dir / f"input{suffix}"

            input_path.write_bytes(
                uploaded_video.getvalue()
            )

            progress_bar = st.progress(
                0.0,
                text="Initializing Vice City AI..."
            )

            status_text = st.empty()

            def update_progress(value, message):

                progress_bar.progress(
                    min(max(float(value), 0.0), 1.0),
                    text=message
                )

                status_text.caption(message)

            try:

                with st.spinner(
                    "Vice City AI is analyzing your video..."
                ):

                    results = analyze_video(
                        video_path=input_path,
                        output_dir=run_dir / "analysis",
                        progress_callback=update_progress,
                        frame_stride=1,
                        target_fps=60
                    )

                results["video_name"] = uploaded_video.name
                st.session_state["dynamic_results"] = results

                progress_bar.progress(
                    1.0,
                    text="AI analysis completed!"
                )

                st.success(
                    "🎉 New video analysis completed successfully!"
                )

            except Exception as error:

                st.error("Video analysis failed.")

                st.exception(error)

    # ======================================
    # DISPLAY DYNAMIC RESULTS
    # ======================================

    results = st.session_state.get(
        "dynamic_results"
    )

    if results is not None:

        metrics = results["metrics"]

        detections = results["detections"]

        peaks = results["peaks"]

        st.divider()

        st.markdown(
            '<div class="section-heading">'
            'NEW VIDEO INTELLIGENCE REPORT'
            '</div>',
            unsafe_allow_html=True
        )

        average_chaos = metrics["chaos_score"].mean()

        maximum_chaos = metrics["chaos_score"].max()

        peak_row = metrics.loc[
            metrics["chaos_score"].idxmax()
        ]

        unique_objects = (
            detections.dropna(
                subset=["track_id"]
            ).drop_duplicates(
                subset=["class", "track_id"]
            ).shape[0]
        )

        # ----------------------------------
        # KPI CARDS
        # ----------------------------------

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Average Chaos",
            f"{average_chaos:.2f}/100"
        )

        c2.metric(
            "Peak Chaos",
            f"{maximum_chaos:.2f}/100"
        )

        c3.metric(
            "Tracked Objects",
            unique_objects
        )

        c4.metric(
            "Peak Timestamp",
            f"{peak_row['timestamp_sec']:.2f}s"
        )

        # ----------------------------------
        # PROCESSED VIDEO
        # ----------------------------------

        st.markdown("### 🎥 AI Processed Gameplay")

        st.caption(
            f"Output: {results.get('output_fps', 60)} FPS · "
            f"{results.get('interpolation_mode', '60 FPS output')}"
        )

        st.video(results["video"])

        # ----------------------------------
        # CHAOS TIMELINE
        # ----------------------------------

        st.markdown("### 📈 Chaos Score Timeline")

        fig = px.line(
            metrics,
            x="timestamp_sec",
            y="chaos_score",
            title="Video-Specific Chaos Analysis"
        )

        fig.update_traces(
            line=dict(
                color="#00D9FF",
                width=3
            ),
            fill="tozeroy",
            fillcolor="rgba(0,217,255,0.10)"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(range=[0, 100])
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        # ----------------------------------
        # ACTIVITY HEATMAPS
        # ----------------------------------

        st.markdown("### 🗺️ ACTIVITY HEATMAPS")

        left, right = st.columns(2)

        with left:

            st.markdown("**Crowd Density**")

            st.image(
                results["crowd_heatmap"],
                width="stretch"
            )

        with right:

            st.markdown("**Vehicle Activity**")

            st.image(
                results["vehicle_heatmap"],
                width="stretch"
            )

        # ----------------------------------
        # PEAK CHAOS MOMENTS
        # ----------------------------------

        st.markdown("### 🔥 PEAK CHAOS MOMENTS")

        st.dataframe(
            peaks,
            width="stretch",
            hide_index=True
        )

        # ----------------------------------
        # DOWNLOAD CENTER
        # ----------------------------------

        st.markdown("### 📥 DOWNLOAD REPORTS")

        d1, d2, d3 = st.columns(3)

        with d1:

            st.download_button(
                "Chaos Summary",
                data=results["summary"],
                file_name="chaos_summary.txt",
                mime="text/plain"
            )

        with d2:

            st.download_button(
                "Detection CSV",
                data=detections.to_csv(
                    index=False
                ),
                file_name="detection_results.csv",
                mime="text/csv"
            )

        with d3:

            st.download_button(
                "Chaos Scores CSV",
                data=metrics.to_csv(
                    index=False
                ),
                file_name="chaos_scores.csv",
                mime="text/csv"
            )



# ==========================================
# 11. OBJECT ANALYTICS - CLEAN DESIGN
# ==========================================

elif page == "Object Analytics":

    st.markdown(
        '<div class="section-heading">'
        'OBJECT DETECTION INTELLIGENCE'
        '</div>',
        unsafe_allow_html=True
    )

    # Summary KPI cards

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Detection Records",
        f"{total_detections:,}"
    )

    c2.metric(
        "Unique Tracked Objects",
        f"{unique_objects:,}"
    )

    c3.metric(
        "Object Categories",
        detections["class"].nunique()
    )

    st.write("")

    # Prepare detection statistics

    detection_distribution = (
        detections.groupby("class")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=True)
    )

    st.markdown("### 🎯 Object Detection Statistics")

    left, right = st.columns(2, gap="large")

    # --------------------------------------
    # UNIQUE TRACKED OBJECTS
    # --------------------------------------

    with left:

        st.markdown("#### Unique Tracked Objects")

        if not object_counts.empty:

            chart_data = object_counts.sort_values(
                "count",
                ascending=True
            )

            fig = px.bar(
                chart_data,
                x="count",
                y="class",
                orientation="h",
                text="count",
                color_discrete_sequence=["#00D9FF"]
            )

            fig.update_traces(
                textposition="outside",
                textfont=dict(size=12),
                cliponaxis=False,
                marker_line_width=0
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                height=380,
                margin=dict(l=15, r=40, t=20, b=35),
                xaxis_title="Number of tracked objects",
                yaxis_title="",
                font=dict(
                    color="#DCE5F5",
                    family="Inter"
                )
            )

            fig.update_xaxes(
                gridcolor="rgba(255,255,255,0.07)"
            )

            fig.update_yaxes(
                gridcolor="rgba(255,255,255,0.03)"
            )

            st.plotly_chart(
                fig,
                width="stretch",
                config={"displayModeBar": False}
            )

        else:

            st.info("No tracking data available.")

    # --------------------------------------
    # TOTAL DETECTION RECORDS
    # --------------------------------------

    with right:

        st.markdown("#### Detection Records by Class")

        fig = px.bar(
            detection_distribution,
            x="count",
            y="class",
            orientation="h",
            text="count",
            color_discrete_sequence=["#FF4FA3"]
        )

        fig.update_traces(
            textposition="outside",
            textfont=dict(size=12),
            cliponaxis=False,
            marker_line_width=0
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            height=380,
            margin=dict(l=15, r=40, t=20, b=35),
            xaxis_title="Detection records",
            yaxis_title="",
            font=dict(
                color="#DCE5F5",
                family="Inter"
            )
        )

        fig.update_xaxes(
            gridcolor="rgba(255,255,255,0.07)"
        )

        fig.update_yaxes(
            gridcolor="rgba(255,255,255,0.03)"
        )

        st.plotly_chart(
            fig,
            width="stretch",
            config={"displayModeBar": False}
        )

    # --------------------------------------
    # DETECTION RECORD TABLE
    # --------------------------------------

    st.divider()

    with st.expander("View Raw Detection Records"):

        st.dataframe(
            detections.head(200),
            width="stretch",
            hide_index=True
        )



# ==========================================
# 12. MOTION INTELLIGENCE
# ==========================================

elif page == "Motion Intelligence":

    st.markdown(
        '<div class="section-heading">MOTION & CROWD INTELLIGENCE</div>',
        unsafe_allow_html=True
    )

    fig = go.Figure()

    series = [
        ("motion_score", "Motion", "#00D9FF"),
        ("crowd_score", "Crowd", "#FF4FA3"),
        ("vehicle_score", "Vehicles", "#FF9D5C")
    ]

    for col, label, color in series:

        if col in activity.columns:

            fig.add_trace(
                go.Scatter(
                    x=activity["timestamp_sec"],
                    y=activity[col],
                    mode="lines",
                    name=label,
                    line=dict(
                        color=color,
                        width=2.5
                    )
                )
            )

    fig.update_layout(
        title="Activity Trends Over Time",
        xaxis_title="Video Time (seconds)",
        yaxis_title="Relative Activity Score",
        yaxis=dict(range=[0, 105])
    )

    st.plotly_chart(
        style_chart(fig, 480),
        width="stretch"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Average Motion",
        f"{activity['motion_score'].mean():.2f}"
    )

    c2.metric(
        "Average Pedestrians",
        f"{activity['person_count'].mean():.2f}"
    )

    c3.metric(
        "Average Vehicles",
        f"{activity['vehicle_count'].mean():.2f}"
    )


# ==========================================
# 13. HEATMAP ANALYSIS
# ==========================================

elif page == "Heatmap Analysis":

    st.markdown(
        '<div class="section-heading">SPATIAL ACTIVITY INTELLIGENCE</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    with left:

        st.markdown("#### 👥 Crowd Density")

        if active_crowd_heatmap.exists():

            st.image(
                str(active_crowd_heatmap),
                width="stretch"
            )

        else:

            st.warning("Crowd heatmap unavailable.")

    with right:

        st.markdown("#### 🚗 Vehicle Activity")

        if active_vehicle_heatmap.exists():

            st.image(
                str(active_vehicle_heatmap),
                width="stretch"
            )

        else:

            st.warning("Vehicle heatmap unavailable.")

    st.info(
        "Heatmaps represent accumulated detected object "
        "locations in image coordinates."
    )


# ==========================================
# 14. CHAOS INTELLIGENCE
# ==========================================

elif page == "Chaos Intelligence":

    st.markdown(
        '<div class="section-heading">VICE CITY CHAOS ENGINE</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns([1, 2])

    with left:

        st.metric(
            "Average Chaos",
            f"{average_chaos:.2f}/100"
        )

        st.metric(
            "Peak Chaos",
            f"{maximum_chaos:.2f}/100"
        )

        st.metric(
            "Peak Timestamp",
            f"{peak_time:.2f}s"
        )

    with right:

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=average_chaos,
                number={"suffix": "/100"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#FF4FA3"},
                    "steps": [
                        {"range": [0, 25], "color": "#153B49"},
                        {"range": [25, 50], "color": "#174D63"},
                        {"range": [50, 75], "color": "#573454"},
                        {"range": [75, 100], "color": "#713044"}
                    ]
                }
            )
        )

        gauge.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=260
        )

        st.plotly_chart(
            gauge,
            width="stretch"
        )

    fig = px.area(
        chaos,
        x="timestamp_sec",
        y="chaos_score",
        title="Chaos Score Timeline"
    )

    fig.update_traces(
        line_color="#FF4FA3",
        fillcolor="rgba(255,79,163,0.15)"
    )

    st.plotly_chart(
        style_chart(fig, 430),
        width="stretch"
    )

    if "chaos_level" in chaos.columns:

        distribution = (
            chaos["chaos_level"]
            .value_counts()
            .reindex(
                ["Low", "Moderate", "High", "Very High"],
                fill_value=0
            )
            .reset_index()
        )

        distribution.columns = ["Level", "Intervals"]

        fig = px.bar(
            distribution,
            x="Level",
            y="Intervals",
            color="Level",
            title="Chaos Level Distribution"
        )

        st.plotly_chart(
            style_chart(fig),
            width="stretch"
        )


# ==========================================
# 15. REPORTS
# ==========================================

elif page == "Reports":

    st.markdown(
        '<div class="section-heading">ANALYTICS & EXPORT CENTER</div>',
        unsafe_allow_html=True
    )

    st.caption(f"Reports for: {active_video_name}")

    if dynamic_results is not None:

        st.markdown("### Latest Video Intelligence Report")

        d1, d2 = st.columns(2)

        with d1:
            st.download_button(
                label="⬇ Download Chaos Summary",
                data=active_summary,
                file_name="chaos_summary.txt",
                mime="text/plain",
                key="dynamic_download_summary",
                width="stretch"
            )

            st.download_button(
                label="⬇ Download Detection Results",
                data=detections.to_csv(index=False),
                file_name="detection_results.csv",
                mime="text/csv",
                key="dynamic_download_detections",
                width="stretch"
            )

        with d2:
            st.download_button(
                label="⬇ Download Activity Metrics",
                data=activity.to_csv(index=False),
                file_name="activity_metrics.csv",
                mime="text/csv",
                key="dynamic_download_activity",
                width="stretch"
            )

            st.download_button(
                label="⬇ Download Chaos Scores",
                data=chaos.to_csv(index=False),
                file_name="chaos_scores.csv",
                mime="text/csv",
                key="dynamic_download_chaos",
                width="stretch"
            )

        st.download_button(
            label="⬇ Download Peak Chaos Moments",
            data=active_peaks.to_csv(index=False),
            file_name="peak_chaos_moments.csv",
            mime="text/csv",
            key="dynamic_download_peaks",
            width="stretch"
        )

        st.markdown("### Chaos Intelligence Summary")
        st.code(active_summary, language="text")

        st.markdown("### Peak Chaos Moments")
        st.dataframe(
            active_peaks,
            width="stretch",
            hide_index=True
        )

    else:

        report_files = {
            "Chaos Scores": CHAOS_PATH,
            "Activity Metrics": ACTIVITY_PATH,
            "Detection Results": DETECTION_PATH,
            "Peak Chaos Moments": PEAK_PATH
        }

        for name, path in report_files.items():

            if path.exists():

                with open(path, "rb") as file:

                    st.download_button(
                        label=f"⬇ Download {name}",
                        data=file.read(),
                        file_name=path.name,
                        mime="text/csv",
                        key=f"download_{name}",
                        width="stretch"
                    )

        st.divider()

        if SUMMARY_PATH.exists():

            st.markdown("### Chaos Intelligence Summary")

            st.code(
                SUMMARY_PATH.read_text(
                    encoding="utf-8"
                ),
                language="text"
            )


# ==========================================
# 16. FOOTER
# ==========================================

st.markdown(
    """
    <div class="footer">
        🌴 VICE CITY CHAOS INTELLIGENCE<br><br>
        POWERED BY COMPUTER VISION & DEEP LEARNING<br>
        GTA6-ViceCity-Vision | AI PROJECT
    </div>
    """,
    unsafe_allow_html=True
)
