"""
╔══════════════════════════════════════════════════════════════╗
║          AI Smart Attendance System — Main App               ║
║          Built with Streamlit + OpenCV + LBPH AI             ║
╚══════════════════════════════════════════════════════════════╝

Entry point for the Streamlit dashboard.
Run with: 


Pages (Sidebar Navigation):
  🏠  Home / Dashboard
  📝  Student Registration
  🧠  Train AI Model
  📸  Mark Attendance
  📋  Attendance Records
  📊  Analytics & Charts
  ⚙️  Settings & Tools

Demonstrates ALL required course topics:
  Variables, Expressions, Operators, Loops, Control Structures,
  Functions, File Handling, Exception Handling, Regex, Strings,
  Lists, Tuples, Dictionaries, OOP, NumPy, Pandas, Matplotlib,
  OpenCV Object Detection, AI/Deep Learning Concepts, Git
"""

# ──────────────────────────────────────────────────────────────
#  STANDARD LIBRARY IMPORTS
# ──────────────────────────────────────────────────────────────
import os
import sys
import io
import time
import datetime

# ──────────────────────────────────────────────────────────────
#  THIRD-PARTY IMPORTS
# ──────────────────────────────────────────────────────────────
import cv2
import numpy  as np
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────
#  PROJECT IMPORTS
# ──────────────────────────────────────────────────────────────
# Make sure the project root is on sys.path so sub-modules resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from utils.helpers     import create_project_directories, logger, model_exists
from utils.validators  import VALID_DEPARTMENTS

from modules.file_handler    import FileHandler
from modules.student_manager import StudentManager
from modules.face_recognizer import FaceRecognizer
from modules.attendance_manager import AttendanceManager
from modules.data_visualizer import DataVisualizer


# ══════════════════════════════════════════════════════════════
#  ONE-TIME SETUP
# ══════════════════════════════════════════════════════════════
# Create data directories before anything else runs
create_project_directories()


# ══════════════════════════════════════════════════════════════
#  STREAMLIT PAGE CONFIG  (must be the FIRST st.* call)
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title = cfg.PAGE_TITLE,
    page_icon  = cfg.PAGE_ICON,
    layout     = cfg.LAYOUT,
    initial_sidebar_state = "expanded",
)


# ══════════════════════════════════════════════════════════════
#  GLOBAL CSS  (injected into the Streamlit app)
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Metric cards ─────────────────────────────── */
div[data-testid="metric-container"] {
    background   : #1A1A2E;
    border       : 1px solid #2C2C2C;
    border-radius: 12px;
    padding      : 16px 20px;
    margin-bottom: 8px;
}
div[data-testid="metric-container"] label {
    color    : #A0A0B0 !important;
    font-size: 13px;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color    : #E0E0FF !important;
    font-size: 2rem;
    font-weight: bold;
}

/* ── Section headers ──────────────────────────── */
.section-header {
    background   : linear-gradient(135deg, #1A1A2E, #16213E);
    border-left  : 4px solid #4CAF50;
    border-radius: 8px;
    padding      : 10px 18px;
    margin-bottom: 16px;
    color        : #FAFAFA;
    font-size    : 1.1rem;
    font-weight  : 600;
}

/* ── Info / success / warning banners ────────── */
.info-box {
    background   : rgba(33,150,243,0.12);
    border       : 1px solid #2196F3;
    border-radius: 8px;
    padding      : 12px 18px;
    color        : #90CAF9;
}
.success-box {
    background   : rgba(76,175,80,0.12);
    border       : 1px solid #4CAF50;
    border-radius: 8px;
    padding      : 12px 18px;
    color        : #A5D6A7;
}
.warning-box {
    background   : rgba(255,152,0,0.12);
    border       : 1px solid #FF9800;
    border-radius: 8px;
    padding      : 12px 18px;
    color        : #FFCC80;
}

/* ── DataFrame table styling ─────────────────── */
div[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow     : hidden;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  CACHED RESOURCE INITIALISATION
#  @st.cache_resource ensures these objects are created once
#  and reused across all user interactions.
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def get_student_manager() -> StudentManager:
    return StudentManager()

@st.cache_resource
def get_attendance_manager() -> AttendanceManager:
    return AttendanceManager()

@st.cache_resource
def get_visualizer() -> DataVisualizer:
    return DataVisualizer()

@st.cache_resource
def get_face_recognizer() -> FaceRecognizer:
    return FaceRecognizer()


# ══════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════

def section_header(icon: str, title: str) -> None:
    """Render a styled section header inside a page."""
    st.markdown(
        f'<div class="section-header">{icon} &nbsp; {title}</div>',
        unsafe_allow_html=True,
    )


def info_box(text: str) -> None:
    st.markdown(f'<div class="info-box">ℹ️ &nbsp; {text}</div>',
                unsafe_allow_html=True)

def success_box(text: str) -> None:
    st.markdown(f'<div class="success-box">✅ &nbsp; {text}</div>',
                unsafe_allow_html=True)

def warning_box(text: str) -> None:
    st.markdown(f'<div class="warning-box">⚠️ &nbsp; {text}</div>',
                unsafe_allow_html=True)


def numpy_from_uploaded(file_bytes) -> np.ndarray:
    """
    Convert bytes from st.camera_input / st.file_uploader to
    an OpenCV-compatible BGR NumPy array.
    """
    file_bytes = np.asarray(bytearray(file_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


# ══════════════════════════════════════════════════════════════
#  SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════
def render_sidebar() -> str:
    """
    Build the sidebar with navigation and quick stats.
    Returns the name of the currently selected page.
    """
    with st.sidebar:
        # ── Logo / title ──────────────────────────────────
        st.markdown("""
        <div style='text-align:center; padding: 10px 0 20px'>
            <div style='font-size:48px'>🎓</div>
            <div style='font-size:1.2rem; font-weight:700; color:#4CAF50'>
                AI Attendance System
            </div>
            <div style='font-size:0.75rem; color:#888; margin-top:4px'>
                Face Recognition Powered
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── Navigation radio buttons ───────────────────────
        pages = {
            "🏠  Home / Dashboard"    : "Home",
            "📝  Student Registration" : "Register",
            "🧠  Train AI Model"       : "Train",
            "📸  Mark Attendance"      : "Attendance",
            "📋  Attendance Records"   : "Records",
            "📊  Analytics & Charts"   : "Analytics",
            "⚙️  Settings & Tools"     : "Settings",
        }

        selected_label = st.radio("Navigate", list(pages.keys()),
                                   label_visibility="collapsed")
        page = pages[selected_label]

        st.divider()

        # ── Quick stats panel ──────────────────────────────
        st.markdown("**📌 Quick Stats**")

        try:
            sm      = get_student_manager()
            am      = get_attendance_manager()
            today   = am.get_today_statistics()

            st.metric("👥 Registered Students", today["total_students"])
            st.metric("✅ Present Today",        today["present"])
            st.metric("❌ Absent Today",          today["absent"])
            st.metric("📅 Date",                 today["date"])

            # Colour-coded attendance rate
            pct = today["percentage"]
            color = "#4CAF50" if pct >= 75 else "#FF9800" if pct >= 50 else "#F44336"
            st.markdown(
                f'<div style="font-size:0.85rem; margin-top:8px">'
                f'Today\'s Rate: <b style="color:{color}">{pct:.1f}%</b></div>',
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.caption(f"Stats unavailable: {e}")

        st.divider()

        # ── Model status indicator ─────────────────────────
        if model_exists():
            st.markdown("🟢 **AI Model:** Trained")
        else:
            st.markdown("🔴 **AI Model:** Not Trained")
            st.caption("Go to 'Train AI Model' after registering students.")

    return page


# ══════════════════════════════════════════════════════════════
#  PAGE 1 — HOME / DASHBOARD
# ══════════════════════════════════════════════════════════════
def page_home() -> None:
    """Landing page with overview metrics and recent activity."""
    # ── Title banner ──────────────────────────────────────
    st.markdown("""
    <div style='background: linear-gradient(135deg, #0F3460, #16213E);
                border-radius: 16px; padding: 28px 36px; margin-bottom: 24px;
                border: 1px solid #1A1A4E;'>
        <h1 style='color:#4CAF50; margin:0; font-size:2.2rem'>
            🎓 AI Smart Attendance System
        </h1>
        <p style='color:#9E9EBE; margin:8px 0 0; font-size:1rem'>
            Face Recognition Based Attendance Tracking &nbsp;·&nbsp;
            Powered by OpenCV LBPH Algorithm
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Metrics Row ───────────────────────────────────
    sm    = get_student_manager()
    am    = get_attendance_manager()
    today = am.get_today_statistics()
    all_df = am.get_all_attendance()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("👥 Total Students", today["total_students"])
    with col2:
        st.metric("✅ Present Today",  today["present"])
    with col3:
        st.metric("❌ Absent Today",   today["absent"])
    with col4:
        pct = today["percentage"]
        st.metric("📊 Today's Rate",   f"{pct:.1f}%")
    with col5:
        total_records = len(all_df) if not all_df.empty else 0
        st.metric("🗂️ Total Records", total_records)

    st.divider()

    # ── Two column layout: charts + table ─────────────────
    col_left, col_right = st.columns([1.5, 1])

    with col_left:
        section_header("📊", "Daily Attendance Overview")
        daily_df = am.get_daily_summary()
        fig = DataVisualizer.plot_daily_attendance(daily_df)
        st.pyplot(fig)

    with col_right:
        section_header("🥧", "Today's Attendance Split")
        fig2 = DataVisualizer.plot_today_pie(today)
        st.pyplot(fig2)

    st.divider()

    # ── Recent attendance table ───────────────────────────
    section_header("🕐", "Recent Attendance (Today)")
    today_df = am.get_today_attendance()

    if today_df.empty:
        info_box("No attendance has been recorded today yet.")
    else:
        st.dataframe(
            today_df.sort_values("Time", ascending=False).head(20),
            use_container_width=True,
            hide_index=True,
        )

    # ── How to use guide ──────────────────────────────────
    st.divider()
    section_header("📖", "How to Use This System")
    steps = [
        ("Step 1 — Register Students",
         "Go to 📝 Student Registration, fill in student details, "
         "then capture 50 face samples using the webcam."),
        ("Step 2 — Train the AI Model",
         "Go to 🧠 Train AI Model and click Train. The LBPH algorithm "
         "will learn each student's face signature from the captured images."),
        ("Step 3 — Mark Attendance",
         "Go to 📸 Mark Attendance, take a photo or run the live webcam. "
         "The AI recognises faces and automatically marks attendance in the CSV."),
        ("Step 4 — View Records",
         "Visit 📋 Attendance Records to filter, search, and export data."),
        ("Step 5 — Analyse",
         "Visit 📊 Analytics for charts, trends, and student rankings."),
    ]
    for title, desc in steps:
        with st.expander(f"**{title}**"):
            st.write(desc)


# ══════════════════════════════════════════════════════════════
#  PAGE 2 — STUDENT REGISTRATION
# ══════════════════════════════════════════════════════════════
def page_register() -> None:
    """Register new students and capture face training images."""
    st.title("📝 Student Registration")
    sm = get_student_manager()

    # ── Tabs: Register | View Students ────────────────────
    tab_register, tab_view = st.tabs(["➕ Register New Student", "👥 View All Students"])

    # ────────────────────────────────────────────────────────
    with tab_register:
        section_header("📋", "Student Details")
        info_box(
            "Fill in all fields below, click 'Register Student', then capture "
            "face samples using your webcam."
        )

        # ── Registration form ──────────────────────────────
        with st.form("registration_form", clear_on_submit=False):
            col1, col2 = st.columns(2)

            with col1:
                student_id = st.text_input(
                    "Student ID *",
                    placeholder="e.g. CS2021001",
                    help="4–12 characters, letters and digits only."
                )
                student_name = st.text_input(
                    "Full Name *",
                    placeholder="e.g. Ali Ahmed",
                    help="Letters and spaces only, 2–50 characters."
                )

            with col2:
                department = st.selectbox(
                    "Department *",
                    options=sorted(VALID_DEPARTMENTS),
                )
                semester = st.selectbox(
                    "Semester *",
                    options=list(range(1, 9)),
                    index=0,
                )

            submitted = st.form_submit_button(
                "Register Student",
                type="primary",
                use_container_width=True,
            )

            if submitted:
                ok, msg, student = sm.register_student(
                    student_id, student_name, department, semester
                )
                if ok:
                    st.success(msg)
                    # Store the new student ID in session state for face capture
                    st.session_state["pending_face_student"] = student.student_id
                else:
                    st.error(msg)

        # ── Face capture section ───────────────────────────
        st.divider()
        section_header("📷", "Capture Face Training Samples")

        # Which student are we capturing for?
        all_students = sm.get_all_students()
        if all_students.empty:
            warning_box("No students registered yet. Register a student first.")
        else:
            # Pre-select the student just registered (if any)
            default_sid = st.session_state.get("pending_face_student", "")
            student_ids = all_students["Student_ID"].tolist()

            default_idx = 0
            if default_sid in student_ids:
                default_idx = student_ids.index(default_sid)

            target_id = st.selectbox(
                "Select Student for Face Capture",
                options=student_ids,
                index=default_idx,
            )

            # Show current sample count
            from utils.helpers import student_face_dir, count_files_in_dir
            face_dir      = student_face_dir(target_id)
            current_count = count_files_in_dir(face_dir, ".jpg")

            col_info, col_target = st.columns(2)
            col_info.metric("Current Face Samples", current_count)
            col_target.metric("Target Samples",     cfg.FACE_SAMPLE_COUNT)

            # Progress bar
            prog = min(current_count / cfg.FACE_SAMPLE_COUNT, 1.0)
            st.progress(prog, text=f"{current_count}/{cfg.FACE_SAMPLE_COUNT} samples")

            st.markdown(
                "📸 **Take snapshot photos** below. Each photo may capture one "
                "face sample. Take photos from different angles for best results."
            )

            # Camera input — user clicks to take a photo
            camera_img = st.camera_input(
                "Take a photo (face the camera clearly)",
                key=f"cam_{target_id}",
            )

            if camera_img is not None:
                # Convert the uploaded photo bytes to a NumPy image array
                img_array = numpy_from_uploaded(camera_img)

                if img_array is not None:
                    # Get the next sample index
                    next_idx  = count_files_in_dir(face_dir, ".jpg")
                    ok, msg   = sm.capture_face_samples_from_image(
                        target_id, img_array, next_idx
                    )
                    if ok:
                        st.success(msg)
                        # Refresh count display
                        new_count = count_files_in_dir(face_dir, ".jpg")
                        if new_count >= cfg.FACE_SAMPLE_COUNT:
                            success_box(
                                f"✅ {cfg.FACE_SAMPLE_COUNT} samples collected for "
                                f"**{target_id}**! You can now train the model."
                            )
                    else:
                        st.warning(msg)
                else:
                    st.error("Could not read the captured image.")

            # Tip for variety
            with st.expander("💡 Tips for best face recognition accuracy"):
                st.markdown("""
                - Take photos in **different lighting conditions**
                - **Slightly move your head** left, right, up, down between photos
                - Avoid covering your face with hands, glasses (at least some shots)
                - Make sure the **entire face is visible** in the frame
                - Aim to collect at least **50 samples** per student
                """)

    # ────────────────────────────────────────────────────────
    with tab_view:
        section_header("👥", "Registered Students")

        # Search box
        search_q = st.text_input("🔍 Search by ID or Name", placeholder="Type to search…")
        df       = sm.search_students(search_q) if search_q else sm.get_all_students()

        if df.empty:
            info_box("No students registered yet.")
        else:
            # Add face status column
            from utils.helpers import student_face_dir, count_files_in_dir
            def face_status(sid):
                n = count_files_in_dir(student_face_dir(str(sid)), ".jpg")
                return f"✅ {n}" if n > 0 else "❌ 0"

            df = df.copy()
            df["Face_Samples_Status"] = df["Student_ID"].apply(face_status)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(f"Showing {len(df)} student(s).")

        # ── Delete student ─────────────────────────────────
        st.divider()
        section_header("🗑️", "Delete Student")

        with st.form("delete_form"):
            del_id  = st.text_input("Student ID to delete")
            confirm = st.checkbox("I confirm I want to delete this student permanently")
            del_btn = st.form_submit_button("Delete Student", type="secondary")

            if del_btn:
                if not del_id:
                    st.warning("Please enter a Student ID.")
                elif not confirm:
                    st.warning("Please check the confirmation box.")
                else:
                    ok, msg = sm.delete_student(del_id)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)


# ══════════════════════════════════════════════════════════════
#  PAGE 3 — TRAIN AI MODEL
# ══════════════════════════════════════════════════════════════
def page_train() -> None:
    """Train the LBPH face recogniser on collected face samples."""
    st.title("🧠 Train AI Model")

    sm = get_student_manager()
    fr = get_face_recognizer()

    section_header("ℹ️", "About the AI Algorithm")
    with st.expander("What is LBPH Face Recognition? (Click to learn)", expanded=False):
        st.markdown("""
        ### LBPH — Local Binary Pattern Histogram

        **LBPH** is a classic machine-learning algorithm for face recognition.

        #### How it works (step by step):
        1. **Divide** the face image into small rectangular cells (e.g. 8×8 grid).
        2. **LBP Encoding:** For each pixel in a cell, compare it to its 8 surrounding
           neighbours. Write **1** if the neighbour is brighter, **0** if darker.
           This gives an 8-bit binary number (0–255) per pixel.
        3. **Histogram:** For each cell, count how many pixels fall into each of the
           256 LBP codes. This produces a 256-bin histogram per cell.
        4. **Concatenate** all cell histograms → one long feature vector per face.
        5. **Recognition:** When a new face arrives, compute its feature vector and
           compare it to all training vectors using **Chi-square distance**.
           The closest stored face is the predicted identity.

        #### Why LBPH is great for this project:
        - ✅ Works well under varying lighting conditions
        - ✅ Runs fast on any laptop (no GPU needed)
        - ✅ Built into OpenCV — no complex installation
        - ✅ Incremental — can update without full retraining
        - ✅ Excellent for a beginner AI project demonstration
        """)

    st.divider()

    # ── Pre-training checks ───────────────────────────────
    section_header("🔍", "Pre-Training Check")

    summary = sm.get_registration_summary()
    col1, col2, col3 = st.columns(3)
    col1.metric("Registered Students", summary["total_students"])
    col2.metric("With Face Data",      summary["with_face_data"])
    col3.metric("Without Face Data",   summary["without_face_data"])

    if summary["total_students"] == 0:
        st.error("❌ No students registered. Please register students before training.")
        return

    if summary["with_face_data"] < 2:
        st.warning(
            "⚠️ Need at least 2 students with face data to train the model. "
            f"Currently only {summary['with_face_data']} student(s) have face samples."
        )

    # ── Model status ──────────────────────────────────────
    st.divider()
    section_header("📦", "Current Model Status")

    if model_exists():
        info = fr.model_info()
        success_box(
            f"Model is trained on **{info['num_students']}** student(s): "
            f"{', '.join(info['students']) or 'none'}"
        )
    else:
        warning_box("No trained model found. Click Train below to create one.")

    # ── Training button ───────────────────────────────────
    st.divider()
    section_header("🚀", "Start Training")

    st.markdown("""
    Click **Train AI Model** to:
    1. Load all face images from disk
    2. Preprocess them (grayscale, resize, equalise)
    3. Run the LBPH training algorithm
    4. Save the model weights to `data/model/trainer.yml`
    """)

    if st.button("🧠 Train AI Model", type="primary", use_container_width=True):
        with st.spinner("Training model — please wait…"):
            try:
                # FaceRecognizer.train() is the core AI training step
                ok, msg = fr.train(cfg.FACES_DIR)
                if ok:
                    st.success(f"✅ {msg}")
                    # Clear the cached recognizer so it reloads
                    get_attendance_manager.clear()
                    get_face_recognizer.clear()
                    st.balloons()
                else:
                    st.error(f"❌ {msg}")
            except Exception as e:
                st.error(f"Training failed: {e}")
                logger.error(f"Training exception: {e}")


# ══════════════════════════════════════════════════════════════
#  PAGE 4 — MARK ATTENDANCE
# ══════════════════════════════════════════════════════════════
def page_attendance() -> None:
    """Mark attendance using the webcam and face recognition."""
    st.title("📸 Mark Attendance")
    am = get_attendance_manager()

    # ── Check model readiness ─────────────────────────────
    if not am.recognizer.is_trained:
        st.error(
            "❌ AI model is not trained yet. "
            "Please go to 🧠 Train AI Model first."
        )
        return

    section_header("ℹ️", "How It Works")
    info_box(
        "Take a photo with the camera below. The AI will detect any faces, "
        "match them against registered students, and mark attendance automatically."
    )

    # ── Mode selector ─────────────────────────────────────
    tab_auto, tab_manual = st.tabs(["📷 Auto (Camera)", "✏️ Manual Entry"])

    # ────────────────────────────────────────────────────────
    with tab_auto:
        st.markdown("### 📷 Capture Photo for Attendance")

        # Show today's stats
        today_stats = am.get_today_statistics()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Students",  today_stats["total_students"])
        c2.metric("✅ Present",       today_stats["present"])
        c3.metric("❌ Absent",        today_stats["absent"])
        c4.metric("📊 Rate",          f"{today_stats['percentage']:.1f}%")

        st.divider()

        # Camera input
        captured = st.camera_input(
            "Click the camera to capture a snapshot for face recognition",
            key="attendance_cam",
        )

        if captured is not None:
            img = numpy_from_uploaded(captured)

            if img is None:
                st.error("Could not read the captured image.")
            else:
                with st.spinner("Analysing face…"):
                    result = am.process_frame(img)

                col_frame, col_result = st.columns([1.2, 1])

                with col_frame:
                    # Display annotated frame (convert BGR→RGB for Streamlit)
                    annotated = result["annotated_frame"]
                    if annotated is not None:
                        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        st.image(rgb, caption="Face Detection Result",
                                 use_container_width=True)

                with col_result:
                    detections = result["detections"]
                    marked_now = result["marked_now"]

                    if not detections:
                        st.warning("No face detected. Try again.")
                    else:
                        for det in detections:
                            if det["recognized"]:
                                if det["marked"]:
                                    st.success(
                                        f"✅ **{det['student_id']}** — "
                                        f"Attendance Marked! "
                                        f"(Confidence: {det['confidence']:.1f})"
                                    )
                                else:
                                    st.info(
                                        f"ℹ️ **{det['student_id']}** — "
                                        "Already marked today."
                                    )
                            else:
                                st.error(
                                    f"❓ Unknown person "
                                    f"(confidence: {det['confidence']:.1f}) — "
                                    "Not marked."
                                )

                # Refresh stats display
                if marked_now:
                    success_box(
                        f"Successfully marked: **{', '.join(marked_now)}**"
                    )
                    st.rerun()

        # ── Today's attendance table ───────────────────────
        st.divider()
        section_header("📋", "Today's Attendance So Far")
        today_df = am.get_today_attendance()
        if today_df.empty:
            info_box("No attendance recorded today yet.")
        else:
            st.dataframe(
                today_df.sort_values("Time", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

    # ────────────────────────────────────────────────────────
    with tab_manual:
        section_header("✏️", "Manual Attendance Entry")
        warning_box(
            "Use manual entry only when the camera is unavailable. "
            "Auto (camera) marking is preferred."
        )

        sm = get_student_manager()
        all_students = sm.get_all_students()

        if all_students.empty:
            st.warning("No students registered.")
        else:
            with st.form("manual_attendance"):
                student_choices = {
                    f"{row['Student_ID']} — {row['Student_Name']}": row["Student_ID"]
                    for _, row in all_students.iterrows()
                }
                selected_label = st.selectbox(
                    "Select Student",
                    options=list(student_choices.keys())
                )
                status_choice = st.radio("Status", ["Present", "Absent"],
                                          horizontal=True)
                date_choice   = st.date_input("Date",
                                               value=datetime.date.today())
                submit_manual = st.form_submit_button("Mark Attendance",
                                                       type="primary")

                if submit_manual:
                    sid  = student_choices[selected_label]
                    date_str = date_choice.strftime(cfg.DATE_FORMAT)
                    ok, msg  = am.mark_attendance_manual(sid, date_str, status_choice)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)


# ══════════════════════════════════════════════════════════════
#  PAGE 5 — ATTENDANCE RECORDS
# ══════════════════════════════════════════════════════════════
def page_records() -> None:
    """View, filter, search, and export attendance records."""
    st.title("📋 Attendance Records")
    am = get_attendance_manager()

    all_df = am.get_all_attendance()

    if all_df.empty:
        info_box("No attendance records found yet.")
        return

    # ── Filters sidebar-like layout ───────────────────────
    section_header("🔍", "Filter Records")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        unique_dates = ["All"] + sorted(all_df["Date"].unique().tolist(),
                                         reverse=True)
        date_filter  = st.selectbox("Date", unique_dates)

    with col2:
        unique_students = ["All"] + sorted(all_df["Student_ID"].unique().tolist())
        student_filter  = st.selectbox("Student ID", unique_students)

    with col3:
        status_filter = st.selectbox("Status", ["All", "Present", "Absent"])

    with col4:
        search_name = st.text_input("Search Name", placeholder="Type name…")

    # ── Apply filters (using Pandas boolean indexing) ─────
    filtered = all_df.copy()

    if date_filter != "All":
        filtered = filtered[filtered["Date"] == date_filter]

    if student_filter != "All":
        filtered = filtered[filtered["Student_ID"] == student_filter]

    if status_filter != "All":
        filtered = filtered[filtered["Status"] == status_filter]

    if search_name.strip():
        q = search_name.strip().lower()
        filtered = filtered[
            filtered["Student_Name"].str.lower().str.contains(q, na=False)
        ]

    st.caption(f"Showing **{len(filtered)}** record(s) out of {len(all_df)} total.")

    # ── Data table ────────────────────────────────────────
    st.dataframe(filtered.sort_values(["Date", "Time"], ascending=[False, False]),
                 use_container_width=True, hide_index=True)

    # ── Export ────────────────────────────────────────────
    st.divider()
    section_header("💾", "Export Records")

    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        # Convert to CSV bytes and offer download button
        csv_bytes = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label    = "⬇️ Download Filtered Records (CSV)",
            data     = csv_bytes,
            file_name = f"attendance_export_{datetime.date.today()}.csv",
            mime     = "text/csv",
            use_container_width=True,
        )
    with col_exp2:
        full_csv = all_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label    = "⬇️ Download ALL Records (CSV)",
            data     = full_csv,
            file_name = "attendance_all_records.csv",
            mime     = "text/csv",
            use_container_width=True,
        )

    # ── Per-student attendance summary ────────────────────
    st.divider()
    section_header("📊", "Attendance Summary (Per Student)")

    summary_df = am.calculate_attendance_percentage()
    if not summary_df.empty:
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # Colour highlight the Status column
        def highlight_status(val):
            if "Good" in str(val):
                return "color: #4CAF50"
            elif "Risk" in str(val):
                return "color: #FF9800"
            else:
                return "color: #F44336"

        styled = summary_df.style.applymap(
            highlight_status, subset=["Status"]
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 6 — ANALYTICS & CHARTS
# ══════════════════════════════════════════════════════════════
def page_analytics() -> None:
    """Data visualisation dashboard with multiple chart types."""
    st.title("📊 Analytics & Charts")
    am = get_attendance_manager()
    sm = get_student_manager()

    all_df     = am.get_all_attendance()
    summary_df = am.calculate_attendance_percentage()
    daily_df   = am.get_daily_summary()
    students_df = sm.get_all_students()

    # ── Row 1: Daily + Pie ────────────────────────────────
    section_header("📅", "Daily Attendance Overview")
    col1, col2 = st.columns([2, 1])

    with col1:
        fig = DataVisualizer.plot_daily_attendance(daily_df)
        st.pyplot(fig)

    with col2:
        today_stats = am.get_today_statistics()
        fig2 = DataVisualizer.plot_today_pie(today_stats)
        st.pyplot(fig2)

    st.divider()

    # ── Row 2: Percentage per student ────────────────────
    section_header("🏅", "Student Attendance Percentages")
    fig3 = DataVisualizer.plot_student_percentages(summary_df)
    st.pyplot(fig3)

    st.divider()

    # ── Row 3: Monthly trend ──────────────────────────────
    section_header("📈", "Monthly Attendance Trend")
    fig4 = DataVisualizer.plot_monthly_trend(all_df)
    st.pyplot(fig4)

    st.divider()

    # ── Row 4: Department pie + Top attendees ─────────────
    col_dept, col_top = st.columns(2)

    with col_dept:
        section_header("🏫", "Students by Department")
        fig5 = DataVisualizer.plot_department_distribution(students_df)
        st.pyplot(fig5)

    with col_top:
        section_header("🥇", "Top Attendees")
        fig6 = DataVisualizer.plot_top_attendees(summary_df, top_n=10)
        st.pyplot(fig6)

    st.divider()

    # ── Row 5: Heatmap ────────────────────────────────────
    section_header("🗓️", "Attendance Heatmap")
    info_box(
        "Each row is a student, each column is a date. "
        "Green = Present, Red = Absent/Not Recorded."
    )
    fig7 = DataVisualizer.plot_attendance_heatmap(all_df)
    st.pyplot(fig7)

    # ── Row 6: Key insights ───────────────────────────────
    st.divider()
    section_header("💡", "Key Insights")

    if not summary_df.empty:
        best_student   = summary_df.iloc[0]
        worst_students = summary_df[summary_df["Attendance_Percentage"] < 50]

        col_i1, col_i2, col_i3 = st.columns(3)
        with col_i1:
            st.metric("🏆 Best Attendance",
                      best_student["Student_ID"],
                      f"{best_student['Attendance_Percentage']:.1f}%")
        with col_i2:
            st.metric("⚠️ Students Below 50%", len(worst_students))
        with col_i3:
            if not all_df.empty:
                avg_pct = summary_df["Attendance_Percentage"].mean()
                st.metric("📊 Class Average", f"{avg_pct:.1f}%")
    else:
        info_box("Record some attendance first to see insights.")


# ══════════════════════════════════════════════════════════════
#  PAGE 7 — SETTINGS & TOOLS
# ══════════════════════════════════════════════════════════════
def page_settings() -> None:
    """System settings, backup tools, and developer info."""
    st.title("⚙️ Settings & Tools")

    # ── System info ───────────────────────────────────────
    section_header("🖥️", "System Information")

    fh = FileHandler()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        | Property | Value |
        |----------|-------|
        | Project | {cfg.PROJECT_NAME} |
        | Version | {cfg.PROJECT_VERSION} |
        | Author | {cfg.AUTHOR} |
        | Face Samples Target | {cfg.FACE_SAMPLE_COUNT} |
        | Recognition Threshold | {cfg.RECOGNITION_THRESHOLD} |
        | Image Size | {cfg.FACE_IMG_SIZE} |
        """)
    with col2:
        import platform
        st.markdown(f"""
        | Property | Value |
        |----------|-------|
        | Python | {sys.version.split()[0]} |
        | OpenCV | {cv2.__version__} |
        | NumPy | {np.__version__} |
        | Pandas | {pd.__version__} |
        | Platform | {platform.system()} |
        | Model Exists | {'Yes ✅' if model_exists() else 'No ❌'} |
        """)

    # ── File paths ────────────────────────────────────────
    st.divider()
    section_header("📁", "Data Paths")

    from utils.helpers import list_attendance_files
    st.code(f"""
Students CSV : {cfg.STUDENTS_CSV}
Faces Dir    : {cfg.FACES_DIR}
Model File   : {cfg.MODEL_FILE}
Label Map    : {cfg.LABEL_MAP_FILE}
Attendance   : {cfg.ATTENDANCE_DIR}
Backup Dir   : {cfg.BACKUP_DIR}
Log File     : {cfg.LOG_FILE}
""")

    att_files = list_attendance_files()
    st.caption(f"Attendance files on disk: {len(att_files)}")

    # ── Backup tools ──────────────────────────────────────
    st.divider()
    section_header("💾", "Backup Data")

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("📦 Backup Students CSV", use_container_width=True):
            path = fh.backup_students()
            if path:
                st.success(f"Backed up to: {os.path.basename(path)}")
            else:
                st.warning("Nothing to back up.")

    with col_b2:
        if st.button("📦 Backup All Attendance", use_container_width=True):
            paths = fh.backup_all_attendance()
            st.success(f"Backed up {len(paths)} attendance file(s).")

    # ── Model management ──────────────────────────────────
    st.divider()
    section_header("🧠", "Model Management")

    fr = get_face_recognizer()

    if model_exists():
        info = fr.model_info()
        st.json(info)

        if st.button("🔄 Reload Model from Disk", use_container_width=True):
            ok = fr.reload_model()
            if ok:
                st.success("Model reloaded successfully.")
            else:
                st.error("Could not reload model.")

        if st.button("🗑️ Delete Trained Model", type="secondary",
                      use_container_width=True):
            if os.path.exists(cfg.MODEL_FILE):
                os.remove(cfg.MODEL_FILE)
                st.warning("Model deleted. Retrain on the Train page.")
    else:
        warning_box("No trained model exists. Go to 🧠 Train AI Model.")

    # ── View log file ─────────────────────────────────────
    st.divider()
    section_header("📜", "System Log (last 50 lines)")

    if os.path.exists(cfg.LOG_FILE):
        with open(cfg.LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        last_lines = "".join(lines[-50:])
        st.code(last_lines, language="text")
    else:
        st.caption("No log file found yet.")

    # ── Git concepts ──────────────────────────────────────
    st.divider()
    section_header("🗂️", "Git Version Control Guide")
    with st.expander("Git commands for this project"):
        st.code("""
# Initialise a new Git repository
git init

# Stage all project files for commit
git add .

# Create the first commit
git commit -m "Initial commit: AI Attendance System"

# Push to GitHub (replace URL with your repo)
git remote add origin https://github.com/your_username/AI-Attendance-System.git
git push -u origin main

# Create a feature branch for new development
git checkout -b feature/new-ui

# Merge feature branch back to main
git checkout main
git merge feature/new-ui

# View commit history
git log --oneline

# Discard uncommitted changes
git restore .
""", language="bash")


# ══════════════════════════════════════════════════════════════
#  MAIN — ROUTER
# ══════════════════════════════════════════════════════════════
def main() -> None:
    """
    Main entry point.
    Renders the sidebar and routes to the selected page function.
    """
    page = render_sidebar()

    # Route to the correct page function using a dictionary
    # (demonstrates using a dictionary instead of a long if-elif chain)
    page_router = {
        "Home"       : page_home,
        "Register"   : page_register,
        "Train"      : page_train,
        "Attendance" : page_attendance,
        "Records"    : page_records,
        "Analytics"  : page_analytics,
        "Settings"   : page_settings,
    }

    # Call the appropriate page function; default to Home if unknown
    page_fn = page_router.get(page, page_home)
    page_fn()


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()