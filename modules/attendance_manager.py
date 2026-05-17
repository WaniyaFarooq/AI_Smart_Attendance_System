"""
=============================================================
  modules/attendance_manager.py — Attendance Management
=============================================================
  Handles the complete attendance workflow:
    - Running the face-recognition loop
    - Marking attendance (auto or manual)
    - Preventing duplicate entries
    - Querying and filtering records

  Demonstrates:
    - OOP (AttendanceManager class)
    - Dictionaries and lists
    - Pandas filtering and aggregation
    - Loops and control structures
    - Exception handling
    - Datetime operations
=============================================================
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from utils.helpers     import (logger, get_current_date, get_current_time,
                                get_current_datetime, minutes_since, percentage)
from modules.file_handler    import FileHandler
from modules.face_detector   import FaceDetector
from modules.face_recognizer import FaceRecognizer


class AttendanceManager:
    """
    Orchestrates face-recognition-based attendance marking.

    Typical flow
    ────────────
    1. User opens the 'Mark Attendance' page in Streamlit.
    2. AttendanceManager.process_frame(image) is called with
       each captured webcam image.
    3. The method detects & recognises faces.
    4. For each recognised face, mark_attendance() is called.
    5. Results are returned to the UI layer.
    """

    def __init__(self):
        """Initialise all sub-components."""
        self.file_handler = FileHandler()
        self.detector     = FaceDetector()
        self.recognizer   = FaceRecognizer()

        # In-memory set of student IDs already marked THIS session
        # (secondary guard on top of the CSV duplicate check)
        self._session_marked: set = set()

        logger.info("AttendanceManager initialised.")

    # ─────────────────────────────────────────
    #  CORE: PROCESS A SINGLE FRAME
    # ─────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process one image frame: detect faces → recognise → mark attendance.

        This is the MAIN method called by the Streamlit UI.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (from cv2.VideoCapture or st.camera_input).

        Returns
        -------
        dict with keys:
            "annotated_frame"  : np.ndarray — frame with boxes drawn
            "detections"       : list[dict] — one dict per detected face
            "marked_now"       : list[str]  — student IDs marked in this call
        """
        if frame is None:
            return {"annotated_frame": None, "detections": [], "marked_now": []}

        annotated_frame = frame.copy()
        detections      = []   # List of recognition results
        marked_now      = []   # IDs newly marked in this call

        # ── Step 1: Detect all faces ──────────────────────
        faces = self.detector.detect_faces(frame)

        if not faces:
            # Draw a status bar to guide the user
            annotated_frame = self.detector.draw_status_bar(
                annotated_frame, "No face detected — please face the camera", (0, 0, 255)
            )
            return {
                "annotated_frame" : annotated_frame,
                "detections"      : [],
                "marked_now"      : [],
            }

        # ── Step 2: Recognise each face ───────────────────
        for face_coords in faces:
            face_roi  = self.detector.extract_face_roi(frame, face_coords)
            face_gray = self.detector.preprocess_face(face_roi)

            result    = self.recognizer.predict(face_gray)
            student_id = result["student_id"]
            confidence = result["confidence"]
            recognized = result["recognized"]

            # ── Step 3: Mark attendance if recognised ─────
            marked = False
            if recognized:
                was_marked = self._mark_attendance_internal(student_id)
                if was_marked:
                    marked_now.append(student_id)
                    marked = True

            # ── Annotate the frame ────────────────────────
            if recognized:
                label = f"{student_id} ({confidence:.1f})"
                color = (0, 255, 0)   # Green = recognised
                if marked:
                    label += " ✓ MARKED"
            else:
                label = f"Unknown ({confidence:.1f})"
                color = (0, 0, 255)   # Red = unknown

            annotated_frame = self.detector.draw_face_boxes(
                annotated_frame, [face_coords], color, label
            )

            detections.append({
                "recognized" : recognized,
                "student_id" : student_id,
                "confidence" : confidence,
                "marked"     : marked,
            })

        # Draw overall status bar
        n_recognised = sum(1 for d in detections if d["recognized"])
        status_msg   = (
            f"Detected: {len(faces)} face(s) | "
            f"Recognised: {n_recognised} | "
            f"Marked: {len(marked_now)}"
        )
        annotated_frame = self.detector.draw_status_bar(
            annotated_frame, status_msg, (0, 200, 0)
        )

        return {
            "annotated_frame" : annotated_frame,
            "detections"      : detections,
            "marked_now"      : marked_now,
        }

    # ─────────────────────────────────────────
    #  MARK ATTENDANCE (internal)
    # ─────────────────────────────────────────

    def _mark_attendance_internal(self, student_id: str) -> bool:
        """
        Core attendance marking — checks duplicates and writes to CSV.

        Returns True if newly marked, False if already marked/cooldown.
        """
        today = get_current_date()

        # ── Guard 1: session-level in-memory check (fastest) ─
        if student_id in self._session_marked:
            return False

        # ── Guard 2: CSV-level check (persists across restarts) ─
        if self.file_handler.record_already_marked(student_id, today):
            # Even if already in CSV, add to session set so we skip next time
            self._session_marked.add(student_id)
            return False

        # ── Get student name for the record ───────────────
        students_df = self.file_handler.load_students()
        name_row    = students_df[
            students_df["Student_ID"].str.upper() == student_id.upper()
        ]
        student_name = (
            name_row.iloc[0]["Student_Name"]
            if not name_row.empty else "Unknown"
        )

        # ── Build attendance record (dictionary) ──────────
        record = {
            "Student_ID"   : student_id,
            "Student_Name" : student_name,
            "Date"         : today,
            "Time"         : get_current_time(),
            "Status"       : "Present",
            "Marked_By"    : "Auto",
        }

        # ── Append to CSV ─────────────────────────────────
        success = self.file_handler.append_attendance(record, today)

        if success:
            self._session_marked.add(student_id)
            logger.info(f"Attendance MARKED: {student_id} — {student_name}")
            return True

        return False

    def mark_attendance_manual(self,
                                student_id   : str,
                                date_str     : str = None,
                                status       : str = "Present") -> tuple:
        """
        Manually mark attendance (without webcam).
        Used by teachers through the dashboard.

        Parameters
        ----------
        student_id : str
        date_str   : str  — date for the entry. Defaults to today.
        status     : str  — 'Present' or 'Absent'

        Returns
        -------
        tuple[bool, str]
        """
        from utils.helpers import format_student_id
        sid   = format_student_id(student_id)
        today = date_str or get_current_date()

        # Check if already marked for that date
        if self.file_handler.record_already_marked(sid, today):
            return False, f"{sid} is already marked for {today}."

        # Look up student name
        students_df  = self.file_handler.load_students()
        name_row     = students_df[
            students_df["Student_ID"].str.upper() == sid.upper()
        ]
        if name_row.empty:
            return False, f"Student '{sid}' not found in registry."

        student_name = name_row.iloc[0]["Student_Name"]

        record = {
            "Student_ID"   : sid,
            "Student_Name" : student_name,
            "Date"         : today,
            "Time"         : get_current_time(),
            "Status"       : status,
            "Marked_By"    : "Manual",
        }

        success = self.file_handler.append_attendance(record, today)
        if success:
            return True, f"Manually marked {sid} as '{status}' for {today}."
        return False, "Failed to save attendance record."

    def reset_session(self) -> None:
        """Clear the in-memory session tracking set."""
        self._session_marked.clear()
        logger.info("Attendance session reset.")

    # ─────────────────────────────────────────
    #  QUERY & ANALYTICS
    # ─────────────────────────────────────────

    def get_today_attendance(self) -> pd.DataFrame:
        """Return today's attendance DataFrame."""
        return self.file_handler.load_attendance(get_current_date())

    def get_attendance_by_date(self, date_str: str) -> pd.DataFrame:
        """Return attendance for a specific date."""
        return self.file_handler.load_attendance(date_str)

    def get_all_attendance(self) -> pd.DataFrame:
        """Return all-time attendance records."""
        return self.file_handler.load_all_attendance()

    def get_student_attendance(self, student_id: str) -> pd.DataFrame:
        """
        Return all attendance records for a specific student.

        Parameters
        ----------
        student_id : str

        Returns
        -------
        pd.DataFrame
        """
        from utils.helpers import format_student_id
        sid = format_student_id(student_id)
        df  = self.get_all_attendance()

        if df.empty:
            return df

        return df[df["Student_ID"].str.upper() == sid.upper()]

    def calculate_attendance_percentage(self) -> pd.DataFrame:
        """
        Compute each student's attendance percentage across all dates.

        Algorithm
        ---------
        total_classes  = number of unique dates in attendance records
        student_present = number of 'Present' entries for that student
        percentage     = (student_present / total_classes) * 100

        Returns
        -------
        pd.DataFrame with columns:
            Student_ID, Student_Name, Present, Total_Classes,
            Attendance_Percentage, Status (Good/At Risk/Low)
        """
        df = self.get_all_attendance()

        if df.empty:
            return pd.DataFrame(columns=[
                "Student_ID", "Student_Name", "Present",
                "Total_Classes", "Attendance_Percentage", "Status"
            ])

        # Total unique class days across all records
        total_dates = df["Date"].nunique()  # nunique = number of unique values

        # Group by student and count Present entries
        summary_rows = []

        # Get all registered students to include those with zero attendance
        students_df = self.file_handler.load_students()

        for _, student in students_df.iterrows():
            sid   = student["Student_ID"]
            sname = student["Student_Name"]

            # Filter for this student's records
            s_df    = df[df["Student_ID"].str.upper() == sid.upper()]
            present = len(s_df[s_df["Status"] == "Present"])
            pct     = percentage(present, total_dates)

            # Classify attendance status
            if pct >= 75:
                status = "Good ✅"
            elif pct >= 50:
                status = "At Risk ⚠️"
            else:
                status = "Low ❌"

            summary_rows.append({
                "Student_ID"            : sid,
                "Student_Name"          : sname,
                "Present"               : present,
                "Total_Classes"         : total_dates,
                "Attendance_Percentage" : pct,
                "Status"                : status,
            })

        result = pd.DataFrame(summary_rows)
        # Sort by percentage descending
        result = result.sort_values("Attendance_Percentage", ascending=False)
        return result.reset_index(drop=True)

    def get_daily_summary(self) -> pd.DataFrame:
        """
        Return a summary of attendance counts per day.

        Returns
        -------
        pd.DataFrame with columns: Date, Present, Absent, Total
        """
        df = self.get_all_attendance()

        if df.empty:
            return pd.DataFrame(columns=["Date", "Present", "Absent", "Total"])

        # Group by date
        daily = df.groupby("Date")["Status"].value_counts().unstack(fill_value=0)

        # Ensure both 'Present' and 'Absent' columns exist
        if "Present" not in daily.columns:
            daily["Present"] = 0
        if "Absent" not in daily.columns:
            daily["Absent"] = 0

        daily["Total"] = daily["Present"] + daily["Absent"]
        daily = daily.reset_index()

        return daily.sort_values("Date")

    def get_today_statistics(self) -> dict:
        """Return a quick statistics dictionary for today's attendance."""
        df    = self.get_today_attendance()
        total = len(self.file_handler.load_students())   # Registered students

        present = len(df[df["Status"] == "Present"]) if not df.empty else 0
        absent  = max(0, total - present)

        return {
            "total_students" : total,
            "present"        : present,
            "absent"         : absent,
            "not_yet_marked" : max(0, total - len(df)),
            "percentage"     : percentage(present, total),
            "date"           : get_current_date(),
        }

    # ─────────────────────────────────────────
    #  LIVE WEBCAM LOOP (CLI / non-Streamlit)
    # ─────────────────────────────────────────

    def run_attendance_loop(self,
                             camera_index   : int = 0,
                             duration_sec   : int = 60) -> list:
        """
        Run a live webcam attendance loop for a fixed duration.
        Intended for testing outside of Streamlit.

        Parameters
        ----------
        camera_index : int
        duration_sec : int  — how long to run (seconds)

        Returns
        -------
        list  — student IDs that were marked during this session.
        """
        if not self.recognizer.is_trained:
            logger.error("Model not trained. Cannot run attendance loop.")
            return []

        try:
            cap = self.detector.open_webcam(camera_index)
        except RuntimeError as e:
            logger.error(str(e))
            return []

        start_time = datetime.now()
        self.reset_session()

        logger.info(f"Attendance loop started — running for {duration_sec}s.")

        try:
            while True:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed >= duration_sec:
                    break

                ret, frame = self.detector.read_frame(cap)
                if not ret:
                    continue

                result = self.process_frame(frame)

                # Show the annotated frame
                cv2.imshow("AI Attendance System — Press Q to quit",
                           result["annotated_frame"])

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

        marked = list(self._session_marked)
        logger.info(f"Attendance loop ended. Marked: {marked}")
        return marked