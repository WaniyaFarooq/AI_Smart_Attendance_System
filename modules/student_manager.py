"""
=============================================================
  modules/student_manager.py — Student Registration System
=============================================================
"""

import os
import sys
import cv2
import numpy as np
from dataclasses import dataclass, field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from utils.helpers  import (logger, get_current_date, student_face_dir,
                             format_student_id, format_student_name,
                             count_files_in_dir)
from utils.validators   import validate_student_form
from modules.file_handler  import FileHandler
from modules.face_detector import FaceDetector


@dataclass
class Student:
    """Represents one student's registration data."""
    student_id        : str
    student_name      : str
    department        : str
    semester          : int
    registration_date : str = field(default_factory=get_current_date)
    face_samples      : int = 0

    def to_dict(self) -> dict:
        """Convert to a plain dictionary for CSV storage."""
        return {
            "Student_ID"        : self.student_id,
            "Student_Name"      : self.student_name,
            "Department"        : self.department,
            "Semester"          : self.semester,
            "Registration_Date" : self.registration_date,
            "Face_Samples"      : self.face_samples,
        }

    def __str__(self) -> str:
        return (
            f"Student({self.student_id}, {self.student_name}, "
            f"Sem-{self.semester}, {self.department})"
        )


class StudentManager:
    """Manages student registration and face image collection."""

    def __init__(self):
        """Set up the manager and ensure all directories exist."""
        os.makedirs(cfg.FACES_DIR, exist_ok=True)
        self.file_handler = FileHandler()
        self.detector     = FaceDetector()
        logger.info("StudentManager initialised.")

    # ─────────────────────────────────────────
    #  REGISTRATION
    # ─────────────────────────────────────────

    def register_student(self,
                         student_id   : str,
                         student_name : str,
                         department   : str,
                         semester     : int) -> tuple:
        """
        Register a new student: validate inputs, save record to CSV.

        Returns
        -------
        tuple[bool, str, Student | None]
        """
        sid  = format_student_id(student_id)
        name = format_student_name(student_name)
        sem  = int(semester)

        ok, msg = validate_student_form(sid, name, department, sem)
        if not ok:
            return False, msg, None

        if self.file_handler.student_exists(sid):
            return (
                False,
                f"Student ID '{sid}' is already registered. "
                "Please use a different ID or delete the existing record.",
                None,
            )

        student = Student(
            student_id        = sid,
            student_name      = name,
            department        = department,
            semester          = sem,
            registration_date = get_current_date(),
            face_samples      = 0,
        )

        saved = self.file_handler.save_student(student.to_dict())
        if not saved:
            return False, "Failed to save student record. Check file permissions.", None

        face_dir = student_face_dir(sid)
        os.makedirs(face_dir, exist_ok=True)

        logger.info(f"Student registered: {student}")
        return True, f"Student '{name}' ({sid}) registered successfully!", student

    # ─────────────────────────────────────────
    #  FACE SAMPLE CAPTURE
    # ─────────────────────────────────────────

    def capture_face_samples_from_image(self,
                                         student_id : str,
                                         image      : np.ndarray,
                                         sample_idx : int) -> tuple:
        """
        Extract a face from a single image and save it as a training sample.
        
        FIXED: Equalize histogram BEFORE saving so that training and
        testing use the same preprocessing.
        """
        sid      = format_student_id(student_id)
        face_dir = student_face_dir(sid)
        os.makedirs(face_dir, exist_ok=True)

        faces = self.detector.detect_faces(image)

        if not faces:
            return False, "No face detected in the image. Please try again."

        if len(faces) > 1:
            return (
                False,
                f"Multiple faces ({len(faces)}) detected. "
                "Please make sure only one person is in the frame.",
            )

        # Extract and preprocess the single detected face
        face_coords = faces[0]
        face_roi    = self.detector.extract_face_roi(image, face_coords)
        face_gray   = self.detector.preprocess_face(face_roi)
        
        # FIXED: The preprocessing in preprocess_face() includes:
        # - Convert to grayscale
        # - Resize to standard size
        # - Equalize histogram
        # We save this already-processed image.
        # During training, we will NOT equalize again.

        filename = f"face_{str(sample_idx).zfill(4)}.jpg"
        filepath = os.path.join(face_dir, filename)
        cv2.imwrite(filepath, face_gray)

        total = count_files_in_dir(face_dir, ".jpg")
        self.file_handler.update_face_sample_count(sid, total)

        return True, f"Sample {sample_idx + 1} saved ({total} total)."

    def capture_face_samples_webcam(self,
                                     student_id      : str,
                                     num_samples     : int = None,
                                     camera_index    : int = 0,
                                     display_window  : bool = True) -> tuple:
        """
        Open the webcam in a loop and automatically capture face samples.
        This is the command-line / standalone version (not Streamlit).
        """
        if num_samples is None:
            num_samples = cfg.FACE_SAMPLE_COUNT

        sid      = format_student_id(student_id)
        face_dir = student_face_dir(sid)
        os.makedirs(face_dir, exist_ok=True)

        try:
            cap = self.detector.open_webcam(camera_index)
        except RuntimeError as e:
            return False, str(e)

        captured_count = 0
        sample_idx     = count_files_in_dir(face_dir, ".jpg")

        try:
            while captured_count < num_samples:
                ret, frame = self.detector.read_frame(cap)
                if not ret:
                    logger.warning("Failed to read frame from webcam.")
                    continue

                faces = self.detector.detect_faces(frame)
                display_frame = frame.copy()

                if faces:
                    face_coords = max(faces, key=lambda f: f[2] * f[3])
                    face_roi    = self.detector.extract_face_roi(frame, face_coords)
                    face_gray   = self.detector.preprocess_face(face_roi)

                    # FIXED: Save every frame (not every 3rd)
                    # because the original condition was buggy: "if ... % 3 == 0 or True:"
                    filename  = f"face_{str(sample_idx).zfill(4)}.jpg"
                    filepath  = os.path.join(face_dir, filename)
                    cv2.imwrite(filepath, face_gray)
                    captured_count += 1
                    sample_idx     += 1

                    display_frame = self.detector.draw_face_boxes(
                        display_frame, [face_coords], (0, 255, 0),
                        label=f"Capturing {captured_count}/{num_samples}"
                    )
                else:
                    display_frame = self.detector.draw_status_bar(
                        display_frame, "No face detected — face the camera", (0, 0, 255)
                    )

                if display_window:
                    cv2.imshow("Face Capture — Press Q to cancel", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        except Exception as e:
            logger.error(f"Face capture error: {e}")
            return False, f"Capture error: {e}"

        finally:
            cap.release()
            if display_window:
                cv2.destroyAllWindows()

        total = count_files_in_dir(face_dir, ".jpg")
        self.file_handler.update_face_sample_count(sid, total)

        msg = f"Captured {captured_count} face samples for {sid}. Total: {total}."
        logger.info(msg)
        return True, msg

    # ─────────────────────────────────────────
    #  QUERY METHODS
    # ─────────────────────────────────────────

    def get_all_students(self):
        """Return a pandas DataFrame of all registered students."""
        return self.file_handler.load_students()

    def get_student(self, student_id: str) -> dict:
        """Return a single student's record as a dict."""
        df  = self.get_all_students()
        sid = format_student_id(student_id)
        row = df[df["Student_ID"].str.upper() == sid]

        if row.empty:
            return {}

        return row.iloc[0].to_dict()

    def search_students(self, query: str):
        """Search students by ID or name (case-insensitive, partial match)."""
        df = self.get_all_students()
        if df.empty or not query.strip():
            return df

        q = query.strip().lower()

        mask = (
            df["Student_ID"].str.lower().str.contains(q, na=False) |
            df["Student_Name"].str.lower().str.contains(q, na=False)
        )
        return df[mask]

    def get_student_count(self) -> int:
        """Return total number of registered students."""
        return len(self.get_all_students())

    def has_face_data(self, student_id: str) -> bool:
        """Return True if the student has at least some face images saved."""
        sid      = format_student_id(student_id)
        face_dir = student_face_dir(sid)
        return count_files_in_dir(face_dir, ".jpg") > 0

    # ─────────────────────────────────────────
    #  DELETION
    # ─────────────────────────────────────────

    def delete_student(self, student_id: str) -> tuple:
        """Remove a student's CSV record AND their face image folder."""
        import shutil

        sid      = format_student_id(student_id)
        face_dir = student_face_dir(sid)

        csv_ok = self.file_handler.delete_student(sid)

        if os.path.isdir(face_dir):
            shutil.rmtree(face_dir)
            logger.info(f"Deleted face folder: {face_dir}")

        if csv_ok:
            return True, f"Student {sid} deleted successfully."
        else:
            return False, f"Student {sid} not found in registry."

    # ─────────────────────────────────────────
    #  SUMMARY
    # ─────────────────────────────────────────

    def get_registration_summary(self) -> dict:
        """Return a summary dictionary for the dashboard."""
        df = self.get_all_students()

        dept_counts = {}
        if not df.empty and "Department" in df.columns:
            for dept in df["Department"]:
                dept_counts[dept] = dept_counts.get(dept, 0) + 1

        with_faces    = 0
        without_faces = 0

        if not df.empty:
            for _, row in df.iterrows():
                sid = str(row.get("Student_ID", ""))
                if sid and self.has_face_data(sid):
                    with_faces += 1
                else:
                    without_faces += 1

        return {
            "total_students"   : len(df),
            "with_face_data"   : with_faces,
            "without_face_data": without_faces,
            "by_department"    : dept_counts,
        }