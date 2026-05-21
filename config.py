
"""
=============================================================
  AI Smart Attendance System — Configuration File
=============================================================
  All project-wide constants, paths, and settings live here.
  Changing a value here updates it everywhere in the project.
  This demonstrates: Variables, Strings, Dictionaries, Tuples
=============================================================
"""

import os  # Built-in module for file/folder path operations
import re
# ─────────────────────────────────────────────
#  PROJECT METADATA
# ─────────────────────────────────────────────
PROJECT_NAME    = "AI Smart Attendance System"
PROJECT_VERSION = "1.0.0"
AUTHOR          = "Waniya Farooq"
UNIVERSITY      = "Comsats University "
COURSE          = "Introduction to Programming / AI"

# ─────────────────────────────────────────────
#  BASE DIRECTORIES  (os.path.join makes paths
#  work on both Windows and Linux/Mac)
# ─────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))  # Folder of this file
DATA_DIR        = os.path.join(BASE_DIR, "data")
FACES_DIR       = os.path.join(DATA_DIR, "faces")       # Saved student face images
ATTENDANCE_DIR  = os.path.join(DATA_DIR, "attendance")  # CSV attendance files
MODEL_DIR       = os.path.join(DATA_DIR, "model")       # Trained recognizer model
BACKUP_DIR      = os.path.join(DATA_DIR, "backups")     # CSV backups

# ─────────────────────────────────────────────
#  FILE NAMES
# ─────────────────────────────────────────────
STUDENTS_CSV    = os.path.join(DATA_DIR, "students.csv")   # Student registry
MODEL_FILE      = os.path.join(MODEL_DIR, "trainer.yml")   # LBPH model weights
LABEL_MAP_FILE  = os.path.join(MODEL_DIR, "labels.pkl")    # {label_int: student_id}

# ─────────────────────────────────────────────
#  FACE DETECTION SETTINGS
# ─────────────────────────────────────────────
HAAR_CASCADE_PATH = cv2_data_path = "haarcascade_frontalface_default.xml"
# ^ OpenCV ships this file; we look it up dynamically in face_detector.py

FACE_DETECTION_CONFIG = {
    "scale_factor"   : 1.3,   # How much image size is reduced at each scale
    "min_neighbors"  : 5,     # How many neighbors each rectangle should retain
    "min_face_size"  : (30, 30),  # Minimum size of detected face (pixels)
}

# ─────────────────────────────────────────────
#  FACE RECOGNITION SETTINGS
# ─────────────────────────────────────────────
FACE_SAMPLE_COUNT   = 50    # Number of face photos captured per student
FACE_IMG_SIZE       = (200, 200)   # All faces resized to this before training
RECOGNITION_THRESHOLD = 70  # Confidence threshold (lower = more strict)
                             # LBPH: 0 = perfect match, higher = worse match
                             # We REJECT if confidence > this threshold

# ─────────────────────────────────────────────
#  ATTENDANCE SETTINGS
# ─────────────────────────────────────────────
ATTENDANCE_COOLDOWN_MINUTES = 30  # Ignore duplicate marks within this window
DATE_FORMAT  = "%Y-%m-%d"         # e.g. 2024-01-15
TIME_FORMAT  = "%H:%M:%S"         # e.g. 14:35:22
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# ─────────────────────────────────────────────
#  ATTENDANCE CSV COLUMNS  (a tuple — immutable)
# ─────────────────────────────────────────────
ATTENDANCE_COLUMNS = (
    "Student_ID",
    "Student_Name",
    "Date",
    "Time",
    "Status",      # 'Present' / 'Absent'
    "Marked_By",   # 'Auto' (camera) or 'Manual'
)

# ─────────────────────────────────────────────
#  STUDENT CSV COLUMNS
# ─────────────────────────────────────────────
STUDENT_COLUMNS = (
    "Student_ID",
    "Student_Name",
    "Department",
    "Semester",
    "Registration_Date",
    "Face_Samples",   # Number of face images stored
)

# ─────────────────────────────────────────────
#  STREAMLIT UI SETTINGS
# ─────────────────────────────────────────────
PAGE_TITLE = "AI Attendance System"
PAGE_ICON  = "🎓"
LAYOUT     = "wide"

# Color palette used across Matplotlib charts
CHART_COLORS = [
    "#4CAF50",  # Green  — Present
    "#F44336",  # Red    — Absent
    "#2196F3",  # Blue   — General
    "#FF9800",  # Orange — Warning
    "#9C27B0",  # Purple — Analytics
]

# ─────────────────────────────────────────────
#  REGEX PATTERNS  (used in validators.py)
# ─────────────────────────────────────────────
import re  # Regular expressions module

PATTERNS = {
    # Student ID: letters + digits, 4–12 chars (e.g. "CS2021001")
    "student_id"   : re.compile(r"^[A-Za-z0-9]{4,12}$"),

    # Name: only letters and spaces, 2–50 chars
    "student_name" : re.compile(r"^[A-Za-z ]{2,50}$"),

    # Date in YYYY-MM-DD format
    "date"         : re.compile(r"^\d{4}-\d{2}-\d{2}$"),
}

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
LOG_FILE = os.path.join(DATA_DIR, "system.log")