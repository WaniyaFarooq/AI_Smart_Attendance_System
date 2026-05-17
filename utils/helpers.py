"""
=============================================================
  utils/helpers.py — General Helper / Utility Functions
=============================================================
  These are small, reusable helper functions used across the
  entire project.  They demonstrate:
    - Functions with default parameters
    - String formatting
    - File / OS operations
    - Logging
    - Datetime manipulation
=============================================================
"""

import os           # File-system operations
import sys          # System-level operations
import logging      # Built-in logging module
import datetime     # Date and time utilities
import shutil       # High-level file operations (copy, move)
import pickle       # Serialize Python objects to disk

# Import our own configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg


# ─────────────────────────────────────────────
#  LOGGING SETUP
#  The logging module lets us write messages
#  to both the console AND a log file.
# ─────────────────────────────────────────────
def setup_logger(name: str = "AttendanceSystem") -> logging.Logger:
    """
    Create and configure a logger.

    Parameters
    ----------
    name : str
        Name of the logger (shows in log messages).

    Returns
    -------
    logging.Logger
        Configured logger object.
    """
    logger = logging.getLogger(name)

    # Prevent adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)  # Capture everything DEBUG and above

    # ── Handler 1: write to log file ──────────────────────
    os.makedirs(os.path.dirname(cfg.LOG_FILE), exist_ok=True)
    file_handler = logging.FileHandler(cfg.LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    # ── Handler 2: print to console ───────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Format: 2024-01-15 14:35:22 — INFO — message
    formatter = logging.Formatter(
        "%(asctime)s — %(levelname)-8s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Singleton logger used everywhere
logger = setup_logger()


# ─────────────────────────────────────────────
#  DIRECTORY CREATION
# ─────────────────────────────────────────────
def create_project_directories() -> None:
    """
    Create all required project directories if they don't exist.
    Called once at startup.  Uses os.makedirs with exist_ok=True
    so it silently skips folders that already exist.
    """
    dirs = [
        cfg.DATA_DIR,
        cfg.FACES_DIR,
        cfg.ATTENDANCE_DIR,
        cfg.MODEL_DIR,
        cfg.BACKUP_DIR,
    ]

    for directory in dirs:          # Loop through list of directory paths
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Directory ready: {directory}")

    logger.info("All project directories are ready.")


# ─────────────────────────────────────────────
#  DATE / TIME HELPERS
# ─────────────────────────────────────────────
def get_current_date() -> str:
    """Return today's date as a formatted string, e.g. '2024-01-15'."""
    return datetime.datetime.now().strftime(cfg.DATE_FORMAT)


def get_current_time() -> str:
    """Return current time as a formatted string, e.g. '14:35:22'."""
    return datetime.datetime.now().strftime(cfg.TIME_FORMAT)


def get_current_datetime() -> str:
    """Return current date-time string, e.g. '2024-01-15 14:35:22'."""
    return datetime.datetime.now().strftime(cfg.DATETIME_FORMAT)


def parse_datetime(dt_string: str) -> datetime.datetime:
    """
    Parse a datetime string back into a Python datetime object.

    Parameters
    ----------
    dt_string : str
        A datetime string in DATETIME_FORMAT.

    Returns
    -------
    datetime.datetime
    """
    return datetime.datetime.strptime(dt_string, cfg.DATETIME_FORMAT)


def minutes_since(dt_string: str) -> float:
    """
    Calculate how many minutes have elapsed since a given datetime string.

    Parameters
    ----------
    dt_string : str
        A past datetime string.

    Returns
    -------
    float
        Minutes elapsed.
    """
    past = parse_datetime(dt_string)
    now  = datetime.datetime.now()
    delta = now - past                         # timedelta object
    return delta.total_seconds() / 60.0        # Convert seconds → minutes


# ─────────────────────────────────────────────
#  STRING HELPERS
# ─────────────────────────────────────────────
def format_student_id(raw_id: str) -> str:
    """
    Normalise a student ID:
      - Strip leading/trailing whitespace
      - Convert to UPPERCASE
    """
    return raw_id.strip().upper()


def format_student_name(raw_name: str) -> str:
    """
    Normalise a student name:
      - Strip whitespace
      - Title-case every word (e.g. 'ali ahmed' → 'Ali Ahmed')
    """
    return raw_name.strip().title()


def truncate_string(text: str, max_len: int = 30) -> str:
    """Shorten a string to max_len chars and add '…' if it was cut."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


# ─────────────────────────────────────────────
#  FILE / PATH HELPERS
# ─────────────────────────────────────────────
def student_face_dir(student_id: str) -> str:
    """Return the folder path where a student's face images are stored."""
    return os.path.join(cfg.FACES_DIR, format_student_id(student_id))


def attendance_csv_path(date_str: str = None) -> str:
    """
    Return path to the attendance CSV file for a given date.
    Defaults to today if no date is provided.

    e.g. data/attendance/attendance_2024-01-15.csv
    """
    if date_str is None:
        date_str = get_current_date()
    filename = f"attendance_{date_str}.csv"
    return os.path.join(cfg.ATTENDANCE_DIR, filename)


def list_attendance_files() -> list:
    """Return a sorted list of all attendance CSV filenames."""
    if not os.path.exists(cfg.ATTENDANCE_DIR):
        return []
    files = [
        f for f in os.listdir(cfg.ATTENDANCE_DIR)
        if f.startswith("attendance_") and f.endswith(".csv")
    ]
    return sorted(files)  # Alphabetical = chronological for YYYY-MM-DD format


def backup_file(source_path: str) -> str:
    """
    Copy a file into the backup directory with a timestamp appended.

    Parameters
    ----------
    source_path : str
        Path of the file to back up.

    Returns
    -------
    str
        Path of the backup file.
    """
    if not os.path.exists(source_path):
        logger.warning(f"Backup skipped — file not found: {source_path}")
        return ""

    # Build backup filename: original_name_YYYY-MM-DD_HH-MM-SS.ext
    basename  = os.path.basename(source_path)
    name, ext = os.path.splitext(basename)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_name = f"{name}_{timestamp}{ext}"
    backup_path = os.path.join(cfg.BACKUP_DIR, backup_name)

    shutil.copy2(source_path, backup_path)
    logger.info(f"Backup created: {backup_path}")
    return backup_path


# ─────────────────────────────────────────────
#  PICKLE HELPERS  (save/load Python objects)
# ─────────────────────────────────────────────
def save_pickle(obj, filepath: str) -> None:
    """Serialize a Python object to disk using pickle."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    logger.debug(f"Pickle saved: {filepath}")


def load_pickle(filepath: str):
    """
    Load a Python object from a pickle file.
    Returns None if the file doesn't exist.
    """
    if not os.path.exists(filepath):
        logger.warning(f"Pickle file not found: {filepath}")
        return None
    with open(filepath, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────
#  MISCELLANEOUS
# ─────────────────────────────────────────────
def count_files_in_dir(directory: str, extension: str = "") -> int:
    """
    Count how many files with the given extension exist in a directory.

    Parameters
    ----------
    directory : str
        Path to look in.
    extension : str
        File extension filter, e.g. '.jpg'. Empty string = all files.

    Returns
    -------
    int
    """
    if not os.path.isdir(directory):
        return 0
    files = os.listdir(directory)
    if extension:
        files = [f for f in files if f.endswith(extension)]
    return len(files)


def percentage(part: int, total: int) -> float:
    """
    Calculate percentage safely (avoids ZeroDivisionError).

    Returns 0.0 if total is 0.
    """
    if total == 0:
        return 0.0
    return round((part / total) * 100, 2)


def model_exists() -> bool:
    """Return True if a trained face-recognition model file exists."""
    return os.path.exists(cfg.MODEL_FILE)