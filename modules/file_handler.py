"""
=============================================================
  modules/file_handler.py — CSV & File Operations
=============================================================
  All reading and writing of CSV / data files is centralised
  here to avoid scattered file-I/O across the codebase.

  Demonstrates:
    - File handling (open, read, write, append)
    - Pandas DataFrames for tabular data
    - Exception handling (try / except / finally)
    - OOP — FileHandler class
    - Context managers (with statement)
=============================================================
"""

import os
import sys
import pandas as pd      # pandas — powerful data analysis library

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from utils.helpers import logger, get_current_datetime, backup_file


class FileHandler:
    """
    Handles all CSV-based persistence for the project.

    Responsibilities
    ----------------
    - Create / read / update the student registry CSV
    - Create / read / append to daily attendance CSV files
    - Load combined attendance data for analytics
    - Provide backup functionality
    """

    # ─────────────────────────────────────────
    #  STUDENT REGISTRY
    # ─────────────────────────────────────────

    @staticmethod
    def init_students_csv() -> None:
        """
        Create the students.csv file with correct column headers
        if it does not already exist.
        Called once during project startup.
        """
        if not os.path.exists(cfg.STUDENTS_CSV):
            # Build an empty DataFrame with the right columns
            df = pd.DataFrame(columns=list(cfg.STUDENT_COLUMNS))
            df.to_csv(cfg.STUDENTS_CSV, index=False)
            logger.info("Created new students.csv")

    @staticmethod
    def load_students() -> pd.DataFrame:
        """
        Load the student registry from disk.

        Returns
        -------
        pd.DataFrame
            All registered students, or an empty DataFrame if none.
        """
        try:
            FileHandler.init_students_csv()
            df = pd.read_csv(cfg.STUDENTS_CSV)

            # Ensure the DataFrame has ALL expected columns
            # (guards against partial/corrupted files)
            for col in cfg.STUDENT_COLUMNS:
                if col not in df.columns:
                    df[col] = ""

            return df

        except pd.errors.EmptyDataError:
            # File exists but has no rows — return empty frame
            return pd.DataFrame(columns=list(cfg.STUDENT_COLUMNS))

        except Exception as e:
            logger.error(f"Could not load students.csv: {e}")
            return pd.DataFrame(columns=list(cfg.STUDENT_COLUMNS))

    @staticmethod
    def student_exists(student_id: str) -> bool:
        """Return True if a student with this ID is already registered."""
        df = FileHandler.load_students()
        return student_id.upper() in df["Student_ID"].str.upper().values

    @staticmethod
    def save_student(student_data: dict) -> bool:
        """
        Append a new student record to students.csv.

        Parameters
        ----------
        student_data : dict
            Must contain keys matching STUDENT_COLUMNS.

        Returns
        -------
        bool
            True on success, False on failure.
        """
        try:
            df = FileHandler.load_students()

            # Convert the dict to a one-row DataFrame
            new_row = pd.DataFrame([student_data])

            # pandas.concat replaces deprecated DataFrame.append()
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(cfg.STUDENTS_CSV, index=False)

            logger.info(f"Student saved: {student_data.get('Student_ID')}")
            return True

        except Exception as e:
            logger.error(f"Failed to save student: {e}")
            return False

    @staticmethod
    def update_face_sample_count(student_id: str, count: int) -> bool:
        """
        Update the Face_Samples column for an existing student.

        Parameters
        ----------
        student_id : str
        count      : int  — number of face images now on disk.

        Returns
        -------
        bool
        """
        try:
            df = FileHandler.load_students()
            mask = df["Student_ID"].str.upper() == student_id.upper()
            df.loc[mask, "Face_Samples"] = count
            df.to_csv(cfg.STUDENTS_CSV, index=False)
            logger.debug(f"Updated face count for {student_id}: {count}")
            return True
        except Exception as e:
            logger.error(f"Failed to update face count: {e}")
            return False

    @staticmethod
    def delete_student(student_id: str) -> bool:
        """
        Remove a student row from students.csv.

        Parameters
        ----------
        student_id : str

        Returns
        -------
        bool
        """
        try:
            df = FileHandler.load_students()
            before = len(df)
            df = df[df["Student_ID"].str.upper() != student_id.upper()]

            if len(df) == before:
                logger.warning(f"Delete: student not found — {student_id}")
                return False

            df.to_csv(cfg.STUDENTS_CSV, index=False)
            logger.info(f"Student deleted: {student_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete student: {e}")
            return False

    # ─────────────────────────────────────────
    #  ATTENDANCE CSV
    # ─────────────────────────────────────────

    @staticmethod
    def init_attendance_csv(csv_path: str) -> None:
        """Create an attendance CSV with headers if it doesn't exist."""
        if not os.path.exists(csv_path):
            df = pd.DataFrame(columns=list(cfg.ATTENDANCE_COLUMNS))
            df.to_csv(csv_path, index=False)
            logger.info(f"Created attendance file: {os.path.basename(csv_path)}")

    @staticmethod
    def load_attendance(date_str: str = None) -> pd.DataFrame:
        """
        Load attendance records for a specific date.

        Parameters
        ----------
        date_str : str, optional
            Date in YYYY-MM-DD format. Defaults to today.

        Returns
        -------
        pd.DataFrame
        """
        from utils.helpers import attendance_csv_path, get_current_date
        path = attendance_csv_path(date_str or get_current_date())

        try:
            FileHandler.init_attendance_csv(path)
            df = pd.read_csv(path)
            return df

        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=list(cfg.ATTENDANCE_COLUMNS))

        except Exception as e:
            logger.error(f"Failed to load attendance: {e}")
            return pd.DataFrame(columns=list(cfg.ATTENDANCE_COLUMNS))

    @staticmethod
    def load_all_attendance() -> pd.DataFrame:
        """
        Load and concatenate ALL attendance CSV files into one DataFrame.
        Used for analytics and the main dashboard.

        Returns
        -------
        pd.DataFrame
        """
        from utils.helpers import list_attendance_files

        all_frames = []   # List to collect individual DataFrames

        for filename in list_attendance_files():
            filepath = os.path.join(cfg.ATTENDANCE_DIR, filename)
            try:
                df = pd.read_csv(filepath)
                if not df.empty:
                    all_frames.append(df)
            except Exception as e:
                logger.warning(f"Skipping unreadable file {filename}: {e}")

        if all_frames:
            # Combine all DataFrames row-wise
            combined = pd.concat(all_frames, ignore_index=True)
            return combined
        else:
            return pd.DataFrame(columns=list(cfg.ATTENDANCE_COLUMNS))

    @staticmethod
    def append_attendance(record: dict, date_str: str = None) -> bool:
        """
        Append a single attendance record to today's CSV file.

        Parameters
        ----------
        record   : dict  — must have keys matching ATTENDANCE_COLUMNS.
        date_str : str   — optional date override.

        Returns
        -------
        bool
        """
        from utils.helpers import attendance_csv_path, get_current_date
        path = attendance_csv_path(date_str or get_current_date())

        try:
            FileHandler.init_attendance_csv(path)
            df = FileHandler.load_attendance(date_str)

            new_row = pd.DataFrame([record])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(path, index=False)

            logger.info(
                f"Attendance marked: {record.get('Student_ID')} "
                f"at {record.get('Time')}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to append attendance record: {e}")
            return False

    @staticmethod
    def record_already_marked(student_id: str, date_str: str = None) -> bool:
        """
        Check whether this student is already marked for today.
        Used to prevent duplicate entries.

        Parameters
        ----------
        student_id : str
        date_str   : str, optional

        Returns
        -------
        bool
        """
        from utils.helpers import get_current_date
        df = FileHandler.load_attendance(date_str or get_current_date())

        if df.empty:
            return False

        already = df["Student_ID"].str.upper() == student_id.upper()
        return already.any()   # True if at least one row matches

    # ─────────────────────────────────────────
    #  BACKUP
    # ─────────────────────────────────────────

    @staticmethod
    def backup_students() -> str:
        """Create a timestamped backup of students.csv."""
        return backup_file(cfg.STUDENTS_CSV)

    @staticmethod
    def backup_all_attendance() -> list:
        """Back up every attendance CSV file. Returns list of backup paths."""
        from utils.helpers import list_attendance_files
        backed_up = []
        for filename in list_attendance_files():
            src  = os.path.join(cfg.ATTENDANCE_DIR, filename)
            dest = backup_file(src)
            if dest:
                backed_up.append(dest)
        return backed_up

    # ─────────────────────────────────────────
    #  EXPORT
    # ─────────────────────────────────────────

    @staticmethod
    def export_attendance_to_csv(df: pd.DataFrame,
                                  output_path: str) -> bool:
        """
        Save an arbitrary attendance DataFrame to a given path.
        Useful for exporting filtered/custom reports.
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(df)} records to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False