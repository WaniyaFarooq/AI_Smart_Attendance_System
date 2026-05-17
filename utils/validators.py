"""
=============================================================
  utils/validators.py — Input Validation Functions
=============================================================
  Every value entered by the user is validated here BEFORE
  it touches the database or file system.

  Demonstrates:
    - Regular expressions (re module)
    - Functions returning bool + error message (tuple)
    - String methods
    - Exception handling
=============================================================
"""

import re       # Regular-expression library
import os
import sys
import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg


# ─────────────────────────────────────────────
#  TYPE ALIAS  (just for documentation clarity)
# ─────────────────────────────────────────────
# Our validators return a 2-tuple: (is_valid: bool, message: str)
ValidationResult = tuple  # (bool, str)


# ─────────────────────────────────────────────
#  STUDENT ID VALIDATOR
# ─────────────────────────────────────────────
def validate_student_id(student_id: str) -> ValidationResult:
    """
    Validate a student ID string.

    Rules:
      - Must not be empty / whitespace-only
      - Must match pattern: letters and digits only, 4–12 characters
      - e.g. 'CS2021001', 'F2024B', 'STU01' are valid

    Returns
    -------
    tuple[bool, str]
        (True, "Valid") on success, or (False, "Error description") on failure.
    """
    # Strip whitespace first
    sid = student_id.strip()

    # Check 1: non-empty
    if not sid:
        return False, "Student ID cannot be empty."

    # Check 2: regex match (letters + digits, 4–12 chars)
    #   re.fullmatch → the ENTIRE string must match (not just a part of it)
    if not re.fullmatch(r"[A-Za-z0-9]{4,12}", sid):
        return False, (
            "Student ID must be 4–12 characters and contain "
            "only letters (A-Z) and digits (0-9). "
            "No spaces or special characters allowed."
        )

    return True, "Valid"


# ─────────────────────────────────────────────
#  STUDENT NAME VALIDATOR
# ─────────────────────────────────────────────
def validate_student_name(name: str) -> ValidationResult:
    """
    Validate a student's full name.

    Rules:
      - Must not be empty
      - Only letters and spaces allowed (no digits, no punctuation)
      - Must be 2–50 characters long
    """
    n = name.strip()

    if not n:
        return False, "Student name cannot be empty."

    # Check length before expensive regex
    if len(n) < 2:
        return False, "Name is too short (minimum 2 characters)."

    if len(n) > 50:
        return False, "Name is too long (maximum 50 characters)."

    # Allow letters and single spaces between words
    #   \s* handles optional spaces at word boundaries
    if not re.fullmatch(r"[A-Za-z]+( [A-Za-z]+)*", n):
        return False, (
            "Name must contain only letters and single spaces. "
            "No digits, symbols, or double spaces allowed."
        )

    return True, "Valid"


# ─────────────────────────────────────────────
#  DEPARTMENT VALIDATOR
# ─────────────────────────────────────────────
VALID_DEPARTMENTS = {
    "Computer Science",
    "Software Engineering",
    "Electrical Engineering",
    "Mechanical Engineering",
    "Business Administration",
    "Mathematics",
    "Physics",
    "Chemistry",
    "Other",
}

def validate_department(dept: str) -> ValidationResult:
    """Check that the department is one of the known options."""
    if not dept or dept.strip() == "":
        return False, "Please select a department."
    if dept not in VALID_DEPARTMENTS:
        return False, f"'{dept}' is not a recognised department."
    return True, "Valid"


# ─────────────────────────────────────────────
#  SEMESTER VALIDATOR
# ─────────────────────────────────────────────
def validate_semester(semester) -> ValidationResult:
    """
    Validate semester number.
    Must be an integer between 1 and 8.
    """
    try:
        sem = int(semester)   # May raise ValueError if not a number
    except (ValueError, TypeError):
        return False, "Semester must be a whole number (1–8)."

    if sem < 1 or sem > 8:
        return False, "Semester must be between 1 and 8."

    return True, "Valid"


# ─────────────────────────────────────────────
#  DATE STRING VALIDATOR
# ─────────────────────────────────────────────
def validate_date_string(date_str: str) -> ValidationResult:
    """
    Validate a date string in YYYY-MM-DD format.

    Checks:
      - Correct format via regex first (cheap check)
      - Then attempts actual datetime parse to catch impossible dates
        like 2024-02-30.
    """
    s = date_str.strip()

    # Quick regex check
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return False, "Date must be in YYYY-MM-DD format (e.g. 2024-01-15)."

    # Deep check: does this calendar date actually exist?
    try:
        datetime.datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        return False, f"'{s}' is not a valid calendar date."

    return True, "Valid"


# ─────────────────────────────────────────────
#  COMBINED STUDENT FORM VALIDATOR
# ─────────────────────────────────────────────
def validate_student_form(student_id: str,
                           name: str,
                           department: str,
                           semester) -> ValidationResult:
    """
    Run all field validators and return the first error found,
    or (True, "Valid") if everything passes.

    Parameters
    ----------
    student_id  : str
    name        : str
    department  : str
    semester    : int | str

    Returns
    -------
    tuple[bool, str]
    """
    # List of (validator_function, argument) pairs
    checks = [
        (validate_student_id,   student_id),
        (validate_student_name, name),
        (validate_department,   department),
        (validate_semester,     semester),
    ]

    # Iterate and stop at first failure
    for validator, value in checks:
        ok, msg = validator(value)
        if not ok:
            return False, msg     # Return the error immediately

    return True, "Valid"


# ─────────────────────────────────────────────
#  PHOTO / IMAGE VALIDATOR
# ─────────────────────────────────────────────
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def validate_image_file(filepath: str) -> ValidationResult:
    """
    Check that a file path points to a supported image format.
    """
    _, ext = os.path.splitext(filepath)
    if ext.lower() not in ALLOWED_EXTENSIONS:
        return False, (
            f"Unsupported image format '{ext}'. "
            f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    if not os.path.exists(filepath):
        return False, f"Image file not found: {filepath}"
    return True, "Valid"


# ─────────────────────────────────────────────
#  QUICK STANDALONE TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Test each validator manually
    tests = [
        ("Student ID valid",    validate_student_id("CS2021001")),
        ("Student ID empty",    validate_student_id("")),
        ("Student ID special",  validate_student_id("CS-2021!")),
        ("Name valid",          validate_student_name("Ali Ahmed")),
        ("Name digits",         validate_student_name("Ali123")),
        ("Date valid",          validate_date_string("2024-01-15")),
        ("Date invalid",        validate_date_string("2024-02-30")),
        ("Semester 4",          validate_semester(4)),
        ("Semester 9",          validate_semester(9)),
    ]

    print("\n--- Validator Test Results ---")
    for label, (ok, msg) in tests:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  [{label}]  →  {msg}")