"""
=============================================================
  modules/face_detector.py — Face Detection with OpenCV
=============================================================
  Uses OpenCV's pre-trained Haar Cascade classifier to detect
  human faces in images or video frames.

  Haar Cascade is a machine-learning-based approach where a
  cascade function (trained on thousands of face/non-face
  images) quickly filters image regions.

  Demonstrates:
    - OpenCV (cv2) library — object detection
    - OOP — FaceDetector class
    - NumPy arrays for image data
    - Exception handling
    - Functions with multiple return values
=============================================================
"""

import os
import sys
import cv2          # OpenCV — computer vision library
import numpy as np  # NumPy — numerical arrays (images = arrays of pixels)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from utils.helpers import logger


class FaceDetector:
    """
    Detects faces in images using OpenCV's Haar Cascade classifier.

    Usage Example
    -------------
    detector = FaceDetector()
    faces = detector.detect_faces(frame)
    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]   # crop the face region
    """

    def __init__(self):
        """
        Initialise the detector by loading the Haar Cascade XML file.
        OpenCV ships this file inside its data directory.
        """
        # Find the path to OpenCV's built-in data files
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

        # Check that the file actually exists before loading
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(
                f"Haar Cascade XML not found at: {cascade_path}\n"
                "Make sure opencv-python (or opencv-contrib-python) is installed."
            )

        # CascadeClassifier loads the pre-trained face model
        self.cascade = cv2.CascadeClassifier(cascade_path)

        if self.cascade.empty():
            raise RuntimeError(
                "Failed to load Haar Cascade classifier. "
                "The XML file may be corrupted."
            )

        logger.info("FaceDetector initialised with Haar Cascade.")

        # Detection hyper-parameters (loaded from config)
        self.scale_factor  = cfg.FACE_DETECTION_CONFIG["scale_factor"]
        self.min_neighbors = cfg.FACE_DETECTION_CONFIG["min_neighbors"]
        self.min_size      = cfg.FACE_DETECTION_CONFIG["min_face_size"]

    # ─────────────────────────────────────────
    #  CORE DETECTION METHODS
    # ─────────────────────────────────────────

    def detect_faces(self, frame: np.ndarray) -> list:
        """
        Detect all faces in a single image frame.

        How it works
        ------------
        1. Convert the colour image to grayscale (detection works on grayscale).
        2. Apply histogram equalisation to improve contrast.
        3. Run the cascade classifier — it scans the image at multiple scales
           looking for face-shaped regions.
        4. Return the (x, y, width, height) rectangles of found faces.

        Parameters
        ----------
        frame : np.ndarray
            A BGR image array (the default format from cv2.VideoCapture).

        Returns
        -------
        list of tuples
            [(x, y, w, h), ...] — one tuple per detected face.
            Returns an empty list if no faces found.
        """
        # ── Step 1: Convert to grayscale ──────────────────
        # cv2.cvtColor changes colour space; BGRA→Gray reduces dimensions
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Step 2: Equalise histogram ────────────────────
        # This improves detection under varying lighting conditions.
        # It stretches the range of pixel intensities.
        gray = cv2.equalizeHist(gray)

        # ── Step 3: Run cascade detection ─────────────────
        # detectMultiScale searches the image at multiple scales
        # Parameters:
        #   scaleFactor  — how much the image is shrunk at each step
        #   minNeighbors — how many overlapping detections to confirm a face
        #   minSize      — smallest face box to consider (pixels)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor  = self.scale_factor,
            minNeighbors = self.min_neighbors,
            minSize      = self.min_size,
        )

        # detectMultiScale returns an empty tuple if nothing found
        if len(faces) == 0:
            return []

        # Convert to a plain Python list of tuples for easier handling
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    def detect_largest_face(self, frame: np.ndarray):
        """
        Detect all faces but return only the largest one (by area).
        Useful for attendance: we assume the closest person is the target.

        Parameters
        ----------
        frame : np.ndarray

        Returns
        -------
        tuple | None
            (x, y, w, h) of the largest face, or None if no face found.
        """
        faces = self.detect_faces(frame)

        if not faces:
            return None

        # Sort faces by area (w * h) in descending order; take first
        largest = max(faces, key=lambda f: f[2] * f[3])
        return largest

    # ─────────────────────────────────────────
    #  ROI & PREPROCESSING
    # ─────────────────────────────────────────

    def extract_face_roi(self, frame: np.ndarray,
                         face_coords: tuple) -> np.ndarray:
        """
        Crop the face region from the full frame.

        Parameters
        ----------
        frame       : np.ndarray  — full image
        face_coords : tuple       — (x, y, w, h)

        Returns
        -------
        np.ndarray  — cropped face region (BGR colour)
        """
        x, y, w, h = face_coords
        # NumPy array slicing: [rows, columns] = [y:y+h, x:x+w]
        return frame[y: y + h, x: x + w]

    @staticmethod
    def preprocess_face(face_roi: np.ndarray,
                        size: tuple = None) -> np.ndarray:
        """
        Prepare a face ROI for the recogniser:
          1. Convert to grayscale
          2. Resize to a standard size
          3. Equalise histogram

        Parameters
        ----------
        face_roi : np.ndarray  — raw face crop (may be BGR or gray)
        size     : tuple       — target (width, height). Defaults to config.

        Returns
        -------
        np.ndarray  — grayscale, resized, equalised face array
        """
        if size is None:
            size = cfg.FACE_IMG_SIZE   # (200, 200) from config

        # Convert to gray if input is colour
        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_roi.copy()

        # Resize to standard dimensions (recogniser requires consistent size)
        resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)

        # Equalise histogram for lighting normalisation
        equalised = cv2.equalizeHist(resized)

        return equalised

    # ─────────────────────────────────────────
    #  DRAWING HELPERS
    # ─────────────────────────────────────────

    @staticmethod
    def draw_face_boxes(frame: np.ndarray,
                        faces: list,
                        color: tuple = (0, 255, 0),
                        label: str = "") -> np.ndarray:
        """
        Draw bounding rectangles around detected faces.

        Parameters
        ----------
        frame  : np.ndarray — image to draw on (will be modified in place)
        faces  : list       — [(x, y, w, h), ...]
        color  : tuple      — BGR colour, default green (0,255,0)
        label  : str        — text to display above the box

        Returns
        -------
        np.ndarray  — same frame with boxes drawn
        """
        annotated = frame.copy()   # Work on a copy to keep original clean

        for (x, y, w, h) in faces:
            # Draw rectangle: (image, top-left, bottom-right, color, thickness)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Draw label text above the box (if provided)
            if label:
                # Black background for text readability
                text_y = y - 10 if y - 10 > 10 else y + h + 20
                cv2.putText(
                    annotated,
                    label,
                    (x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,  # Font style
                    0.8,                        # Font scale
                    (0, 0, 0),                  # Black shadow
                    3,                          # Shadow thickness
                )
                cv2.putText(
                    annotated,
                    label,
                    (x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )

        return annotated

    @staticmethod
    def draw_status_bar(frame: np.ndarray,
                        message: str,
                        color: tuple = (0, 200, 0)) -> np.ndarray:
        """
        Draw a semi-transparent status message bar at the bottom of the frame.

        Parameters
        ----------
        frame   : np.ndarray
        message : str
        color   : tuple — BGR

        Returns
        -------
        np.ndarray
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Draw filled rectangle at the bottom
        cv2.rectangle(overlay, (0, h - 40), (w, h), (0, 0, 0), -1)

        # Blend with the original (semi-transparent)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw the message text
        cv2.putText(
            frame,
            message,
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )
        return frame

    # ─────────────────────────────────────────
    #  WEBCAM CAPTURE (static helper)
    # ─────────────────────────────────────────

    @staticmethod
    def open_webcam(camera_index: int = 0) -> cv2.VideoCapture:
        """
        Open the webcam and return a VideoCapture object.

        Parameters
        ----------
        camera_index : int  — 0 = default (built-in) webcam.

        Returns
        -------
        cv2.VideoCapture

        Raises
        ------
        RuntimeError if the webcam cannot be opened.
        """
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open webcam at index {camera_index}. "
                "Check that:\n"
                "  1. Your webcam is connected and not in use by another app.\n"
                "  2. You have granted camera permission to this application."
            )

        # Set preferred resolution (720p)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

        logger.info(f"Webcam opened (index={camera_index}).")
        return cap

    @staticmethod
    def read_frame(cap: cv2.VideoCapture):
        """
        Read one frame from an open VideoCapture object.

        Returns
        -------
        tuple[bool, np.ndarray | None]
            (success, frame) — frame is None if read failed.
        """
        ret, frame = cap.read()
        if not ret or frame is None:
            return False, None
        return True, frame