"""
=============================================================
  modules/face_recognizer.py — Face Recognition with LBPH
=============================================================
  Uses OpenCV's LBPH (Local Binary Pattern Histogram) face
  recogniser — a classic, lightweight AI algorithm.

  HOW LBPH WORKS (explain in viva):
  ──────────────────────────────────
  1. Divide the face image into small cells.
  2. For each cell, compute an LBP (Local Binary Pattern):
     compare each pixel to its 8 neighbours; write 1 if
     neighbour > centre, else 0. This gives an 8-bit binary
     number per pixel.
  3. Build a histogram of these binary codes for each cell.
  4. Concatenate all cell histograms → one feature vector.
  5. At prediction time, compare the query vector to every
     training vector using Chi-square distance.
     The closest match wins (lowest distance = best match).

  Advantages of LBPH:
    ✅ Works under different lighting conditions
    ✅ Runs fast on a regular laptop (no GPU needed)
    ✅ Built into OpenCV — no extra installation
    ✅ Incremental training (can add new students without
       retraining from scratch)

  Demonstrates:
    - AI / Machine Learning concepts
    - OpenCV face recogniser API
    - OOP with state management
    - NumPy arrays for training data
    - Exception handling
    - File I/O (model save / load)
=============================================================
"""

import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from utils.helpers import logger, save_pickle, load_pickle, model_exists


class FaceRecognizer:
    """
    Wraps OpenCV's LBPH Face Recogniser with a clean interface.

    Attributes
    ----------
    recognizer  : cv2.face.LBPHFaceRecognizer
    label_map   : dict  {int_label → student_id_string}
    is_trained  : bool
    """

    def __init__(self):
        """Initialise the LBPH recogniser with default hyper-parameters."""
        # LBPHFaceRecognizer_create is in opencv-contrib-python
        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create(
                radius=1,       # Radius of the LBP circle
                neighbors=8,    # Number of sample points on the circle
                grid_x=8,       # Number of cells in horizontal direction
                grid_y=8,       # Number of cells in vertical direction
                threshold=cfg.RECOGNITION_THRESHOLD,  # Max confidence to accept
            )
        except AttributeError:
            raise ImportError(
                "cv2.face module not found!\n"
                "Install opencv-contrib-python:\n"
                "    pip install opencv-contrib-python"
            )

        # label_map: {0: 'CS2021001', 1: 'CS2021002', ...}
        # We need this because LBPH works with integer labels,
        # but we want to display student IDs (strings).
        self.label_map: dict = {}

        self.is_trained: bool = False

        # Try to load a previously saved model
        self._load_if_exists()

    # ─────────────────────────────────────────
    #  TRAINING
    # ─────────────────────────────────────────

    def train(self, faces_dir: str = None) -> tuple:
        """
        Train (or retrain) the LBPH recogniser on all stored face images.

        Training pipeline
        -----------------
        1. Scan faces_dir for student sub-folders.
        2. For each student folder, load every .jpg image.
        3. Preprocess each image (grayscale, resize, equalise).
        4. Assign an integer label to each student.
        5. Call recognizer.train(images, labels).
        6. Save model weights + label map to disk.

        Parameters
        ----------
        faces_dir : str, optional
            Root directory containing student face sub-folders.
            Defaults to cfg.FACES_DIR.

        Returns
        -------
        tuple[bool, str]
            (success, message)
        """
        if faces_dir is None:
            faces_dir = cfg.FACES_DIR

        images = []   # List of grayscale face numpy arrays
        labels = []   # List of corresponding integer labels
        label_map = {}  # {int_label: student_id}

        # ── Scan student folders ──────────────────────────
        try:
            student_folders = [
                d for d in os.listdir(faces_dir)
                if os.path.isdir(os.path.join(faces_dir, d))
            ]
        except FileNotFoundError:
            return False, f"Faces directory not found: {faces_dir}"

        if not student_folders:
            return False, "No student face data found. Please register students first."

        label_counter = 0   # Integer labels start at 0

        for student_id in student_folders:
            student_folder = os.path.join(faces_dir, student_id)
            img_files = [
                f for f in os.listdir(student_folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            if not img_files:
                logger.warning(f"No images found for student: {student_id}")
                continue

            # Assign this student an integer label
            label_map[label_counter] = student_id

            for img_file in img_files:
                img_path = os.path.join(student_folder, img_file)

                # Load image in grayscale (0 = IMREAD_GRAYSCALE)
                img = cv2.imread(img_path, 0)

                if img is None:
                    logger.warning(f"Could not read image: {img_path}")
                    continue

                # Resize to standard size required by LBPH
                img = cv2.resize(img, cfg.FACE_IMG_SIZE)

                # Equalise histogram for lighting normalisation
                img = cv2.equalizeHist(img)

                images.append(img)                   # Add image array
                labels.append(label_counter)         # Add matching label

            logger.debug(
                f"Loaded {len(img_files)} images for {student_id} "
                f"(label={label_counter})"
            )
            label_counter += 1

        # ── Check we have enough data ─────────────────────
        if len(images) < 2:
            return (
                False,
                "Need face images for at least 2 students to train the model.",
            )

        # ── Train LBPH ────────────────────────────────────
        labels_array = np.array(labels, dtype=np.int32)
        self.recognizer.train(images, labels_array)

        # ── Save to disk ──────────────────────────────────
        os.makedirs(cfg.MODEL_DIR, exist_ok=True)
        self.recognizer.write(cfg.MODEL_FILE)       # Save LBPH weights
        save_pickle(label_map, cfg.LABEL_MAP_FILE)  # Save label map

        self.label_map = label_map
        self.is_trained = True

        msg = (
            f"Model trained successfully on {len(images)} images "
            f"from {label_counter} student(s)."
        )
        logger.info(msg)
        return True, msg

    # ─────────────────────────────────────────
    #  PREDICTION (RECOGNITION)
    # ─────────────────────────────────────────

    def predict(self, face_gray: np.ndarray) -> dict:
        """
        Recognise a face and return structured prediction results.

        Parameters
        ----------
        face_gray : np.ndarray
            A grayscale, preprocessed face image (same size as training).

        Returns
        -------
        dict with keys:
            "recognized"   : bool   — did we find a match?
            "student_id"   : str    — matched student ID (or "Unknown")
            "confidence"   : float  — LBPH confidence (lower = better)
            "label"        : int    — internal integer label
        """
        if not self.is_trained:
            return {
                "recognized" : False,
                "student_id" : "Unknown",
                "confidence" : 999.0,
                "label"      : -1,
            }

        try:
            # Ensure correct size
            face_resized = cv2.resize(face_gray, cfg.FACE_IMG_SIZE)
            face_input = face_resized
           # face_eq      = cv2.equalizeHist(face_resized)

            # LBPH predict returns (label, confidence)
            # confidence: 0 = perfect match, > threshold = no match
            label, confidence = self.recognizer.predict(face_input )

            # Check if confidence is within acceptable threshold
            recognized = confidence < cfg.RECOGNITION_THRESHOLD

            student_id = (
                self.label_map.get(label, "Unknown")
                if recognized else "Unknown"
            )

            return {
                "recognized" : recognized,
                "student_id" : student_id,
                "confidence" : round(float(confidence), 2),
                "label"      : int(label),
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                "recognized" : False,
                "student_id" : "Unknown",
                "confidence" : 999.0,
                "label"      : -1,
            }

    # ─────────────────────────────────────────
    #  MODEL PERSISTENCE
    # ─────────────────────────────────────────

    def _load_if_exists(self) -> None:
        """
        Load model weights and label map from disk (if they exist).
        Called automatically in __init__.
        """
        if model_exists() and os.path.exists(cfg.LABEL_MAP_FILE):
            try:
                self.recognizer.read(cfg.MODEL_FILE)
                self.label_map = load_pickle(cfg.LABEL_MAP_FILE) or {}
                self.is_trained = True
                logger.info(
                    f"Loaded trained model — "
                    f"{len(self.label_map)} student(s) in memory."
                )
            except Exception as e:
                logger.warning(f"Could not load existing model: {e}")
                self.is_trained = False
        else:
            logger.info("No trained model found. Train the model after registration.")

    def reload_model(self) -> bool:
        """
        Force-reload the model from disk.
        Call this after training to update the running instance.

        Returns
        -------
        bool  — True if loaded successfully.
        """
        if not model_exists():
            logger.warning("reload_model: no model file found.")
            return False
        try:
            self.recognizer.read(cfg.MODEL_FILE)
            self.label_map = load_pickle(cfg.LABEL_MAP_FILE) or {}
            self.is_trained = True
            logger.info("Model reloaded from disk.")
            return True
        except Exception as e:
            logger.error(f"Failed to reload model: {e}")
            return False

    # ─────────────────────────────────────────
    #  STATUS / INFO
    # ─────────────────────────────────────────

    def get_registered_students(self) -> list:
        """Return list of student IDs the model was trained on."""
        return list(self.label_map.values())

    def model_info(self) -> dict:
        """Return a summary dict about the current model state."""
        return {
            "is_trained"   : self.is_trained,
            "num_students" : len(self.label_map),
            "students"     : self.get_registered_students(),
            "model_path"   : cfg.MODEL_FILE,
            "threshold"    : cfg.RECOGNITION_THRESHOLD,
        }