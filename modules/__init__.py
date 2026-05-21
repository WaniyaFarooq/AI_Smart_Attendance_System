"""modules package — AI Smart Attendance System sub-modules."""

from collections import defaultdict, deque

_face_votes = defaultdict(lambda: deque(maxlen=5))