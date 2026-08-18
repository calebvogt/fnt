"""Subject identification for recordings.

Every recording is tagged with a subject code so sessions can be grouped,
compared within a person, and kept apart between people.

Two deliberate choices:

* **A code, not a name.** ``S01`` / ``CV01`` rather than "Caleb Vogt". The code
  ends up in folder names, config files and any report you might share, so
  keeping identifying details out of it by default is simply the cheaper habit.
  Nothing enforces this — it is your data — but the field is shaped for it.

* **Handedness is stored per subject.** For most recordings that would be
  padding; for this project it is a real covariate. Hemispheric lateralization
  differs systematically between right- and left-handers, so any left/right
  asymmetry or interhemispheric-synchrony result has to be read against it.
  Asked once per subject, then remembered.

The registry lives beside the recordings (``subjects.json`` in the recording
folder) so it travels with the data rather than being stranded in app settings.
"""

import json
import os
import re
from datetime import datetime

REGISTRY_NAME = "subjects.json"
HANDEDNESS = ["right", "left", "ambidextrous", "unspecified"]
SEX = ["unspecified", "female", "male", "other"]
# Session 1 (M01) lost both ear electrodes within two minutes; the cause was
# long hair between the sensor and the skin. That is a property of the person,
# it predicts which electrodes will be usable before a session starts, and it
# explains a failure that otherwise looks like bad technique — so it is worth
# one field.
HAIR_OVER_EARS = ["unspecified", "clear", "some", "long/thick"]


def sanitize_id(text):
    """Filesystem-safe subject code: letters, digits, dash, underscore."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", str(text or "").strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-_")
    return cleaned[:32]


class SubjectRegistry:
    """Known subjects and their per-person metadata."""

    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.path = os.path.join(base_dir, REGISTRY_NAME)
        self.data = {}
        self.load()

    def load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    self.data = loaded
        except Exception:
            self.data = {}
        return self.data

    def save(self):
        try:
            os.makedirs(self.base_dir, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, sort_keys=True)
            return True
        except Exception:
            return False

    def ids(self):
        return sorted(self.data)

    def get(self, subject_id):
        return dict(self.data.get(subject_id, {}))

    def upsert(self, subject_id, handedness=None, notes=None, sex=None,
               hair_over_ears=None):
        """Create or update a subject; returns the stored record."""
        subject_id = sanitize_id(subject_id)
        if not subject_id:
            return None
        record = self.data.setdefault(
            subject_id, {"created": datetime.now().isoformat(timespec="seconds")})
        if handedness:
            record["handedness"] = handedness
        if sex:
            record["sex"] = sex
        if hair_over_ears:
            record["hair_over_ears"] = hair_over_ears
        if notes is not None:
            record["notes"] = notes
        record["last_seen"] = datetime.now().isoformat(timespec="seconds")
        record["sessions"] = int(record.get("sessions", 0))
        self.save()
        return dict(record)

    def note_session(self, subject_id):
        """Increment the session counter for a subject."""
        subject_id = sanitize_id(subject_id)
        if not subject_id:
            return
        record = self.data.setdefault(
            subject_id, {"created": datetime.now().isoformat(timespec="seconds")})
        record["sessions"] = int(record.get("sessions", 0)) + 1
        record["last_seen"] = datetime.now().isoformat(timespec="seconds")
        self.save()
