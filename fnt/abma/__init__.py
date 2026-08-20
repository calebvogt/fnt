"""Animal Behavior Modeling Arena (ABMA).

A GUI-driven agent-based platform for running *in silico* animal-behaviour
experiments inside the FieldNeuroToolbox. Design an arena, populate it
with genetically and pharmacologically manipulated agents, run replicate trials,
and export tracking data in FNT's canonical schema for post-hoc analysis in R.

The engine (:mod:`fnt.abma.core`) is fully headless; the GUI
(:mod:`fnt.abma.gui`) is a thin PyQt5 wrapper. Keep this init import-light so the
package can be imported before a QApplication exists.
"""

__all__ = []
