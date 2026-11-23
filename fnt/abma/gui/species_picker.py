"""Species cards — pick the animal, then tune it.

The card grid is the first thing an experimenter meets in the Agent Builder,
because "which animal is this?" is the question they can actually answer
without reading anything. Choosing a card fills in the *body*: mass, length,
speed, metabolic rate, olfactory acuity, scent output. It never fills in a home
range or a territory size — those are what the experiment is for.
"""
from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QFrame, QLabel, QVBoxLayout, QGridLayout, QSizePolicy,
)

from ..core.species import SPECIES, Species

_CARD_QSS = """
QFrame#species_card {
    background: #242830; border: 1px solid #343a44; border-radius: 7px;
}
QFrame#species_card:hover { border-color: #4a90d9; background: #272c35; }
QFrame#species_card[picked="true"] {
    border: 2px solid #58c470; background: #24302a;
}
"""


class SpeciesCard(QFrame):
    """One clickable animal card."""

    clicked = pyqtSignal(str)          # species key

    def __init__(self, sp: Species, parent=None):
        super().__init__(parent)
        self.sp = sp
        self.setObjectName("species_card")
        self.setCursor(Qt.PointingHandCursor)
        self.setProperty("picked", "false")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        v = QVBoxLayout(self)
        v.setContentsMargins(9, 7, 9, 8)
        v.setSpacing(1)
        name = QLabel(sp.name)
        name.setStyleSheet("font-weight:bold; font-size:12px; color:#e9edf2;")
        v.addWidget(name)
        latin = QLabel(sp.latin)
        latin.setStyleSheet("font-style:italic; font-size:10px; color:#8a9099;")
        v.addWidget(latin)
        facts = QLabel(f"{sp.mass_g.get('M', '?')} g · {sp.body_length_cm:g} cm"
                       f" · {sp.scent_rate:g} marks/h")
        facts.setStyleSheet("font-size:10px; color:#6f9fd0;")
        facts.setWordWrap(True)
        v.addWidget(facts)
        self.setToolTip(f"{sp.summary}\n\n{sp.notes}" if sp.notes
                        else sp.summary)

    def set_picked(self, on: bool) -> None:
        self.setProperty("picked", "true" if on else "false")
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.clicked.emit(self.sp.key)


class SpeciesPicker(QWidget):
    """Grid of species cards plus the selected animal's natural-history line."""

    species_picked = pyqtSignal(str)   # species key

    def __init__(self, columns: int = 3, parent=None):
        super().__init__(parent)
        self.setStyleSheet(_CARD_QSS)
        self._cards: dict[str, SpeciesCard] = {}
        self._key: str | None = None
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)
        hint = QLabel("Pick the animal — this sets its body (mass, size, "
                      "speed, nose, scent output). Space use is not set here: "
                      "it emerges from marking and competition.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#8a9099; font-size:10px;")
        v.addWidget(hint)
        grid = QGridLayout()
        grid.setSpacing(6)
        for i, sp in enumerate(SPECIES):
            card = SpeciesCard(sp)
            card.clicked.connect(self._on_pick)
            self._cards[sp.key] = card
            grid.addWidget(card, i // columns, i % columns)
        v.addLayout(grid)
        self.detail = QLabel("")
        self.detail.setWordWrap(True)
        self.detail.setStyleSheet("color:#9aa2ac; font-size:10px; "
                                  "padding:2px 1px;")
        v.addWidget(self.detail)

    def _on_pick(self, key: str):
        self.set_key(key)
        self.species_picked.emit(key)

    def key(self) -> str | None:
        return self._key

    def set_key(self, key: str | None) -> None:
        self._key = key
        for k, card in self._cards.items():
            card.set_picked(k == key)
        from ..core.species import get_species
        sp = get_species(key) if key else None
        self.detail.setText(f"{sp.summary}  {sp.notes}" if sp else
                            "Custom animal — body values below are yours to "
                            "set.")
