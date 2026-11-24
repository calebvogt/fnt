"""The brain map, run by the ABMA engine.

``open_brain_map(cfg)`` puts one animal with the config's anatomical brain in
its arena, steps the real engine in real time (times a speed), and shows the
brain map page (:mod:`fnt.abma.core.brainmap`) drawing it: every model
neuron's firing, single-neuron traces, the body in the box.

The page is the display only. A bridge object named ``engine`` carries frames
to it and controls back (play, speed, restart, another wiring draw, a change
to a projection's strength or a receptor level), so the one network that runs
is the Python one.

Where the page appears:

*In an FNT window*, when Qt's web engine is installed
(``pip install fieldneurotoolbox[brainmap]``, which adds PyQtWebEngine).

*In your own browser* otherwise: FNT writes the page to a temporary file,
serves the bridge on a local WebSocket (this computer only), opens the
browser, and keeps a small window up whose closing stops the engine.
"""
from __future__ import annotations

import base64
import json
import os
import tempfile
import time

import numpy as np
from PyQt5.QtCore import QFile, QIODevice, QObject, QTimer, QUrl, pyqtSignal, pyqtSlot
from PyQt5.QtWebChannel import QWebChannel, QWebChannelAbstractTransport
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow

from ..core import anatomy, brainmap

#: the network's own tick; the engine is stepped at this grain so the page
#: sees every tick
STEP_S = 0.25


_APP = None


def _app():
    """The Qt application, kept alive here: one made inside a function and
    dropped when it returns takes every window and timer with it."""
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


def webengine_available() -> bool:
    try:
        from PyQt5 import QtWebEngineWidgets  # noqa: F401
        return True
    except ImportError:
        return False


def qwebchannel_js() -> str:
    """Qt's own JavaScript side of the bridge, read from Qt's resources."""
    f = QFile(":/qtwebchannel/qwebchannel.js")
    if not f.open(QIODevice.ReadOnly):
        raise RuntimeError("Qt's qwebchannel.js is not available in this installation")
    try:
        return bytes(f.readAll()).decode("utf-8")
    finally:
        f.close()


class Engine(QObject):
    """One animal, the config's anatomical brain, stepped as the clock runs."""

    #: one engine step: position, heading, bladder, and every unit's rate
    frame = pyqtSignal(str)
    #: the run started again (with this wiring draw)
    cleared = pyqtSignal(int)

    def __init__(self, cfg, parent=None, save: bool = True):
        super().__init__(parent)
        import copy

        if not getattr(cfg.brain, "anatomy", ""):
            raise ValueError("the brain map needs a config with brain.anatomy set")
        self.cfg = copy.deepcopy(cfg)
        self.cfg.dt, self.cfg.days, self.cfg.n_trials = STEP_S, max(float(cfg.days), 1.0), 1
        self.cfg.brain.enabled = True
        self.name = self.cfg.brain.anatomy
        self.spec = anatomy.load_spec(self.name)
        self.save = bool(save)
        self.playing, self.rate, self._acc, self._last = False, 4.0, 0.0, time.perf_counter()
        self.step_ms: list[float] = []
        self.draws_per_s: list[float] = []
        self.sent = 0
        self._build()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(15)

    def _build(self) -> None:
        from ..core.simulation import Simulation

        self.sim, self.k = Simulation(self.cfg, 0), 0

    # ---- the clock ------------------------------------------------------ #
    def tick(self) -> None:
        now = time.perf_counter()
        if self.playing:
            self._acc += (now - self._last) * self.rate / STEP_S
        self._last = now
        n = min(40, int(self._acc))            # never more than 40 steps in one go
        self._acc -= n
        for _ in range(n):
            self.step()

    def step(self) -> None:
        t0 = time.perf_counter()
        self.sim.step(self.k * STEP_S, STEP_S)
        self.k += 1
        self.step_ms.append((time.perf_counter() - t0) * 1e3)
        del self.step_ms[:-2000]
        pol, sim = self.sim.policy, self.sim
        rates = np.clip(pol.r[0] * 255.0, 0, 255).astype(np.uint8)
        self.frame.emit(json.dumps({
            "x": float(sim.P[0, 0]), "y": float(sim.P[0, 1]), "h": float(sim.H[0]),
            "t": self.k * STEP_S, "dt": STEP_S, "turn": float(pol.turn[0]),
            "bladder": float(sim.bladder[0]) / 100.0,
            "rates": base64.b64encode(rates.tobytes()).decode("ascii")}))
        self.sent += 1

    # ---- what the page can ask for ------------------------------------- #
    @pyqtSlot(result=str)
    def hello(self) -> str:
        return json.dumps({"seed": int(self.cfg.brain.anatomy_seed), "units": self.sim.policy.brain.n,
                           "arena": [self.cfg.arena.width, self.cfg.arena.height],
                           "notes": brainmap.load_notes(self.name) if self.save else None})

    @pyqtSlot(result=str)
    def weights(self) -> str:
        return base64.b64encode(self.sim.policy.brain.W.astype("<f4").tobytes()).decode("ascii")

    @pyqtSlot(bool)
    def setPlaying(self, on: bool) -> None:
        self.playing, self._last = bool(on), time.perf_counter()

    @pyqtSlot(float)
    def setRate(self, v: float) -> None:
        self.rate = min(60.0, max(0.25, float(v)))

    @pyqtSlot(int)
    def restart(self, seed: int) -> None:
        self.cfg.brain.anatomy_seed = int(seed)
        self._build()
        self.cleared.emit(int(seed))

    @pyqtSlot(str, result=str)
    def setEdits(self, text: str) -> str:
        """Apply the page's edits to the same wiring draw; the animal carries
        on from where it is. Returns the weights now in force."""
        try:
            strengths, receptors = brainmap.edits_from_notes(json.loads(text or "{}"))
            brain = anatomy.AnatomicalBrain(self.spec, seed=int(self.cfg.brain.anatomy_seed),
                                            strengths=strengths, receptors=receptors)
        except ValueError:
            return self.weights()               # an edit that names nothing real changes nothing
        self.cfg.brain.strengths, self.cfg.brain.receptors = strengths, receptors
        self.sim.policy.brain = brain
        return self.weights()

    @pyqtSlot(str)
    def saveNotes(self, text: str) -> None:
        if not self.save:
            return
        try:
            notes = json.loads(text)
        except ValueError:
            return
        if isinstance(notes, dict) and len(text) < 2_000_000:
            brainmap.save_notes(self.name, notes)

    @pyqtSlot(int)
    def report(self, draws: int) -> None:
        self.draws_per_s.append(draws / 2.0)
        del self.draws_per_s[:-50]


class _SocketTransport(QWebChannelAbstractTransport):
    """Carries the bridge over one WebSocket, for a page in an ordinary browser."""

    def __init__(self, socket, parent=None):
        super().__init__(parent)
        self.socket = socket
        socket.textMessageReceived.connect(self._got)

    def sendMessage(self, message) -> None:
        from PyQt5.QtCore import QJsonDocument
        try:
            self.socket.sendTextMessage(bytes(QJsonDocument(message).toJson(QJsonDocument.Compact)).decode("utf-8"))
        except RuntimeError:
            pass                    # the browser tab has gone; an exception here would abort the program

    def _got(self, text: str) -> None:
        from PyQt5.QtCore import QJsonDocument
        doc = QJsonDocument.fromJson(text.encode("utf-8"))
        if doc.isObject():
            self.messageReceived.emit(doc.object(), self)


def _temp_page(html: str) -> str:
    d = tempfile.mkdtemp(prefix="abma_brain_map_")
    path = os.path.join(d, "brain_map.html")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(html)
    return path


def open_brain_map(cfg, where: str = "auto", exec_: bool = True, open_browser: bool = True):
    """Show the brain map for ``cfg`` and run it.

    ``where`` is ``"window"`` (inside FNT; needs Qt's web engine),
    ``"browser"``, or ``"auto"`` (the window when it can, else the browser).
    With ``exec_=False`` nothing blocks and ``(window, engine, page_path)`` is
    returned, which is what a test wants.
    """
    if where not in ("auto", "window", "browser"):
        raise ValueError("where is 'auto', 'window' or 'browser'")
    inside = webengine_available() if where == "auto" else where == "window"
    if inside and not webengine_available():
        raise RuntimeError("showing the map inside FNT needs Qt's web engine: "
                           "pip install PyQtWebEngine (or fieldneurotoolbox[brainmap]); "
                           "--brain-map browser works without it")
    app = _app()
    engine = Engine(cfg)
    channel = QWebChannel(engine)
    channel.registerObject("engine", engine)
    win = QMainWindow()
    win._engine, win._channel = engine, channel          # keep them alive with the window

    if inside:
        from PyQt5.QtWebEngineWidgets import QWebEngineSettings, QWebEngineView

        view = QWebEngineView()
        view.page().setWebChannel(channel)
        view.settings().setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls, True)
        page = _temp_page(brainmap.build_page(engine.spec, bridge_js=qwebchannel_js()))
        view.load(QUrl.fromLocalFile(page))
        win.setCentralWidget(view)
        win.setWindowTitle(f"ABMA brain map — {engine.spec.get('name', engine.name)}")
        win.resize(1500, 1000)
    else:
        from PyQt5.QtNetwork import QHostAddress
        from PyQt5.QtWebSockets import QWebSocketServer

        server = QWebSocketServer("ABMA brain map", QWebSocketServer.NonSecureMode, win)
        if not server.listen(QHostAddress.LocalHost, 0):
            raise RuntimeError("could not open a local port for the brain map")

        win._links = []

        def connected():
            sock = server.nextPendingConnection()
            link = (sock, _SocketTransport(sock))
            win._links.append(link)
            channel.connectTo(link[1])

            def gone():
                # a closed tab must stop receiving frames before its socket is destroyed
                channel.disconnectFrom(link[1])
                if link in win._links:
                    win._links.remove(link)
                sock.deleteLater()
            sock.disconnected.connect(gone)
        server.newConnection.connect(connected)
        page = _temp_page(brainmap.build_page(engine.spec, bridge_js=qwebchannel_js(),
                                              ws_url=f"ws://127.0.0.1:{server.serverPort()}"))
        win._server = server
        label = QLabel("The brain map is open in your browser.\n\n"
                       "The animal is being run here, by the ABMA engine.\n"
                       "Close this window to stop it.\n\n" + page)
        label.setMargin(18)
        label.setTextInteractionFlags(label.textInteractionFlags() | 0x1)   # selectable
        win.setCentralWidget(label)
        win.setWindowTitle("ABMA brain map (engine)")
        if open_browser:
            import webbrowser
            webbrowser.open(QUrl.fromLocalFile(page).toString())
    win.show()
    if not exec_:
        return win, engine, page
    return app.exec_()
