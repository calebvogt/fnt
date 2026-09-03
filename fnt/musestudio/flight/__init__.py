"""Flight Mode — piloting a craft from cortical EEG alone.

Sub-package layout:

* ``pipeline`` — pure signal → control translation. No Qt, no hardware.
* ``sim``      — craft physics. Pure, deterministic, unit-testable.
* ``telemetry``— per-tick CSV of every pipeline stage, on the LSL clock.

The split exists so the whole control path can be exercised against synthetic
or replayed EEG without a headband, which is how every other feature in this
codebase has been de-risked.
"""

from fnt.musestudio.flight.pipeline import (
    ControlConfig, ControlPipeline, PipelineTrace, ChannelTrace,
)
from fnt.musestudio.flight.sim import CraftConfig, CraftSim, CraftState, FlightPhase

__all__ = [
    "ControlConfig", "ControlPipeline", "PipelineTrace", "ChannelTrace",
    "CraftConfig", "CraftSim", "CraftState", "FlightPhase",
]
