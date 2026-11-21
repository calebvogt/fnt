"""Simulated FED3s for trying the FED3 window without hardware.

    python -m fnt.fed3.ui.window --demo 4            # four simulated devices
    python -m fnt.fed3.ui.window --demo 4 --speed 60 # a minute per second

Each device is a :class:`~.device_sim.SimDevice` behind a fake port, with a
restless simulated animal poking at it. The engine and the devices share one
clock, so ``--speed`` runs a day in minutes.
"""
import random
import time
from datetime import datetime, timedelta

from . import schedule as sch
from .device_sim import SimDevice, SimTransport


class DemoWorld:
    def __init__(self, count=4, speed=1.0, seed=1):
        self.speed = speed
        self.rng = random.Random(seed)
        self.start = datetime.now(sch.standard_timezone()).replace(microsecond=0)
        self.elapsed = 0.0
        self.wall = time.monotonic()
        self.transports = {
            f"SIM{n + 1}": SimTransport(SimDevice(device_id=n + 1, now=self.start, seed=seed + n))
            for n in range(count)
        }
        # Each animal has its own habits, so the devices don't move in step.
        self.animals = {port: random.Random(seed * 100 + n)
                        for n, port in enumerate(self.transports)}

    # Engine hooks
    def now(self):
        return self.start + timedelta(seconds=self.elapsed)

    def monotonic(self):
        return self.elapsed

    def ports(self):
        return {port: "simulated FED3" for port in self.transports}

    def open(self, port):
        transport = self.transports[port]
        transport.closed = False
        return transport

    def tick(self):
        """Move the world on by however much (scaled) time has passed."""
        wall = time.monotonic()
        dt = (wall - self.wall) * self.speed
        self.wall = wall
        if dt <= 0:
            return
        self.elapsed += dt
        for port, transport in self.transports.items():
            rng = self.animals[port]
            remaining = dt
            while remaining > 0:      # small steps, so a fast demo still looks like animals
                chunk = min(remaining, 5.0)
                transport.act(lambda device, c=chunk: self._live(device, c, rng))
                remaining -= chunk

    def _live(self, device, dt, rng):
        # Pokes about every 20 s, more often on the left; a pellet in the
        # well is taken within a minute or so.
        # A poke takes time on the device's clock too, so it comes out of dt;
        # otherwise the device's clock would run ahead of the engine's.
        used = 0.0
        expected = dt / 20.0
        while expected > 0 and used < dt:
            if rng.random() < min(expected, 1.0):
                ms = rng.randint(80, 600)
                device.poke("L" if rng.random() < 0.7 else "R", ms=ms)
                used += ms / 1000
            expected -= 1.0
        device.advance(max(dt - used, 0.0))
        if device.well == "full" and rng.random() < min(dt / 40.0, 1.0):
            device.retrieve()
