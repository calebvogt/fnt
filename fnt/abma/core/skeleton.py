"""A jointed skeleton built from published bone lengths and joint ranges.

A skeleton spec (``fnt/abma/bodies/<name>.json``, or the user's own in
``~/ABMA_bodies``) holds:

``lengths``
    One entry per bone or region with its length in mm and a ``basis``:
    ``published`` (a number printed in a paper), ``derived`` (arithmetic on
    published numbers) or ``layout`` (our own choice, where nothing is in
    print). Each names its source.
``spine``, ``tail``, ``head``
    The standing posture of the backbone, one pitch per vertebra, with the
    printed range of each joint between two vertebrae.
``joints``
    The limb joints, each with its printed range, how the angle is measured
    here, and what had to be assumed to read the printed range.
``layout``, ``gait``, ``radius_mm``
    Where bones attach, how the animal stands and steps, and how thick the
    bones are drawn. None of this is in a paper.

``Skeleton(spec).pose(phase, amp, bend, sniff)`` places every joint in space
(mm; x forwards, y up, z to the animal's right, the ground at y = 0, the hip
joints over x = 0). The legs are placed by reaching for the ground: a foot
position is chosen from the stepping pattern and the joints above it are
solved, staying inside the printed ranges where that is possible.
``gait_check`` reports how much of each range a walk uses.

This is geometry only. Nothing here has mass or muscle, and the network does
not move these joints: it sets the animal's speed and turning, and the legs
step in a fixed pattern.
"""
from __future__ import annotations

import glob
import json
import math
import os

import numpy as np

BASES = ("published", "derived", "layout")
SIDES = (("R", 1.0), ("L", -1.0))
#: where in the stepping cycle each leg is; diagonal legs step together (a trot)
OFFSET = {("fore", "R"): 0.0, ("fore", "L"): math.pi, ("hind", "R"): math.pi, ("hind", "L"): 0.0}


def package_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bodies")


def user_dir() -> str:
    return os.path.join(os.path.expanduser("~"), "ABMA_bodies")


def list_skeletons() -> list[tuple[str, str]]:
    """Every skeleton that can be asked for by name, as ``(name, path)``; the
    user's own first, so an edited copy shadows the shipped one."""
    out, seen = [], set()
    for d in (user_dir(), package_dir()):
        for p in sorted(glob.glob(os.path.join(d, "*.json"))):
            name = os.path.splitext(os.path.basename(p))[0]
            if name not in seen:
                seen.add(name)
                out.append((name, p))
    return out


def find_skeleton(name: str) -> str:
    if name and os.path.isfile(name):
        return name
    for n, p in list_skeletons():
        if n == name:
            return p
    have = ", ".join(n for n, _ in list_skeletons()) or "none"
    raise FileNotFoundError(f"no skeleton named {name!r} (available: {have}); or give a path to a .json spec")


def check_spec(spec: dict) -> None:
    """Raise ``ValueError`` naming the first thing wrong with a spec."""
    for key in ("name", "lengths", "spine", "tail", "head", "joints", "layout", "gait", "radius_mm", "sources"):
        if key not in spec:
            raise ValueError(f"skeleton spec has no {key!r}")
    need = ("skull", "sacrum", "tail", "pelvis", "scapula", "humerus", "forearm", "forepaw", "femur", "tibia",
            "foot", "toes") + tuple(r["id"] for r in spec["spine"])
    for k in need:
        v = spec["lengths"].get(k)
        if not v or not float(v.get("mm") or 0) > 0:
            raise ValueError(f"skeleton spec has no length for {k!r}")
        if v.get("basis") not in BASES:
            raise ValueError(f"length {k!r}: basis must be one of {BASES}")
        for part in str(v.get("source", "")).replace(";", ",").split(","):
            key = part.strip().split(" ")[0]
            if v["basis"] == "published" and key not in spec["sources"]:
                raise ValueError(f"length {k!r} is marked published but its source {key!r} is not listed")
    prev = float(spec["layout"]["sacrum_pitch_deg"])
    for r in spec["spine"]:
        if len(r["pitch_deg"]) != int(r["count"]):
            raise ValueError(f"spine region {r['id']!r}: {len(r['pitch_deg'])} pitches for {r['count']} vertebrae")
        lo, hi = r["flex_limit_deg"]
        for i, p in enumerate(r["pitch_deg"]):
            if not lo - 1e-6 <= p - prev <= hi + 1e-6 and not (r is spine_first(spec) and i == 0):
                raise ValueError(f"spine region {r['id']!r}, vertebra {i + 1}: a bend of {p - prev:g} degrees "
                                 f"is outside the printed range {lo:g} to {hi:g}")
            prev = p
    lo, hi = spec["head"]["limit_deg"]
    for p in (spec["head"]["pitch_deg"], spec["head"]["pitch_deg"] + spec["head"].get("sniff_pitch_deg", 0.0)):
        if not lo - 1e-6 <= p - prev <= hi + 1e-6:
            raise ValueError(f"head: a bend of {p - prev:g} degrees is outside the printed range {lo:g} to {hi:g}")
    t = spec["tail"]
    if len(t["pitch_deg"]) != int(t["pieces"]):
        raise ValueError("tail: one pitch per piece is needed")
    ids = [j["id"] for j in spec["joints"]]
    for k in ("hip_flex", "hip_out", "knee", "ankle", "shoulder", "elbow", "wrist"):
        if k not in ids:
            raise ValueError(f"skeleton spec has no joint {k!r}")


def spine_first(spec: dict) -> dict:
    """The spine region next to the sacrum. Its first joint (loin to sacrum)
    has no printed range of its own, so its bend is not checked."""
    return spec["spine"][0]


def load_spec(name: str = "c57_male") -> dict:
    with open(find_skeleton(name), encoding="utf-8") as f:
        spec = json.load(f)
    check_spec(spec)
    return spec


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def _dir(pitch_deg: float, back: bool = False) -> np.ndarray:
    a = math.radians(pitch_deg)
    return np.array([-math.cos(a) if back else math.cos(a), math.sin(a), 0.0])


def _between(u: np.ndarray, v: np.ndarray) -> float:
    c = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _two_bones(root: np.ndarray, end: np.ndarray, l1: float, l2: float, pole: np.ndarray):
    """The middle joint of two bones that run from ``root`` to ``end``, bent
    towards ``pole``. Returns ``(middle, end, reached)``; if the two bones are
    too short or too long to span the gap, ``end`` is moved along the line."""
    d = end - root
    dist = float(np.linalg.norm(d))
    u = d / (dist + 1e-12)
    want = min(max(dist, abs(l1 - l2) + 1e-3), l1 + l2 - 1e-3)
    reached = abs(want - dist) < 1e-6
    a = (l1 * l1 - l2 * l2 + want * want) / (2.0 * want)
    h = math.sqrt(max(0.0, l1 * l1 - a * a))
    v = pole - float(np.dot(pole, u)) * u
    v = v / (np.linalg.norm(v) + 1e-12)
    return root + a * u + h * v, root + want * u, reached


def _yaw(v: np.ndarray, th: float) -> np.ndarray:
    c, s = math.cos(th), math.sin(th)
    return np.array([v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c])


class Skeleton:
    """A skeleton spec turned into joint positions. See the module text."""

    def __init__(self, spec: dict | str = "c57_male"):
        self.spec = load_spec(spec) if isinstance(spec, str) else spec
        check_spec(self.spec)
        s = self.spec
        self.len = {k: float(v["mm"]) for k, v in s["lengths"].items()}
        self.limit = {j["id"]: (float(j["min_deg"]), float(j["max_deg"])) for j in s["joints"]}
        self.enforced = {j["id"]: bool(j.get("enforced", True)) for j in s["joints"]}
        self._axial()

    # ---- the backbone, standing ------------------------------------------ #
    def _axial(self) -> None:
        s, lay, L = self.spec, self.spec["layout"], self.len
        P: dict[str, np.ndarray] = {}
        bones: list[tuple[str, str, str]] = []
        pel = _dir(lay["pelvis_pitch_deg"])
        at = float(lay["acetabulum_at"])
        for side, sg in SIDES:
            hip = np.array([0.0, float(lay["hip_height_mm"]), sg * float(lay["hip_half_width_mm"])])
            P["hip_" + side] = hip
            P["crest_" + side] = hip + at * L["pelvis"] * pel
            P["ischium_" + side] = hip - (1.0 - at) * L["pelvis"] * pel
            bones.append(("crest_" + side, "ischium_" + side, "pelvis"))
        mid = 0.5 * (P["crest_R"] + P["crest_L"])
        P["ls"] = mid + np.array([float(lay["lumbosacral_dx_mm"]), float(lay["lumbosacral_dy_mm"]), 0.0])
        # sacrum and tail, towards the tail
        P["tail_0"] = P["ls"] + L["sacrum"] * _dir(float(lay["sacrum_pitch_deg"]), back=True)
        bones.append(("ls", "tail_0", "sacrum"))
        self.tail_chain = ["tail_0"]
        n = int(s["tail"]["pieces"])
        for i, p in enumerate(s["tail"]["pitch_deg"]):
            name = f"tail_{i + 1}"
            P[name] = P[self.tail_chain[-1]] + (L["tail"] / n) * _dir(float(p), back=True)
            bones.append((self.tail_chain[-1], name, "tail"))
            self.tail_chain.append(name)
        # loin, chest, neck and skull, towards the head
        self.chain = ["ls"]
        self.region_of: dict[str, str] = {}
        for r in s["spine"]:
            step = L[r["id"]] / int(r["count"])
            for i, p in enumerate(r["pitch_deg"]):
                name = f"{r['id'][:3]}_{i + 1}"
                P[name] = P[self.chain[-1]] + step * _dir(float(p))
                bones.append((self.chain[-1], name, r["id"]))
                self.region_of[name] = r["id"]
                self.chain.append(name)
        self.skull_base = self.chain[-1]
        P["snout"] = P[self.skull_base] + L["skull"] * _dir(float(s["head"]["pitch_deg"]))
        bones.append((self.skull_base, "snout", "skull"))
        self.chain.append("snout")
        # shoulder blades ride on one chest joint
        self.scap_on = f"tho_{int(lay['scapula_on_thoracic'])}"
        sc = _dir(float(lay["scapula_pitch_deg"]))
        self.riders: list[str] = []
        for side, sg in SIDES:
            top = P[self.scap_on] + np.array([0.0, -float(lay["scapula_drop_mm"]),
                                              sg * float(lay["scapula_half_width_mm"])])
            sh = top + L["scapula"] * sc
            out = sg * float(lay["shoulder_half_width_mm"]) - sh[2]
            sh = top + L["scapula"] * (sh - top + np.array([0.0, 0.0, out])) / np.linalg.norm(
                sh - top + np.array([0.0, 0.0, out]))
            P["scap_" + side], P["shoulder_" + side] = top, sh
            bones.append(("scap_" + side, "shoulder_" + side, "scapula"))
            bones += [("shoulder_" + side, "elbow_" + side, "humerus"), ("elbow_" + side, "wrist_" + side, "forearm"),
                      ("wrist_" + side, "paw_" + side, "forepaw"), ("hip_" + side, "knee_" + side, "femur"),
                      ("knee_" + side, "ankle_" + side, "tibia"), ("ankle_" + side, "mtp_" + side, "foot"),
                      ("mtp_" + side, "toe_" + side, "toes")]
            self.riders += [k + side for k in ("scap_", "shoulder_", "elbow_", "wrist_", "paw_")]
        self._P0 = P
        self.bones = bones
        self.moving = [k + side for side, _ in SIDES
                       for k in ("elbow_", "wrist_", "paw_", "knee_", "ankle_", "mtp_", "toe_")]
        self.names = list(P) + self.moving
        self.origin_x = 0.5 * float(P["shoulder_R"][0])

    # ---- one leg ----------------------------------------------------------- #
    def _cycle(self, p: float, amp: float, pitch: dict):
        """Where the foot is in its step: ``(x from its resting place, lift,
        wanted pitch of the foot, share of the swing done)``."""
        g = self.spec["gait"]
        duty = float(g["duty"])
        travel = amp * float(g["stride_mm"]) * duty
        c = (p / (2.0 * math.pi)) % 1.0
        st, td, to = float(pitch["stand"]), float(pitch["touch_down"]), float(pitch["toe_off"])
        if c < duty:
            q = c / duty
            return travel * (0.5 - q), 0.0, st + amp * ((td - st) + (to - td) * q * q), 0.0
        u = (c - duty) / (1.0 - duty)
        e = u * u * (3.0 - 2.0 * u)
        return (travel * (e - 0.5), amp * float(g["clearance_mm"]) * math.sin(math.pi * u),
                st + amp * ((to - st) * (1.0 - e) + (td - st) * e), u)

    def _over(self, ang: dict) -> float:
        """How far a set of joint angles is outside the printed ranges (degrees)."""
        bad = 0.0
        for k, v in ang.items():
            lo, hi = self.limit[k[:-2]]
            if self.enforced[k[:-2]]:
                bad += max(0.0, lo - v) + max(0.0, v - hi)
        return bad

    def _hind(self, side: str, sg: float, p: float, amp: float):
        g, lay, L = self.spec["gait"], self.spec["layout"], self.len
        hip = self._P0["hip_" + side]
        x, lift, want, u = self._cycle(p, amp, g["foot_pitch_deg"])
        mtp = np.array([float(g["hind_neutral_x_mm"]) + x, lift, sg * float(lay["hind_track_half_width_mm"])])
        zero = float(lay["pelvis_pitch_deg"]) - 90.0
        best = None
        for k in range(0, 45):
            for sgn in ((1,) if k == 0 else (1, -1)):
                a = want + sgn * k * 2.0
                if not 0.0 <= a <= 88.0:
                    continue
                ankle = mtp + L["foot"] * _dir(a, back=True)
                knee, ankle2, ok = _two_bones(hip, ankle, L["femur"], L["tibia"], np.array([1.0, 0.2, 0.35 * sg]))
                f = knee - hip
                ang = {"hip_flex_" + side: (math.degrees(math.atan2(f[1], f[0])) - zero + 180.0) % 360.0 - 180.0,
                       "hip_out_" + side: math.degrees(math.asin(max(-1.0, min(1.0, sg * f[2] / L["femur"])))),
                       "knee_" + side: _between(f, ankle2 - knee),
                       "ankle_" + side: _between(knee - ankle2, mtp - ankle) - 90.0}
                bad = self._over(ang) + (0.0 if ok else 100.0)
                if best is None or bad < best[0] - 1e-9:
                    best = (bad, knee, ankle2, ang, ok, a)
                if bad == 0.0:
                    break
            if best is not None and best[0] == 0.0:
                break
        _, knee, ankle, ang, ok, a = best
        mtp = ankle - L["foot"] * _dir(a, back=True)
        toe = mtp + L["toes"] * _dir(-amp * 30.0 * math.sin(math.pi * u))
        return {"knee_" + side: knee, "ankle_" + side: ankle, "mtp_" + side: mtp, "toe_" + side: toe}, ang, ok

    def _fore(self, side: str, sg: float, p: float, amp: float):
        g, lay, L = self.spec["gait"], self.spec["layout"], self.len
        sh, top = self._P0["shoulder_" + side], self._P0["scap_" + side]
        x, lift, want, _ = self._cycle(p, amp, g["paw_pitch_deg"])
        tip = np.array([sh[0] + float(g["fore_neutral_x_mm"]) + x, lift, sg * float(lay["fore_track_half_width_mm"])])
        best = None
        for k in range(0, 45):
            for sgn in ((1,) if k == 0 else (1, -1)):
                a = want + sgn * k * 2.0
                if not 0.0 <= a <= 88.0:
                    continue
                wrist = tip + L["forepaw"] * _dir(a, back=True)
                elbow, wrist2, ok = _two_bones(sh, wrist, L["humerus"], L["forearm"], np.array([-1.0, 0.1, 0.3 * sg]))
                fa, paw = wrist2 - elbow, tip - wrist
                bent = math.degrees(math.atan2(paw[1], paw[0]) - math.atan2(fa[1], fa[0]))
                ang = {"shoulder_" + side: _between(top - sh, elbow - sh) - 90.0,
                       "elbow_" + side: _between(sh - elbow, wrist2 - elbow),
                       "wrist_" + side: (bent + 180.0) % 360.0 - 180.0}
                bad = self._over(ang) + (0.0 if ok else 100.0)
                if best is None or bad < best[0] - 1e-9:
                    best = (bad, elbow, wrist2, ang, ok, a)
                if bad == 0.0:
                    break
            if best is not None and best[0] == 0.0:
                break
        _, elbow, wrist, ang, ok, a = best
        return {"elbow_" + side: elbow, "wrist_" + side: wrist,
                "paw_" + side: wrist - L["forepaw"] * _dir(a, back=True)}, ang, ok

    # ---- the whole animal -------------------------------------------------- #
    def spine_yaw(self, bend: float) -> tuple[float, float]:
        """Sideways bend per joint (radians) for a turn: ``(spine, tail)``.
        ``bend`` is the same number the schematic body takes."""
        lim = min(min(abs(r["bend_limit_deg"][0]), abs(r["bend_limit_deg"][1])) for r in self.spec["spine"])
        tl = min(abs(v) for v in self.spec["tail"]["limit_deg"])
        per = 1.9 * bend / (len(self.chain) - 1)
        return (max(-math.radians(lim), min(math.radians(lim), per)),
                max(-math.radians(tl), min(math.radians(tl), -0.35 * bend)))

    def pose(self, phase: float = 0.0, amp: float = 0.0, bend: float = 0.0, sniff: bool = False) -> dict:
        """Every joint in space. ``phase`` is the stepping cycle in radians,
        ``amp`` 0 (standing) to 1 (a full walk), ``bend`` the turn of the
        spine. Returns ``{"points", "angles", "reached"}``."""
        amp = max(0.0, min(1.0, float(amp)))
        P = {k: v.copy() for k, v in self._P0.items()}
        angles: dict[str, float] = {}
        reached = True
        for side, sg in SIDES:
            for limb, solve in (("fore", self._fore), ("hind", self._hind)):
                pts, ang, ok = solve(side, sg, phase + OFFSET[(limb, side)], amp)
                P.update(pts)
                angles.update(ang)
                reached = reached and ok
        if sniff:
            P["snout"] = P[self.skull_base] + self.len["skull"] * _dir(
                float(self.spec["head"]["pitch_deg"]) + float(self.spec["head"].get("sniff_pitch_deg", 0.0)))
        if bend:
            O = {k: v.copy() for k, v in P.items()}
            per, tper = self.spine_yaw(bend)
            for chain, step, riders, on in ((self.chain, per, self.riders, self.chain.index(self.scap_on)),
                                            (self.tail_chain, tper, (), 0)):
                for i in range(1, len(chain)):
                    P[chain[i]] = P[chain[i - 1]] + _yaw(O[chain[i]] - O[chain[i - 1]], i * step)
                for k in riders:
                    P[k] = P[chain[on]] + _yaw(O[k] - O[chain[on]], on * step)
        return {"points": {k: P[k] for k in self.names}, "angles": angles, "reached": reached}

    def gait_check(self, amp: float = 1.0, n: int = 48) -> list[dict]:
        """How much of each printed joint range one stepping cycle uses."""
        seen: dict[str, list[float]] = {}
        reached = True
        for i in range(n):
            got = self.pose(2.0 * math.pi * i / n, amp)
            reached = reached and got["reached"]
            for k, v in got["angles"].items():
                seen.setdefault(k[:-2], []).append(v)
        rows = []
        for j in self.spec["joints"]:
            v = seen[j["id"]]
            lo, hi = float(j["min_deg"]), float(j["max_deg"])
            rows.append({"joint": j["id"], "used_min": min(v), "used_max": max(v), "min": lo, "max": hi,
                         "inside": min(v) >= lo - 1e-6 and max(v) <= hi + 1e-6, "basis": j["basis"],
                         "assumed": j["assumed"], "enforced": bool(j.get("enforced", True))})
        rows.append({"joint": "reach", "inside": reached})
        return rows

    # ---- for the brain map page -------------------------------------------- #
    def page_payload(self, phases: int = 24, amps: tuple = (0.0, 0.5, 1.0)) -> dict:
        """The skeleton as the brain map page draws it: joint positions in
        tenths of a millimetre, the legs as a table over the stepping cycle,
        and the chains the page bends when the animal turns."""
        names = self.names
        ix = {k: i for i, k in enumerate(names)}

        def q(v):
            return [int(round(float(v[0] - self.origin_x) * 10)), int(round(float(v[1]) * 10)),
                    int(round(float(v[2]) * 10))]

        rest = self.pose()["points"]
        frames = []
        for a in amps:
            row = []
            for k in range(phases):
                pts = self.pose(2.0 * math.pi * k / phases, a)["points"]
                row.append([c for m in self.moving for c in q(pts[m])])
            frames.append(row)
        rad = self.spec["radius_mm"]
        n_tail = len(self.tail_chain) - 1
        bones = []
        for a, b, kind in self.bones:
            r0, r1 = rad[kind]
            if kind == "tail":
                i = self.tail_chain.index(a)
                r0, r1 = (rad[kind][0] + (rad[kind][1] - rad[kind][0]) * j / n_tail for j in (i, i + 1))
            bones.append([ix[a], ix[b], int(round(r0 * 10)), int(round(r1 * 10))])
        balls = [[ix[k + side], int(round(r * 10))] for side, _ in SIDES
                 for k, r in (("shoulder_", 1.0), ("elbow_", 0.9), ("wrist_", 0.8), ("hip_", 1.2), ("knee_", 1.1),
                              ("ankle_", 0.9), ("mtp_", 0.7))]
        lens = self.spec["lengths"]
        count = {b: sum(1 for v in lens.values() if v["basis"] == b) for b in BASES}
        per, tper = self.spine_yaw(1e9)
        return {"name": self.spec["name"], "title": self.spec.get("title", self.spec["name"]),
                "unit_mm": 0.1, "pts": names, "rest": [q(rest[k]) for k in names], "bones": bones, "balls": balls,
                "moving": [ix[k] for k in self.moving], "frames": frames,
                "chain": [ix[k] for k in self.chain], "tail": [ix[k] for k in self.tail_chain],
                "ride": [ix[k] for k in self.riders], "ride_on": self.chain.index(self.scap_on),
                "yaw_lim": round(abs(per), 5), "tail_yaw_lim": round(abs(tper), 5),
                "sniff": round(math.radians(float(self.spec["head"].get("sniff_pitch_deg", 0.0))), 5),
                "skull": [ix[self.skull_base], ix["snout"]],
                "note": f"{count['published']} of {len(lens)} lengths are printed in a paper, {count['derived']} "
                        f"worked out from printed numbers, and {count['layout']} (the spine) are our own split "
                        "of the printed body length. Posture and stepping are our own."}


def describe(spec: dict) -> str:
    """A plain-text account of a skeleton: every length with where it comes
    from, and how much of each joint range a walk uses."""
    sk = Skeleton(spec)
    out = [f"{spec.get('title', spec['name'])} (draft {spec.get('draft', '?')})", spec.get("subject", ""), ""]
    out.append(f"{'bone or region':<12}{'mm':>7}  basis      source")
    for k, v in spec["lengths"].items():
        out.append(f"{k:<12}{float(v['mm']):>7.2f}  {v['basis']:<10} {v['source']}")
    out += ["", "A full walk, against the printed range of each joint (degrees):",
            f"{'joint':<10}{'used':>16}{'printed':>18}  inside  note"]
    for r in sk.gait_check():
        if r["joint"] == "reach":
            out.append("every foot reaches its place on the ground" if r["inside"]
                       else "a leg is too short or too long to reach its place on the ground")
            continue
        used = f"{r['used_min']:.0f} to {r['used_max']:.0f}"
        printed = f"{r['min']:g} to {r['max']:g}"
        out.append(f"{r['joint']:<10}{used:>16}{printed:>18}  {'yes' if r['inside'] else 'NO':<6}  {r['basis']}")
    out += ["", "What had to be assumed to read each printed range:"]
    out += [f"  {j['id']}: {j['assumed']}" for j in spec["joints"]]
    if spec.get("left_out"):
        out += ["", "Left out:"] + [f"  - {x}" for x in spec["left_out"]]
    return "\n".join(out)


def plot(spec: dict, path: str) -> str:
    """Save a picture of the skeleton: from the side through one step, from
    above, and the share of each printed joint range a walk uses."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sk = Skeleton(spec)
    colour = {"published": "#1f6fb4", "derived": "#c77700", "layout": "#8a8f98"}
    basis = {k: v["basis"] for k, v in spec["lengths"].items()}
    fig = plt.figure(figsize=(13, 8.2))
    grid = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 0.95])

    def draw(ax, pts, a, b, flat=False):
        for u, v, kind in sk.bones:
            ax.plot([pts[u][a], pts[v][a]], [pts[u][b], pts[v][b]], color=colour[basis[kind]],
                    lw=3.2 if kind == "skull" else 1.8, solid_capstyle="round")
        ax.set_aspect("equal")
        ax.set_xlim(-95, 95)
        if not flat:
            ax.axhline(0, color="#b9bec6", lw=0.8)
            ax.set_ylim(-4, 40)
        ax.tick_params(labelsize=7)

    for i in range(4):
        ax = fig.add_subplot(grid[0, i])
        draw(ax, sk.pose(2 * math.pi * i / 4, 1.0)["points"], 0, 1)
        ax.set_title(f"walking, step {i + 1} of 4", fontsize=9)
        if i == 0:
            ax.set_ylabel("mm")
    ax = fig.add_subplot(grid[1, 0:2])
    draw(ax, sk.pose()["points"], 0, 1)
    ax.set_title("standing, from the side", fontsize=9)
    ax.set_ylabel("mm")
    for b, c in colour.items():
        ax.plot([], [], color=c, lw=2.5, label={"published": "length printed in a paper",
                                               "derived": "worked out from printed numbers",
                                               "layout": "our own split (no printed length)"}[b])
    ax.legend(fontsize=7, loc="upper right", frameon=False)
    ax = fig.add_subplot(grid[1, 2:4])
    draw(ax, sk.pose(0.0, 1.0, 0.35)["points"], 0, 2, flat=True)
    ax.set_ylim(-30, 45)
    ax.set_title("turning, from above", fontsize=9)
    ax = fig.add_subplot(grid[2, :])
    rows = [r for r in sk.gait_check() if r["joint"] != "reach"]
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([r["min"], r["max"]], [y, y], color="#d3d7dd", lw=9, solid_capstyle="butt")
        ax.plot([r["used_min"], r["used_max"]], [y, y], color="#1f6fb4" if r["inside"] else "#c0392b", lw=5,
                solid_capstyle="butt")
        ax.text(r["max"] + 3, y, r["basis"][:70], fontsize=7, va="center", color="#4a4f57")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["joint"] for r in rows][::-1], fontsize=8)
    ax.set_xlabel("degrees (grey: range printed in the papers; blue: what a full walk uses)", fontsize=8)
    ax.set_xlim(-60, 260)
    ax.tick_params(labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.suptitle(f"{spec.get('title', spec['name'])}: bone lengths from the literature, posture and stepping our own",
                 fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
