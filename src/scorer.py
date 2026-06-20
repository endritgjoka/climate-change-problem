"""Validate a submission against an instance and compute its score.

Mirrors the rules in the README so we can self-check solutions before
uploading them to the official validator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .model import Instance, CAPACITY


@dataclass
class ScoreReport:
    valid: bool
    errors: List[str]
    coverage: float = 0.0
    efficiency: float = 0.0
    score: float = 0.0
    cleaned_length: float = 0.0
    water_wasted: float = 0.0
    l_max: float = 0.0
    w_max: float = 0.0
    mandatory_total: int = 0
    mandatory_cleaned: int = 0
    optional_total: int = 0
    optional_cleaned: int = 0

    def summary(self) -> str:
        if not self.valid:
            return "INVALID:\n  " + "\n  ".join(self.errors)
        return (
            f"VALID  score={self.score:.4f}  "
            f"coverage={self.coverage:.4f}  efficiency={self.efficiency:.4f}\n"
            f"  cleaned length = {self.cleaned_length:.0f} / {self.l_max:.0f} m\n"
            f"  water wasted   = {self.water_wasted:.2f} / {self.w_max:.2f} L\n"
            f"  mandatory cleaned = {self.mandatory_cleaned}/{self.mandatory_total}"
            f"   optional cleaned = {self.optional_cleaned}/{self.optional_total}"
        )


def _build_edge_map(inst: Instance):
    """(u, v) -> street index, respecting direction."""
    em = {}
    for s in inst.streets:
        em[(s.a, s.b)] = s.idx
        if not s.one_way:
            em[(s.b, s.a)] = s.idx
    return em


def score_submission(inst: Instance, sub_path: str) -> ScoreReport:
    with open(sub_path) as f:
        tokens = f.read().split("\n")
    # flatten into a line iterator that tolerates trailing blank lines
    lines = [ln for ln in tokens]

    errors: List[str] = []
    edge_map = _build_edge_map(inst)

    idx = 0

    def read_line():
        nonlocal idx
        ln = lines[idx]
        idx += 1
        return ln

    try:
        c = int(read_line().strip())
    except Exception:
        return ScoreReport(False, ["could not read vehicle count C on line 1"])

    if c != inst.c:
        errors.append(f"vehicle count {c} != instance C {inst.c}")

    cleaned_by = {}          # street idx -> list of vehicle capacities that cleaned it
    cleaned_set = set()

    for vi in range(c):
        try:
            n = int(read_line().strip())
            node_line = read_line().strip()
            clean_line = read_line().strip()
        except Exception:
            errors.append(f"vehicle {vi}: truncated route block")
            break

        nodes = [int(x) for x in node_line.split()] if node_line else []
        cleans = [int(x) for x in clean_line.split()] if clean_line else []

        # Per-vehicle count is the edge count = len(nodes) - 1 (matches the
        # official validator, not the misleading README prose).
        if len(nodes) - 1 != n:
            errors.append(
                f"vehicle {vi}: declared n={n} but route has {len(nodes)} nodes "
                f"(expected {len(nodes) - 1})"
            )

        if not nodes:
            errors.append(f"vehicle {vi}: empty route")
            continue

        if nodes[0] != inst.depot or nodes[-1] != inst.depot:
            errors.append(
                f"vehicle {vi}: route must start and end at depot {inst.depot} "
                f"(got {nodes[0]}..{nodes[-1]})"
            )

        # validate connectivity + accumulate time + record traversed streets
        traversed = set()
        time_used = 0
        ok_path = True
        for a, b in zip(nodes, nodes[1:]):
            sid = edge_map.get((a, b))
            if sid is None:
                errors.append(f"vehicle {vi}: no valid street {a}->{b}")
                ok_path = False
                break
            traversed.add(sid)
            time_used += inst.streets[sid].time

        if ok_path and time_used > inst.t:
            errors.append(
                f"vehicle {vi}: time {time_used} exceeds limit {inst.t}"
            )
            ok_path = False

        if not ok_path:
            continue

        vtype = inst.vehicles[vi] if vi < len(inst.vehicles) else None
        cap = CAPACITY.get(vtype, 0)
        for sid in cleans:
            s = inst.streets[sid]
            if not s.cleanable:
                errors.append(f"vehicle {vi}: street {sid} is a connector, not cleanable")
                continue
            if sid not in traversed:
                errors.append(f"vehicle {vi}: cleans street {sid} it never traverses")
                continue
            if cap < s.req:
                errors.append(
                    f"vehicle {vi} ({vtype}) cap {cap} < street {sid} req {s.req}"
                )
                continue
            cleaned_by.setdefault(sid, []).append(cap)
            cleaned_set.add(sid)

    # mandatory coverage
    mand = [s.idx for s in inst.streets if s.mandatory]
    opt = [s.idx for s in inst.streets if s.category == "O"]
    missing = [m for m in mand if m not in cleaned_set]
    if missing:
        errors.append(f"{len(missing)} mandatory street(s) uncleaned: {missing[:10]}")

    l_max = inst.l_max() or 1.0
    w_max = inst.w_max() or 1.0

    cleaned_length = sum(inst.streets[sid].length for sid in cleaned_set)
    # waste counts every cleaning event; length counts once
    water_wasted = 0.0
    for sid, caps in cleaned_by.items():
        s = inst.streets[sid]
        for cap in caps:
            water_wasted += (cap - s.req) * s.length_km

    coverage = cleaned_length / l_max
    efficiency = 1 - water_wasted / w_max
    score = inst.alpha * coverage + (1 - inst.alpha) * efficiency

    valid = len(errors) == 0
    return ScoreReport(
        valid=valid,
        errors=errors,
        coverage=coverage,
        efficiency=efficiency,
        score=score if valid else 0.0,
        cleaned_length=cleaned_length,
        water_wasted=water_wasted,
        l_max=l_max,
        w_max=w_max,
        mandatory_total=len(mand),
        mandatory_cleaned=sum(1 for m in mand if m in cleaned_set),
        optional_total=len(opt),
        optional_cleaned=sum(1 for o in opt if o in cleaned_set),
    )
