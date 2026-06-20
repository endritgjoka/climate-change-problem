"""Data model for the Street Cleaning optimisation problem.

A city is a directed graph. Each street is an edge with a category
(Mandatory / Optional / Connector) and a cleaning requirement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

# Vehicle capacities (litres per km) keyed by type letter.
CAPACITY = {"S": 10, "M": 20, "L": 30}

# Cleaning-requirement levels.
LIGHT, MEDIUM, HEAVY = 10, 20, 30


@dataclass
class Street:
    idx: int            # street index (0..M-1)
    a: int              # endpoint A
    b: int              # endpoint B
    one_way: bool       # True => only A->B, False => both directions
    time: int           # traversal time in seconds
    length: int         # length in metres (contributes to coverage)
    category: str       # 'M', 'O' or 'C'
    req: int            # cleaning requirement in L/km (0 for connectors)

    @property
    def cleanable(self) -> bool:
        return self.category in ("M", "O")

    @property
    def mandatory(self) -> bool:
        return self.category == "M"

    @property
    def length_km(self) -> float:
        return self.length / 1000.0


@dataclass
class Instance:
    n: int                      # number of junctions
    m: int                      # number of streets
    t: int                      # time limit (seconds) per vehicle
    c: int                      # number of vehicles
    depot: int                  # depot junction index
    alpha: float                # coverage vs efficiency weight
    coords: List[tuple]         # junction coordinates
    streets: List[Street]
    vehicles: List[str]         # list of 'S'/'M'/'L', length c

    # ---- normalisation constants (cached) ----
    def l_max(self) -> float:
        """Total length of all mandatory + optional streets (metres)."""
        return sum(s.length for s in self.streets if s.cleanable)

    def w_max(self) -> float:
        """Worst-case wasted water: every cleanable street cleaned by Large."""
        return sum((30 - s.req) * s.length_km for s in self.streets if s.cleanable)


@dataclass
class Route:
    """A single vehicle's plan."""
    vehicle_idx: int
    vtype: str
    nodes: List[int] = field(default_factory=list)   # junction sequence, depot..depot
    cleaned: List[int] = field(default_factory=list)  # street indices cleaned
    time_used: int = 0


@dataclass
class Solution:
    routes: List[Route]
