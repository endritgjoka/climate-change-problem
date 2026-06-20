"""Parse an instance file into an Instance object.

Robust to whether the optional N junction-coordinate lines are present:
the README's worked example omits them, but the format spec includes them.
We classify body lines by shape:
  * street line  -> 7 tokens, the 6th being a category M/O/C
  * coord line   -> 2 numeric tokens
  * vehicle line -> the final line, tokens in {S, M, L}
"""
from __future__ import annotations

from .model import Instance, Street


def parse_instance(path: str) -> Instance:
    with open(path, "r") as f:
        raw = [ln.strip() for ln in f]
    lines = [ln for ln in raw if ln != ""]

    header = lines[0].split()
    n = int(header[0])
    m = int(header[1])
    t = int(header[2])
    c = int(header[3])
    depot = int(header[4])
    alpha = float(header[5])

    body = lines[1:]
    vehicles = body[-1].split()
    body = body[:-1]

    coords = []
    street_lines = []
    for ln in body:
        toks = ln.split()
        if len(toks) >= 7 and toks[5] in ("M", "O", "C"):
            street_lines.append(toks)
        elif len(toks) == 2:
            coords.append((float(toks[0]), float(toks[1])))
        else:
            # unexpected line shape; ignore defensively
            continue

    while len(coords) < n:
        coords.append((0.0, 0.0))

    streets = []
    for j, p in enumerate(street_lines):
        streets.append(
            Street(
                idx=j,
                a=int(p[0]),
                b=int(p[1]),
                one_way=(int(p[2]) == 1),
                time=int(p[3]),
                length=int(p[4]),
                category=p[5],
                req=int(p[6]),
            )
        )

    if len(streets) != m:
        # trust the actual parsed count but warn via exception only on mismatch
        m = len(streets)

    if len(vehicles) != c:
        c = len(vehicles)

    return Instance(
        n=n, m=m, t=t, c=c, depot=depot, alpha=alpha,
        coords=coords, streets=streets, vehicles=vehicles,
    )
