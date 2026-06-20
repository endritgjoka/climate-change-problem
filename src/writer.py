"""Write a Solution to the submission file format."""
from __future__ import annotations

from .model import Solution


def write_solution(sol: Solution, path: str) -> None:
    lines = [str(len(sol.routes))]
    for r in sol.routes:
        # The validator's per-vehicle count is the number of edges/transitions
        # in the route, i.e. (#junctions - 1). Despite the README prose calling
        # it "number of junctions", the README's own examples and the official
        # validator both use len(nodes) - 1.
        lines.append(str(max(len(r.nodes) - 1, 0)))
        lines.append(" ".join(str(x) for x in r.nodes))
        lines.append(" ".join(str(x) for x in r.cleaned))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
