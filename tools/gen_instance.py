#!/usr/bin/env python3
"""Generate a random but solvable test instance.

Builds a connected graph (spanning tree + extra edges, mostly two-way so the
depot can reach and return from everywhere), then adds categories/requirements.

Usage: python tools/gen_instance.py N M T C alpha seed > input/randNNN.txt
"""
import random
import sys


def main():
    n = int(sys.argv[1])
    m = int(sys.argv[2])
    t = int(sys.argv[3])
    c = int(sys.argv[4])
    alpha = float(sys.argv[5])
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    rng = random.Random(seed)

    depot = 0
    edges = []
    used = set()

    def add(a, b):
        key = (min(a, b), max(a, b))
        if a == b or key in used:
            return False
        used.add(key)
        edges.append((a, b))
        return True

    # spanning tree for connectivity
    nodes = list(range(n))
    rng.shuffle(nodes)
    for i in range(1, n):
        a = nodes[i]
        b = nodes[rng.randint(0, i - 1)]
        add(a, b)

    # extra random edges up to m
    tries = 0
    while len(edges) < m and tries < m * 20:
        a = rng.randint(0, n - 1)
        b = rng.randint(0, n - 1)
        add(a, b)
        tries += 1

    out = []
    out.append(f"{n} {len(edges)} {t} {c} {depot} {alpha}")
    for _ in range(n):
        out.append(f"{rng.uniform(0,100):.4f} {rng.uniform(0,100):.4f}")

    reqs = [10, 20, 30]
    for (a, b) in edges:
        # keep tree edges mostly two-way for return feasibility
        direction = 1 if rng.random() < 0.15 else 2
        time = rng.randint(5, 60)
        roll = rng.random()
        if roll < 0.25:
            cat, req, length = "M", rng.choice(reqs), rng.randint(50, 400)
        elif roll < 0.65:
            cat, req, length = "O", rng.choice(reqs), rng.randint(50, 400)
        else:
            cat, req, length = "C", 0, 0
        out.append(f"{a} {b} {direction} {time} {length} {cat} {req}")

    vtypes = [rng.choice("SML") for _ in range(c)]
    out.append(" ".join(vtypes))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
