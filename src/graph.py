"""Directed graph built from streets, plus Dijkstra helpers.

Edges respect street direction:
  - two-way street j: A<->B (two directed edges)
  - one-way street j: A->B only
Each directed edge carries (to_node, traversal_time, street_idx).
"""
from __future__ import annotations

import heapq
from typing import List, Tuple

from .model import Instance

INF = float("inf")


class Graph:
    def __init__(self, inst: Instance):
        self.n = inst.n
        # adjacency: adj[u] = list of (v, time, street_idx)
        self.adj: List[List[Tuple[int, int, int]]] = [[] for _ in range(inst.n)]
        # reverse adjacency for "distance to depot" computation
        self.radj: List[List[Tuple[int, int, int]]] = [[] for _ in range(inst.n)]
        for s in inst.streets:
            self.adj[s.a].append((s.b, s.time, s.idx))
            self.radj[s.b].append((s.a, s.time, s.idx))
            if not s.one_way:
                self.adj[s.b].append((s.a, s.time, s.idx))
                self.radj[s.a].append((s.b, s.time, s.idx))

    def dijkstra(self, src: int, cutoff: float = INF, max_pops: int = 0):
        """Forward shortest paths from src (times). Stops exploring beyond cutoff.

        If max_pops > 0, exploration halts after that many nodes are settled —
        a locality bound that keeps per-step cost constant on huge graphs.

        Returns (dist, prev) where prev[v] is the predecessor node on the
        shortest path from src to v (-1 if none / unreachable).
        """
        dist = [INF] * self.n
        prev = [-1] * self.n
        settled = []
        dist[src] = 0
        pq = [(0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if d > cutoff:
                continue
            settled.append(u)
            if max_pops and len(settled) > max_pops:
                break
            for v, w, _sid in self.adj[u]:
                nd = d + w
                if nd < dist[v] and nd <= cutoff:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        return dist, prev, settled

    def dijkstra_to(self, target: int, cutoff: float = INF):
        """Shortest path *to* target from every node (runs on reverse graph).

        Returns (dist, succ) where dist[u] = shortest time u->target and
        succ[u] = next node after u on the shortest path toward target.
        """
        dist = [INF] * self.n
        succ = [-1] * self.n
        dist[target] = 0
        pq = [(0, target)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if d > cutoff:
                continue
            for v, w, _sid in self.radj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    succ[v] = u
                    heapq.heappush(pq, (nd, v))
        return dist, succ


def reconstruct_forward(prev: List[int], src: int, dst: int) -> List[int]:
    """Path src..dst (inclusive) using forward prev array. Empty if unreachable."""
    if dst == src:
        return [src]
    path = []
    cur = dst
    while cur != -1 and cur != src:
        path.append(cur)
        cur = prev[cur]
    if cur != src:
        return []  # unreachable
    path.append(src)
    path.reverse()
    return path


def reconstruct_toward(succ: List[int], src: int, dst: int) -> List[int]:
    """Path src..dst (inclusive) using succ array from dijkstra_to(dst)."""
    if src == dst:
        return [src]
    path = [src]
    cur = src
    while cur != dst:
        nxt = succ[cur]
        if nxt == -1:
            return []  # unreachable
        path.append(nxt)
        cur = nxt
    return path
