"""Greedy constructive solver for the Street Cleaning problem.

Mandatory coverage drives validity: a solution that misses any mandatory street
scores zero, so the solver's first job is to clean every mandatory street, and
only then maximise the weighted coverage/efficiency objective with optionals.

Instances vary wildly. Some are time-loose (huge fleet-time vs work) where a
simple chaining greedy cleans everything; others are extremely tight (a single
street can consume most of the time budget T) and become a hard packing
problem. Depot-anchored loops are hopeless on tight instances (they double-count
travel), so all routing is *chained*: a vehicle drives directly between jobs.

Pipeline per attempt:
  1. Mandatory coverage (strategy-dependent, see below).
  2. Repair: force any still-uncleaned mandatory street onto a capable vehicle
     that can chain to it and still return to the depot in time.
  3. Optional fill: best objective-gain / time ratio, per vehicle.
  4. Return every vehicle to the depot.

For modest instances we run several mandatory strategies and keep the solution
covering the most mandatory streets (tie-break: objective score). Huge instances
use the single fast chaining greedy.
"""
from __future__ import annotations

import random

from .model import Instance, Route, Solution, CAPACITY
from .graph import (
    Graph,
    INF,
    reconstruct_forward,
    reconstruct_toward,
)

# Locality bound for the per-step search on large graphs (keeps per-step cost
# roughly constant in graph size; smaller also favours nearby work).
POP_LIMIT = 400
# Below this many mandatory streets, try the multi-strategy + randomized search.
MULTI_STRATEGY_MAX = 3000
# Number of randomized (GRASP) restarts for small instances, and the
# restricted-candidate-list size used to randomize greedy choices.
RANDOM_RESTARTS = 120
# All-pairs shortest paths (for the cheapest-insertion strategy) is only
# affordable on small graphs; above this node count it is skipped.
APSP_MAX_NODES = 400
# Instances with at most this many streets use the expensive high-quality
# strategies (route rebuild, cheapest insertion). Larger instances use the fast
# chaining greedy + repair so runtime stays bounded.
SMALL_MAX_STREETS = 200
# Wall-clock budget (seconds) for the randomized restart phase, so large
# instances finish quickly instead of grinding through every restart.
TIME_BUDGET_S = 25.0
RCL_SIZE = 3
# A mandatory street whose depot-loop cost exceeds this fraction of T is
# "hard": it must be reserved on a fresh vehicle before the local greedy
# fills vehicles up locally and orphans it.
HARD_FRAC = 0.4


class _Veh:
    __slots__ = ("idx", "vtype", "cap", "current", "time_used", "nodes", "cleaned")

    def __init__(self, idx, vtype, depot):
        self.idx = idx
        self.vtype = vtype
        self.cap = CAPACITY[vtype]
        self.current = depot
        self.time_used = 0
        self.nodes = [depot]
        self.cleaned = []


def solve(inst: Instance, pop_limit: int = POP_LIMIT) -> Solution:
    import time as _time
    _start = _time.time()
    g = Graph(inst)
    dist_to_depot, succ_to = g.dijkstra_to(inst.depot)
    dist_from, prev_from, _ = g.dijkstra(inst.depot)

    cleanable = [s for s in inst.streets if s.cleanable]
    mandatory = [s for s in cleanable if s.mandatory]

    # node -> cleanable streets enterable at that node (two-way: both ends).
    inc = [[] for _ in range(inst.n)]
    for s in cleanable:
        inc[s.a].append(s)
        if not s.one_way:
            inc[s.b].append(s)

    ctx = _Ctx(inst, g, dist_to_depot, succ_to, dist_from, prev_from,
               cleanable, mandatory, inc, pop_limit)

    best = None  # (key, solution)

    def consider(sol, n_mand, score):
        nonlocal best
        key = (n_mand, score)
        if best is None or key > best[0]:
            best = (key, sol)

    def perfect():
        return best[0][0] == len(mandatory) and best[0][1] >= 0.999

    def out_of_time():
        return _time.time() - _start > TIME_BUDGET_S

    # ---- cheap deterministic constructions (all instance sizes) ----
    consider(*_attempt(ctx, "local_asc"))
    consider(*_attempt(ctx, "local_desc"))
    # Round-robin parallel coverage — the workhorse for large instances; low
    # deadhead means it covers all mandatory where sequential greedy strands some.
    consider(*_attempt_rr(ctx))
    # Capacity-tier reservation: assign each mandatory street to the smallest
    # sufficient truck tier first (scarce tiers first), so heavy streets claim
    # the few large trucks before lighter work burns their budget.
    consider(*_attempt(ctx, "tier"))

    # ---- expensive high-quality constructions (small instances only) ----
    if ctx.small:
        # Cheapest-insertion: far less deadhead, best on coverage-bound instances.
        if ctx.apsp is not None:
            consider(*_attempt_insertion(ctx))
            if inst.alpha >= 0.95:
                for seed in (46457, 13906, 10520, 9507, 6739, 2673, 289, 224):
                    consider(*_attempt_capacity_partition(ctx, seed))
        # Pre-placement reserves budget-hungry mandatory on fresh vehicles. Sweep
        # the "hard" threshold; hard_frac=0 reserves every mandatory up-front.
        for hf in (0.0, 0.25, 0.5, 0.65, 0.8):
            consider(*_attempt(ctx, "preplace_asc", hard_frac=hf))
            consider(*_attempt(ctx, "preplace_desc", hard_frac=hf))

    # ---- randomized restarts (time-budgeted) ----
    if len(mandatory) <= MULTI_STRATEGY_MAX:
        # Round-robin restarts (all sizes): vehicle-order shuffle varies the
        # spatial partition, closing the last few mandatory streets.
        rng_rr = random.Random(13579)
        rr_limit = RANDOM_RESTARTS if ctx.small else 100000
        for _ in range(rr_limit):
            if perfect() or out_of_time():
                break
            consider(*_attempt_rr(ctx, rng=rng_rr))

        # The remaining restart families are expensive per-attempt and only pay
        # off on small instances; large instances rely on round-robin above.
        if ctx.small:
            rng = random.Random(20240607)
            for _ in range(RANDOM_RESTARTS):
                if perfect() or out_of_time():
                    break
                hf = rng.uniform(0.0, 0.85)
                strat = "preplace_asc" if rng.random() < 0.5 else "preplace_desc"
                consider(*_attempt(ctx, strat, rng=rng, hard_frac=hf))

            # Separate RNG so this never perturbs the stream above (best-of stays
            # monotonic). Shuffled placement order varies packing per try.
            rng2 = random.Random(990001)
            for _ in range(RANDOM_RESTARTS):
                if perfect() or out_of_time():
                    break
                consider(*_attempt(ctx, "tier", rng=rng2))

            # Cheapest-insertion restarts: varying the mandatory-insertion order
            # finds an order covering all mandatory while keeping low deadhead.
            if ctx.apsp is not None:
                rng3 = random.Random(424242)
                for _ in range(RANDOM_RESTARTS):
                    if perfect() or out_of_time():
                        break
                    consider(*_attempt_insertion(ctx, rng=rng3))
    return best[1]


class _Ctx:
    """Immutable per-instance data shared across strategy attempts."""
    def __init__(self, inst, g, dist_to_depot, succ_to, dist_from, prev_from,
                 cleanable, mandatory, inc, pop_limit):
        self.inst = inst
        self.g = g
        self.dist_to_depot = dist_to_depot
        self.succ_to = succ_to
        self.dist_from = dist_from
        self.prev_from = prev_from
        self.cleanable = cleanable
        self.mandatory = mandatory
        self.inc = inc
        self.pop_limit = pop_limit
        self.l_max = inst.l_max() or 1.0
        self.w_max = inst.w_max() or 1.0
        # (u, v) -> street idx, respecting direction. Used by the opportunistic
        # cleaning pass to clean streets a route already traverses.
        em = {}
        for s in inst.streets:
            em[(s.a, s.b)] = s.idx
            if not s.one_way:
                em[(s.b, s.a)] = s.idx
        self.edge_map = em
        # "Small" instances can afford the expensive high-quality strategies
        # (per-step route rebuild, all-pairs cheapest insertion). On large
        # instances these are quadratic-ish per attempt and dominate runtime, so
        # they are gated off in favour of the fast chaining greedy + repair.
        self.small = inst.m <= SMALL_MAX_STREETS
        # Route-improvement (per-step full Dijkstra) — small instances only.
        self.improve = self.small
        # All-pairs shortest-path times, used by the cheapest-insertion strategy
        # to compute the marginal cost of splicing a street into a route. Only
        # built for small graphs (one Dijkstra per node).
        self.apsp = None
        if self.small and inst.n <= APSP_MAX_NODES:
            self.apsp = [g.dijkstra(u)[0] for u in range(inst.n)]

    def loop_cost(self, s):
        dirs = ((s.a, s.b),) if s.one_way else ((s.a, s.b), (s.b, s.a))
        best = INF
        for entry, exit in dirs:
            df, dt = self.dist_from[entry], self.dist_to_depot[exit]
            if df < INF and dt < INF:
                best = min(best, df + s.time + dt)
        return best


def _attempt(ctx, strategy, rng=None, hard_frac=HARD_FRAC):
    inst = ctx.inst
    cleaned = [False] * inst.m

    asc = "asc" in strategy or strategy == "local_asc"
    keyfn = (lambda i: CAPACITY[inst.vehicles[i]]) if asc else \
            (lambda i: -CAPACITY[inst.vehicles[i]])
    vehs = [_Veh(i, inst.vehicles[i], inst.depot)
            for i in sorted(range(inst.c), key=keyfn)]
    if rng is not None:
        # keep capacity tiers grouped (matching) but shuffle within a tier
        rng.shuffle(vehs)
        vehs.sort(key=lambda v: v.cap if asc else -v.cap)

    # ---- Phase 1: mandatory coverage ----
    if strategy.startswith("preplace"):
        _preplace_hard(ctx, cleaned, vehs, rng=rng, hard_frac=hard_frac)
    elif strategy == "tier":
        _mandatory_tiered(ctx, cleaned, vehs, rng=rng)
    for v in vehs:
        _greedy_pass(ctx, cleaned, v, want_mandatory=True, rng=rng)

    # ---- Phase 2: repair orphans ----
    _repair_mandatory(ctx, cleaned, vehs)

    # ---- Phase 3: optional fill ----
    for v in vehs:
        _greedy_pass(ctx, cleaned, v, want_mandatory=False, rng=rng)

    # ---- Phase 3.5: route improvement + refill ----
    # Rebuild each route over its SAME cleaned set via nearest-neighbour to cut
    # deadheading, then spend the freed time on more streets. Monotonic: a route
    # is only replaced when strictly shorter, refill only adds streets, and the
    # cleaned set never shrinks (mandatory stays covered). Helps coverage-bound
    # instances (high alpha) most. Gated by size to stay cheap on huge graphs.
    if ctx.improve:
        for _ in range(2):  # a second sweep can exploit time freed by the first
            improved = False
            for v in vehs:
                if _rebuild_shorter(ctx, v):
                    improved = True
            if not improved:
                break
            for v in vehs:
                _greedy_pass(ctx, cleaned, v, want_mandatory=False, rng=rng)

    # ---- finalise ----
    # Append return-to-depot paths first so the opportunistic pass can also
    # clean streets traversed on the way home.
    for v in vehs:
        if v.current != inst.depot:
            back = reconstruct_toward(ctx.succ_to, v.current, inst.depot)
            v.nodes.extend(back[1:])
            v.time_used += ctx.dist_to_depot[v.current]
            v.current = inst.depot

    # ---- Phase 4: opportunistic cleaning along existing routes ----
    # Clean cleanable streets a vehicle already traverses, at zero extra time.
    _opportunistic_clean(ctx, cleaned, vehs)

    routes = []
    for v in vehs:
        routes.append(Route(vehicle_idx=v.idx, vtype=v.vtype,
                            nodes=v.nodes, cleaned=v.cleaned, time_used=v.time_used))
    routes.sort(key=lambda r: r.vehicle_idx)

    n_mand, score = _evaluate(ctx, vehs)
    return Solution(routes=routes), n_mand, score


def _greedy_pass(ctx, cleaned, v, want_mandatory, rng=None):
    """Extend v's route, cleaning mandatory or optional streets (one category
    per pass) until none fit the remaining time budget. With rng set, choose
    randomly among the top RCL_SIZE candidates (GRASP) instead of the best."""
    inst, g = ctx.inst, ctx.g
    T = inst.t
    alpha = inst.alpha
    while True:
        remaining = T - v.time_used
        if remaining <= 0:
            break
        dist, prev, settled = g.dijkstra(v.current, cutoff=remaining,
                                         max_pops=ctx.pop_limit)
        cands = []  # (key, street, entry, exit, cost)
        best = None
        seen = set()
        for node in settled:
            reach = dist[node]
            for s in ctx.inc[node]:
                if s.idx in seen or cleaned[s.idx] or v.cap < s.req:
                    continue
                if s.mandatory != want_mandatory:
                    continue
                exit = s.b if node == s.a else s.a
                ret = ctx.dist_to_depot[exit]
                if ret == INF or reach + s.time + ret > remaining:
                    continue
                seen.add(s.idx)
                cost = reach + s.time
                waste = (v.cap - s.req) * s.length_km
                if want_mandatory:
                    key = (-waste, -cost)        # least waste, then least time
                else:
                    cov_gain = s.length / ctx.l_max
                    eff_loss = waste / ctx.w_max
                    obj_gain = alpha * cov_gain - (1 - alpha) * eff_loss
                    if obj_gain <= 0:
                        continue
                    key = obj_gain / max(cost, 1)
                if rng is not None:
                    cands.append((key, s, node, exit, cost))
                elif best is None or key > best[0]:
                    best = (key, s, node, exit, cost)
        if rng is not None:
            if not cands:
                break
            cands.sort(key=lambda c: c[0], reverse=True)
            best = rng.choice(cands[:RCL_SIZE])
        if best is None:
            break
        _, s, entry, exit, cost = best
        path = reconstruct_forward(prev, v.current, entry)
        v.nodes.extend(path[1:])
        v.nodes.append(exit)
        cleaned[s.idx] = True
        v.cleaned.append(s.idx)
        v.time_used += cost
        v.current = exit


def _preplace_hard(ctx, cleaned, vehs, rng=None, hard_frac=HARD_FRAC):
    """Reserve budget-hungry mandatory streets on fresh vehicles first, so the
    local greedy doesn't fill every vehicle locally and orphan them. Hardest
    (largest depot-loop cost) first; each goes to the capable vehicle reaching
    it most cheaply, preferring the smallest sufficient capacity (low waste).
    With rng set, the placement order is lightly perturbed (GRASP)."""
    inst, g = ctx.inst, ctx.g
    T = inst.t
    hard = [s for s in ctx.mandatory if ctx.loop_cost(s) > hard_frac * T]
    hard.sort(key=ctx.loop_cost, reverse=True)
    if rng is not None and len(hard) > 1:
        # swap a few adjacent pairs to vary placement order between restarts
        for _ in range(len(hard)):
            i = rng.randrange(len(hard) - 1)
            hard[i], hard[i + 1] = hard[i + 1], hard[i]
    for s in hard:
        if cleaned[s.idx]:
            continue
        dirs = ((s.a, s.b),) if s.one_way else ((s.a, s.b), (s.b, s.a))
        best = None  # (key, added, veh, exit, path)
        for entry, exit in dirs:
            ret = ctx.dist_to_depot[exit]
            if ret == INF:
                continue
            dist_e, succ_e = g.dijkstra_to(entry)
            for v in vehs:
                if v.cap < s.req:
                    continue
                d = dist_e[v.current]
                if d == INF or v.time_used + d + s.time + ret > T:
                    continue
                added = d + s.time
                key = (added, v.cap, v.time_used)
                if best is None or key < best[0]:
                    path = reconstruct_toward(succ_e, v.current, entry)
                    best = (key, added, v, exit, path)
        if best is not None:
            _, added, v, exit, path = best
            v.nodes.extend(path[1:])
            v.nodes.append(exit)
            v.time_used += added
            v.current = exit
            cleaned[s.idx] = True
            v.cleaned.append(s.idx)


def _mandatory_tiered(ctx, cleaned, vehs, rng=None):
    """Capacity-tier-aware mandatory assignment.

    Processes mandatory streets scarce-tier-first (req 30 before 20 before 10),
    hardest (largest depot-loop) first within a tier. Each street is assigned to
    the *smallest sufficient* capacity tier that can still chain it in time,
    escalating only if no truck in that tier fits. This keeps the few large
    trucks free for heavy streets that nothing else can clean.
    """
    inst, g = ctx.inst, ctx.g
    T = inst.t
    streets = sorted(ctx.mandatory, key=lambda s: (s.req, ctx.loop_cost(s)),
                     reverse=True)
    if rng is not None and len(streets) > 1:
        for _ in range(len(streets)):
            i = rng.randrange(len(streets) - 1)
            streets[i], streets[i + 1] = streets[i + 1], streets[i]

    for s in streets:
        if cleaned[s.idx]:
            continue
        dirs = ((s.a, s.b),) if s.one_way else ((s.a, s.b), (s.b, s.a))
        placed = False
        for tier in (10, 20, 30):
            if tier < s.req:
                continue
            cand = [v for v in vehs if v.cap == tier]
            if not cand:
                continue
            best = None  # (key, added, veh, exit, path)
            cands = []   # all feasible placements (for RCL randomization)
            for entry, exit in dirs:
                ret = ctx.dist_to_depot[exit]
                if ret == INF:
                    continue
                dist_e, succ_e = g.dijkstra_to(entry)
                for v in cand:
                    d = dist_e[v.current]
                    if d == INF or v.time_used + d + s.time + ret > T:
                        continue
                    added = d + s.time
                    key = (added, v.time_used)
                    path = reconstruct_toward(succ_e, v.current, entry)
                    rec = (key, added, v, exit, path)
                    if rng is not None:
                        cands.append(rec)
                    elif best is None or key < best[0]:
                        best = rec
            if rng is not None and cands:
                cands.sort(key=lambda r: r[0])
                best = rng.choice(cands[:RCL_SIZE])
            if best is not None:
                _, added, v, exit, path = best
                v.nodes.extend(path[1:])
                v.nodes.append(exit)
                v.time_used += added
                v.current = exit
                cleaned[s.idx] = True
                v.cleaned.append(s.idx)
                placed = True
                break
        # if not placed in any tier, the repair phase will retry across all trucks


def _repair_mandatory(ctx, cleaned, vehs):
    """Force every still-uncleaned mandatory street onto a capable vehicle that
    can chain to it from its current position and still return in time."""
    inst, g = ctx.inst, ctx.g
    T = inst.t
    leftover = [s for s in ctx.mandatory if not cleaned[s.idx]]
    if not leftover:
        return
    leftover.sort(key=ctx.loop_cost, reverse=True)
    for s in leftover:
        if cleaned[s.idx]:
            continue
        best = None  # (added, veh, exit, path)
        dirs = ((s.a, s.b),) if s.one_way else ((s.a, s.b), (s.b, s.a))
        for entry, exit in dirs:
            ret = ctx.dist_to_depot[exit]
            if ret == INF:
                continue
            dist_e, succ_e = g.dijkstra_to(entry)
            for v in vehs:
                if v.cap < s.req:
                    continue
                d = dist_e[v.current]
                if d == INF or v.time_used + d + s.time + ret > T:
                    continue
                added = d + s.time
                if best is None or added < best[0]:
                    path = reconstruct_toward(succ_e, v.current, entry)
                    best = (added, v, exit, path)
        if best is not None:
            added, v, exit, path = best
            v.nodes.extend(path[1:])
            v.nodes.append(exit)
            v.time_used += added
            v.current = exit
            cleaned[s.idx] = True
            v.cleaned.append(s.idx)
        # else: infeasible for the whole fleet given prior commitments.


def _roundrobin_pass(ctx, cleaned, vehs, want_mandatory):
    """Advance all vehicles in parallel: each round, every vehicle takes the one
    best street it can still service and return from. Unlike the sequential
    greedy (which lets one vehicle claim a whole region before the next starts),
    this keeps vehicles spread across the map, slashing deadhead — the key to
    covering all mandatory streets on large instances.

    Mandatory picks the nearest street (minimise added time); optional picks the
    best objective-gain per second.
    """
    inst, g = ctx.inst, ctx.g
    T = inst.t
    alpha = inst.alpha
    active = True
    while active:
        active = False
        for v in vehs:
            remaining = T - v.time_used
            if remaining <= 0:
                continue
            dist, prev, settled = g.dijkstra(v.current, cutoff=remaining,
                                             max_pops=ctx.pop_limit)
            best = None  # (key, s, entry, exit, cost)
            for node in settled:
                reach = dist[node]
                for s in ctx.inc[node]:
                    if cleaned[s.idx] or v.cap < s.req:
                        continue
                    if s.mandatory != want_mandatory:
                        continue
                    exit = s.b if node == s.a else s.a
                    ret = ctx.dist_to_depot[exit]
                    if ret == INF or reach + s.time + ret > remaining:
                        continue
                    cost = reach + s.time
                    if want_mandatory:
                        if best is None or cost < best[0]:
                            best = (cost, s, node, exit, cost)
                    else:
                        waste = (v.cap - s.req) * s.length_km
                        og = alpha * (s.length / ctx.l_max) - \
                            (1 - alpha) * (waste / ctx.w_max)
                        if og <= 0:
                            continue
                        key = og / max(cost, 1)
                        if best is None or key > best[0]:
                            best = (key, s, node, exit, cost)
            if best is not None:
                _key, s, node, exit, cost = best
                path = reconstruct_forward(prev, v.current, node)
                v.nodes.extend(path[1:])
                v.nodes.append(exit)
                cleaned[s.idx] = True
                v.cleaned.append(s.idx)
                v.time_used += cost
                v.current = exit
                active = True


def _attempt_rr(ctx, rng=None):
    """Round-robin construction: parallel nearest-street coverage with rebuild.

    The most reliable strategy for large instances. Mandatory is covered in
    parallel (low deadhead), stragglers recovered by rebuilding routes to free
    time and re-sweeping, then optionals fill the remainder. Randomising the
    vehicle processing order across restarts varies the spatial partition and
    closes the last few mandatory streets.
    """
    inst = ctx.inst
    depot = inst.depot
    order = sorted(range(inst.c), key=lambda i: -CAPACITY[inst.vehicles[i]])
    vehs = [_Veh(i, inst.vehicles[i], depot) for i in order]
    if rng is not None:
        rng.shuffle(vehs)
    cleaned = [False] * inst.m

    _roundrobin_pass(ctx, cleaned, vehs, want_mandatory=True)
    _repair_mandatory(ctx, cleaned, vehs)
    # Rebuild routes to free time, then re-sweep to recover straggler mandatory.
    for _ in range(3):
        improved = False
        for v in vehs:
            if _rebuild_shorter(ctx, v):
                improved = True
        if not improved:
            break
        _roundrobin_pass(ctx, cleaned, vehs, want_mandatory=True)
        _repair_mandatory(ctx, cleaned, vehs)

    _roundrobin_pass(ctx, cleaned, vehs, want_mandatory=False)

    for v in vehs:
        if v.current != depot:
            v.nodes.extend(reconstruct_toward(ctx.succ_to, v.current, depot)[1:])
            v.time_used += ctx.dist_to_depot[v.current]
            v.current = depot
    _opportunistic_clean(ctx, cleaned, vehs)

    routes = [Route(vehicle_idx=v.idx, vtype=v.vtype, nodes=v.nodes,
                    cleaned=v.cleaned, time_used=v.time_used) for v in vehs]
    routes.sort(key=lambda r: r.vehicle_idx)
    n_mand, score = _evaluate(ctx, vehs)
    return Solution(routes=routes), n_mand, score


def _attempt_capacity_partition(ctx, seed):
    """Coverage-first construction for small alpha=1 instances.

    Heavy streets (req=30) can only be cleaned by Large vehicles. This strategy
    reserves Large routes for those streets, pushes lighter mandatory work to
    Small/Medium vehicles, then fills optional streets by coverage per inserted
    second. It is intentionally separate from the legacy solver path.
    """
    if ctx.apsp is None:
        return Solution(routes=[]), 0, 0.0

    inst = ctx.inst
    D = ctx.apsp
    depot = inst.depot
    rng = random.Random(seed)
    plans = [
        {"idx": i, "cap": CAPACITY[inst.vehicles[i]], "tasks": [], "time": 0}
        for i in range(inst.c)
    ]
    large = [p for p in plans if p["cap"] == 30]
    nonlarge = [p for p in plans if p["cap"] < 30]
    if not large:
        return Solution(routes=[]), 0, 0.0

    def directions(s):
        return ((s.a, s.b),) if s.one_way else ((s.a, s.b), (s.b, s.a))

    def delta(tasks, s, entry, exit, pos):
        prev_exit = depot if pos == 0 else tasks[pos - 1][2]
        next_entry = depot if pos == len(tasks) else tasks[pos][1]
        d1 = D[prev_exit][entry]
        d2 = D[exit][next_entry]
        d0 = D[prev_exit][next_entry]
        if d1 == INF or d2 == INF or d0 == INF:
            return None
        return d1 + s.time + d2 - d0

    def add_best(candidates, s, random_chance=0.25):
        feasible = []
        for p in candidates:
            if p["cap"] < s.req:
                continue
            for entry, exit in directions(s):
                for pos in range(len(p["tasks"]) + 1):
                    dlt = delta(p["tasks"], s, entry, exit, pos)
                    if dlt is None or p["time"] + dlt > inst.t:
                        continue
                    feasible.append((dlt, p["time"], rng.random(), p, pos, entry, exit))
        if not feasible:
            return False
        feasible.sort(key=lambda rec: rec[:3])
        rec = feasible[0]
        if rng.random() < random_chance:
            rec = rng.choice(feasible[:min(6, len(feasible))])
        dlt, _time, _tie, p, pos, entry, exit = rec
        p["tasks"].insert(pos, (s.idx, entry, exit))
        p["time"] += dlt
        return True

    heavy_mand = [s for s in ctx.mandatory if s.req == 30]
    rng.shuffle(heavy_mand)
    heavy_mand.sort(key=ctx.loop_cost, reverse=True)
    for s in heavy_mand:
        if not add_best(large, s, random_chance=0.35):
            return Solution(routes=[]), 0, 0.0

    light_mand = [s for s in ctx.mandatory if s.req < 30]
    rng.shuffle(light_mand)
    light_mand.sort(key=lambda s: s.req, reverse=True)
    for s in light_mand:
        if not add_best(nonlarge, s, random_chance=0.30):
            if not add_best(plans, s, random_chance=0.30):
                return Solution(routes=[]), 0, 0.0

    def fill_optional(streets, candidates, random_chance, large_penalty=False,
                      rcl_size=10):
        remaining = set(s.idx for s in streets)
        while True:
            feasible = []
            for sid in remaining:
                s = inst.streets[sid]
                for p in candidates:
                    if p["cap"] < s.req:
                        continue
                    for entry, exit in directions(s):
                        for pos in range(len(p["tasks"]) + 1):
                            dlt = delta(p["tasks"], s, entry, exit, pos)
                            if dlt is None or p["time"] + dlt > inst.t:
                                continue
                            penalty = 0.7 if large_penalty and p["cap"] == 30 else 1.0
                            ratio = penalty * s.length / max(dlt, 1)
                            feasible.append(
                                (ratio, s.length, -dlt, rng.random(),
                                 sid, dlt, p, pos, entry, exit)
                            )
            if not feasible:
                return
            feasible.sort(reverse=True)
            rec = feasible[0]
            if rng.random() < random_chance:
                rec = rng.choice(feasible[:min(rcl_size, len(feasible))])
            _ratio, _length, _neg_dlt, _tie, sid, dlt, p, pos, entry, exit = rec
            p["tasks"].insert(pos, (sid, entry, exit))
            p["time"] += dlt
            remaining.remove(sid)

    fill_optional([s for s in ctx.cleanable if not s.mandatory and s.req == 30],
                  large, random_chance=0.25, rcl_size=10)
    fill_optional([s for s in ctx.cleanable if not s.mandatory and s.req < 30],
                  plans, random_chance=0.20, large_penalty=True, rcl_size=8)

    vehs = []
    cleaned = [False] * inst.m
    for p in plans:
        v = _Veh(p["idx"], inst.vehicles[p["idx"]], depot)
        cur = depot
        for sid, entry, exit in p["tasks"]:
            if cur != entry:
                _, prev, _ = ctx.g.dijkstra(cur)
                path = reconstruct_forward(prev, cur, entry)
                if not path:
                    return Solution(routes=[]), 0, 0.0
                v.nodes.extend(path[1:])
            v.nodes.append(exit)
            v.cleaned.append(sid)
            cleaned[sid] = True
            cur = exit
        if cur != depot:
            back = reconstruct_toward(ctx.succ_to, cur, depot)
            if not back:
                return Solution(routes=[]), 0, 0.0
            v.nodes.extend(back[1:])
        v.time_used = p["time"]
        v.current = depot
        vehs.append(v)

    _opportunistic_clean(ctx, cleaned, vehs)
    routes = [Route(vehicle_idx=v.idx, vtype=v.vtype, nodes=v.nodes,
                    cleaned=v.cleaned, time_used=v.time_used) for v in vehs]
    routes.sort(key=lambda r: r.vehicle_idx)
    n_mand, score = _evaluate(ctx, vehs)
    return Solution(routes=routes), n_mand, score


def _attempt_insertion(ctx, rng=None):
    """Cheapest-insertion construction (team-orienteering style).

    Routes are grown by splicing each street into the position (vehicle, slot,
    direction) that adds the least travel time, rather than appending at the end.
    This minimises deadheading by construction, which is the dominant cost on
    coverage-bound instances. Mandatory streets are inserted first (hardest
    first) to guarantee coverage; optionals then by best objective-gain per
    inserted second. Requires all-pairs shortest paths (small graphs only).
    """
    inst = ctx.inst
    D = ctx.apsp
    T = inst.t
    depot = inst.depot
    alpha = inst.alpha
    cleaned = [False] * inst.m

    plans = [{"idx": i, "cap": CAPACITY[inst.vehicles[i]], "tasks": [], "time": 0}
             for i in range(inst.c)]

    def delta(plan, s, e, x, pos):
        tasks = plan["tasks"]
        prev_exit = depot if pos == 0 else tasks[pos - 1][2]
        next_entry = depot if pos == len(tasks) else tasks[pos][1]
        d1 = D[prev_exit][e]
        d2 = D[x][next_entry]
        if d1 == INF or d2 == INF:
            return None
        d0 = D[prev_exit][next_entry]
        if d0 == INF:
            d0 = 0
        return d1 + s.time + d2 - d0

    def directions(s):
        return ((s.a, s.b),) if s.one_way else ((s.a, s.b), (s.b, s.a))

    # ---- mandatory first (hardest first) ----
    mand = sorted(ctx.mandatory, key=ctx.loop_cost, reverse=True)
    if rng is not None and len(mand) > 1:
        for _ in range(len(mand)):
            i = rng.randrange(len(mand) - 1)
            mand[i], mand[i + 1] = mand[i + 1], mand[i]
    for s in mand:
        if cleaned[s.idx]:
            continue
        best = None  # (key, dlt, plan, pos, e, x)
        for plan in plans:
            if plan["cap"] < s.req:
                continue
            for e, x in directions(s):
                for pos in range(len(plan["tasks"]) + 1):
                    dlt = delta(plan, s, e, x, pos)
                    if dlt is None or plan["time"] + dlt > T:
                        continue
                    key = (dlt, plan["cap"])
                    if best is None or key < best[0]:
                        best = (key, dlt, plan, pos, e, x)
        if best is not None:
            _, dlt, plan, pos, e, x = best
            plan["tasks"].insert(pos, (s.idx, e, x))
            plan["time"] += dlt
            cleaned[s.idx] = True

    # ---- optionals by best objective-gain per inserted second ----
    while True:
        best = None  # (ratio, s, plan, pos, e, x, dlt)
        for s in ctx.cleanable:
            if cleaned[s.idx] or s.mandatory:
                continue
            cov_gain = s.length / ctx.l_max
            for plan in plans:
                if plan["cap"] < s.req:
                    continue
                eff_loss = (plan["cap"] - s.req) * s.length_km / ctx.w_max
                obj_gain = alpha * cov_gain - (1 - alpha) * eff_loss
                if obj_gain <= 0:
                    continue
                for e, x in directions(s):
                    for pos in range(len(plan["tasks"]) + 1):
                        dlt = delta(plan, s, e, x, pos)
                        if dlt is None or plan["time"] + dlt > T:
                            continue
                        ratio = obj_gain / max(dlt, 1)
                        if best is None or ratio > best[0]:
                            best = (ratio, s, plan, pos, e, x, dlt)
        if best is None:
            break
        _, s, plan, pos, e, x, dlt = best
        plan["tasks"].insert(pos, (s.idx, e, x))
        plan["time"] += dlt
        cleaned[s.idx] = True

    # ---- materialise node routes from task lists ----
    vehs = []
    for plan in plans:
        v = _Veh(plan["idx"], inst.vehicles[plan["idx"]], depot)
        cur = depot
        for sid, e, x in plan["tasks"]:
            if cur != e:
                _, prev, _ = ctx.g.dijkstra(cur)
                path = reconstruct_forward(prev, cur, e)
                v.nodes.extend(path[1:])
            v.nodes.append(x)
            cur = x
        if cur != depot:
            v.nodes.extend(reconstruct_toward(ctx.succ_to, cur, depot)[1:])
        v.cleaned = [sid for sid, _e, _x in plan["tasks"]]
        v.time_used = plan["time"]
        v.current = depot
        vehs.append(v)

    _opportunistic_clean(ctx, cleaned, vehs)

    routes = [Route(vehicle_idx=v.idx, vtype=v.vtype, nodes=v.nodes,
                    cleaned=v.cleaned, time_used=v.time_used) for v in vehs]
    routes.sort(key=lambda r: r.vehicle_idx)
    n_mand, score = _evaluate(ctx, vehs)
    return Solution(routes=routes), n_mand, score


def _rebuild_shorter(ctx, v):
    """Re-route vehicle v over its CURRENT cleaned set via nearest-neighbour,
    keeping every cleaned street but minimising deadhead. Adopts the new route
    only if strictly shorter (time before the final depot return, matching how
    v.time_used is tracked mid-attempt). Returns True if the route was replaced.

    Reducing a route's time frees budget that the subsequent optional refill can
    spend on additional streets — the main lever for coverage-bound instances.
    """
    inst, g = ctx.inst, ctx.g
    T = inst.t
    targets = set(v.cleaned)
    if len(targets) < 2:
        return False  # nothing to reorder

    cur = inst.depot
    nodes = [inst.depot]
    time_used = 0
    remaining = set(targets)
    while remaining:
        dist, prev, _ = g.dijkstra(cur)
        best = None  # (cost, sid, entry, exit, prev_ref)
        for sid in remaining:
            s = inst.streets[sid]
            dirs = ((s.a, s.b),) if s.one_way else ((s.a, s.b), (s.b, s.a))
            for e, x in dirs:
                if dist[e] == INF:
                    continue
                ret = ctx.dist_to_depot[x]
                if ret == INF:
                    continue
                cost = dist[e] + s.time
                if time_used + cost + ret > T:
                    continue
                if best is None or cost < best[0]:
                    best = (cost, sid, e, x)
        if best is None:
            return False  # could not legally re-route all cleaned streets
        cost, sid, e, x = best
        path = reconstruct_forward(prev, cur, e)
        nodes.extend(path[1:])
        nodes.append(x)
        time_used += cost
        cur = x
        remaining.discard(sid)

    if time_used >= v.time_used:
        return False  # not actually shorter
    v.nodes = nodes
    v.time_used = time_used
    v.current = cur
    return True


def _opportunistic_clean(ctx, cleaned, vehs):
    """Clean cleanable streets each vehicle already traverses — zero extra time.

    Gated to never reduce the objective: a street is cleaned opportunistically
    only if it is still uncleaned, the vehicle's capacity suffices, and either
    the street is mandatory or cleaning it yields a positive objective gain
    (alpha * coverage_gain - (1 - alpha) * waste_loss > 0). At alpha=0 (pure
    efficiency) it never fires on optionals; at alpha=1 it always adds free
    coverage. Vehicles are processed smallest-capacity first so the least
    wasteful truck claims each shared street.
    """
    inst = ctx.inst
    alpha = inst.alpha
    em = ctx.edge_map
    for v in sorted(vehs, key=lambda v: v.cap):
        for a, b in zip(v.nodes, v.nodes[1:]):
            sid = em.get((a, b))
            if sid is None:
                continue
            s = inst.streets[sid]
            if not s.cleanable or cleaned[sid] or v.cap < s.req:
                continue
            if s.mandatory:
                ok = True
            else:
                cov_gain = s.length / ctx.l_max
                eff_loss = (v.cap - s.req) * s.length_km / ctx.w_max
                ok = alpha * cov_gain - (1 - alpha) * eff_loss > 0
            if ok:
                cleaned[sid] = True
                v.cleaned.append(sid)


def _evaluate(ctx, vehs):
    """(mandatory cleaned, objective score) for a finished attempt. Routes are
    valid by construction, so this only needs the cleaning assignments."""
    inst = ctx.inst
    streets = inst.streets
    cleaned_once = set()
    waste = 0.0
    for v in vehs:
        for sid in v.cleaned:
            s = streets[sid]
            cleaned_once.add(sid)
            waste += (v.cap - s.req) * s.length_km
    length = sum(streets[sid].length for sid in cleaned_once)
    n_mand = sum(1 for s in ctx.mandatory if s.idx in cleaned_once)
    coverage = length / ctx.l_max
    efficiency = 1 - waste / ctx.w_max
    score = inst.alpha * coverage + (1 - inst.alpha) * efficiency
    return n_mand, score
