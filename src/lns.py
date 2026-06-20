"""Large-Neighbourhood-Search post-optimiser for the Street Cleaning problem.

Takes a *valid* starting solution (all mandatory cleaned) and improves the
weighted coverage/efficiency objective by reorganising routes and reshuffling
optional cleaning, while never dropping a mandatory street.

Representation
--------------
Each vehicle is a *task list* ``[(street_id, entry, exit), ...]``. A task means
"traverse street ``street_id`` from junction ``entry`` to ``exit`` and clean it".
Between consecutive tasks the vehicle deadheads along the shortest path, so the
closed-tour time is computed from all-pairs shortest paths (``ctx.apsp``):

    depot -> t0.entry -> t0.exit -> t1.entry -> ... -> t_last.exit -> depot

Because coverage counts a street once and waste is per cleaning event, the
score is a pure function of the *set* of cleaned streets and which vehicle
cleans each one. Route time only enters as a per-vehicle feasibility constraint
(<= T). So the marginal score of cleaning optional ``s`` with a capacity-``c``
vehicle is independent of where it is inserted:

    dscore(s, c) = alpha * len(s)/Lmax  -  (1-alpha) * (c-req(s))*len_km(s)/Wmax

and insertion *position* only affects the added time (the feasibility budget).
This is what makes the objective alpha-adaptive without per-instance branches:
at alpha=1 waste is free (clean everything reachable); at low alpha a high-waste
optional has negative dscore and is rejected / removed.

Operators
---------
* nn_rebuild      - re-order one route greedily (nearest task entry) to cut
                    deadhead; the freed time lets more optionals fit. The key
                    lever on time-bound, coverage-heavy instances.
* relocate        - move a task to its cheapest slot across all vehicles
                    (mandatory only to capable vehicles); rebalances load.
* swap            - exchange two optional tasks between vehicles.
* fill            - global cheapest insertion of positive-dscore optionals,
                    ranked by dscore/added-time.
* destroy/repair  - remove a fraction of optionals (biased to low value) then
                    re-fill globally; the randomised LNS kick out of local optima.

Every accepted change keeps the cleaned set valid (mandatory intact) and each
route within T. The best-scoring snapshot is returned.
"""
from __future__ import annotations

import random
import time as _time

from .model import Route, Solution, CAPACITY
from .graph import INF, reconstruct_forward


def optimize(ctx, solution, deadline, rng=None):
    """Improve ``solution`` in place-ish and return (Solution, n_mand, score).

    Requires ``ctx.apsp`` (all-pairs shortest path times). If unavailable the
    input solution is returned unchanged.
    """
    if ctx.apsp is None:
        from .solver import _evaluate, _Veh
        return solution, *_eval_from_solution(ctx, solution)

    inst = ctx.inst
    D = ctx.apsp
    T = inst.t
    depot = inst.depot
    alpha = inst.alpha
    Lmax = ctx.l_max
    Wmax = ctx.w_max
    streets = inst.streets
    if rng is None:
        rng = random.Random(0xC0FFEE)

    def dirs(sid):
        s = streets[sid]
        return ((s.a, s.b),) if s.one_way else ((s.a, s.b), (s.b, s.a))

    def dscore(sid, cap):
        s = streets[sid]
        return alpha * (s.length / Lmax) - \
            (1 - alpha) * ((cap - s.req) * s.length_km / Wmax)

    # ---- build plans from the starting solution ----
    em = ctx.edge_map
    plans = []
    taken = set()
    for r in solution.routes:
        cap = CAPACITY[r.vtype]
        # direction each cleaned street is actually traversed (first occurrence)
        occ = {}
        order = {}
        for j, (a, b) in enumerate(zip(r.nodes, r.nodes[1:])):
            sid = em.get((a, b))
            if sid is not None and sid not in occ:
                occ[sid] = (a, b)
                order[sid] = j
        tasks = []
        for sid in r.cleaned:
            if sid in taken:
                continue  # a street cleaned twice -> keep a single cleaner
            taken.add(sid)
            if sid in occ:
                e, x = occ[sid]
            else:
                s = streets[sid]
                e, x = s.a, s.b
            tasks.append((sid, e, x))
        tasks.sort(key=lambda t: order.get(t[0], 1 << 30))
        plans.append({"idx": r.vehicle_idx, "cap": cap, "tasks": tasks,
                      "time": 0.0})

    def route_time(tasks):
        if not tasks:
            return 0.0
        sid0, e0, _x0 = tasks[0]
        t = D[depot][e0]
        if t == INF:
            return INF
        for i, (sid, e, x) in enumerate(tasks):
            t += streets[sid].time
            ne = depot if i == len(tasks) - 1 else tasks[i + 1][1]
            leg = D[x][ne]
            if leg == INF:
                return INF
            t += leg
        return t

    for p in plans:
        p["time"] = route_time(p["tasks"])

    mand_ids = set(s.idx for s in ctx.mandatory)
    cleaned = set()
    for p in plans:
        for sid, _e, _x in p["tasks"]:
            cleaned.add(sid)

    optionals = [s.idx for s in ctx.cleanable if not s.mandatory]

    def total_score():
        waste = 0.0
        length = 0.0
        for p in plans:
            cap = p["cap"]
            for sid, _e, _x in p["tasks"]:
                s = streets[sid]
                waste += (cap - s.req) * s.length_km
        for sid in cleaned:
            length += streets[sid].length
        cov = length / Lmax
        eff = 1 - waste / Wmax
        return alpha * cov + (1 - alpha) * eff

    def delta_time(plan, sid, e, x, pos):
        tasks = plan["tasks"]
        pe = depot if pos == 0 else tasks[pos - 1][2]
        ne = depot if pos == len(tasks) else tasks[pos][1]
        d1 = D[pe][e]
        if d1 == INF:
            return None
        d2 = D[x][ne]
        if d2 == INF:
            return None
        d0 = D[pe][ne]
        if d0 == INF:
            d0 = 0.0
        return d1 + streets[sid].time + d2 - d0

    def best_insertion(plan, sid):
        """Cheapest feasible (added_time, pos, e, x) for sid in plan, or None."""
        best = None
        for e, x in dirs(sid):
            for pos in range(len(plan["tasks"]) + 1):
                dl = delta_time(plan, sid, e, x, pos)
                if dl is None or plan["time"] + dl > T + 1e-6:
                    continue
                if best is None or dl < best[0]:
                    best = (dl, pos, e, x)
        return best

    # ---- operator: greedy global fill of positive-margin optionals ----
    def fill():
        added = False
        while True:
            if _time.time() > deadline:
                break
            best = None  # (ratio, sid, plan, dl, pos, e, x)
            for sid in optionals:
                if sid in cleaned:
                    continue
                for plan in plans:
                    cap = plan["cap"]
                    if cap < streets[sid].req:
                        continue
                    ds = dscore(sid, cap)
                    if ds <= 1e-12:
                        continue
                    bi = best_insertion(plan, sid)
                    if bi is None:
                        continue
                    dl, pos, e, x = bi
                    ratio = ds / max(dl, 1.0)
                    if best is None or ratio > best[0]:
                        best = (ratio, sid, plan, dl, pos, e, x)
            if best is None:
                break
            _r, sid, plan, dl, pos, e, x = best
            plan["tasks"].insert(pos, (sid, e, x))
            plan["time"] += dl
            cleaned.add(sid)
            added = True
        return added

    # ---- operator: nearest-neighbour re-order of one route (cut deadhead) ----
    def nn_rebuild(plan):
        sids = [t[0] for t in plan["tasks"]]
        if len(sids) < 2:
            return False
        remaining = set(sids)
        cur = depot
        new_tasks = []
        t = 0.0
        ok = True
        while remaining:
            choice = None  # (cost, sid, e, x)
            for sid in remaining:
                for e, x in dirs(sid):
                    d = D[cur][e]
                    if d == INF or D[x][depot] == INF:
                        continue
                    cost = d + streets[sid].time
                    if choice is None or cost < choice[0]:
                        choice = (cost, sid, e, x)
            if choice is None:
                ok = False
                break
            cost, sid, e, x = choice
            new_tasks.append((sid, e, x))
            t += cost
            cur = x
            remaining.discard(sid)
        if not ok:
            return False
        t += D[cur][depot]
        if t < plan["time"] - 1e-6:
            plan["tasks"] = new_tasks
            plan["time"] = t
            return True
        return False

    # ---- operator: relocate each task to its cheapest slot anywhere ----
    def relocate():
        moved = False
        for src in plans:
            i = 0
            while i < len(src["tasks"]):
                sid, e0, x0 = src["tasks"][i]
                s = streets[sid]
                pe = depot if i == 0 else src["tasks"][i - 1][2]
                ne = depot if i == len(src["tasks"]) - 1 else src["tasks"][i + 1][1]
                cur_cost = D[pe][e0] + s.time + D[x0][ne] - D[pe][ne]
                src["tasks"].pop(i)
                src["time"] -= cur_cost
                best = None  # (added, plan, pos, e, x, ddelta_waste)
                for p2 in plans:
                    if p2["cap"] < s.req:
                        continue
                    bi = best_insertion(p2, sid)
                    if bi is None:
                        continue
                    add, pos, e, x = bi
                    # accept if it shortens travel OR reduces waste (smaller cap)
                    waste_delta = (p2["cap"] - s.req) * s.length_km - \
                                  (src["cap"] - s.req) * s.length_km
                    # objective change: -(1-alpha)*waste_delta/Wmax ; travel change
                    # is feasibility only, but prefer shorter to free time.
                    key = ((1 - alpha) * waste_delta / Wmax, add)
                    if best is None or key < best[0]:
                        best = (key, add, p2, pos, e, x, waste_delta)
                accept = False
                if best is not None:
                    _key, add, p2, pos, e, x, wd = best
                    # accept when not worse on objective and (frees time or same veh)
                    better_eff = wd < -1e-12
                    shorter = (p2 is not src and add < cur_cost - 1e-6) or \
                              (p2 is src and add < cur_cost - 1e-6)
                    if better_eff or shorter:
                        accept = True
                if accept:
                    _key, add, p2, pos, e, x, wd = best
                    p2["tasks"].insert(pos, (sid, e, x))
                    p2["time"] += add
                    moved = True
                    if p2 is src:
                        i += 1
                    # else src shrank; stay at i
                else:
                    src["tasks"].insert(i, (sid, e0, x0))
                    src["time"] += cur_cost
                    i += 1
        return moved

    # ---- operator: swap two optional tasks between different vehicles ----
    def swap_pass():
        moved = False
        for a in range(len(plans)):
            pa = plans[a]
            for b in range(a + 1, len(plans)):
                pb = plans[b]
                for i in range(len(pa["tasks"])):
                    sid_a = pa["tasks"][i][0]
                    if sid_a in mand_ids:
                        continue
                    if pb["cap"] < streets[sid_a].req:
                        continue
                    for j in range(len(pb["tasks"])):
                        sid_b = pb["tasks"][j][0]
                        if sid_b in mand_ids:
                            continue
                        if pa["cap"] < streets[sid_b].req:
                            continue
                        # objective change from swapping cleaners (waste only)
                        old_w = (pa["cap"] - streets[sid_a].req) * streets[sid_a].length_km \
                            + (pb["cap"] - streets[sid_b].req) * streets[sid_b].length_km
                        new_w = (pa["cap"] - streets[sid_b].req) * streets[sid_b].length_km \
                            + (pb["cap"] - streets[sid_a].req) * streets[sid_a].length_km
                        if new_w >= old_w - 1e-12:
                            continue  # no efficiency gain
                        # try the swap, checking time feasibility after re-fit
                        ta = list(pa["tasks"]); tb = list(pb["tasks"])
                        ta[i] = (sid_b, *_dir_for(sid_b, dirs))
                        tb[j] = (sid_a, *_dir_for(sid_a, dirs))
                        na = route_time(ta); nb = route_time(tb)
                        if na <= T + 1e-6 and nb <= T + 1e-6:
                            pa["tasks"] = ta; pa["time"] = na
                            pb["tasks"] = tb; pb["time"] = nb
                            moved = True
        return moved

    def snapshot():
        return [(list(p["tasks"]), p["time"]) for p in plans], set(cleaned)

    def restore(snap):
        tasks_times, cl = snap
        for p, (tk, tm) in zip(plans, tasks_times):
            p["tasks"] = list(tk)
            p["time"] = tm
        cleaned.clear()
        cleaned.update(cl)

    def local_improve():
        for _ in range(6):
            changed = False
            for p in plans:
                if nn_rebuild(p):
                    changed = True
            if relocate():
                changed = True
            if swap_pass():
                changed = True
            if fill():
                changed = True
            if _time.time() > deadline or not changed:
                break

    # ---- main LNS ----
    local_improve()
    best_snap = snapshot()
    best_sc = total_score()

    while _time.time() < deadline:
        # destroy: remove a random fraction of optionals, biased to low value/time
        opt_tasks = [(pi, ti, t[0]) for pi, p in enumerate(plans)
                     for ti, t in enumerate(p["tasks"]) if t[0] not in mand_ids]
        if not opt_tasks:
            break
        frac = rng.uniform(0.10, 0.30)
        k = max(1, int(len(opt_tasks) * frac))
        # bias: rank by dscore/time ascending (worst first), then sample
        def val(item):
            sid = item[2]
            # use the cleaning vehicle's cap
            cap = plans[item[0]]["cap"]
            return dscore(sid, cap) / max(streets[sid].time, 1)
        opt_tasks.sort(key=val)
        pool = opt_tasks[:max(k * 2, k)]
        rng.shuffle(pool)
        victims = set(item[2] for item in pool[:k])
        for p in plans:
            kept = [t for t in p["tasks"] if t[0] not in victims]
            if len(kept) != len(p["tasks"]):
                p["tasks"] = kept
                p["time"] = route_time(p["tasks"])
        cleaned.difference_update(victims)

        local_improve()
        sc = total_score()
        if sc > best_sc + 1e-12 and _valid(plans, mand_ids, T, route_time):
            best_sc = sc
            best_snap = snapshot()
        else:
            restore(best_snap)

    restore(best_snap)

    # ---- materialise back to a Solution ----
    routes = []
    for p in plans:
        vtype = inst.vehicles[p["idx"]]
        nodes = [depot]
        cur = depot
        for sid, e, x in p["tasks"]:
            if cur != e:
                path = reconstruct_forward(ctx.apsp_prev[cur], cur, e)
                nodes.extend(path[1:])
            nodes.append(x)
            cur = x
        if cur != depot:
            nodes.extend(reconstruct_forward(ctx.apsp_prev[cur], cur, depot)[1:])
        cleaned_list = [sid for sid, _e, _x in p["tasks"]]
        routes.append(Route(vehicle_idx=p["idx"], vtype=vtype, nodes=nodes,
                            cleaned=cleaned_list, time_used=p["time"]))

    # opportunistic free cleaning along the realised paths (never hurts objective)
    _opportunistic(ctx, routes)

    routes.sort(key=lambda r: r.vehicle_idx)
    n_mand, score = _eval_routes(ctx, routes)
    return Solution(routes=routes), n_mand, score


def _dir_for(sid, dirs):
    # default direction for a swapped-in task (first valid); nn_rebuild fixes it.
    return dirs(sid)[0]


def _valid(plans, mand_ids, T, route_time):
    covered = set()
    for p in plans:
        if p["time"] > T + 1e-6:
            return False
        for sid, _e, _x in p["tasks"]:
            covered.add(sid)
    return mand_ids.issubset(covered)


def _opportunistic(ctx, routes):
    """Clean cleanable streets a vehicle already traverses for free, when it
    improves the objective (always at alpha=1; gated by positive gain otherwise).
    Smallest-capacity vehicle that passes through claims the street."""
    inst = ctx.inst
    alpha = inst.alpha
    em = ctx.edge_map
    cleaned = set()
    for r in routes:
        cleaned.update(r.cleaned)
    for r in sorted(routes, key=lambda r: CAPACITY[r.vtype]):
        cap = CAPACITY[r.vtype]
        for a, b in zip(r.nodes, r.nodes[1:]):
            sid = em.get((a, b))
            if sid is None:
                continue
            s = inst.streets[sid]
            if not s.cleanable or sid in cleaned or cap < s.req:
                continue
            if s.mandatory:
                ok = True
            else:
                cov_gain = s.length / ctx.l_max
                eff_loss = (cap - s.req) * s.length_km / ctx.w_max
                ok = alpha * cov_gain - (1 - alpha) * eff_loss > 0
            if ok:
                cleaned.add(sid)
                r.cleaned.append(sid)


def _eval_routes(ctx, routes):
    inst = ctx.inst
    streets = inst.streets
    cleaned_once = set()
    waste = 0.0
    for r in routes:
        cap = CAPACITY[r.vtype]
        for sid in r.cleaned:
            s = streets[sid]
            cleaned_once.add(sid)
            waste += (cap - s.req) * s.length_km
    length = sum(streets[sid].length for sid in cleaned_once)
    n_mand = sum(1 for s in ctx.mandatory if s.idx in cleaned_once)
    coverage = length / ctx.l_max
    efficiency = 1 - waste / ctx.w_max
    score = inst.alpha * coverage + (1 - inst.alpha) * efficiency
    return n_mand, score


def _eval_from_solution(ctx, solution):
    routes = solution.routes
    return _eval_routes(ctx, routes)
