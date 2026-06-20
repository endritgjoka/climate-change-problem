# Street Cleaning — Solver

An optimisation solver for the street-cleaning challenge (a Capacitated Arc
Routing Problem variant): route a fleet of cleaning vehicles from a depot to
clean as much street length as possible, clean **all mandatory** streets, waste
as little water as possible, and bring every vehicle back to the depot within a
shared time limit `T`.

## Layout

```
input/            instance files (you drop them here)
output/           submissions written here, SAME filename as the input
src/
  model.py        Instance / Street / Route / Solution data classes + constants
  parser.py       reads an instance file (robust to optional coord lines)
  graph.py        directed graph + Dijkstra (forward, reverse-to-depot, bounded)
  solver.py       greedy constructive heuristic
  writer.py       writes the submission format
  scorer.py       self-validator + score (mirrors the README rules)
main.py           CLI: solve input/* -> output/*, then self-score
tools/gen_instance.py   random solvable instance generator (for testing)
```

## Usage

```bash
python3 main.py                    # solve every file in input/, write to output/
python3 main.py input/example.txt  # solve one file
```

Each input file `input/NAME` produces `output/NAME` (identical filename, as the
validator expects). The CLI prints a self-scored report for each.

## Approach (greedy constructive)

1. **Pre-compute** shortest *return* times to the depot for every node (one
   reverse Dijkstra), used to guarantee every vehicle can get home in time.
2. **Process vehicles in ascending capacity (S → M → L).** This makes each
   vehicle naturally clean the streets it matches best — Small does Light,
   Medium does Medium — minimising water waste, while Large vehicles mop up
   Heavy and any leftover streets.
3. **Per vehicle, repeatedly pick the next street to clean:**
   - A locality-bounded Dijkstra from the current position finds nearby
     reachable streets.
   - **Mandatory streets are always preferred** (validity first), choosing the
     one with least waste, then least added time.
   - Otherwise pick the **optional** street with the best
     `objective-gain / time` ratio, skipping ones that would lower the score
     (e.g. a wasteful big-vehicle-on-light-street clean when `α` is low).
   - A street is only taken if `reach + clean + return-to-depot ≤ remaining`.
   - If nothing is reachable nearby but a serviceable mandatory street still
     exists, a full (unbounded) Dijkstra is run as a safety fallback so
     mandatory coverage is never silently dropped.

### Why it's robust on unseen instances

- The feasibility gate (`reach + clean + return ≤ remaining`) makes every
  produced route valid by construction: on time and returning to the depot.
- The ascending-capacity ordering + mandatory-first rule keeps water waste low
  and protects validity regardless of the `α` weighting.
- The locality bound (`POP_LIMIT`) keeps per-step cost roughly constant in
  graph size, so it scales to the largest instances (N≤10⁴, M≤10⁵, C≤10²).

## Tuning

`POP_LIMIT` in `src/solver.py` (default **400**) bounds how far each per-step
search explores. Smaller = faster and tends to favour nearby work (often
*better* coverage per unit time); larger = considers farther tasks. Lowering it
barely changes score but speeds large instances up several-fold.

## Self-scoring

`src/scorer.py` re-implements the README scoring rules (validity checks,
coverage, efficiency, weighted score) so solutions can be verified locally
before uploading to the official validator. On the README example it reports
`score=1.0000` (all streets cleaned, zero waste).

## Ideas to improve score next

- **Local search / 2-opt** on each route to cut wasted travel time and free
  budget for more optional streets.
- **Clean-on-traversal:** opportunistically clean cleanable streets that lie on
  a repositioning path (currently only the targeted street is cleaned).
- **Smarter optional selection** via a global savings/insertion heuristic
  instead of pure nearest-ratio greedy.
- **Vehicle-to-region assignment** (cluster streets geographically using the
  junction coordinates) to reduce overlap between vehicles.
