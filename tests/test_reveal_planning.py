"""Tests for the reveal-planning changes:

  #3  plan_reveal_maximize prefers the most-accumulated matching destination.
  #2c plan_reveal_maximize diverges across seeds (randomized tie-breaking).
  #4  score_reveal_batch scores a batch by solvability preservation.

Dual-mode: runs under pytest if installed, but also standalone with plain
`python tests/test_reveal_planning.py`.
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver import UNKNOWN, plan_reveal_maximize, score_reveal_batch

U = UNKNOWN


# ── #3: purposeful reveal destinations ───────────────────────────────

def test_maximize_prefers_most_accumulated_dest():
    """Two matching destinations for the same revealed colour → the fuller tube
    wins, deterministically, across many random tie-break draws."""
    state = (
        ('a',),                 # 0  dst: 1 'a'
        ('a', 'a'),             # 1  dst: 2 'a'  -> preferred
        ('x', U, 'a'),          # 2  src: top 'a' sits on an UNKNOWN
    )
    for seed in range(25):
        random.seed(seed)
        moves = plan_reveal_maximize(state, 4)
        assert moves, f"seed {seed}: expected a reveal move"
        # The single source must pour into the more-accumulated tube (index 1).
        assert moves[0] == (2, 1, 1), f"seed {seed}: chose {moves[0]}, want (2,1,1)"


def test_maximize_uses_empty_only_without_matching_dest():
    """When no matching top exists, an empty is used; when one exists, it wins."""
    only_empty = (
        (),                     # 0  empty
        ('b', U, 'a'),          # 1  src: top 'a', no other 'a' top anywhere
    )
    random.seed(0)
    moves = plan_reveal_maximize(only_empty, 4)
    assert moves and moves[0] == (1, 0, 1), f"expected pour into empty, got {moves}"

    with_match = (
        (),                     # 0  empty (last resort)
        ('a', 'a'),             # 1  matching dst
        ('b', U, 'a'),          # 2  src
    )
    for seed in range(15):
        random.seed(seed)
        moves = plan_reveal_maximize(with_match, 4)
        assert moves and moves[0] == (2, 1, 1), \
            f"seed {seed}: matching dst should beat empty, got {moves}"


# ── #2c: randomized divergence ───────────────────────────────────────

def test_maximize_diverges_across_seeds():
    """Two symmetric sources (equal top-run) must be ordered differently under
    different seeds — otherwise patience retries would replay the same path."""
    state = (
        ('p', U, 'a'),          # 0  src A -> dst 2
        ('q', U, 'b'),          # 1  src B -> dst 3
        ('a',),                 # 2
        ('b',),                 # 3
    )
    orders = set()
    for seed in range(30):
        random.seed(seed)
        orders.add(tuple(plan_reveal_maximize(state, 4)))
    assert len(orders) > 1, f"maximizer never diverged across seeds: {orders}"


# ── #4: score_reveal_batch ───────────────────────────────────────────

# Board (cap 3) with two solvable completions of the two hidden slots. One move
# preserves solvability in both worlds; another strands one of them.
_BASE = (('a', 'b', 'a'), (U, U, 'b'), ())


def test_score_safe_batch_is_one():
    """A move that keeps every solvable completion solvable scores 1.0
    (verified: (1,2,1) is solvability-preserving in both worlds)."""
    random.seed(0)
    s = score_reveal_batch(_BASE, [(1, 2, 1)], 3, num_samples=40)
    assert s == 1.0, f"safe batch scored {s}, expected 1.0"


def test_score_stranding_batch_below_one():
    """A move that strands one of the two solvable worlds scores < 1.0 — stable
    across seeds (both completions are sampled out of 40 draws)."""
    for seed in range(5):
        random.seed(seed)
        s = score_reveal_batch(_BASE, [(0, 2, 1)], 3, num_samples=40)
        assert s is not None and s < 1.0, \
            f"seed {seed}: stranding batch scored {s}, expected < 1.0"


def test_score_returns_none_when_unsamplable():
    """Structurally inconsistent board (a colour overflows capacity) → None,
    so callers fall back to the full reveal batch."""
    bad = (('a', 'a', 'a', 'a', 'a'),)
    assert score_reveal_batch(bad, [], 4) is None


# ── Standalone runner ────────────────────────────────────────────────

def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed.")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
