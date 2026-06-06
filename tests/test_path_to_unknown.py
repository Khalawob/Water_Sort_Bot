"""Tests for find_path_to_unknown — the BFS that exposes a buried UNKNOWN.

Unlike the A* solver, this search only needs SOME tube's top to become UNKNOWN.
It guarantees the shortest exposing path and may shuffle balls across multiple
tubes to clear room for a peel.

Dual-mode: runs under pytest if installed, but also standalone with plain
`python tests/test_path_to_unknown.py` (pytest is not a project dependency).
"""

import os
import sys

# Make `import solver` work when run as `python tests/test_path_to_unknown.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver import UNKNOWN, find_path_to_unknown, apply_move

U = UNKNOWN
CAP = 4


def _replay(state, moves, capacity=CAP):
    """Apply (src, dst, n) moves in order, returning the final state."""
    cur = tuple(tuple(t) for t in state)
    for src, dst, _n in moves:
        cur, _ = apply_move(cur, src, dst, capacity)
    return cur


def _has_exposed_unknown(state):
    return any(t and t[-1] == U for t in state)


# ── 1: already exposed ───────────────────────────────────────────────

def test_already_exposed_returns_empty():
    """A tube already topped by UNKNOWN needs no moves → []."""
    state = (('a', U), ('b', 'b'), ())
    assert find_path_to_unknown(state, CAP) == []


# ── 2: no unknowns at all ────────────────────────────────────────────

def test_no_unknowns_returns_none():
    """A fully-known board can never expose an UNKNOWN → None."""
    state = (('a',), ('a',), ())
    assert find_path_to_unknown(state, CAP) is None


# ── 3: one-move peel ─────────────────────────────────────────────────

def test_one_move_peel():
    """A single known ball sitting directly on an UNKNOWN, with an empty tube
    available → exactly one move that exposes the hidden slot."""
    state = (
        (U, 'a'),   # 0: 'a' on top of a hidden slot
        (),         # 1: empty destination
    )
    moves = find_path_to_unknown(state, CAP)
    assert moves is not None
    assert len(moves) == 1, f"expected a 1-move peel, got {moves}"
    final = _replay(state, moves)
    assert _has_exposed_unknown(final), f"move did not expose an unknown: {final}"


# ── 4: two-move dig ──────────────────────────────────────────────────

def test_two_move_dig():
    """An UNKNOWN buried under two differently-coloured layers needs two pours
    (one per colour) to reach — BFS returns the shortest such sequence."""
    state = (
        (U, 'b', 'a'),   # 0: 'a' over 'b' over a hidden slot
        ('a', 'a'),      # 1: matching destination for 'a'
        ('b',),          # 2: matching destination for 'b'
        (),              # 3: spare empty
    )
    moves = find_path_to_unknown(state, CAP)
    assert moves is not None
    assert len(moves) == 2, f"expected a 2-move dig, got {moves}"
    final = _replay(state, moves)
    assert _has_exposed_unknown(final), f"dig did not expose an unknown: {final}"


# ── 5: returned moves are valid and shortest ─────────────────────────

def test_returned_moves_valid_and_shortest():
    """Replaying the path must legally expose an unknown, and its length must
    equal the hand-computed minimum for the board."""
    state = (
        (U, 'b', 'a'),   # unknown buried under two layers → minimum 2 pours
        ('a', 'a'),
        ('b',),
        (),
    )
    moves = find_path_to_unknown(state, CAP)
    assert moves is not None
    # Every move pours a positive amount (apply_move would return 0 otherwise).
    cur = tuple(tuple(t) for t in state)
    for src, dst, _n in moves:
        nxt, poured = apply_move(cur, src, dst, CAP)
        assert poured > 0, f"move {(src, dst)} poured nothing on {cur}"
        cur = nxt
    assert _has_exposed_unknown(cur)
    assert len(moves) == 2, f"shortest dig is 2 moves, got {len(moves)}"


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
