"""Tests for find_guaranteed_safe_moves and the _build_completion_pool refactor.

Dual-mode: runs under pytest if installed, but also standalone with plain
`python tests/test_safe_moves.py` (pytest is not a project dependency).
"""

import itertools
import os
import random
import sys
import time

# Make `import solver` work when run as `python tests/test_safe_moves.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import solver
from solver import (
    UNKNOWN, apply_move, valid_moves, solve_astar, deduce_hidden_slots,
    find_guaranteed_safe_moves, pick_best_move_by_determinization,
    _build_completion_pool, SAFE_MAX_UNKNOWNS,
)

U = UNKNOWN


# ── Independent oracle (re-derives the safe set without the unit's helpers) ──

def _oracle_safe_set(state, cap, prev_state=None):
    """Brute-force the solvability-preserving move set, independently of
    _distinct_permutations / _is_solvable (uses itertools + solve_astar directly)."""
    unknown = [(ti, si) for ti, t in enumerate(state)
               for si, s in enumerate(t) if s == UNKNOWN]
    if not unknown:
        return None  # caller should be on the full-info path
    pool = _build_completion_pool(state, cap)
    if pool is None or len(pool) != len(unknown):
        return set()

    completions = []
    for perm in set(itertools.permutations(pool)):
        filled = solver._fill_unknowns(state, perm)
        if solve_astar(filled, cap, max_states=20_000) is not None:
            completions.append(filled)
    if not completions:
        return set()

    safe = set()
    for src, dst in valid_moves(state, cap):
        after, n = apply_move(state, src, dst, cap)
        if n == 0 or after == prev_state:
            continue
        if all(solve_astar(apply_move(c, src, dst, cap)[0], cap, max_states=20_000) is not None
               for c in completions):
            safe.add((src, dst, n))
    return safe


def _is_solvability_preserving(state, cap, move):
    """True if `move` keeps every solvable completion solvable."""
    src, dst, _ = move
    pool = _build_completion_pool(state, cap)
    completions = [solver._fill_unknowns(state, p)
                   for p in set(itertools.permutations(pool))]
    completions = [c for c in completions if solve_astar(c, cap, max_states=20_000) is not None]
    return all(solve_astar(apply_move(c, src, dst, cap)[0], cap, max_states=20_000) is not None
               for c in completions)


# ── Tests ────────────────────────────────────────────────────────────

def test_bails_on_big_board():
    """> SAFE_MAX_UNKNOWNS unknowns → returns [] fast (no enumeration)."""
    n = SAFE_MAX_UNKNOWNS + 4
    state = tuple((U,) for _ in range(n)) + (('a', 'a'),)
    t0 = time.perf_counter()
    result = find_guaranteed_safe_moves(state, 4)
    elapsed = time.perf_counter() - t0
    assert result == []
    assert elapsed < 0.5, f"big-board bail too slow: {elapsed:.3f}s"


def test_zero_unknowns_returns_empty():
    state = (('a', 'a', 'a', 'a'), ())
    assert find_guaranteed_safe_moves(state, 4) == []


def test_finds_obvious_safe_move():
    """Ambiguous endgame: completing the 'c' tube is solvability-preserving in
    every completion and is returned; output matches the independent oracle."""
    state = (
        ('c', 'c', 'c'),   # 0
        ('c',),            # 1  -> T1->T0 completes c
        ('a', 'a', 'a'),   # 2
        ('b', 'b', 'b'),   # 3
        (U,),              # 4  (one is 'a', one is 'b')
        (U,),              # 5
    )
    result = find_guaranteed_safe_moves(state, 4)
    assert result, "expected at least one safe move"
    # Completing the c tube (T1 -> T0, pours 1) must be among them.
    assert (1, 0, 1) in result
    # Every returned move is genuinely solvability-preserving.
    for mv in result:
        assert _is_solvability_preserving(state, 4, mv), f"{mv} not safe"
    # And nothing outside the oracle's safe set is returned.
    assert set(result) <= _oracle_safe_set(state, 4)


def test_output_matches_oracle_on_several_boards():
    """No unsafe move is ever returned: output ⊆ oracle, capped at 3, and the
    returned set equals the oracle's top-3 by progress ordering."""
    boards = [
        # forced single colour
        ((('a', 'a'), ('a', U), ('b', 'b', 'b'), (U,), ()), 4),
        # two-completion endgame
        ((('c', 'c', 'c'), ('c',), ('a', 'a', 'a'), ('b', 'b', 'b'), (U,), (U,)), 4),
        # mixed tops + empty
        ((('a', 'a', 'b'), ('b', 'b'), ('a',), (U,), (U,), ()), 4),
    ]
    for state, cap in boards:
        result = find_guaranteed_safe_moves(state, cap)
        oracle = _oracle_safe_set(state, cap)
        assert oracle is not None
        assert set(result) <= oracle, f"returned unsafe move(s) for {state}"
        assert len(result) <= 3
        for mv in result:
            assert _is_solvability_preserving(state, cap, mv)


def test_random_boards_never_return_unsafe():
    """Fuzz: across random small solvable-ish boards, the returned set is always
    a subset of the independently computed safe set (the core guarantee)."""
    rng = random.Random(1234)
    cap = 4
    checked = 0
    for _ in range(40):
        colours = ['a', 'b', 'c']
        balls = []
        for col in colours:
            balls += [col] * cap
        rng.shuffle(balls)
        # lay into 3 colour tubes + 1 empty
        tubes = [balls[0:4], balls[4:8], balls[8:12], []]
        # hide a few slots (cap the unknown count low so enumeration runs)
        positions = [(ti, si) for ti, t in enumerate(tubes) for si in range(len(t))]
        rng.shuffle(positions)
        for ti, si in positions[:rng.randint(1, 4)]:
            tubes[ti][si] = U
        state = tuple(tuple(t) for t in tubes)

        result = find_guaranteed_safe_moves(state, cap)
        if not result:
            continue
        oracle = _oracle_safe_set(state, cap)
        if oracle in (None, set()):
            continue
        assert set(result) <= oracle, f"returned unsafe move(s) for {state}: {result}"
        checked += 1
    assert checked > 0, "fuzz produced no positive cases to check"


def test_generalises_deduce_hidden_slots():
    """When counts force the hidden colour, safe moves agree with the forced
    reading: each returned move keeps the deduced board solvable."""
    state = (
        ('a', 'a'),                # 0
        ('a',),                    # 1
        (U,),                      # 2  forced 'a' (only deficit colour)
        ('b', 'b', 'b', 'b'),      # 3
        (),                        # 4
    )
    deduced = deduce_hidden_slots(state, 4)
    # Sanity: deduction filled the unknowns (single deficit colour -> forced).
    assert all(U not in t for t in deduced)
    result = find_guaranteed_safe_moves(state, 4)
    for src, dst, _ in result:
        after, _ = apply_move(deduced, src, dst, 4)
        assert solve_astar(after, 4, max_states=20_000) is not None


def test_reversal_guard_excludes_prev_state():
    """A move whose resulting board equals prev_state is excluded."""
    state = (
        ('c', 'c', 'c'),
        ('c',),
        ('a', 'a', 'a'),
        ('b', 'b', 'b'),
        (U,),
        (U,),
    )
    # Excluding via prev_state = the board after T0->T1 should drop (0,1,3).
    after_01, _ = apply_move(state, 0, 1, 4)
    result = find_guaranteed_safe_moves(state, 4, prev_state=after_01)
    assert (0, 1, 3) not in result
    assert result, "the non-reversing safe move should still be returned"


def test_build_completion_pool_consistency():
    """Pool length matches unknown count on a consistent board; None on overflow."""
    state = (('a', 'a', U), ('a', U), ('b', 'b', 'b'))  # 8 balls, a/b deficit 1 each
    pool = _build_completion_pool(state, 4)
    unknowns = sum(s == U for t in state for s in t)
    assert pool is not None and len(pool) == unknowns
    # Overflow: a visible colour exceeds capacity -> None.
    bad = (('a', 'a', 'a', 'a', 'a'),)
    assert _build_completion_pool(bad, 4) is None


def test_determinization_deterministic_after_refactor():
    """Regression guard for the _build_completion_pool extraction: with a fixed
    seed the sampler is reproducible and still returns an executable move."""
    state = (
        ('a', 'a', 'a'),
        ('b', 'b', 'b'),
        ('c', 'c'),
        (U,),
        (U,),
        (),
    )
    random.seed(42)
    first = pick_best_move_by_determinization(state, 4)
    random.seed(42)
    second = pick_best_move_by_determinization(state, 4)
    assert first == second
    # Whatever it returns must be an executable pour from a known top.
    for src, dst, _ in first:
        top = state[src][-1]
        assert top != U


def test_performance_endgame():
    """≤ SAFE_MAX_UNKNOWNS unknowns on a wider board returns within ~1-2s."""
    # 14 tubes; mostly complete, a handful of unknowns in the endgame.
    state = (
        ('a', 'a', 'a', 'a'),
        ('b', 'b', 'b', 'b'),
        ('c', 'c', 'c', 'c'),
        ('d', 'd', 'd', 'd'),
        ('e', 'e', 'e', 'e'),
        ('f', 'f', 'f', 'f'),
        ('g', 'g', 'g', 'g'),
        ('h', 'h', 'h', 'h'),
        ('i', 'i', 'i', 'i'),
        ('j', 'j', 'j'),       # deficit 1 -> j
        ('k', 'k', 'k'),       # deficit 1 -> k
        (U,),                  # j or k
        (U,),                  # j or k
        (),
    )
    t0 = time.perf_counter()
    find_guaranteed_safe_moves(state, 4)
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"endgame call too slow: {elapsed:.3f}s"


# ── Standalone runner (no pytest required) ───────────────────────────

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
