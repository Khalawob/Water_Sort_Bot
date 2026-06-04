"""Tests for the memory-integrity and patience changes:

  #1  poisoned-read guards (_reads_agree, _batch_exposed_unknown) and the
      corruption conditions (poisoned overlay overflows / unsolvable board).
  #2  the pure end-of-attempt retry decision (decide_retry).
  #4  select_reveal_prefix skips scoring above the empty gate.

Dual-mode: runs under pytest if installed, but also standalone with plain
`python tests/test_retry_and_memory.py`.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main
from main import (
    decide_retry, _reads_agree, _batch_exposed_unknown,
    select_reveal_prefix, PATIENCE,
)
from solver import UNKNOWN as U, solve_astar
from level_memory import LevelMemory

# ── #2: decide_retry ─────────────────────────────────────────────────


def test_retry_on_new_knowledge():
    retry, barren, reason = decide_retry(
        "stuck", learned=2, fully_mapped=False, barren_attempts=1,
        attempt=1, max_attempts=12, give_up=False)
    assert retry and barren == 0 and reason == "learned"


def test_retry_when_fully_mapped_even_if_barren():
    retry, barren, reason = decide_retry(
        "stuck", learned=0, fully_mapped=True, barren_attempts=9,
        attempt=4, max_attempts=12, give_up=False)
    assert retry and reason == "fully_mapped"


def test_patience_boundary():
    """With PATIENCE consecutive barren attempts, retry is granted until the
    incremented count reaches PATIENCE, then the level is abandoned."""
    barren = 0
    retries = 0
    for attempt in range(1, 12):
        retry, barren, reason = decide_retry(
            "stuck", learned=0, fully_mapped=False, barren_attempts=barren,
            attempt=attempt, max_attempts=12, give_up=False)
        if retry:
            assert reason == "patience"
            retries += 1
        else:
            assert reason == "give_up"
            break
    assert retries == PATIENCE - 1, f"expected {PATIENCE - 1} patience retries, got {retries}"


def test_heal_always_retries():
    retry, _, reason = decide_retry(
        "heal", learned=0, fully_mapped=False, barren_attempts=99,
        attempt=5, max_attempts=12, give_up=False)
    assert retry and reason == "heal"


def test_give_up_is_terminal():
    retry, _, reason = decide_retry(
        "stuck", learned=0, fully_mapped=True, barren_attempts=0,
        attempt=2, max_attempts=12, give_up=True)
    assert not retry and reason == "give_up"


def test_max_attempts_caps_retry():
    retry, _, reason = decide_retry(
        "stuck", learned=5, fully_mapped=False, barren_attempts=0,
        attempt=12, max_attempts=12, give_up=False)
    assert not retry and reason == "give_up"


# ── #1: poisoned-read guards ─────────────────────────────────────────


def test_reads_agree_within_tolerance():
    a = [["c1"], [U]]
    la = {"c1": (10, 20, 30)}
    b = [["c9"], [U]]               # different label, near-identical RGB
    lb = {"c9": (11, 21, 31)}
    assert _reads_agree(a, la, b, lb)


def test_reads_disagree_on_colour():
    a = [["c1"], [U]]
    la = {"c1": (10, 20, 30)}
    b = [["c9"], [U]]
    lb = {"c9": (200, 20, 30)}      # far-apart RGB
    assert not _reads_agree(a, la, b, lb)


def test_reads_disagree_on_unknown_pattern_and_shape():
    a = [["c1"], [U]]
    la = {"c1": (10, 20, 30)}
    # UNKNOWN where the other has a colour
    b = [["c1"], ["c1"]]
    assert not _reads_agree(a, la, b, la)
    # different shape
    assert not _reads_agree(a, la, [["c1"]], la)


def test_batch_exposed_unknown():
    # Pouring tube0's 'c' top reveals an UNKNOWN beneath it → exposed.
    state = (("a", U, "c"), ("c",))
    assert _batch_exposed_unknown(state, [(0, 1, 1)], 3)
    # Pouring a pure tube empties it (no UNKNOWN top revealed) → not exposed.
    assert not _batch_exposed_unknown((("a", "a"), ("a",)), [(0, 1, 1)], 4)


# ── #1: corruption conditions ────────────────────────────────────────


def test_poisoned_overlay_makes_board_unsolvable():
    """A solvable hidden layout, mislabelled by one slot, becomes unsolvable —
    the exact condition that triggers self-heal on a fully-known board."""
    true_board = (("a", "a", "a", "b"), ("b", "b", "b", "a"), ())
    assert solve_astar(true_board, 4, max_states=20_000) is not None
    # Corrupt: the buried 'b' in tube0 misremembered as 'a' (now 5 a's / 3 b's).
    poisoned = (("a", "a", "a", "a"), ("b", "b", "b", "a"), ())
    assert solve_astar(poisoned, 4, max_states=20_000) is None


def test_level_memory_roundtrip_and_delete():
    with tempfile.TemporaryDirectory() as d:
        mem = LevelMemory(path=os.path.join(d, "mem.json"))
        sig = "deadbeef"
        assert mem.count(sig) == 0
        mem.record_slot(sig, 0, 1, (10, 20, 30), 4)
        mem.record_slot(sig, 0, 2, (40, 50, 60), 4)
        assert mem.count(sig) == 2
        assert mem.get_initial_slots(sig)[(0, 1)] == (10, 20, 30)
        # Reload from disk: persistence holds.
        assert LevelMemory(path=mem.path).count(sig) == 2
        mem.delete(sig)
        assert mem.count(sig) == 0


# ── #4: select_reveal_prefix gating ──────────────────────────────────


def test_prefix_skipped_above_gate():
    """With empties above the gate, no solvability scoring runs and the full
    batch is returned unchanged."""
    reveal = [(0, 1, 1), (2, 3, 1)]
    calls = []
    orig = main.score_reveal_batch
    main.score_reveal_batch = lambda *a, **k: (calls.append(1), 1.0)[1]
    try:
        out = select_reveal_prefix((), reveal, 4, empties=2)  # gate is 1
        assert out == reveal
        assert calls == [], "scored despite being above the empty gate"
        # At/below the gate, scoring runs and the shortest fully-safe prefix wins.
        out = select_reveal_prefix((), reveal, 4, empties=1)
        assert out == reveal[:1] and calls, "expected scoring + trim at the gate"
    finally:
        main.score_reveal_batch = orig


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
