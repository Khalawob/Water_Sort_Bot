# Architectural Patterns

## Pipeline Architecture

The bot follows a strict linear pipeline per level:

```
screenshot() → auto_calibrate() → read_tubes() → solve() → execute_move_list()
```

Each stage is a pure function consuming the previous stage's output. `main.py:109` (`solve_one_level`) orchestrates this pipeline per round; it re-runs all stages from the screenshot step on each iteration (to handle hidden-slot reveals).

## Separation of Concerns

Each module owns exactly one stage with no cross-stage dependencies (all imports are one-directional down the pipeline):

| Module | Responsibility |
|---|---|
| `capture.py` | ADB I/O, scrcpy process management, window detection |
| `auto_calibrate.py` | Vision-based tube geometry detection from a screenshot |
| `screen_reader.py` | Pixel colour reading and hidden-slot detection |
| `solver.py` | Pure search algorithms — no I/O, no imports from other project modules |
| `automator.py` | Device coordinate → screen coordinate mapping and tap execution |
| `main.py` | CLI entry point and pipeline orchestration |

`solver.py` has zero project imports, making it independently testable.

## Config as Shared State

`config.json` is the single source of truth for all calibration data. Its `tubes[].sample_points` list — pixel coordinates on the device — is the contract between pipeline stages:

- `auto_calibrate.py:255` produces it per-frame
- `screen_reader.py:92` reads colours from those points
- `automator.py:103` derives tap zones from those same points

Config is regenerated from the screenshot each round (`main.py:132`) but preserves `next_button` across rounds (`main.py:100`).

## Fallback Chains

Three independent fallback chains handle failure gracefully:

**Screenshot capture** (`capture.py:148` → `capture.py:178`):  
ADB fast pipe (`exec-out screencap -p`) → two-step legacy pull (`screencap` + `adb pull`)

**Solver** (`solver.py:290`):  
A* with `max_states=1_000_000` → DFS with depth limits [50, 100, 150, 200] (for >12 tubes only)

**Auto-calibration** (`main.py:93`):  
Fresh per-frame detection → saved config from previous level

## UNKNOWN Sentinel

`UNKNOWN = "unknown"` (defined at `solver.py:16`, mirrored at `screen_reader.py:32`) is a string sentinel threaded through the entire pipeline to represent hidden/unrevealed ball slots. The solver never pours unknowns (`solver.py:57`), the heuristic penalises them (`solver.py:112`), and when a full solve is impossible, `find_safe_moves()` (`solver.py:223`) scores moves by how many unknowns they reveal.

## Module-Level Lazy Caches

Two module-level singletons cache expensive lookups across calls within the same process run:

- `automator.py:26`: `_cached_mapping` — scrcpy window bounds + scale factors; refreshed before each move batch via `refresh_mapping()`
- `capture.py:196`: `_device_resolution` — device screen dimensions from `adb shell wm size`

Both use `None` as the uninitialized sentinel and recompute on first use.

## Device Coordinate Abstraction

All game coordinates (tube positions, next-button location) are stored in **device space** (native Android pixels). Conversion to screen coordinates happens only at tap time in `automator.py:62` (`_device_to_screen`), using the cached scale factors. This means `config.json` is independent of the scrcpy window size or position.

## Human-Like Input Randomisation

Tap positions and timing are randomised at two levels, both in `automator.py`:

- **Position jitter** (`automator.py:129`): ±8–12px around the tube center, scaled to tube spread
- **Timing jitter** (`automator.py:136`): `base ± 0.15s` via `human_delay()`; pour wait scales linearly with `num_poured` (`0.83 + 0.52 * n`)

## Immutable State in Solver

The solver represents puzzle state as `tuple(tuple(t) for t in state)` — a tuple of tuples. This makes states hashable for the `seen` set in A* and the `visited` set in DFS, preventing revisits without needing equality checks.
