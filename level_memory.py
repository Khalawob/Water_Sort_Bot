"""
Level Memory — cross-restart learning of hidden-slot colours.

The game is deterministic on restart: the same level always has the identical
hidden layout. This module persists discovered hidden-slot colours (keyed by
RGB, never by the unstable ``colour_N`` labels) so the solver accumulates
knowledge across restart attempts and across program runs.

Two pieces:
  - ``LevelMemory``  — JSON persistence keyed by a stable board signature.
  - ``AttemptSim``   — an origin-tracking simulation of one attempt that
                       reconciles the planned board against what was actually
                       read, attributing newly-revealed colours to the slots
                       they originated from.
"""

import hashlib
import json
from pathlib import Path

# Mirror the sentinel rather than importing, keeping dependencies one-directional
# (same pattern as screen_reader.py).
UNKNOWN = "unknown"

MEMORY_PATH = Path(__file__).parent / "level_memory.json"


def colour_distance(c1, c2):
    """Euclidean distance between two RGB tuples."""
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5


# ── Persistence ──────────────────────────────────────────────────────

class LevelMemory:
    """Persistent store of originally-hidden slot colours, per level signature.

    On-disk schema::

        { signature: { "initial_slots": { "tube,depth": [r,g,b] },
                       "capacity": N } }
    """

    def __init__(self, path=None):
        self.path = Path(path) if path else MEMORY_PATH
        self.data = self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    @staticmethod
    def compute_signature(tubes_labels, label_to_rgb, tube_capacity):
        """Stable hash of a round-1 raw read identifying the level.

        Uses tube count, capacity, and each tube's visible top descriptor:
        "empty", "unknown", or the top colour's RGB tuple. Does not rely on the
        in-game level number.
        """
        descriptors = []
        for tube in tubes_labels:
            if not tube:
                descriptors.append("empty")
            elif tube[-1] == UNKNOWN:
                descriptors.append("unknown")
            else:
                descriptors.append(tuple(label_to_rgb[tube[-1]]))
        key = repr((len(tubes_labels), tube_capacity, tuple(descriptors)))
        return hashlib.sha1(key.encode()).hexdigest()

    def get_initial_slots(self, signature):
        """Return learned slots as ``{(tube, depth): (r, g, b)}``."""
        entry = self.data.get(signature)
        if not entry:
            return {}
        result = {}
        for key, rgb in entry.get("initial_slots", {}).items():
            tube_str, depth_str = key.split(",")
            result[(int(tube_str), int(depth_str))] = tuple(rgb)
        return result

    def record_slot(self, signature, tube, depth, rgb, capacity):
        """Record a revealed originally-hidden slot's RGB and persist."""
        entry = self.data.setdefault(
            signature, {"initial_slots": {}, "capacity": capacity}
        )
        entry["capacity"] = capacity
        entry["initial_slots"][f"{tube},{depth}"] = [int(c) for c in rgb]
        self._save()

    def delete(self, signature):
        """Remove a level's entry and persist. No-op if the signature is absent."""
        if signature in self.data:
            del self.data[signature]
            self._save()

    def count(self, signature):
        """Number of learned slots for the signature (0 if unseen)."""
        if not signature:
            return 0
        entry = self.data.get(signature)
        if not entry:
            return 0
        return len(entry.get("initial_slots", {}))


# ── Origin-tracking simulation ───────────────────────────────────────

class AttemptSim:
    """Simulated board for one attempt.

    Each ball is ``{"colour": rgb_tuple-or-UNKNOWN, "origin": (tube, depth)}``.
    Seeded from a round-1 read (every ball gets its position as origin, with any
    already-learned colours filled in). Moves are mirrored onto the sim; after
    each round the real read is reconciled against the sim to attribute newly
    revealed colours to their origin slot.
    """

    def __init__(self, tubes_labels, label_to_rgb, learned_slots):
        self.valid = True
        self.board = []
        for ti, tube in enumerate(tubes_labels):
            stack = []
            for depth, label in enumerate(tube):
                origin = (ti, depth)
                if label == UNKNOWN:
                    colour = learned_slots.get(origin, UNKNOWN)
                else:
                    colour = tuple(label_to_rgb[label])
                stack.append({"colour": colour, "origin": origin})
            self.board.append(stack)

    def apply_move(self, src, dst, num_poured):
        """Mirror a real pour: move the top ``num_poured`` balls src→dst."""
        for _ in range(num_poured):
            if not self.board[src]:
                break
            self.board[dst].append(self.board[src].pop())

    def reconcile(self, tubes_labels, label_to_rgb):
        """Attribute newly-revealed colours.

        Returns a list of ``((tube, depth), rgb)`` for balls whose colour was
        UNKNOWN in the sim but is now visible in the read. If any tube's sim
        length diverges from the read length, execution drifted from the plan:
        mark the sim invalid and attribute nothing (conservative by design).
        """
        if not self.valid:
            return []

        for ti, tube in enumerate(tubes_labels):
            if len(self.board[ti]) != len(tube):
                self.valid = False
                return []

        revealed = []
        for ti, tube in enumerate(tubes_labels):
            for depth, label in enumerate(tube):
                if label == UNKNOWN:
                    continue
                ball = self.board[ti][depth]
                if ball["colour"] == UNKNOWN:
                    rgb = tuple(label_to_rgb[label])
                    ball["colour"] = rgb
                    revealed.append((ball["origin"], rgb))
        return revealed
