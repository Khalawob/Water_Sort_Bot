"""
Water Sort Puzzle Solver — A* with unknown/hidden slot support.

Strategies:
  - A* search (primary): colour-transition heuristic, fast + optimal
  - DFS (fallback): for very large puzzles where A* runs out of memory
  - Safe moves (partial): when unknowns prevent a full solve, find moves
    that reveal hidden slots or make obvious progress

Moves are returned as (src, dst, num_poured).
"""

import random
from collections import deque
from heapq import heappush, heappop

UNKNOWN = "unknown"


# ── Shared helpers ───────────────────────────────────────────────────

def is_solved(state, tube_capacity):
    for tube in state:
        if len(tube) == 0:
            continue
        if UNKNOWN in tube:
            return False
        if len(tube) != tube_capacity or len(set(tube)) != 1:
            return False
    return True


def get_top_run(tube):
    """Return (colour, count) of consecutive same-colour layers on top."""
    if not tube:
        return None, 0
    colour = tube[-1]
    if colour == UNKNOWN:
        return UNKNOWN, 1
    count = 0
    for c in reversed(tube):
        if c == colour:
            count += 1
        else:
            break
    return colour, count


def apply_move(state, src, dst, tube_capacity):
    """
    Pour top matching layers from src to dst.
    Returns (new_state, num_poured).
    Never pours unknown slots.
    """
    tubes = [list(t) for t in state]
    colour, _ = get_top_run(tubes[src])

    # Never pour unknowns
    if colour == UNKNOWN:
        return state, 0

    num_poured = 0
    while (tubes[src] and tubes[src][-1] == colour
           and len(tubes[dst]) < tube_capacity):
        tubes[dst].append(tubes[src].pop())
        num_poured += 1
    return tuple(tuple(t) for t in tubes), num_poured


def valid_moves(state, tube_capacity, restrict_empties=True):
    """Yield all legal (src_idx, dst_idx) moves."""
    n = len(state)
    for src in range(n):
        if not state[src]:
            continue
        src_colour, _ = get_top_run(state[src])

        # Can't pour unknowns
        if src_colour == UNKNOWN:
            continue

        for dst in range(n):
            if dst == src:
                continue
            if len(state[dst]) >= tube_capacity:
                continue
            if len(state[dst]) == 0:
                if restrict_empties:
                    # Don't pour a single-colour tube into empty (pointless)
                    known = [c for c in state[src] if c != UNKNOWN]
                    if len(set(known)) <= 1 and UNKNOWN not in state[src]:
                        continue
                yield src, dst
            elif state[dst][-1] == src_colour:
                yield src, dst


# ── Heuristic ────────────────────────────────────────────────────────

def heuristic(state):
    """
    Count colour transitions in each tube.
    Unknowns always count as transitions.
    """
    total = 0
    for tube in state:
        if not tube or (len(set(tube)) == 1 and UNKNOWN not in tube):
            continue
        transitions = 0
        for i in range(1, len(tube)):
            if tube[i] != tube[i - 1]:
                transitions += 1
        # Each unknown adds extra penalty — we want to reveal them
        unknowns = tube.count(UNKNOWN)
        total += transitions + unknowns
    return total


# ── A* search ────────────────────────────────────────────────────────

def solve_astar(initial_state, tube_capacity=4, max_states=1_000_000):
    """A* search with colour-transition heuristic."""
    start = tuple(tuple(t) for t in initial_state)

    if is_solved(start, tube_capacity):
        return []

    counter = 0
    queue = [(heuristic(start), 0, counter, start, [])]
    seen = set()
    seen.add(start)

    while queue:
        if len(seen) > max_states:
            return None

        f, g, _, state, moves = heappop(queue)

        if is_solved(state, tube_capacity):
            return moves

        for src, dst in valid_moves(state, tube_capacity):
            new_state, num_poured = apply_move(state, src, dst, tube_capacity)
            if num_poured == 0:
                continue

            if new_state in seen:
                continue
            seen.add(new_state)

            new_moves = moves + [(src, dst, num_poured)]
            h = heuristic(new_state)
            counter += 1
            heappush(queue, (g + 1 + h, g + 1, counter, new_state, new_moves))

    return None


# ── DFS fallback ─────────────────────────────────────────────────────

def solve_dfs(initial_state, tube_capacity=4, max_states=2_000_000,
              restrict_empties=True):
    start = tuple(tuple(t) for t in initial_state)
    if is_solved(start, tube_capacity):
        return []

    best_solution = None

    for depth_limit in [50, 100, 150, 200]:
        visited = set()
        states_explored = 0

        def dfs(state, moves, last_move=None):
            nonlocal best_solution, states_explored
            if states_explored > max_states:
                return
            if is_solved(state, tube_capacity):
                if best_solution is None or len(moves) < len(best_solution):
                    best_solution = list(moves)
                return
            if len(moves) >= depth_limit:
                return
            if best_solution and len(moves) >= len(best_solution) - 1:
                return

            canon = tuple(sorted(state))
            if canon in visited:
                return
            visited.add(canon)
            states_explored += 1

            move_list = []
            used_empty = False
            for src, dst in valid_moves(state, tube_capacity):
                if last_move and src == last_move[1] and dst == last_move[0]:
                    continue
                if restrict_empties and len(state[dst]) == 0:
                    if used_empty:
                        continue
                    used_empty = True
                new_state, num_poured = apply_move(state, src, dst, tube_capacity)
                if num_poured == 0:
                    continue
                h = heuristic(new_state)
                move_list.append((-h, src, dst, num_poured))

            move_list.sort()
            for _, src, dst, num_poured in move_list:
                new_state, _ = apply_move(state, src, dst, tube_capacity)
                moves.append((src, dst, num_poured))
                dfs(new_state, moves, (src, dst))
                moves.pop()
                if states_explored > max_states:
                    return

        print(f"    Trying depth limit {depth_limit}...")
        dfs(start, [])
        if best_solution is not None:
            return best_solution
    return None


# ── Constraint deduction ─────────────────────────────────────────────

def deduce_unknowns(state, tube_capacity=4):
    """
    Fill in forced unknowns using colour counting.

    Each colour appears exactly tube_capacity times. If a colour has been
    seen tube_capacity - 1 times, the one remaining UNKNOWN slot must be
    that colour. Repeats until no more deductions can be made (one fill-in
    can enable the next).

    Returns a new state (list of lists) and the number of slots deduced.
    """
    tubes = [list(t) for t in state]
    total_deduced = 0

    while True:
        # Count visible colours across all tubes
        colour_counts = {}
        for tube in tubes:
            for slot in tube:
                if slot != UNKNOWN:
                    colour_counts[slot] = colour_counts.get(slot, 0) + 1

        # Find colours that are one short of full count
        missing_one = [
            colour for colour, count in colour_counts.items()
            if count == tube_capacity - 1
        ]

        if not missing_one:
            break

        # Count total unknowns to detect single-unknown-remaining case
        total_unknowns = sum(t.count(UNKNOWN) for t in tubes)

        deduced_this_round = 0
        for colour in missing_one:
            # If only one unknown slot remains anywhere, it must be this colour
            if total_unknowns == 1:
                for tube in tubes:
                    if UNKNOWN in tube:
                        idx = tube.index(UNKNOWN)
                        tube[idx] = colour
                        deduced_this_round += 1
                        total_unknowns -= 1
                        break
            else:
                # Multiple unknowns remain — can we narrow down which one?
                # A tube with no unknowns can't hold the missing slot.
                # If exactly one tube has unknowns AND could contain this
                # colour, the missing instance must be there.
                candidate_tubes = []
                for i, tube in enumerate(tubes):
                    if UNKNOWN in tube:
                        candidate_tubes.append(i)

                if len(candidate_tubes) == 1:
                    # Only one tube has any unknowns — all missing colours
                    # must be in that tube. Fill the topmost unknown.
                    tube = tubes[candidate_tubes[0]]
                    idx = tube.index(UNKNOWN)
                    tube[idx] = colour
                    deduced_this_round += 1
                    total_unknowns -= 1

        if deduced_this_round == 0:
            break
        total_deduced += deduced_this_round

    if total_deduced > 0:
        print(f"  🧩 Deduced {total_deduced} hidden slot(s) via colour counting")

    return tubes, total_deduced


# ── Safe moves (when unknowns prevent full solve) ────────────────────

def find_safe_moves(initial_state, tube_capacity=4):
    """
    Reveal-focused strategy. Called each round while unknowns remain.

    Priority order:
      1. Matching reveal — pour from a tube with hidden slots below into
         a tube whose top matches the source colour. Free information
         without spending an empty tube.
      2. Empty reveal — pour from a tube with hidden slots below into an
         empty tube. Safe (always reversible) but spends an empty.
      3. Matching consolidation — pour matching colours together even if
         neither tube has hidden slots. Frees up space for future reveals.
      4. Non-reveal empty park — pour into an empty to create room,
         only as a last resort.

    Only one move per call — the loop in solve_one_level will re-screenshot
    and re-assess after each move, so we don't need to chain moves blindly.

    Returns a list of (src, dst, num_poured) moves (0 or 1 move).
    """
    state = [list(t) for t in initial_state]

    def has_hidden_below(tube):
        """Check if pouring the visible top would reveal an UNKNOWN."""
        if not tube:
            return False
        top_colour = tube[-1]
        if top_colour == UNKNOWN:
            return False
        # Walk down past the top run to see if UNKNOWN is directly below
        for i in range(len(tube) - 1, -1, -1):
            if tube[i] == top_colour:
                continue
            return tube[i] == UNKNOWN
        return False

    def do_pour(src, dst):
        """Execute a pour and return (num_poured, move_tuple)."""
        frozen = tuple(tuple(t) for t in state)
        new_frozen, num_poured = apply_move(frozen, src, dst, tube_capacity)
        if num_poured > 0:
            return num_poured, (src, dst, num_poured)
        return 0, None

    # Categorise tubes
    empty_tubes = [i for i, t in enumerate(state) if len(t) == 0]

    # Build candidate moves in priority order
    # Each candidate: (priority, src, dst)
    candidates = []

    for src in range(len(state)):
        if not state[src]:
            continue
        top_colour, top_count = get_top_run(state[src])
        if top_colour == UNKNOWN:
            continue

        reveals = has_hidden_below(state[src])

        for dst in range(len(state)):
            if dst == src:
                continue
            if len(state[dst]) >= tube_capacity:
                continue

            if len(state[dst]) == 0:
                # Pouring a fully-visible single-colour tube into empty
                # is pointless — skip it
                known = [c for c in state[src] if c != UNKNOWN]
                if len(set(known)) <= 1 and UNKNOWN not in state[src]:
                    continue

                if reveals:
                    # Priority 2: empty reveal
                    candidates.append((2, src, dst))
                else:
                    # Priority 4: non-reveal empty park
                    candidates.append((4, src, dst))

            elif state[dst][-1] == top_colour:
                if reveals:
                    # Priority 1: matching reveal (best move)
                    candidates.append((1, src, dst))
                else:
                    # Priority 3: matching consolidation
                    candidates.append((3, src, dst))

    # Sort by priority and pick the best one
    candidates.sort(key=lambda c: c[0])

    for priority, src, dst in candidates:
        num_poured, move = do_pour(src, dst)
        if num_poured > 0:
            kind = {1: "matching reveal", 2: "empty reveal",
                    3: "consolidation", 4: "empty park"}
            print(f"  Strategy: {kind.get(priority, '?')} "
                  f"(Tube {src+1} → Tube {dst+1})")
            return [move]

    return []


# ── Auto-selecting solver ────────────────────────────────────────────

def solve(initial_state, tube_capacity=4, max_states=500_000):
    """
    Solve using A* first, falling back to DFS.
    Returns list of (src, dst, num_poured) moves, or None.
    """
    num_tubes = len(initial_state)

    print(f"    Using A* solver ({num_tubes} tubes)")
    result = solve_astar(initial_state, tube_capacity, max_states=max_states * 2)
    if result is not None:
        return result

    if num_tubes > 12:
        print("    A* hit limit, trying DFS...")
        result = solve_dfs(initial_state, tube_capacity,
                           max_states=max_states * 4, restrict_empties=True)
        if result:
            return result
        print("    Retrying DFS relaxed...")
        return solve_dfs(initial_state, tube_capacity,
                         max_states=max_states * 4, restrict_empties=False)

    return None
