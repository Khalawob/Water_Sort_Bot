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
from collections import Counter, deque
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


def survey_visible_tops(state):
    """Return (colour_map, empties) scanning all tube tops.

    colour_map maps each visible (non-UNKNOWN) top colour to the list of
    tube indices that show it.  empties is the list of empty tube indices.
    """
    colour_map = {}
    empties = []
    for i, tube in enumerate(state):
        if not tube:
            empties.append(i)
            continue
        colour, _ = get_top_run(tube)
        if colour is None or colour == UNKNOWN:
            continue
        colour_map.setdefault(colour, []).append(i)
    return colour_map, empties


def count_colour_occurrences(state):
    """Return a dict mapping each known colour to its total slot count across all tubes."""
    counts = {}
    for tube in state:
        for slot in tube:
            if slot == UNKNOWN:
                continue
            counts[slot] = counts.get(slot, 0) + 1
    return counts


def find_matching_tops(state, tube_capacity):
    """Return sorted (src, dst) pairs where same-colour tops can pour without using an empty."""
    colour_map, _ = survey_visible_tops(state)
    working = []
    for colour, indices in colour_map.items():
        if len(indices) < 2:
            continue
        for dst in indices:
            if len(state[dst]) >= tube_capacity:
                continue
            is_pure = all(c == colour for c in state[dst])
            score = (not is_pure, -state[dst].count(colour))
            for src in indices:
                if src != dst:
                    working.append((score, src, dst))
    working.sort(key=lambda x: x[0])
    return [(src, dst) for _, src, dst in working]


def _pick_by_count(counts, candidates, pick_min):
    """Return a randomly chosen colour from candidates with the min or max count."""
    target = min(counts[c] for c in candidates) if pick_min else max(counts[c] for c in candidates)
    tied = [c for c in candidates if counts[c] == target]
    return random.choice(tied)


def plan_reveal_round(state, tube_capacity, force_park=False, prev_state=None):
    """Return (src, dst, num_poured) moves for one reveal round based on empty-tube count."""
    colour_map, empties = survey_visible_tops(state)
    moves = []

    def _free_match_moves(cur_state):
        result = []
        s = cur_state
        for src, dst in find_matching_tops(s, tube_capacity):
            new_s, n = apply_move(s, src, dst, tube_capacity)
            if n > 0:
                if prev_state is not None and new_s == prev_state:
                    continue
                src_after = new_s[src]
                empties_src = len(src_after) == 0
                reveals_hidden = bool(src_after) and src_after[-1] == UNKNOWN
                if empties_src or reveals_hidden:
                    result.append((src, dst, n))
                    s = new_s
        return result

    if len(empties) >= 2:
        counts = count_colour_occurrences(state)
        visible_colours = list(colour_map)

        # Park least-occurring colour into empties[0]
        park_dst = empties[0]
        least = _pick_by_count(counts, visible_colours, pick_min=True)
        park_src = random.choice(colour_map[least])
        n = min(get_top_run(state[park_src])[1], tube_capacity)
        moves.append((park_src, park_dst, n))

        # Consolidate most-occurring colour (2+ tops) into empties[1]
        cons_dst = empties[1]
        multi_top = [c for c in colour_map if len(colour_map[c]) >= 2 and c != least]
        if multi_top:
            most = _pick_by_count(counts, multi_top, pick_min=False)
            cur_state = state
            for src in colour_map[most]:
                if len(cur_state[cons_dst]) >= tube_capacity:
                    break
                cur_state, n = apply_move(cur_state, src, cons_dst, tube_capacity)
                if n > 0:
                    moves.append((src, cons_dst, n))

    elif len(empties) == 1:
        if not force_park:
            matches = _free_match_moves(state)
            if matches:
                return matches
        counts = count_colour_occurrences(state)
        least = _pick_by_count(counts, list(colour_map), pick_min=True)
        park_src = random.choice(colour_map[least])
        n = min(get_top_run(state[park_src])[1], tube_capacity)
        moves.append((park_src, empties[0], n))

    else:
        # 0 empties: score by reveal proximity, filter anti-shuffle
        candidates = []
        for src, dst in find_matching_tops(state, tube_capacity):
            new_s, n = apply_move(state, src, dst, tube_capacity)
            if n == 0:
                continue
            if prev_state is not None and new_s == prev_state:
                continue
            src_after = new_s[src]
            empties_src = len(src_after) == 0
            reveals_hidden = bool(src_after) and src_after[-1] == UNKNOWN
            if not (empties_src or reveals_hidden):
                continue
            depth = next(
                (i for i, c in enumerate(reversed(state[src])) if c == UNKNOWN),
                float('inf'),
            )
            candidates.append((depth, src, dst, n))
        candidates.sort()
        return [(src, dst, n) for _, src, dst, n in candidates[:3]]

    return moves


def find_reclaim_moves(state, tube_capacity):
    """Return (src, dst, num_poured) moves that pour single-colour parking tubes into matching tops.

    Fully-emptying moves (reclaiming the tube as empty) are sorted first.
    """
    colour_map, _ = survey_visible_tops(state)
    candidates = []
    for i, tube in enumerate(state):
        if not tube or UNKNOWN in tube or len(set(tube)) != 1:
            continue
        colour = tube[0]
        for dst in colour_map.get(colour, []):
            if dst == i or len(state[dst]) >= tube_capacity:
                continue
            _, num_poured = apply_move(state, i, dst, tube_capacity)
            if num_poured > 0:
                fully_empties = num_poured == len(tube)
                candidates.append((not fully_empties, -num_poured, i, dst, num_poured))
    candidates.sort(key=lambda x: (x[0], x[1]))
    return [(src, dst, n) for _, _, src, dst, n in candidates]


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


# ── Determinization ───────────────────────────────────────────────────

def _fill_unknowns(state, color_pool):
    pool_iter = iter(color_pool)
    new_state = []
    for tube in state:
        new_tube = []
        for slot in tube:
            if slot == UNKNOWN:
                new_tube.append(next(pool_iter))
            else:
                new_tube.append(slot)
        new_state.append(tuple(new_tube))
    return tuple(new_state)


def pick_best_move_by_determinization(state, tube_capacity, num_samples=20):
    """
    Estimate the best next move(s) under hidden-slot uncertainty by sampling
    random completions of UNKNOWN slots and running A* on each.

    Returns up to 3 (src, dst, num_poured) tuples that win the most votes
    across successful sample solves. Returns [] if no sample produced a solution.
    """
    visible_counts = count_colour_occurrences(state)

    for colour, cnt in visible_counts.items():
        if cnt > tube_capacity:
            return []

    pool_base = []
    for colour, cnt in visible_counts.items():
        needed = tube_capacity - cnt
        if needed > 0:
            pool_base.extend([colour] * needed)

    num_unknowns = sum(slot == UNKNOWN for tube in state for slot in tube)

    if num_unknowns == 0:
        return []

    if len(pool_base) != num_unknowns:
        return []

    solutions = []
    for _ in range(num_samples):
        pool = pool_base[:]
        random.shuffle(pool)
        filled = _fill_unknowns(state, pool)
        sol = solve_astar(filled, tube_capacity)
        if sol:
            solutions.append(sol[:3])

    if not solutions:
        return []

    first_moves = [s[0] for s in solutions if s]
    if not first_moves:
        return []

    best_first = Counter(first_moves).most_common(1)[0][0]
    result = [best_first]

    sharing_first = [s for s in solutions if s and s[0] == best_first]
    second_candidates = [s[1] for s in sharing_first if len(s) >= 2]
    if second_candidates:
        top_second, top_second_count = Counter(second_candidates).most_common(1)[0]
        if top_second_count > len(sharing_first) / 2:
            result.append(top_second)

            sharing_two = [s for s in sharing_first if len(s) >= 3 and s[1] == top_second]
            third_candidates = [s[2] for s in sharing_two]
            if third_candidates:
                top_third, top_third_count = Counter(third_candidates).most_common(1)[0]
                if top_third_count > len(sharing_two) / 2:
                    result.append(top_third)

    return result


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


# ── Safe moves (when unknowns prevent full solve) ────────────────────

def find_safe_moves(initial_state, tube_capacity=4, prev_state=None):
    """
    When the puzzle can't be fully solved (hidden slots), find moves
    that make obvious progress:
      1. Complete a tube (fill it with one colour)
      2. Pour onto matching colour
      3. Pour to reveal a hidden slot (empty a tube above unknowns)

    Returns a short list of (src, dst, num_poured) moves.
    """
    state = tuple(tuple(t) for t in initial_state)
    safe_moves = []

    # Score each possible move — use permissive generator so single-colour
    # tubes can pour into empty slots (may be the only way to make progress
    # when unknowns are blocking everything else)
    scored = []
    for src, dst in valid_moves(state, tube_capacity, restrict_empties=False):
        new_state, num_poured = apply_move(state, src, dst, tube_capacity)
        if num_poured == 0:
            continue
        if prev_state is not None and new_state == prev_state:
            continue

        score = 0
        src_tube_after = new_state[src]
        dst_tube_after = new_state[dst]

        # Completing a tube = great
        if (len(dst_tube_after) == tube_capacity
                and len(set(dst_tube_after)) == 1
                and UNKNOWN not in dst_tube_after):
            score += 100

        # Emptying a tube = great (especially reveals unknowns)
        if len(src_tube_after) == 0:
            score += 80

        # Revealing an unknown (top of source becomes unknown after pour)
        if src_tube_after and src_tube_after[-1] == UNKNOWN:
            score += 60

        # Pouring onto matching colour = good
        if state[dst] and state[dst][-1] == state[src][-1]:
            score += 30

        # Pouring more balls at once = efficient
        score += num_poured * 5

        if score > 0:
            scored.append((score, src, dst, num_poured))

    scored.sort(reverse=True)

    # Fallback: when all tops are UNKNOWN, pour into empty tubes to reveal hidden balls.
    # The real game always allows any tube → empty tube regardless of the top colour.
    if not scored:
        empty_dsts = [i for i, t in enumerate(state) if len(t) == 0]
        for src, tube in enumerate(state):
            if not tube or not empty_dsts:
                break
            top, _ = get_top_run(tube)
            if top == UNKNOWN:
                dst = empty_dsts.pop(0)
                scored.append((10, src, dst, 1))

    # Take the best moves but avoid conflicts
    used_srcs = set()
    for score, src, dst, num_poured in scored:
        if src in used_srcs:
            continue
        safe_moves.append((src, dst, num_poured))
        used_srcs.add(src)

        # Limit to a few moves — we'll re-screenshot after
        if len(safe_moves) >= 3:
            break

    return safe_moves


# ── Constraint-based hidden-slot deduction ───────────────────────────

def deduce_hidden_slots(state, tube_capacity):
    """
    Replace UNKNOWN slots whose value is forced by colour-count constraints.
    Each colour appears exactly tube_capacity times total; the unknowns fill
    the remainder. Any slot whose value is identical across all valid
    assignments is replaced; ambiguous slots remain UNKNOWN.
    """
    unknown_positions = [
        (ti, si)
        for ti, tube in enumerate(state)
        for si, slot in enumerate(tube)
        if slot == UNKNOWN
    ]
    if not unknown_positions:
        return state

    visible_counts = {}
    for tube in state:
        for slot in tube:
            if slot != UNKNOWN:
                visible_counts[slot] = visible_counts.get(slot, 0) + 1

    needed = {c: tube_capacity - visible_counts.get(c, 0) for c in visible_counts}
    needed = {c: n for c, n in needed.items() if n > 0}

    # If visible colours don't account for all unknowns, a colour is 100%
    # hidden and we can't build a complete constraint system — skip.
    if not needed or sum(needed.values()) != len(unknown_positions):
        return state

    try:
        from constraint import Problem
    except ImportError:
        return state

    problem = Problem()
    var_names = [f"u{i}" for i in range(len(unknown_positions))]

    for var in var_names:
        problem.addVariable(var, list(needed.keys()))

    for colour, count in needed.items():
        problem.addConstraint(
            lambda *args, col=colour, cnt=count: args.count(col) == cnt,
            var_names,
        )

    solutions = problem.getSolutions()
    if not solutions:
        return state

    new_state = [list(tube) for tube in state]
    for i, (ti, si) in enumerate(unknown_positions):
        values = {sol[var_names[i]] for sol in solutions}
        if len(values) == 1:
            new_state[ti][si] = values.pop()

    return tuple(tuple(tube) for tube in new_state)


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
