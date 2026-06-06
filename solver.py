"""
Water Sort Puzzle Solver — A* with unknown/hidden slot support.

Strategies:
  - A* search (primary): colour-transition heuristic, fast + optimal
  - DFS (fallback): for very large puzzles where A* runs out of memory
  - Safe moves (partial): when unknowns prevent a full solve, find moves
    that reveal hidden slots or make obvious progress

Moves are returned as (src, dst, num_poured).
"""

import math
import random
from collections import Counter, deque
from functools import lru_cache
from heapq import heappush, heappop

UNKNOWN = "unknown"

# Guaranteed-safe-move enumeration tuning (see find_guaranteed_safe_moves).
SAFE_MAX_UNKNOWNS = 5       # only enumerate at/below this many unknowns; 6-10
                            # ran full enumeration for ~0% hit rate, late-game
                            # handles <5-unknown boards anyway
# Bail (→ maximizer fallback) when the multiset has more completions than this.
# Was 2000, but a near-ceiling board ran the exhaustive move×completion check for
# ~14 min in one call; 200 caps a single call to a few seconds. When we *do* run
# the check it stays exhaustive over all completions, so the safety guarantee is
# preserved — we just decline to compute it (and fall through to the maximizer,
# the path taken ~89% of the time regardless) on large completion spaces.
SAFE_MAX_PERMS    = 200
SAFE_MAX_STATES   = 20_000  # A* budget per completion solve

# ── Info-gain reveal scorer tuning (see plan_reveal_info_gain) ────────
# Base-score weights. Cascade (forced-deduction unlocks) is weighted highest
# because it removes uncertainty outright; completion and heuristic delta are
# softer progress signals. EMPTY_PENALTY discourages spending the last empty on
# a reveal that neither completes nor empties a tube.
IG_W_CASCADE     = 3.0
IG_W_COMPLETION  = 2.0
IG_W_HEURISTIC   = 1.0
IG_EMPTY_PENALTY = 2.0
# Outcome-memory adjustments (advisory soft nudges, never hard blocks).
IG_DEADEND_PENALTY = 4.0    # decayed by 1/(1 + new_knowledge_since_that_attempt)
IG_EXPLORE_BONUS   = 1.0    # decayed by 1/(1 + times tube was a reveal source)
IG_OPENER_PENALTY  = 1.5    # repeating a past attempt's first move


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
        sim_state = cur_state
        seen_states = {cur_state}
        added_pairs = set()
        # find_matching_tops scores/sorts candidates against the static state;
        # acceptance is gated through a running sim so the batch stays executable.
        for src, dst in find_matching_tops(cur_state, tube_capacity):
            # Can't pour a known colour onto a hidden ball.
            if sim_state[dst] and sim_state[dst][-1] == UNKNOWN:
                continue
            # Prevent A→B then B→A cancellation.
            if (dst, src) in added_pairs:
                continue
            next_state, n = apply_move(sim_state, src, dst, tube_capacity)
            if n == 0:
                continue
            # Prevent cycles.
            if next_state in seen_states:
                continue
            if prev_state is not None and next_state == prev_state:
                continue
            src_after = next_state[src]
            empties_src = len(src_after) == 0
            reveals_hidden = bool(src_after) and src_after[-1] == UNKNOWN
            if empties_src or reveals_hidden:
                result.append((src, dst, n))
                sim_state = next_state
                seen_states.add(next_state)
                added_pairs.add((src, dst))
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
        # 0 empties: score by reveal proximity against the static state, then
        # accept through a running sim so the batch is executable in order.
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
            candidates.append((depth, src, dst))
        candidates.sort()

        sim_state = state
        seen_states = {state}
        added_pairs = set()
        batch = []
        for _, src, dst in candidates:
            # Can't pour a known colour onto a hidden ball.
            if sim_state[dst] and sim_state[dst][-1] == UNKNOWN:
                continue
            # Prevent A→B then B→A cancellation.
            if (dst, src) in added_pairs:
                continue
            next_state, n = apply_move(sim_state, src, dst, tube_capacity)
            if n == 0:
                continue
            # Prevent cycles.
            if next_state in seen_states:
                continue
            batch.append((src, dst, n))
            sim_state = next_state
            seen_states.add(next_state)
            added_pairs.add((src, dst))
            if len(batch) >= 3:
                break
        return batch

    return moves


def _exposes_unknown(tube):
    """True if pouring the top run off ``tube`` would uncover an UNKNOWN slot."""
    colour, run = get_top_run(tube)
    if colour is None or colour == UNKNOWN:
        return False
    beneath = len(tube) - run - 1          # slot directly under the top run
    return beneath >= 0 and tube[beneath] == UNKNOWN


def _legal_reveal_dests(sim, src, capacity):
    """Ranked legal destinations for evicting ``src``'s top run (best first).

    Prefers a matching top where the colour is most accumulated / closest to pure
    (so a reveal also makes sorting progress and conserves empties); an empty tube
    is a last resort. Random tail breaks exact ties so barren retries diverge.
    """
    colour, _ = get_top_run(sim[src])
    outs = []
    for dst in range(len(sim)):
        if dst == src or len(sim[dst]) >= capacity:
            continue
        if not sim[dst]:
            outs.append((dst, (1, 0, 0, random.random())))
        elif sim[dst][-1] == UNKNOWN:
            continue                         # can't pour onto a hidden top
        elif sim[dst][-1] == colour:
            cnt = sim[dst].count(colour)
            pure = all(c == colour for c in sim[dst])
            outs.append((dst, (0, -cnt, 0 if pure else 1, random.random())))
    outs.sort(key=lambda d: d[1])
    return [d for d, _ in outs]


def plan_reveal_maximize(state, tube_capacity, prev_state=None, max_moves=4):
    """Greedy reveal-maximizer.

    Each round, pick moves that expose still-UNKNOWN slots, ignoring solvability.
    Assumes learned slots are already overlaid, so every remaining UNKNOWN is
    genuinely unrecorded. Only evicts KNOWN top runs that sit directly on an
    UNKNOWN — never pours a guessed colour. Returns [(src, dst, num_poured), ...]
    or [] if nothing can be revealed this round.
    """
    # Shallow unknowns first (they unlock deeper ones next round); ties broken by
    # smaller top run (cheaper to evict, burns less destination capacity), then
    # randomly so barren retries explore divergent paths.
    sources = sorted(
        (get_top_run(t)[1], random.random(), i)
        for i, t in enumerate(state) if _exposes_unknown(t)
    )

    sim = tuple(tuple(t) for t in state)
    seen = {sim}
    moves = []
    for _, _, src in sources:
        if not _exposes_unknown(sim[src]):       # sim moved on; re-check
            continue
        for dst in _legal_reveal_dests(sim, src, tube_capacity):
            nxt, n = apply_move(sim, src, dst, tube_capacity)
            if n == 0 or nxt in seen or nxt == prev_state:
                continue
            moves.append((src, dst, n))
            sim = nxt
            seen.add(nxt)
            break
        if len(moves) >= max_moves:
            break
    return moves


def _count_unknowns(state):
    return sum(slot == UNKNOWN for tube in state for slot in tube)


def _outcome_memory_indices(attempt_log):
    """Build the three outcome-memory indices the reveal scorers share.

    Returns ``(deadend_new_knowledge, explore_used, past_openers)``:
      - ``deadend_new_knowledge``: ``{(src, dst): knowledge_since}`` — for each move
        played into a past dead-end, how many slots were learnt in *later* attempts
        (penalty decays as that grows). The most-recent dead-end wins per move.
      - ``explore_used``: ``{src: count}`` — how often a tube was a *productive*
        reveal source across past attempts (bonus decays as it grows).
      - ``past_openers``: set of ``(src, dst)`` first moves tried before.
    """
    attempt_log = attempt_log or []
    deadend_log = [e for e in attempt_log
                   if e.get("outcome") in ("stuck", "no_moves")]
    deadend_new_knowledge = {}
    for i, e in enumerate(deadend_log):
        later_knowledge = sum(x.get("total_reveals", 0) for x in deadend_log[i + 1:])
        for m in e.get("moves", []):
            deadend_new_knowledge[(m[0], m[1])] = later_knowledge  # later i overwrites
    explore_used = {}
    for e in attempt_log:
        for m, rev in zip(e.get("moves", []), e.get("reveals_per_move", [])):
            if rev > 0:
                explore_used[m[0]] = explore_used.get(m[0], 0) + 1
    past_openers = {(e["moves"][0][0], e["moves"][0][1])
                    for e in attempt_log if e.get("moves")}
    return deadend_new_knowledge, explore_used, past_openers


def _reveal_base_proxies(after, src, exposed_pos, possible_colours,
                         known_counts, base_h, capacity):
    """Score a freshly-exposed UNKNOWN slot by the shared info-gain proxies.

    ``after`` is a board where ``after[src][exposed_pos]`` is the UNKNOWN slot the
    reveal uncovers (the new source top). Trying each colour the slot could hold,
    returns ``(cascade, completion, heuristic_delta)``:
      - ``cascade``: how many candidate colours force *additional* deductions.
      - ``completion``: best colour-completion proximity (one-short-of-full ≈ 1.0).
      - ``heuristic_delta``: average heuristic improvement (positive = better).
    """
    cascade = 0
    completion = 0.0
    deltas = []
    for colour in possible_colours:
        trial = [list(t) for t in after]
        trial[src][exposed_pos] = colour
        trial = tuple(tuple(t) for t in trial)

        # Cascade: did setting this one slot force *additional* deductions?
        deduced = deduce_hidden_slots(trial, capacity)
        if _count_unknowns(deduced) < _count_unknowns(trial):
            cascade += 1
        # Heuristic improvement (positive = better).
        deltas.append(base_h - heuristic(trial))
        # Completion proximity: this colour one short of full scores ~1.0.
        completion = max(completion,
                         (known_counts.get(colour, 0) + 1) / capacity)

    heuristic_delta = sum(deltas) / len(deltas) if deltas else 0.0
    return cascade, completion, heuristic_delta


def plan_reveal_info_gain(tubes, capacity, attempt_log=None):
    """Rank candidate reveal moves by cheap information-gain proxies (no A*).

    A *candidate* pours the top run off a tube whose run sits directly on an
    UNKNOWN slot (same condition as ``_exposes_unknown``), uncovering that slot.
    One candidate per source tube, using its single best legal destination.

    Each candidate is scored against the set of colours the hidden slot could
    hold (the distinct completion-pool colours) by four base proxies — cascade
    potential (forced deductions unlocked), colour-completion proximity, average
    heuristic improvement, and empty conservation — then softly adjusted using
    ``attempt_log`` (dead-end penalty, exploration bonus, opener diversity).

    Returns ``(moves, top_score)`` where ``moves`` is a list of ``(src, dst, n)``
    tuples sorted by score descending. Returns ``([], 0.0)`` when no tube exposes
    an unknown (caller falls back to the maximizer).
    """
    state = tuple(tuple(t) for t in tubes)

    # Colours the hidden slot could take: distinct completion-pool colours
    # (visible deficits + phantom groups for fully-hidden colours). Fall back to
    # visible known colours if the board can't be pooled (structurally odd).
    pool = _build_completion_pool(state, capacity)
    possible_colours = sorted(set(pool)) if pool else sorted(count_colour_occurrences(state))

    known_counts = count_colour_occurrences(state)
    base_h = heuristic(state)

    # ── Outcome-memory indices (built once, shared with plan_reveal_deep) ──
    deadend_new_knowledge, explore_used, past_openers = \
        _outcome_memory_indices(attempt_log)

    scored = []
    for src in range(len(state)):
        if not _exposes_unknown(state[src]):
            continue
        dests = _legal_reveal_dests(state, src, capacity)
        if not dests:
            continue
        dst = dests[0]
        after, n = apply_move(state, src, dst, capacity)
        if n == 0:
            continue
        # Exposed slot is the new top of the source after the pour.
        exposed_pos = len(after[src]) - 1
        if exposed_pos < 0 or after[src][exposed_pos] != UNKNOWN:
            continue

        cascade, completion, heuristic_delta = _reveal_base_proxies(
            after, src, exposed_pos, possible_colours,
            known_counts, base_h, capacity)

        # Empty conservation: penalise spending the last empty on a reveal that
        # neither completes a tube nor empties the source.
        empties_after = sum(1 for t in after if len(t) == 0)
        src_after = after[src]
        dst_after = after[dst]
        completes = (len(dst_after) == capacity
                     and len(set(dst_after)) == 1
                     and UNKNOWN not in dst_after)
        empties_src = len(src_after) == 0
        empty_pen = (IG_EMPTY_PENALTY
                     if empties_after < 1 and not (completes or empties_src)
                     else 0.0)

        score = (IG_W_CASCADE * cascade
                 + IG_W_COMPLETION * completion
                 + IG_W_HEURISTIC * heuristic_delta
                 - empty_pen)

        # ── Outcome-memory adjustments (advisory, never hard blocks) ──
        if attempt_log:
            if (src, dst) in deadend_new_knowledge:
                nk = deadend_new_knowledge[(src, dst)]
                score -= IG_DEADEND_PENALTY / (1 + nk)
            score += IG_EXPLORE_BONUS / (1 + explore_used.get(src, 0))
            if (src, dst) in past_openers:
                score -= IG_OPENER_PENALTY

        scored.append((score, src, dst, n))

    if not scored:
        return [], 0.0

    scored.sort(key=lambda x: x[0], reverse=True)
    moves = [(src, dst, n) for _, src, dst, n in scored]
    return moves, scored[0][0]


def plan_reveal_deep(state, capacity, max_depth=3, attempt_log=None):
    """Dig 2–3 known top runs off one source tube to expose a *buried* UNKNOWN.

    Once memory overlay fills most hidden slots, the survivors sit 2–3 layers
    under known balls, so the planners gated on ``_exposes_unknown`` (info-gain,
    maximizer) all return empty and the bot cycles. This planner peels the top
    run off a single source repeatedly — into the best legal destination each
    time (``_legal_reveal_dests``) — until a pour surfaces the buried UNKNOWN as
    the new source top, never pouring an unknown itself.

    Each candidate digs ONE source only and is scored with the *same* proxies as
    ``plan_reveal_info_gain`` (evaluated on the finally-exposed unknown slot),
    minus a depth penalty so shorter digs win, the empty-conservation penalty,
    and the shared outcome-memory adjustments keyed on the dig's opener/source.

    Returns ``(moves, top_score)`` for the single best peel sequence, or
    ``([], 0.0)`` if no tube can be dug to a buried unknown. No A*.
    """
    state = tuple(tuple(t) for t in state)

    pool = _build_completion_pool(state, capacity)
    possible_colours = sorted(set(pool)) if pool else sorted(count_colour_occurrences(state))
    known_counts = count_colour_occurrences(state)
    base_h = heuristic(state)
    deadend_new_knowledge, explore_used, past_openers = \
        _outcome_memory_indices(attempt_log)

    candidates = []
    for src in range(len(state)):
        tube = state[src]
        # Only buried unknowns: a tube that has an UNKNOWN but doesn't expose one
        # directly (the directly-exposing case is info-gain's job).
        if UNKNOWN not in tube or _exposes_unknown(tube):
            continue

        sim = state
        moves = []
        completed_a_tube = False
        exposed = False
        for _ in range(max_depth):
            top_colour, _run = get_top_run(sim[src])
            if top_colour == UNKNOWN:
                break                                  # can't pour unknowns
            dests = _legal_reveal_dests(sim, src, capacity)
            if not dests:
                break                                  # no legal destination
            dst = dests[0]
            after, n = apply_move(sim, src, dst, capacity)
            if n == 0:
                break
            moves.append((src, dst, n))
            sim = after
            dst_after = sim[dst]
            if (len(dst_after) == capacity and len(set(dst_after)) == 1
                    and UNKNOWN not in dst_after):
                completed_a_tube = True
            if not sim[src]:
                break                                  # emptied before reaching unknown
            if sim[src][-1] == UNKNOWN:
                exposed = True                         # buried unknown surfaced
                break

        if not exposed or not moves:
            continue

        exposed_pos = len(sim[src]) - 1
        cascade, completion, heuristic_delta = _reveal_base_proxies(
            sim, src, exposed_pos, possible_colours,
            known_counts, base_h, capacity)

        score = (IG_W_CASCADE * cascade
                 + IG_W_COMPLETION * completion
                 + IG_W_HEURISTIC * heuristic_delta)

        # Depth penalty: prefer shorter digs (single-pour digs unpenalised).
        score -= 0.5 * (len(moves) - 1)

        # Empty conservation: penalise a dig that spends the last empty without
        # completing a tube (source-emptying digs are already skipped above).
        empties_after = sum(1 for t in sim if len(t) == 0)
        if empties_after < 1 and not completed_a_tube:
            score -= IG_EMPTY_PENALTY

        # Outcome-memory adjustments (advisory), keyed on the dig opener/source.
        if attempt_log:
            opener = (moves[0][0], moves[0][1])
            if opener in deadend_new_knowledge:
                score -= IG_DEADEND_PENALTY / (1 + deadend_new_knowledge[opener])
            score += IG_EXPLORE_BONUS / (1 + explore_used.get(src, 0))
            if opener in past_openers:
                score -= IG_OPENER_PENALTY

        candidates.append((score, moves))

    if not candidates:
        return [], 0.0

    candidates.sort(key=lambda c: c[0], reverse=True)
    return candidates[0][1], candidates[0][0]


def plan_consolidate(state, capacity, needed_empties=None):
    """BFS short sorting sequences among KNOWN-ONLY tubes to free empties.

    Used when deep-reveal can't dig because empties are the bottleneck. Finds a
    move sequence that merges known balls to reach >= needed_empties empty tubes,
    so next round's deep-reveal has a destination for its deepest peel.

    Constraint: never pours an unknown ball. The BFS may freely use tubes with
    unknowns buried below known balls — only tubes whose current TOP is UNKNOWN
    are excluded (they can't be a source or destination). Buried unknowns never
    move (apply_move stops at the UNKNOWN layer; valid_moves only pours onto a
    matching known top), so the memory overlay stays valid. Returns
    (moves, empties_created); ([], 0) if no path.
    """
    state = tuple(tuple(t) for t in state)

    def count_empties(s):
        return sum(1 for t in s if not t)

    start_empties = count_empties(state)
    if needed_empties is None:
        # Free just one more tube than currently available (0→1, 1→2, 2→3).
        # A fixed target of 3 from 0 empties is unreachable in MAX_MOVES; an
        # adaptive +1 target lets the bot incrementally create space per round.
        needed_empties = start_empties + 1
    if start_empties >= needed_empties:
        return [], 0

    # Excludes only tubes whose current TOP is UNKNOWN (can't pour from/to them).
    # Tubes with unknowns buried below known balls are fair game: the unknown's
    # position never changes (apply_move stops at the UNKNOWN layer, and pours
    # only land on matching known tops), so membership stays constant.
    allowed = frozenset(
        i for i, t in enumerate(state)
        if not t or t[-1] != UNKNOWN  # empty tubes OK, known-top tubes OK
    )

    print(f"  🔧 Consolidation BFS: {len([i for i in allowed if state[i]])} usable tubes, "
          f"{start_empties} empties, target {needed_empties}")

    MAX_STATES = 50_000        # dequeued expansions
    MAX_MOVES = 8              # longest sorting sequence considered

    queue = deque([(state, [])])
    seen = {state}
    explored = 0
    best_moves = None
    best_empties = -1

    while queue and explored < MAX_STATES:
        cur, path = queue.popleft()
        # A deeper parent can only produce longer/equal goals — can't improve on
        # a goal we already have. Shorter parents still expand (filling the level
        # so the max-empties tiebreak sees every equal-length goal).
        if best_moves is not None and len(path) >= len(best_moves):
            continue
        if len(path) >= MAX_MOVES:
            continue
        explored += 1

        for src, dst in valid_moves(cur, capacity, restrict_empties=True):
            if src not in allowed or dst not in allowed:
                continue
            nxt, n = apply_move(cur, src, dst, capacity)
            if n == 0 or nxt in seen:
                continue
            seen.add(nxt)
            npath = path + [(src, dst, n)]
            emp = count_empties(nxt)
            if emp >= needed_empties:
                created = emp - start_empties
                if (best_moves is None
                        or len(npath) < len(best_moves)
                        or (len(npath) == len(best_moves) and created > best_empties)):
                    best_moves = npath
                    best_empties = created
                # Goal state is terminal — don't expand it further.
            else:
                queue.append((nxt, npath))

    if best_moves is None:
        return [], 0
    return best_moves, best_empties


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


def validate_move_sequence(state, moves, tube_capacity):
    """Truncate a planned batch at the first move the real game can't execute.

    After a pour, the exposed ball is UNKNOWN in our model but gets revealed
    to an unpredictable colour in-game, so any move pouring onto an UNKNOWN
    top must wait for a re-screenshot. Returns the executable prefix.
    """
    sim = tuple(tuple(t) for t in state)
    valid = []
    for (src, dst, _) in moves:
        # Can't pour onto a tube whose top is hidden — outcome is unknowable.
        if sim[dst] and sim[dst][-1] == UNKNOWN:
            break
        src_colour, _ = get_top_run(sim[src])
        # Pouring a hidden top is only allowed into an empty tube (a reveal move);
        # include it but stop, since we can't simulate what gets revealed.
        if src_colour == UNKNOWN:
            if sim[dst]:
                break
            valid.append((src, dst, 1))
            break
        # Can't pour onto a mismatched known colour.
        if sim[dst] and sim[dst][-1] != src_colour:
            break
        new_sim, poured = apply_move(sim, src, dst, tube_capacity)
        if poured == 0:
            break
        valid.append((src, dst, poured))
        sim = new_sim
    return valid


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


def _build_completion_pool(state, tube_capacity):
    """Multiset of colours that must fill the UNKNOWN slots, or None if the
    board is structurally inconsistent.

    Phantom-colour pool sizing: each colour fills exactly tube_capacity slots, so
    the true colour count is derivable from board structure. The pool is built
    from visible colours' deficits, plus a phantom group of tube_capacity balls
    per colour that is entirely hidden (0 visible slots). This makes the pool
    length match the unknown count even when a colour never appears visibly.
    """
    visible_counts = count_colour_occurrences(state)

    # Counts are inconsistent if any visible colour overflows a tube.
    for cnt in visible_counts.values():
        if cnt > tube_capacity:
            return None

    total_balls = sum(len(t) for t in state)
    if total_balls % tube_capacity != 0:
        return None
    num_colours = total_balls // tube_capacity

    pool = []
    for colour, cnt in visible_counts.items():
        needed = tube_capacity - cnt
        if needed > 0:
            pool.extend([colour] * needed)
    for i in range(num_colours - len(visible_counts)):
        pool.extend([f"_phantom_{i}"] * tube_capacity)
    return pool


def is_late_game(state, max_unknowns=5):
    """True when few enough unknowns remain for the full-solve strategy.

    Returns True when the total number of UNKNOWN slots across all tubes is
    between 1 and ``max_unknowns`` inclusive.  Called after the memory overlay,
    so recalled slots are already filled in.  Unlike the previous version, this
    does NOT require unknowns to be at depth 0 — the late-game execution loop
    handles unknowns at any depth by detecting when a pour exposes one as the
    tube's new top.
    """
    total = sum(1 for t in state for ball in t if ball == UNKNOWN)
    return 0 < total <= max_unknowns


def plan_late_game_solve(state, capacity, max_samples=20):
    """Sample a completion of the UNKNOWN slots and A*-solve the full board.

    Returns ``(moves, completed_state)`` for the first solvable sample, or
    ``None`` if the board can't be pooled or no sample (of up to ``max_samples``)
    is solvable. ``moves`` is the full solution (typically 30–50 moves).
    """
    pool = _build_completion_pool(state, capacity)
    num_unknowns = _count_unknowns(state)
    # None: structurally inconsistent. Length mismatch is defensive (misread).
    if pool is None or len(pool) != num_unknowns:
        return None
    for _ in range(max_samples):
        p = pool[:]
        random.shuffle(p)
        filled = _fill_unknowns(state, p)
        solution = solve(filled, capacity)
        if solution:
            return solution, filled
    return None


def find_path_to_unknown(state, capacity, max_moves=8, max_states=50_000):
    """Shortest move sequence that exposes ANY hidden UNKNOWN as a tube's top.

    Unlike the A* solver (goal = fully solved) this BFS only needs to surface one
    buried unknown so the reveal/read loop can make progress. It may shuffle balls
    across MULTIPLE tubes to clear room for a peel — something the single-source
    digging planners (deep-reveal, maximize) can't do. Reuses valid_moves /
    apply_move, never pours an UNKNOWN (apply_move enforces this).

    Returns a list of (src, dst, num_poured) tuples (possibly []), or None if no
    exposing path exists within budget.
    """
    state = tuple(tuple(t) for t in state)

    def exposes(s):                       # any tube whose current top is hidden
        return any(t and t[-1] == UNKNOWN for t in s)

    if exposes(state):                    # already exposed → no moves needed
        return []

    queue = deque([(state, [])])
    seen = {state}
    explored = 0
    while queue and explored < max_states:
        cur, path = queue.popleft()
        explored += 1
        if len(path) >= max_moves:
            continue
        for src, dst in valid_moves(cur, capacity, restrict_empties=True):
            nxt, n = apply_move(cur, src, dst, capacity)
            if n == 0 or nxt in seen:
                continue
            npath = path + [(src, dst, n)]
            # Goal: this move uncovered an unknown as src's NEW top.
            if nxt[src] and nxt[src][-1] == UNKNOWN:
                return npath
            seen.add(nxt)
            queue.append((nxt, npath))
    return None


def pick_best_move_by_determinization(state, tube_capacity, num_samples=10):
    """
    Estimate the best next move(s) under hidden-slot uncertainty by sampling
    random completions of UNKNOWN slots and running A* on each.

    Returns up to 3 (src, dst, num_poured) tuples that win the most votes
    across successful sample solves. Returns [] if no sample produced a solution.
    """
    num_unknowns = sum(slot == UNKNOWN for tube in state for slot in tube)

    if num_unknowns == 0:
        return []

    # After memory overlay the unknown count sits around tube_capacity * 3; only
    # bail when it's large enough that sampling becomes unreliable/slow.
    if num_unknowns > tube_capacity * 5:
        return []

    pool_base = _build_completion_pool(state, tube_capacity)

    # None: structurally inconsistent board. Length mismatch is defensive — it
    # only triggers on a structurally impossible board (e.g. more visible colours
    # than num_colours), where _fill_unknowns would otherwise run the pool dry.
    if pool_base is None or len(pool_base) != num_unknowns:
        return []

    solutions = []
    for i in range(num_samples):
        print(f"    Determinization sample {i+1}/{num_samples}...")
        pool = pool_base[:]
        random.shuffle(pool)
        filled = _fill_unknowns(state, pool)
        sol = solve_astar(filled, tube_capacity, max_states=20_000)
        if sol:
            solutions.append(sol[:3])

    if not solutions:
        return []

    def _executable(real_state, move):
        """True if `move` can be played on the real (still-UNKNOWN) board.

        A sampled solution's move may rely on a slot only known in the sample,
        so we accept it only when the real top is known and the destination
        genuinely accepts the pour.
        """
        src, dst = move[0], move[1]
        src_top, _ = get_top_run(real_state[src])
        if src_top is None or src_top == UNKNOWN:
            return False
        if not real_state[dst]:
            return True
        dst_top, _ = get_top_run(real_state[dst])
        return dst_top == src_top and len(real_state[dst]) < tube_capacity

    # First move: vote only among sampled first moves executable for real.
    executable_firsts = [s[0] for s in solutions if s and _executable(state, s[0])]
    if not executable_firsts:
        return []

    best_first = Counter(executable_firsts).most_common(1)[0][0]
    result = [best_first]
    # Advance the real board; newly-exposed slots stay UNKNOWN, so subsequent
    # executability tests correctly reject moves that depended on the sample.
    real_state, _ = apply_move(state, best_first[0], best_first[1], tube_capacity)

    sharing_first = [s for s in solutions if s and s[0] == best_first]
    second_candidates = [
        s[1] for s in sharing_first
        if len(s) >= 2 and _executable(real_state, s[1])
    ]
    if second_candidates:
        top_second, top_second_count = Counter(second_candidates).most_common(1)[0]
        if top_second_count > len(sharing_first) / 2:
            result.append(top_second)
            real_state, _ = apply_move(real_state, top_second[0], top_second[1], tube_capacity)

            sharing_two = [s for s in sharing_first if len(s) >= 3 and s[1] == top_second]
            third_candidates = [
                s[2] for s in sharing_two if _executable(real_state, s[2])
            ]
            if third_candidates:
                top_third, top_third_count = Counter(third_candidates).most_common(1)[0]
                if top_third_count > len(sharing_two) / 2:
                    result.append(top_third)

    return result


def sample_solvable_completions(state, tube_capacity,
                                num_samples=10, max_states=20_000):
    """Sample random completions of the UNKNOWN slots, keep the solvable ones.

    Returns a list of fully-known (no-UNKNOWN) states the game could actually win,
    or ``None`` if the board can't be sampled (structurally inconsistent) or no
    sampled completion was solvable. Factored out of ``score_reveal_batch`` so a
    caller scoring several move-batches against the *same* board can reuse one base
    set instead of resampling per batch.
    """
    pool = _build_completion_pool(state, tube_capacity)
    if pool is None:
        return None

    solvable = []
    for _ in range(num_samples):
        p = pool[:]
        random.shuffle(p)
        c = _fill_unknowns(state, p)
        if solve_astar(c, tube_capacity, max_states=max_states) is not None:
            solvable.append(c)
    return solvable or None


def score_reveal_batch(state, moves, tube_capacity,
                       num_samples=10, max_states=20_000, solvable=None):
    """Fraction of solvable sampled completions that remain solvable after replaying
    ``moves`` (``(src, dst, num_poured)`` tuples) on them.

    Samples random completions of the UNKNOWN slots, keeps the ones the game could
    actually win, then replays the planned tap sequence on each winning world and
    measures how many stay winnable. A completion has no unknowns, so each
    ``(src, dst)`` is an ordinary pour — replaying the taps faithfully models what
    the game does in that world. Returns ``None`` if the board can't be sampled
    (structurally inconsistent / no solvable completion found).

    Pass a precomputed ``solvable`` base (from ``sample_solvable_completions``) to
    score several batches against one shared sample set; when omitted the base is
    sampled here, preserving the original standalone behaviour.
    """
    if solvable is None:
        solvable = sample_solvable_completions(state, tube_capacity,
                                               num_samples, max_states)
    if not solvable:
        return None

    kept = 0
    for c in solvable:
        cur = c
        for (src, dst, _n) in moves:        # replay taps; apply_move pours what's legal
            cur, _ = apply_move(cur, src, dst, tube_capacity)
        if solve_astar(cur, tube_capacity, max_states=max_states) is not None:
            kept += 1
    return kept / len(solvable)


# ── Guaranteed-safe moves (exhaustive endgame) ───────────────────────

def _distinct_permutations(items):
    """Yield each distinct ordering of a multiset exactly once.

    Avoids materialising n! tuples the way set(itertools.permutations(...))
    would. Items must be sortable (colour labels are strings).
    """
    items = sorted(items)
    n = len(items)
    used = [False] * n
    cur = []

    def rec():
        if len(cur) == n:
            yield tuple(cur)
            return
        prev = None
        for i in range(n):
            if used[i] or items[i] == prev:
                continue
            used[i] = True
            prev = items[i]
            cur.append(items[i])
            yield from rec()
            cur.pop()
            used[i] = False

    yield from rec()


@lru_cache(maxsize=None)
def _is_solvable(state, tube_capacity=4, max_states=SAFE_MAX_STATES):
    """Cached solvability test. `state` is a tuple-of-tuples (hashable)."""
    return solve_astar(state, tube_capacity, max_states=max_states) is not None


def find_guaranteed_safe_moves(state, tube_capacity,
                               prev_state=None,
                               max_unknowns=SAFE_MAX_UNKNOWNS,
                               max_perms=SAFE_MAX_PERMS,
                               max_states=SAFE_MAX_STATES):
    """Return solvability-preserving moves [(src, dst, n), ...], best first.

    A move is *safe* iff, for every hidden completion of the board in which the
    level is solvable at all, applying the move leaves the board solvable. This
    exhaustively enumerates the consistent hidden layouts (the endgame
    counterpart to the sampling-based pick_best_move_by_determinization).

    Returns [] when:
      - there are no unknowns (caller should be on the full-info solve path),
      - there are too many unknowns to enumerate (caller falls back to sampling),
      - the multiset of completions exceeds max_perms,
      - the board is structurally inconsistent / misread, or
      - no completion is solvable (board already lost).
    """
    _is_solvable.cache_clear()

    unknown_pos = [
        (ti, si)
        for ti, tube in enumerate(state)
        for si, slot in enumerate(tube)
        if slot == UNKNOWN
    ]
    if not unknown_pos or len(unknown_pos) > max_unknowns:
        return []

    pool = _build_completion_pool(state, tube_capacity)
    if pool is None or len(pool) != len(unknown_pos):
        return []

    # Multiset-permutation count up front: len(pool)! / prod(count_i!). Bail
    # before generating anything if it blows the budget.
    num_perms = math.factorial(len(pool))
    for cnt in Counter(pool).values():
        num_perms //= math.factorial(cnt)
    if num_perms > max_perms:
        return []

    solvable = []
    for perm in _distinct_permutations(pool):
        filled = _fill_unknowns(state, perm)
        if _is_solvable(filled, tube_capacity, max_states):
            solvable.append(filled)
    if not solvable:
        return []

    safe = []
    for src, dst in valid_moves(state, tube_capacity):
        after, n = apply_move(state, src, dst, tube_capacity)
        if n == 0:
            continue
        if after == prev_state:          # no-progress / reversal guard
            continue

        # Safe iff every solvable completion stays solvable after the move.
        # Early-exit on the first completion where it doesn't.
        is_safe = True
        for completion in solvable:
            moved, _ = apply_move(completion, src, dst, tube_capacity)
            if not _is_solvable(moved, tube_capacity, max_states):
                is_safe = False
                break
        if is_safe:
            safe.append((src, dst, n, after))

    if not safe:
        return []

    def _progress_key(move):
        src, dst, n, after = move
        src_after = after[src]
        dst_after = after[dst]
        reveals = bool(src_after) and src_after[-1] == UNKNOWN
        completes = (len(dst_after) == tube_capacity
                     and len(set(dst_after)) == 1
                     and UNKNOWN not in dst_after)
        empties = len(src_after) == 0
        # Lower key sorts first: prioritise reveal > complete > empty, then
        # smallest heuristic.
        return (not reveals, not completes, not empties, heuristic(after))

    safe.sort(key=_progress_key)
    return [(src, dst, n) for src, dst, n, _ in safe[:3]]


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

    # Take the best moves but avoid conflicts. Gate acceptance through a running
    # sim so the batch stays executable in order (no reversals across different
    # sources, no pours onto freshly-revealed hidden tops, no cycles).
    used_srcs = set()
    sim_state = state
    seen_states = {state}
    accepted_pairs = set()
    for score, src, dst, num_poured in scored:
        if src in used_srcs:
            continue
        next_state, _ = apply_move(sim_state, src, dst, tube_capacity)
        # Can't pour a known colour onto a hidden ball.
        if sim_state[dst] and sim_state[dst][-1] == UNKNOWN:
            continue
        # Prevent A→B then B→A reversal.
        if (dst, src) in accepted_pairs:
            continue
        # Prevent cycles (real pours only; UNKNOWN-top fallback pours are no-ops).
        is_noop = next_state == sim_state
        if not is_noop and next_state in seen_states:
            continue

        safe_moves.append((src, dst, num_poured))
        used_srcs.add(src)
        accepted_pairs.add((src, dst))
        if not is_noop:
            sim_state = next_state
            seen_states.add(next_state)

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

    # Constraints are count-only with no positional dependence, so every
    # unknown position is interchangeable. A position is forced (identical
    # across all valid assignments) iff exactly one colour is still needed.
    if len(needed) == 1:
        colour = next(iter(needed))
        new_state = [list(tube) for tube in state]
        for ti, si in unknown_positions:
            new_state[ti][si] = colour
        return tuple(tuple(tube) for tube in new_state)

    return state


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
