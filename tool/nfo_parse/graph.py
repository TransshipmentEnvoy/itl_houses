"""
graph.py — Build the Action 2 directed graph and traverse it to produce
           :class:`~nodes.HouseTileGraphics`.

Building the graph
------------------
    :func:`build_graph` combines all parsed node types into a single
    ``dict[int, Action2Node]`` keyed by set ID.

Traversal
---------
    :func:`build_house_tile_graphics` starts from the Action 3 root group ID
    for one house tile and walks the graph to accumulate a
    :class:`~nodes.HouseTileGraphics`.

    The traversal honours the following node types in order of precedence:

    1. **Callback router** (type-81/85 var 0x0C) — follow the *default* branch
       to the graphics chain; all other branches (callback handlers) are ignored
       during sprite generation.
    2. **Climate router** (type-81 var 0x03) — fan out to temperate / arctic /
       tropic paths.
    3. **Terrain / snow check** (type-81 var 0x43) — fan out to snow / no-snow.
    4. **Construction state** (type-81 var 0x40) — per stage 0-3; stage 3 is
       the "completed" state.
    5. **Animation frame** (type-81 var 0x46) — per-frame layouts.
    6. **Random selection** (type-80) — record all unique entries as
       ``random_variants``.
    7. **Type-00 layout** — terminal node; record as a :class:`~nodes.FrameLayout`.
    8. **Computation** (type-89) — follow default / any subroutine calls
       (best-effort, records what it can).
    9. **Re-random** (type-82) — follow the default branch.

    Visited-set prevents infinite loops.  Callback-result sentinel values
    (bit 15 set) are never followed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .nodes import (
    VAR_ANIMATION_COUNTER, VAR_ANIMATION_FRAME, VAR_CALLBACK_ID,
    VAR_CLIMATE, VAR_CONSTRUCTION_STATE, VAR_TERRAIN_TYPE,
    CLIMATE_ARCTIC, CLIMATE_TROPIC,
    TERRAIN_SNOW,
    Action2Graph,
    Action2Node, ClimateGraphics, ComputationNode, FrameLayout,
    HouseTileGraphics, LayoutNode, RandomNode, VariationalNode,
    is_callback_result,
)
from .parse_layout import parse_all_layout_nodes
from .parse_variational import parse_all_variational_nodes
from .parse_random import parse_all_random_nodes
from .parse_computation import parse_all_computation_nodes
from .parse_raw import (
    RawSprite,
    collect_action2_house_entries,
    collect_action2_in_range,
    collect_action3_entries,
    collect_house_section_ranges,
)


# ============================================================================
# Graph construction
# ============================================================================


def build_graph(sprites: list[RawSprite]) -> Action2Graph:
    """
    Merge all parsed Action 2 node types into one lookup dict.

    When more than one parser claims the same set ID (shouldn't happen for a
    valid NFO but can occur when two types share an ID) the priority order is:
        layout > variational > random (80) > rerand (82) > computation
    """
    graph: Action2Graph = {}

    # Lowest priority first so higher-priority parsers overwrite.
    comp_nodes = parse_all_computation_nodes(sprites)
    rand_nodes, rerand_nodes = parse_all_random_nodes(sprites)
    var_nodes   = parse_all_variational_nodes(sprites)
    lay_nodes   = parse_all_layout_nodes(sprites)

    for nid, node in comp_nodes.items():
        graph[nid] = node
    for nid, node in rand_nodes.items():
        graph[nid] = node
    for nid, node in rerand_nodes.items():
        if nid not in rand_nodes:   # don't overwrite random with rerand
            graph[nid] = node
    for nid, node in var_nodes.items():
        graph[nid] = node
    for nid, node in lay_nodes.items():
        graph[nid] = node

    return graph


def build_graph_from_lines(lines: list[str]) -> Action2Graph:
    """Convenience: collect raw sprites from NFO lines then build the graph."""
    sprites = collect_action2_house_entries(lines)
    return build_graph(sprites)


# ============================================================================
# Traversal context
# ============================================================================

_DEFAULT_CLIMATE = "temperate"
_CLIMATES = ("temperate", "snow", "tropic", "arctic_v2")


@dataclass
class _TraversalState:
    """Mutable state passed down through recursive traversal."""
    result:   HouseTileGraphics
    visited:  set[int]  = field(default_factory=set)
    max_depth: int = 40


def _climate_graphics(result: HouseTileGraphics, climate: str) -> ClimateGraphics:
    """Return (creating if needed) the ClimateGraphics for *climate*."""
    attr_map = {
        "temperate": "temperate",
        "snow":      "snow",
        "tropic":    "tropic",
        "arctic_v2": "arctic_v2",
    }
    attr = attr_map.get(climate, "temperate")
    cg = getattr(result, attr, None)
    if cg is None:
        cg = ClimateGraphics()
        setattr(result, attr, cg)
    return cg


def _to_frame_layout(node: LayoutNode) -> FrameLayout:
    return FrameLayout(ground=node.ground, building=node.building, bbox=node.bbox)


# ============================================================================
# Core recursive traversal
# ============================================================================


def _traverse(
    node_id: int,
    graph: Action2Graph,
    state: _TraversalState,
    climate: str,                       # "temperate" | "snow" | "tropic" | "arctic_v2"
    in_constr: bool,                    # True when inside a construction-stage branch
    constr_idx: int,                    # 0–2; meaningful only when in_constr=True
    anim_frame: int | None,             # current animation frame, or None
    in_random: bool,                    # True when inside random-variant sub-tree
    random_variant_idx: int | None,     # index into result.random_variants
    depth: int,
) -> None:
    """
    Recursively walk the graph.  All terminal type-00 nodes are recorded into
    *state.result*.
    """
    if depth > state.max_depth:
        return
    if is_callback_result(node_id):
        return
    if node_id in state.visited:
        return

    state.visited.add(node_id)
    node = graph.get(node_id)
    if node is None:
        return

    # ------------------------------------------------------------------
    # 1. Terminal: type-00 layout
    # ------------------------------------------------------------------
    if isinstance(node, LayoutNode):
        fl = _to_frame_layout(node)

        if in_random:
            # Record as a random variant for the given climate
            idx = random_variant_idx if random_variant_idx is not None else 0
            # Grow random_variants list as needed
            while len(state.result.random_variants) <= idx:
                state.result.random_variants.append(ClimateGraphics())
            rv_cg = state.result.random_variants[idx]
            if in_constr:
                _record_constr(rv_cg, constr_idx, fl)
            elif anim_frame is not None:
                rv_cg.animation_frames.append(fl)
                # Pad to correct position if needed
                while len(rv_cg.animation_frames) <= anim_frame:
                    rv_cg.animation_frames.append(fl)
                rv_cg.animation_frames[anim_frame] = fl
            else:
                rv_cg.completed = fl
        else:
            cg = _climate_graphics(state.result, climate)
            if in_constr:
                _record_constr(cg, constr_idx, fl)
            elif anim_frame is not None:
                # Pad animation_frames to the correct slot
                while len(cg.animation_frames) <= anim_frame:
                    cg.animation_frames.append(fl)
                cg.animation_frames[anim_frame] = fl
            else:
                if cg.completed is None:
                    cg.completed = fl
        state.visited.discard(node_id)
        return

    # ------------------------------------------------------------------
    # 2. Variational node: dispatch on the variable
    # ------------------------------------------------------------------
    if isinstance(node, VariationalNode):
        var = node.variable
        var_type = node.var_type

        # 2a. Callback router (var 0x0C) — we only care about the *default*
        #     branch (= the graphics chain).
        if var == VAR_CALLBACK_ID:
            _follow(node.default, graph, state, climate, in_constr, constr_idx,
                    anim_frame, in_random, random_variant_idx, depth + 1)

        # 2b. Climate split (var 0x03)
        elif var == VAR_CLIMATE:
            # Collect per-climate targets from ranges
            climate_targets: dict[str, int] = {}
            for rng in node.ranges:
                if rng.range_lo == CLIMATE_TROPIC and rng.range_hi == CLIMATE_TROPIC:
                    climate_targets["tropic"] = rng.result_id
                elif rng.range_lo == CLIMATE_ARCTIC and rng.range_hi == CLIMATE_ARCTIC:
                    climate_targets["arctic_v2"] = rng.result_id
            # Default = temperate (or current climate if not at top level)
            climate_targets["temperate"] = node.default

            for cli, target in climate_targets.items():
                _follow(target, graph, state, cli, in_constr, constr_idx,
                        anim_frame, in_random, random_variant_idx, depth + 1)

        # 2c. Terrain / snow check (var 0x43)
        elif var == VAR_TERRAIN_TYPE:
            snow_target: int | None = None
            default_target = node.default

            for rng in node.ranges:
                if rng.range_lo <= TERRAIN_SNOW <= rng.range_hi:
                    snow_target = rng.result_id

            # No-snow path  (= default)
            _follow(default_target, graph, state, climate, in_constr, constr_idx,
                    anim_frame, in_random, random_variant_idx, depth + 1)

            # Snow path
            if snow_target is not None:
                _follow(snow_target, graph, state, "snow", in_constr, constr_idx,
                        anim_frame, in_random, random_variant_idx, depth + 1)

        # 2d. Construction state (var 0x40)
        elif var == VAR_CONSTRUCTION_STATE:
            # Stage 3 = completed → follow default
            _follow(node.default, graph, state, climate, False, -1,
                    anim_frame, in_random, random_variant_idx, depth + 1)

            # Stages 0-2 = under construction
            covered: set[int] = set()
            for rng in node.ranges:
                lo = max(0, rng.range_lo)
                hi = min(2, rng.range_hi)
                for stage in range(lo, hi + 1):
                    if stage not in covered:
                        covered.add(stage)
                        _follow(rng.result_id, graph, state, climate, True, stage,
                                anim_frame, in_random, random_variant_idx, depth + 1)

        # 2e. Animation frame (var 0x46)
        elif var == VAR_ANIMATION_COUNTER:
            # Follow each frame range
            for rng in node.ranges:
                for frame in range(rng.range_lo, rng.range_hi + 1):
                    _follow(rng.result_id, graph, state, climate, in_constr, constr_idx,
                            frame, in_random, random_variant_idx, depth + 1)
            # Default frame (also used as a representative if ranges cover all)
            _follow(node.default, graph, state, climate, in_constr, constr_idx,
                    None, in_random, random_variant_idx, depth + 1)

        # 2f. Animation info (var 0x44) — used in callbacks; follow default
        elif var == VAR_ANIMATION_FRAME:
            _follow(node.default, graph, state, climate, in_constr, constr_idx,
                    anim_frame, in_random, random_variant_idx, depth + 1)

        # 2g. Re-random (var_type 0x82) — follow the default branch
        elif var_type == 0x82:
            _follow(node.default, graph, state, climate, in_constr, constr_idx,
                    anim_frame, in_random, random_variant_idx, depth + 1)

        # 2h. Other variational (town zone, building age, …) — follow all branches
        else:
            all_targets: set[int] = {node.default}
            for rng in node.ranges:
                all_targets.add(rng.result_id)
            for target in sorted(all_targets):
                _follow(target, graph, state, climate, in_constr, constr_idx,
                        anim_frame, in_random, random_variant_idx, depth + 1)

        state.visited.discard(node_id)
        return

    # ------------------------------------------------------------------
    # 3. Random node (type-80): record each unique entry as a variant
    # ------------------------------------------------------------------
    if isinstance(node, RandomNode):
        seen_entries: dict[int, int] = {}   # entry_id → variant_index
        for entry_id in node.entries:
            if is_callback_result(entry_id):
                continue
            if entry_id not in seen_entries:
                idx = len(seen_entries)
                seen_entries[entry_id] = idx
                _follow(entry_id, graph, state, climate, in_constr, constr_idx,
                        anim_frame, True, idx, depth + 1)
        state.visited.discard(node_id)
        return

    # ------------------------------------------------------------------
    # 4. Computation node (type-89): best-effort — follow default and any
    #    subroutine-call targets found in the steps.
    # ------------------------------------------------------------------
    if isinstance(node, ComputationNode):
        # Follow default
        if not is_callback_result(node.default):
            _follow(node.default, graph, state, climate, in_constr, constr_idx,
                    anim_frame, in_random, random_variant_idx, depth + 1)
        # Follow subroutine targets embedded in steps
        for step in node.steps:
            if step.var == 0x7E and step.add_val != 0:
                _follow(step.add_val, graph, state, climate, in_constr, constr_idx,
                        anim_frame, in_random, random_variant_idx, depth + 1)
        # Also follow range results (they are return values, so typically skip)
        state.visited.discard(node_id)
        return

    state.visited.discard(node_id)


def _follow(
    target: int,
    graph: Action2Graph,
    state: _TraversalState,
    climate: str,
    in_constr: bool,
    constr_idx: int,
    anim_frame: int | None,
    in_random: bool,
    random_variant_idx: int | None,
    depth: int,
) -> None:
    """Helper: guard callback-result sentinels, then recurse."""
    if is_callback_result(target):
        return
    _traverse(target, graph, state, climate, in_constr, constr_idx,
              anim_frame, in_random, random_variant_idx, depth)


def _record_constr(cg: ClimateGraphics, idx: int, fl: FrameLayout) -> None:
    """Record a construction stage frame layout into *cg*."""
    while len(cg.construction_stages) <= idx:
        cg.construction_stages.append(None)
    if cg.construction_stages[idx] is None:
        cg.construction_stages[idx] = fl


# ============================================================================
# Public API
# ============================================================================


def build_house_tile_graphics(
    root_id: int,
    graph: Action2Graph,
) -> HouseTileGraphics:
    """
    Traverse the Action 2 graph starting from *root_id* and return a fully
    resolved :class:`~nodes.HouseTileGraphics` for one tile.

    *root_id* is typically the group ID read from the house's Action 3 entry.
    """
    result = HouseTileGraphics()
    state  = _TraversalState(result=result)
    _traverse(
        root_id, graph, state,
        climate="temperate",
        in_constr=False, constr_idx=-1,
        anim_frame=None,
        in_random=False, random_variant_idx=None,
        depth=0,
    )
    return result


def build_all_house_graphics(
    lines: list[str],
) -> tuple[dict[int, HouseTileGraphics], dict[int, Action2Graph]]:
    """
    Parse the NFO *lines* and return:

        house_graphics : {house_id: HouseTileGraphics}
        house_graphs   : {house_id: Action2Graph}  (per-house, for debugging)

    Each house's graph is built from only the lines in its own NFO section
    (the ~400-line window before its Action 3 entry).  This ensures that
    set IDs that are reused across houses are scoped correctly — the last
    definition for each set ID within a section wins.
    """
    sections = collect_house_section_ranges(lines)

    house_graphics: dict[int, HouseTileGraphics] = {}
    house_graphs:   dict[int, Action2Graph]       = {}

    for house_id, start_idx, end_idx, root_id in sections:
        sec_sprites = collect_action2_in_range(lines, start_idx, end_idx + 1)
        # For per-house graphs use LAST-wins: later definitions (snow, tropic
        # overrides) replace earlier ones when the same set ID appears twice.
        sec_graph   = _build_graph_last_wins(sec_sprites)
        htg         = build_house_tile_graphics(root_id, sec_graph)
        house_graphics[house_id] = htg
        house_graphs[house_id]   = sec_graph

    return house_graphics, house_graphs


def _build_graph_last_wins(sprites: list[RawSprite]) -> Action2Graph:
    """
    Build an Action2Graph where the **last** definition for each set ID wins.

    Within a single house's NFO section the ordering is:
      1. Temperate type-00 entries first (small set IDs like 0x00-0x03, 0x30)
      2. Snow type-00 entries next (same IDs reused OR IDs 0x31, 0x10-0x13)
      3. Tropic/arctic type-00 entries (IDs 0x32, 0x33 — don't collide)
      4. Routing variational nodes (callback router, constr-state, snow check …)

    By keeping the last definition we get the full picture: the variational
    nodes correctly reference all set IDs including snow/tropic variants.

    Note that the collision of set IDs for snow vs temperate (e.g. both have
    ID 0x30; temperate first, snow second) is intentional in Pattern B: the
    variational (var 0x43 terrain-type) node routes to 0x30 vs 0x31, where
    0x31 is the snow key.  The split_climate mechanism in the original heuristic
    handled this; here we preserve it by NOT collapsing on first-wins.
    """
    from .parse_layout import parse_layout_node
    from .parse_variational import parse_variational_node
    from .parse_random import parse_random_node
    from .parse_computation import parse_computation_node
    from .nodes import LayoutNode, VariationalNode, RandomNode, ComputationNode

    graph: Action2Graph = {}

    for rs in sprites:
        node: Action2Node | None = None
        if len(rs.bytes) < 4:
            continue
        t = rs.bytes[3]
        if t == 0x00:
            node = parse_layout_node(rs)
        elif t in (0x81, 0x85):
            node = parse_variational_node(rs)
        elif t in (0x80, 0x82):
            node = parse_random_node(rs)
        elif t == 0x89:
            node = parse_computation_node(rs)

        if node is not None:
            # Last-wins: always overwrite
            graph[node.node_id] = node  # type: ignore[assignment]

    return graph
