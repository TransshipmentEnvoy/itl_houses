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
    9. **Related-object variational** (type-82) — follow the default branch.

    Visited-set prevents infinite loops.  Callback-result sentinel values
    (bit 15 set) are never followed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .nodes import (
    VAR_BUILDING_COUNTS, VAR_ANIMATION_FRAME, VAR_CALLBACK_ID,
    VAR_CLIMATE, VAR_CONSTRUCTION_STATE, VAR_TERRAIN_TYPE,
    CLIMATE_ARCTIC, CLIMATE_TROPIC,
    TERRAIN_DESERT, TERRAIN_RAINFOREST, TERRAIN_SNOW,
    Action2Graph,
    Action2Node, ClimateGraphics, ComputationNode, FrameLayout,
    HouseTileGraphics, LayoutNode, RandomNode, RandomVariantGraphics,
    VariationalNode,
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
        layout > variational > random (80) > computation
    """
    graph: Action2Graph = {}

    # Lowest priority first so higher-priority parsers overwrite.
    comp_nodes = parse_all_computation_nodes(sprites)
    rand_nodes = parse_all_random_nodes(sprites)
    var_nodes  = parse_all_variational_nodes(sprites)
    lay_nodes  = parse_all_layout_nodes(sprites)

    for nid, node in comp_nodes.items():
        graph[nid] = node
    for nid, node in rand_nodes.items():
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


def _rvg_climate_graphics(rvg: RandomVariantGraphics, climate: str) -> ClimateGraphics:
    """Return (creating if needed) the ClimateGraphics for *climate* inside a random variant."""
    attr = {"temperate": "temperate", "snow": "snow", "tropic": "tropic", "arctic_v2": "arctic_v2"}.get(climate, "temperate")
    cg = getattr(rvg, attr, None)
    if cg is None:
        cg = ClimateGraphics()
        setattr(rvg, attr, cg)
    return cg


# ============================================================================
# Colour callback extraction (CB 0x1E)
# ============================================================================

# CBID_HOUSE_COLOUR in OpenTTD
_CBID_HOUSE_COLOUR = 0x1E


def _extract_colour_values(
    target_id: int,
    resolve_graph: Action2Graph,
    final_graph: Action2Graph,
) -> tuple[list[int], int]:
    """Follow *target_id* and extract colour callback return values.

    The NFO colour callback branch typically points to a type-80 random node
    whose entries are all callback-result sentinels (bit 15 set).  We extract
    the 15-bit callback result of each entry.  Duplicates are **preserved** so
    that downstream code can derive probability weights from entry repetition.

    Also handles the case of a type-82 (re-randomise) node which contains
    ranges that are callback results, and the case of a direct callback result.

    Returns ``(colour_values, triggers)`` where *triggers* is the NFO trigger
    byte from a RandomNode (0 when the source is not random).
    """
    if is_callback_result(target_id):
        return [target_id & 0x7FFF], 0

    node = resolve_graph.get(target_id)
    if node is None:
        node = final_graph.get(target_id)
    if node is None:
        return [], 0

    if isinstance(node, RandomNode):
        colours: list[int] = []
        for entry_id in node.entries:
            if is_callback_result(entry_id):
                colours.append(entry_id & 0x7FFF)
        return colours, node.triggers

    # Type-82/86 re-randomise — treated as VariationalNode by the parser.
    # Extract callback-result values from ranges and default.
    if isinstance(node, VariationalNode) and node.var_type in (0x82, 0x86):
        colours = []
        # Check default
        if is_callback_result(node.default):
            colours.append(node.default & 0x7FFF)
        # Check range results
        for rng in node.ranges:
            if is_callback_result(rng.result_id):
                colours.append(rng.result_id & 0x7FFF)
        return colours, 0

    return [], 0


# ============================================================================
# Core recursive traversal
# ============================================================================


def _traverse(
    node_id: int,
    resolve_graph: Action2Graph,        # graph to look up node_id in
    snapshots: dict[int, Action2Graph],  # keyed by id(node)
    final_graph: Action2Graph,           # complete graph (fallback)
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

    *resolve_graph* is the graph used to look up *node_id*.  For the initial
    entry (from Action 3 root) this is the *final_graph*.  When a non-terminal
    node N references a target, the target is looked up in N's **snapshot** —
    the graph state at N's definition position — ensuring that reused set IDs
    resolve to their correct, position-specific definitions.

    The visited set tracks ``id(node)`` (Python object identity) rather than
    ``node_id``, so the same set_id can be visited twice when it maps to
    different node objects in different resolution contexts (e.g. a LayoutNode
    at set_id 0x11 versus a VariationalNode at the same set_id 0x11).
    """
    if depth > state.max_depth:
        return
    if is_callback_result(node_id):
        return

    # Look up node: try resolve_graph first, then final_graph
    node = resolve_graph.get(node_id)
    if node is None:
        node = final_graph.get(node_id)
    if node is None:
        return

    # Visited guard uses object identity to handle set_id reuse
    node_ident = id(node)
    if node_ident in state.visited:
        return

    state.visited.add(node_ident)

    # For non-layout nodes, the snapshot is the graph to pass downstream
    node_snapshot = snapshots.get(node_ident, final_graph)

    # ------------------------------------------------------------------
    # 1. Terminal: type-00 layout
    # ------------------------------------------------------------------
    if isinstance(node, LayoutNode):
        fl = _to_frame_layout(node)

        if in_random:
            idx = random_variant_idx if random_variant_idx is not None else 0
            while len(state.result.random_variants) <= idx:
                state.result.random_variants.append(RandomVariantGraphics())
            rvg = state.result.random_variants[idx]
            cg = _rvg_climate_graphics(rvg, climate)
            if in_constr:
                _record_constr(cg, constr_idx, fl)
            elif anim_frame is not None:
                while len(cg.animation_frames) <= anim_frame:
                    cg.animation_frames.append(None)
                cg.animation_frames[anim_frame] = fl
            else:
                if cg.completed is None:
                    cg.completed = fl
        else:
            cg = _climate_graphics(state.result, climate)
            if in_constr:
                _record_constr(cg, constr_idx, fl)
            elif anim_frame is not None:
                while len(cg.animation_frames) <= anim_frame:
                    cg.animation_frames.append(None)
                cg.animation_frames[anim_frame] = fl
            else:
                if cg.completed is None:
                    cg.completed = fl
        state.visited.discard(node_ident)
        return

    # ------------------------------------------------------------------
    # 2. Variational node: dispatch on the variable
    # ------------------------------------------------------------------
    if isinstance(node, VariationalNode):
        var = node.variable
        var_type = node.var_type

        # 2a. Callback router (var 0x0C) — follow the *default* branch
        #     (= the graphics chain); extract colour values from CB 0x1E;
        #     store all other callback handler targets for later translation.
        if var == VAR_CALLBACK_ID:
            _follow(node.default, node_snapshot, snapshots, final_graph, state,
                    climate, in_constr, constr_idx,
                    anim_frame, in_random, random_variant_idx, depth + 1)
            # Extract colour callback values (CB 0x1E) and record other CBs
            for rng in node.ranges:
                cb_lo, cb_hi = rng.range_lo, rng.range_hi
                if cb_lo <= _CBID_HOUSE_COLOUR <= cb_hi:
                    if not state.result.colour_values:
                        colours, col_triggers = _extract_colour_values(
                            rng.result_id, node_snapshot, final_graph,
                        )
                        if colours:
                            state.result.colour_values = colours
                            state.result.colour_triggers = col_triggers
                else:
                    # Record other callback handler targets (e.g. CB 0x17,
                    # 0x1B, 0x1F, 0x21, 0x2E, 0x143, …).
                    cb_id = cb_lo if cb_lo == cb_hi else cb_lo
                    if cb_id not in state.result.callback_handlers:
                        state.result.callback_handlers[cb_id] = rng.result_id

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
                _follow(target, node_snapshot, snapshots, final_graph, state,
                        cli, in_constr, constr_idx,
                        anim_frame, in_random, random_variant_idx, depth + 1)

        # 2c. Terrain / snow check (var 0x43)
        #     Per spec: 0=normal, 1=desert, 2=rainforest, 4=snow
        elif var == VAR_TERRAIN_TYPE:
            snow_target: int | None = None
            desert_target: int | None = None
            rainforest_target: int | None = None
            default_target = node.default

            for rng in node.ranges:
                if rng.range_lo <= TERRAIN_SNOW <= rng.range_hi:
                    snow_target = rng.result_id
                if rng.range_lo <= TERRAIN_DESERT <= rng.range_hi:
                    desert_target = rng.result_id
                if rng.range_lo <= TERRAIN_RAINFOREST <= rng.range_hi:
                    rainforest_target = rng.result_id

            # No-snow / no-desert path (= default → temperate)
            _follow(default_target, node_snapshot, snapshots, final_graph, state,
                    climate, in_constr, constr_idx,
                    anim_frame, in_random, random_variant_idx, depth + 1)

            # Snow path
            if snow_target is not None:
                _follow(snow_target, node_snapshot, snapshots, final_graph, state,
                        "snow", in_constr, constr_idx,
                        anim_frame, in_random, random_variant_idx, depth + 1)

            # Desert path → maps to "tropic" climate slot
            if desert_target is not None and desert_target != default_target:
                _follow(desert_target, node_snapshot, snapshots, final_graph, state,
                        "tropic", in_constr, constr_idx,
                        anim_frame, in_random, random_variant_idx, depth + 1)

            # Rainforest path — usually same as temperate; only follow if distinct
            if (rainforest_target is not None
                    and rainforest_target != default_target
                    and rainforest_target != desert_target):
                # No dedicated slot; treat as temperate (already covered by default)
                pass

        # 2d. Construction state (var 0x40)
        elif var == VAR_CONSTRUCTION_STATE:
            effective_shift = node.shift_count
            effective_mask = node.mask

            # Standard construction state: shift=0, mask covers bits 0-1
            if effective_shift == 0 and (effective_mask & 0x03) == 0x03:
                # Stage 3 = completed → follow default
                _follow(node.default, node_snapshot, snapshots, final_graph, state,
                        climate, False, -1,
                        anim_frame, in_random, random_variant_idx, depth + 1)

                # Stages 0-2 = under construction
                covered: set[int] = set()
                for rng in node.ranges:
                    lo = max(0, rng.range_lo)
                    hi = min(2, rng.range_hi)
                    for stage in range(lo, hi + 1):
                        if stage not in covered:
                            covered.add(stage)
                            _follow(rng.result_id, node_snapshot, snapshots,
                                    final_graph, state, climate, True, stage,
                                    anim_frame, in_random, random_variant_idx,
                                    depth + 1)
            else:
                # Non-standard extraction (e.g. pseudo-random bits 2-3):
                # follow all branches without construction-state semantics.
                all_targets: set[int] = {node.default}
                for rng in node.ranges:
                    all_targets.add(rng.result_id)
                for target in sorted(all_targets):
                    _follow(target, node_snapshot, snapshots, final_graph, state,
                            climate, in_constr, constr_idx,
                            anim_frame, in_random, random_variant_idx, depth + 1)

        # 2e. Animation frame (var 0x46)
        elif var == VAR_ANIMATION_FRAME:
            # Follow each frame range
            for rng in node.ranges:
                for frame in range(rng.range_lo, rng.range_hi + 1):
                    _follow(rng.result_id, node_snapshot, snapshots, final_graph,
                            state, climate, in_constr, constr_idx,
                            frame, in_random, random_variant_idx, depth + 1)
            # Default frame (also used as a representative if ranges cover all)
            _follow(node.default, node_snapshot, snapshots, final_graph, state,
                    climate, in_constr, constr_idx,
                    None, in_random, random_variant_idx, depth + 1)

        # 2f. Building counts (var 0x44) — used in callbacks; follow default
        elif var == VAR_BUILDING_COUNTS:
            _follow(node.default, node_snapshot, snapshots, final_graph, state,
                    climate, in_constr, constr_idx,
                    anim_frame, in_random, random_variant_idx, depth + 1)

        # 2g. Related-object variational (var_type 0x82/0x86) — follow default
        elif var_type in (0x82, 0x86):
            _follow(node.default, node_snapshot, snapshots, final_graph, state,
                    climate, in_constr, constr_idx,
                    anim_frame, in_random, random_variant_idx, depth + 1)

        # 2h. Other variational (town zone, building age, …) — follow all branches
        else:
            all_targets: set[int] = {node.default}
            for rng in node.ranges:
                all_targets.add(rng.result_id)
            for target in sorted(all_targets):
                _follow(target, node_snapshot, snapshots, final_graph, state,
                        climate, in_constr, constr_idx,
                        anim_frame, in_random, random_variant_idx, depth + 1)

        state.visited.discard(node_ident)
        return

    # ------------------------------------------------------------------
    # 3. Random node (type-80): record each unique entry as a variant
    #
    #    When randoms are nested (outer random → inner random), we compose
    #    variant indices multiplicatively so that a 2×2 tree yields 4 flat
    #    variant slots instead of overwriting slots 0-1 twice.
    # ------------------------------------------------------------------
    if isinstance(node, RandomNode):
        # Deduplicate entries for graph traversal, but also count occurrences
        # to derive per-variant probability weights.
        seen_entries: dict[int, int] = {}   # entry_id → local_index
        unique_order: list[int] = []
        entry_counts: dict[int, int] = {}   # entry_id → occurrence count
        for entry_id in node.entries:
            if is_callback_result(entry_id):
                continue
            if entry_id not in seen_entries:
                seen_entries[entry_id] = len(unique_order)
                unique_order.append(entry_id)
                entry_counts[entry_id] = 1
            else:
                entry_counts[entry_id] += 1

        num_branches = len(unique_order)

        # Store random triggers from the outermost random node.
        if not in_random:
            state.result.random_triggers = node.triggers

        # Build per-variant weights from entry repetition counts.
        for entry_id in unique_order:
            local_idx = seen_entries[entry_id]
            weight = entry_counts[entry_id]
            if in_random and random_variant_idx is not None:
                # Nested random: compose multiplicatively
                effective_idx = random_variant_idx * num_branches + local_idx
            else:
                effective_idx = local_idx
            # Extend random_weights list to cover effective_idx
            while len(state.result.random_weights) <= effective_idx:
                state.result.random_weights.append(1)
            if in_random and random_variant_idx is not None:
                # Nested: multiply outer weight by inner count
                outer_weight = state.result.random_weights[random_variant_idx] if random_variant_idx < len(state.result.random_weights) else 1
                state.result.random_weights[effective_idx] = outer_weight * weight
            else:
                state.result.random_weights[effective_idx] = weight
            _follow(entry_id, node_snapshot, snapshots, final_graph, state,
                    climate, in_constr, constr_idx,
                    anim_frame, True, effective_idx, depth + 1)
        state.visited.discard(node_ident)
        return

    # ------------------------------------------------------------------
    # 4. Computation node (type-89): best-effort — follow default and any
    #    subroutine-call targets found in the steps.
    # ------------------------------------------------------------------
    if isinstance(node, ComputationNode):
        # Follow default
        if not is_callback_result(node.default):
            _follow(node.default, node_snapshot, snapshots, final_graph, state,
                    climate, in_constr, constr_idx,
                    anim_frame, in_random, random_variant_idx, depth + 1)
        # Follow subroutine targets embedded in steps
        for step in node.steps:
            if step.var == 0x7E and step.add_val != 0:
                _follow(step.add_val, node_snapshot, snapshots, final_graph, state,
                        climate, in_constr, constr_idx,
                        anim_frame, in_random, random_variant_idx, depth + 1)
        # Also follow range results (they are return values, so typically skip)
        state.visited.discard(node_ident)
        return

    state.visited.discard(node_ident)


def _follow(
    target: int,
    resolve_graph: Action2Graph,
    snapshots: dict[int, Action2Graph],
    final_graph: Action2Graph,
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
    _traverse(target, resolve_graph, snapshots, final_graph, state,
              climate, in_constr, constr_idx,
              anim_frame, in_random, random_variant_idx, depth)


def _record_constr(cg: ClimateGraphics, idx: int, fl: FrameLayout) -> None:
    """Record a construction stage frame layout into *cg*."""
    while len(cg.construction_stages) <= idx:
        cg.construction_stages.append(None)
    if cg.construction_stages[idx] is None:
        cg.construction_stages[idx] = fl


def _fixup_construction_replication(result: HouseTileGraphics) -> None:
    """Replicate construction stages to align with nested random's completed variants.

    When the completed state uses a nested random (N variants) but the
    construction state uses a flat (non-nested) random (M < N variants),
    the construction stages end up at indices 0..M-1 instead of being
    distributed across the correct stride.  This function detects the
    mismatch and replicates construction data to fill all variant slots.

    Example: completed random 2×2 → 4 variants (0,1,2,3).
    Construction random flat    → 2 variants at (0,1).
    After fixup: construction[0] → slots 0,1 ; construction[1] → slots 2,3.
    """
    n = len(result.random_variants)
    if n <= 1:
        return

    for attr in ("temperate", "snow", "tropic", "arctic_v2"):
        # Collect construction sources and count completed variants
        constr_sources: list[tuple[int, list]] = []  # (index, stages_copy)
        total_with_content = 0

        for i, rvg in enumerate(result.random_variants):
            cg = getattr(rvg, attr)
            if cg is None:
                continue
            has_content = (cg.completed is not None or
                           any(fl is not None for fl in cg.animation_frames))
            if has_content:
                total_with_content += 1
            has_constr = any(fl is not None for fl in cg.construction_stages)
            if has_constr:
                constr_sources.append((i, list(cg.construction_stages)))

        m = len(constr_sources)
        if m == 0 or m >= total_with_content or total_with_content <= 1:
            continue  # No replication needed

        # Check for clean divisibility (expected from multiplicative nesting)
        if n % m != 0:
            continue

        # Verify sources are at contiguous indices 0..m-1
        if not all(constr_sources[j][0] == j for j in range(m)):
            continue  # Sources not at expected positions, skip

        stride = n // m

        # Replicate: source j fills slots [j*stride, (j+1)*stride)
        for j, (_, stages) in enumerate(constr_sources):
            for slot in range(j * stride, min((j + 1) * stride, n)):
                rvg = result.random_variants[slot]
                cg = getattr(rvg, attr)
                if cg is None:
                    cg = ClimateGraphics()
                    setattr(rvg, attr, cg)
                cg.construction_stages = list(stages)


def _fixup_snow_construction_fallback(result: HouseTileGraphics) -> None:
    """Copy construction stages from temperate to snow/tropic/arctic_v2 when missing.

    In Pattern A houses (NFO checks construction_state before terrain_type),
    construction stages 0–2 branch directly to layout nodes without passing
    through the terrain check.  The traversal records these layouts only under
    ``climate="temperate"``, leaving the snow/tropic/arctic_v2 construction
    stages empty.  The NFO's intended behaviour is that construction sprites
    are climate-independent — only the completed building differs.

    This fixup copies temperate construction stages to any climate variant
    that has a completed sprite (or animation frames) but no construction
    stages of its own.
    """
    def _has_constr(cg: ClimateGraphics | None) -> bool:
        return cg is not None and any(fl is not None for fl in cg.construction_stages)

    def _has_content(cg: ClimateGraphics | None) -> bool:
        if cg is None:
            return False
        return (cg.completed is not None
                or any(fl is not None for fl in cg.animation_frames))

    # Non-random path
    if _has_constr(result.temperate):
        for attr in ("snow", "tropic", "arctic_v2"):
            cg = getattr(result, attr)
            if _has_content(cg) and not _has_constr(cg):
                cg.construction_stages = list(result.temperate.construction_stages)  # type: ignore[union-attr]

    # Random variants path
    for rvg in result.random_variants:
        if _has_constr(rvg.temperate):
            for attr in ("snow", "tropic", "arctic_v2"):
                cg = getattr(rvg, attr)
                if _has_content(cg) and not _has_constr(cg):
                    cg.construction_stages = list(rvg.temperate.construction_stages)  # type: ignore[union-attr]


# ============================================================================
# Callback sub-graph traversal
# ============================================================================

# Mapping from NFO variable byte to a human-readable NML expression.
_VAR_EXPR_MAP: dict[int, str] = {
    0x40: "construction_state",
    0x41: "age",
    0x42: "town_zone",
    0x43: "terrain_type",
    0x44: "var[0x44, 0, 0xFFFFFFFF]",   # building counts — raw
    0x46: "animation_frame",
    0x47: "var[0x47, 0, 0xFF]",          # xy coordinate of building
    0x60: "var[0x60, 0, 0xFFFFFFFF]",
    0x61: "var[0x61, 0, 0xFFFFFFFF]",
    0x7D: "var[0x7D, 0, 0xFFFFFFFF]",    # temp register
}


def _nml_var_expr(variable: int, shift: int, mask: int, param: int | None = None) -> str:
    """Build an NML variable expression from NFO variable / shift / mask.

    For variables in the 0x60-0x7F range that require an extra parameter byte,
    *param* is included as the **fourth** argument:
    ``var[0xNN, shift, mask, param]`` (matching nmlc's Variable(num, shift, mask, param) constructor).
    """
    if variable in _VAR_EXPR_MAP and shift == 0 and mask in (0xFF, 0xFFFF, 0xFFFFFFFF) and param is None:
        return _VAR_EXPR_MAP[variable]
    if param is not None:
        return f"var[0x{variable:02X}, {shift}, 0x{mask:X}, {param}]"
    return f"var[0x{variable:02X}, {shift}, 0x{mask:X}]"


def traverse_callback_subgraph(
    root_id: int,
    resolve_graph: Action2Graph,
    final_graph: Action2Graph,
    house_id_hex: str,
    cb_id: int,
    *,
    cb_label: str | None = None,
    _depth: int = 0,
    _visited: set[int] | None = None,
    _counter: list[int] | None = None,
) -> tuple[list[str], str | None]:
    """Recursively translate a callback decision-tree into NML switch blocks.

    Callback sub-graphs are pure decision trees whose leaf nodes are callback-
    result sentinels (``is_callback_result() == True``).  They never reach
    layout nodes.

    Parameters
    ----------
    root_id : int
        Entry node ID (or callback-result sentinel) to translate.
    resolve_graph, final_graph : Action2Graph
        Per-section and global graph for node lookup.
    house_id_hex : str
        Used for naming emitted switches.
    cb_id : int
        NFO callback ID (for naming).
    cb_label : str or None
        Human-readable label for the callback (e.g. ``"anim_speed"``).
        When provided, switch names use this label instead of ``cb{hex}``.
    _depth, _visited, _counter : internal recursion state.

    Returns
    -------
    (lines, switch_name)
        *lines* is a list of NML code lines; *switch_name* is the top-level
        identifier (or ``None`` if translation failed).
    """
    if _counter is None:
        _counter = [0]
    if _visited is None:
        _visited = set()
    name_tag = cb_label or f"cb{cb_id:X}"
    if _depth > 20:
        return [f"/* WARN: {name_tag} sub-graph depth limit reached */"], None

    # -- Callback-result terminal ------------------------------------------
    if is_callback_result(root_id):
        val = root_id & 0x7FFF
        name = f"switch_ttrs_{house_id_hex}_{name_tag}_{_counter[0]}"
        _counter[0] += 1
        lines = [
            f"switch (FEAT_HOUSES, SELF, {name}, 0) {{ return {val}; }}"
        ]
        return lines, name

    # -- Look up node ------------------------------------------------------
    node = resolve_graph.get(root_id)
    if node is None:
        node = final_graph.get(root_id)
    if node is None:
        return [f"/* WARN: {name_tag} target 0x{root_id:02X} not found in any graph section */"], None

    node_ident = id(node)
    if node_ident in _visited:
        return [f"/* WARN: {name_tag} cycle at 0x{root_id:02X} */"], None
    _visited.add(node_ident)

    # -- VariationalNode: emit switch with ranges --------------------------
    if isinstance(node, VariationalNode):
        var_expr = _nml_var_expr(node.variable, node.shift_count, node.mask, getattr(node, 'param', None))
        switch_name = f"switch_ttrs_{house_id_hex}_{name_tag}_{_counter[0]}"
        _counter[0] += 1
        all_lines: list[str] = []

        # Build cases
        cases: list[str] = []
        for rng in node.ranges:
            child_lines, child_name = traverse_callback_subgraph(
                rng.result_id, resolve_graph, final_graph,
                house_id_hex, cb_id,
                cb_label=cb_label, _depth=_depth + 1, _visited=_visited, _counter=_counter,
            )
            all_lines.extend(child_lines)
            if child_name is None:
                continue
            if rng.range_lo == rng.range_hi:
                cases.append(f"{rng.range_lo}: {child_name}")
            else:
                cases.append(f"{rng.range_lo}..{rng.range_hi}: {child_name}")

        # Default branch
        def_lines, def_name = traverse_callback_subgraph(
            node.default, resolve_graph, final_graph,
            house_id_hex, cb_id,
            cb_label=cb_label, _depth=_depth + 1, _visited=_visited, _counter=_counter,
        )
        all_lines.extend(def_lines)
        def_ref = def_name or "0"

        cases_str = "; ".join(cases)
        if cases_str:
            cases_str += "; "
        all_lines.append(
            f"switch (FEAT_HOUSES, SELF, {switch_name}, {var_expr}) {{ {cases_str}return {def_ref}; }}"
        )
        _visited.discard(node_ident)
        return all_lines, switch_name

    # -- RandomNode: emit random_switch with weights -----------------------
    if isinstance(node, RandomNode):
        switch_name = f"switch_ttrs_{house_id_hex}_{name_tag}_{_counter[0]}"
        _counter[0] += 1
        all_lines: list[str] = []

        # Count entry occurrences for weights
        entry_order: list[int] = []
        entry_weights: dict[int, int] = {}
        for entry_id in node.entries:
            if entry_id not in entry_weights:
                entry_order.append(entry_id)
                entry_weights[entry_id] = 1
            else:
                entry_weights[entry_id] += 1

        weight_parts: list[str] = []
        for entry_id in entry_order:
            w = entry_weights[entry_id]
            if is_callback_result(entry_id):
                val = entry_id & 0x7FFF
                weight_parts.append(f"{w}: return {val}")
            else:
                child_lines, child_name = traverse_callback_subgraph(
                    entry_id, resolve_graph, final_graph,
                    house_id_hex, cb_id,
                    cb_label=cb_label, _depth=_depth + 1, _visited=_visited, _counter=_counter,
                )
                all_lines.extend(child_lines)
                if child_name is not None:
                    weight_parts.append(f"{w}: {child_name}")

        if not weight_parts:
            _visited.discard(node_ident)
            return [f"/* WARN: {name_tag} random node 0x{root_id:02X} empty */"], None

        entries_str = "; ".join(weight_parts)
        # Map trigger byte
        trigger_clause = ""
        if node.triggers != 0:
            trigger_clause = f" /* NFO triggers: 0x{node.triggers:02X} */"
        all_lines.append(
            f"random_switch (FEAT_HOUSES, SELF, {switch_name}) {{ {entries_str}; }}{trigger_clause}"
        )
        _visited.discard(node_ident)
        return all_lines, switch_name

    # -- ComputationNode: best-effort — follow default / subroutine calls --
    if isinstance(node, ComputationNode):
        # For computation nodes, try to resolve the result from ranges/default
        all_lines: list[str] = []
        if node.ranges:
            # Has ranges — treat like a variational node
            switch_name = f"switch_ttrs_{house_id_hex}_{name_tag}_{_counter[0]}"
            _counter[0] += 1
            cases = []
            for rng in node.ranges:
                child_lines, child_name = traverse_callback_subgraph(
                    rng.result_id, resolve_graph, final_graph,
                    house_id_hex, cb_id,
                    cb_label=cb_label, _depth=_depth + 1, _visited=_visited, _counter=_counter,
                )
                all_lines.extend(child_lines)
                if child_name is None:
                    continue
                if rng.range_lo == rng.range_hi:
                    cases.append(f"{rng.range_lo}: {child_name}")
                else:
                    cases.append(f"{rng.range_lo}..{rng.range_hi}: {child_name}")

            def_lines, def_name = traverse_callback_subgraph(
                node.default, resolve_graph, final_graph,
                house_id_hex, cb_id,
                cb_label=cb_label, _depth=_depth + 1, _visited=_visited, _counter=_counter,
            )
            all_lines.extend(def_lines)
            def_ref = def_name or "0"

            # Use a generic expression — computation semantics are approximated
            cases_str = "; ".join(cases)
            if cases_str:
                cases_str += "; "
            all_lines.append(
                f"switch (FEAT_HOUSES, SELF, {switch_name}, 0) "
                f"{{ {cases_str}return {def_ref}; }}"
                f" /* NOTE: computation node 0x{root_id:02X}, result approximated */"
            )
            _visited.discard(node_ident)
            return all_lines, switch_name
        else:
            # No ranges — just follow default
            _visited.discard(node_ident)
            return traverse_callback_subgraph(
                node.default, resolve_graph, final_graph,
                house_id_hex, cb_id,
                cb_label=cb_label, _depth=_depth + 1, _visited=_visited, _counter=_counter,
            )

    # -- LayoutNode: callback chain falls through to default graphics ------
    # In NFO, reaching a layout node from a callback means "use the default
    # graphics result" — equivalent to CB_FAILED.  We omit this branch so the
    # callback is simply not listed in the NML graphics block.
    _visited.discard(node_ident)
    return [f"/* NOTE: {name_tag} fell through to layout node 0x{root_id:02X} (default graphics) */"], None


# ============================================================================
# Public API
# ============================================================================


def build_house_tile_graphics(
    root_id: int,
    graph: Action2Graph,
    snapshots: dict[int, Action2Graph] | None = None,
) -> HouseTileGraphics:
    """
    Traverse the Action 2 graph starting from *root_id* and return a fully
    resolved :class:`~nodes.HouseTileGraphics` for one tile.

    *root_id* is typically the group ID read from the house's Action 3 entry.
    *snapshots* provides position-aware graph views for each non-terminal node;
    pass ``None`` (or omit) to use the global graph for all lookups.
    """
    result = HouseTileGraphics()
    state  = _TraversalState(result=result)
    snaps  = snapshots if snapshots is not None else {}
    _traverse(
        root_id, graph, snaps, graph, state,
        climate="temperate",
        in_constr=False, constr_idx=-1,
        anim_frame=None,
        in_random=False, random_variant_idx=None,
        depth=0,
    )
    _fixup_construction_replication(result)
    _fixup_snow_construction_fallback(result)
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
    set IDs that are reused across houses are scoped correctly.

    The graph builder is **position-aware**: when a set ID is defined multiple
    times within a section (e.g. temperate then snow), each non-terminal node
    receives a snapshot of the graph as it existed at the node's definition
    position.  This correctly resolves references to set IDs that were reused
    for different climate variants — the typical TTRS Pattern A pattern.
    """
    sections = collect_house_section_ranges(lines)

    house_graphics: dict[int, HouseTileGraphics] = {}
    house_graphs:   dict[int, Action2Graph]       = {}

    for house_id, start_idx, end_idx, root_id in sections:
        sec_sprites = collect_action2_in_range(lines, start_idx, end_idx + 1)
        sec_graph, snapshots = _build_graph_positional(sec_sprites)
        htg = build_house_tile_graphics(root_id, sec_graph, snapshots)
        house_graphics[house_id] = htg
        house_graphs[house_id]   = sec_graph

    return house_graphics, house_graphs


def _parse_one_sprite(rs: RawSprite) -> Action2Node | None:
    """Parse a single RawSprite into its Action 2 node type.

    Routing table
    -------------
    - ``0x00``               → basic sprite layout (parse_layout)
    - ``0x01..0x3F``         → extended sprite layout (parse_layout, warns)
    - ``0x40..0x7F``         → advanced sprite layout (parse_layout, warns)
    - ``0x80``, ``0x83``     → random action (parse_random)
    - ``0x81``, ``0x82``     → variational byte (parse_variational)
    - ``0x85``, ``0x86``     → variational dword (parse_variational)
    - ``0x89``, ``0x8A``     → computation chain (parse_computation)
    """
    from .parse_layout import parse_layout_node
    from .parse_variational import parse_variational_node
    from .parse_random import parse_random_node
    from .parse_computation import parse_computation_node

    if len(rs.bytes) < 4:
        return None
    t = rs.bytes[3]

    # --- Sprite layouts (type byte < 0x80) ---------------------------------
    if t < 0x80:
        return parse_layout_node(rs)

    # --- Callback / decision types (type byte >= 0x80) ---------------------
    if t in (0x80, 0x83):
        # 0x80 = self scope random; 0x83 = related object random
        return parse_random_node(rs)
    elif t in (0x81, 0x82, 0x85, 0x86):
        return parse_variational_node(rs)
    elif t in (0x89, 0x8A):
        # 0x89 = self scope computation; 0x8A = related scope computation
        return parse_computation_node(rs)
    return None


def _build_graph_positional(
    sprites: list[RawSprite],
) -> tuple[Action2Graph, dict[int, Action2Graph]]:
    """
    Build an Action2Graph with **position-aware snapshots** for correct NFO
    sequential semantics.

    NFO set IDs are a mutable sequential namespace: when a variational node
    at position P references set_id S, it means the definition of S that
    existed at position P in the byte stream — not the final/global definition.

    Returns:
        final_graph : complete graph with last-wins for all set_ids
            (used for root entry point resolution from Action 3)
        snapshots : dict keyed by ``id(node)`` for each non-layout node,
            containing only definitions that existed before that node's
            position.  During traversal, when a non-terminal node N
            references target T, T is looked up in ``snapshots[id(N)]``.
    """
    # Parse all nodes in definition order
    parsed: list[tuple[int, Action2Node]] = []  # (position, node)
    for pos, rs in enumerate(sprites):
        node = _parse_one_sprite(rs)
        if node is not None:
            parsed.append((pos, node))

    # Build final graph (last-wins for every set_id)
    final_graph: Action2Graph = {}
    for _, node in parsed:
        final_graph[node.node_id] = node  # type: ignore[union-attr]

    # Build per-node snapshots.
    # running_graph accumulates definitions in order.  Before adding a
    # non-layout node, we snapshot the running state — this is the graph
    # that was "visible" at the point where the non-layout node was defined.
    snapshots: dict[int, Action2Graph] = {}  # keyed by id(node)
    running_graph: Action2Graph = {}

    for _pos, node in parsed:
        if not isinstance(node, LayoutNode):
            # Snapshot the graph *before* this node's own definition
            snapshots[id(node)] = dict(running_graph)
        # Add / overwrite in running graph (all types, last-wins within order)
        running_graph[node.node_id] = node  # type: ignore[union-attr]

    return final_graph, snapshots
