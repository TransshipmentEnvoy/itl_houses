"""
nml_emit.py — Emit NML code (spriteset / spritelayout / switch blocks) from
              :class:`~nodes.HouseTileGraphics`.

This module provides the **new** NML generation pipeline that understands
tropic sprites, animation frames, and random variants — the limitations of
the heuristic-based pipeline in the original ``generate_ttrs_from_nfo.py``.

Public API
----------
    emit_house_tile_nml(
        house_id_hex, htg, sprite_table, pcx_path
    ) -> (lines, entry_name)

    *entry_name* is always ``sl_ttrs_{house_id_hex}`` so it is a drop-in
    replacement for the existing ``emit_sprite_blocks()`` call.

Sprite table format
-------------------
The *sprite_table* argument must be a ``dict[int, SpriteCoord]`` where the
key is an Action 1 set index and the value is a 6-tuple of integers:
    (xpos, ypos, xsize, ysize, xrel, yrel)
(This matches the ``SpriteCoord`` namedtuple in the main tool.)

Terrain-type switch ordering
-----------------------------
In NML the ``terrain_type`` variable in house scope returns:
    TILETYPE_NORMAL  (0) — temperate / no snow / no desert
    TILETYPE_DESERT  (2) — tropic / sub-tropical desert
    TILETYPE_RAINFOREST (?) — tropic rain-forest (same sprite as normal)
    TILETYPE_SNOW    (4) — arctic / above snowline

We use ``terrain_type`` to route between temperate, snow, and tropic variants.
(This corresponds to NFO var 0x43; the separate var 0x03 climate check for
arctic_v2 / tropic that appears in Pattern B houses is approximated by the
``TILETYPE_DESERT`` check for tropic and ``TILETYPE_SNOW`` for arctic.)
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence

from .nodes import (
    BoundingBox, ClimateGraphics, FrameLayout, HouseTileGraphics, LayoutSprite,
    RandomVariantGraphics,
)


# ---------------------------------------------------------------------------
# Type alias for a sprite coordinate record
# ---------------------------------------------------------------------------

# (xpos, ypos, xsize, ysize, xrel, yrel)  — matches SpriteCoord in main tool
SpriteCoord = Any   # avoid hard dependency; caller passes whatever it has


def _sprite_literal(sc: Any) -> str:
    """Format a SpriteCoord-like object as an NML sprite literal."""
    # Supports both named-tuple style (sc.xpos …) and plain tuple (sc[0] …)
    try:
        return f"[{sc.xpos}, {sc.ypos}, {sc.xsize}, {sc.ysize}, {sc.xrel}, {sc.yrel}]"
    except AttributeError:
        return f"[{sc[0]}, {sc[1]}, {sc[2]}, {sc[3]}, {sc[4]}, {sc[5]}]"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

SPRITE_ACTION1_FLAG  = 0x80000000
SPRITE_RECOLOUR_FLAG = 0x00008000
SPRITE_INDEX_MASK    = 0x00003FFF


def _ground_expr(
    gsprite: LayoutSprite,
    ss_name: str,
    sprite_table: dict[int, Any],
    pcx_path: str,
    out: list[str],
) -> tuple[str, LayoutSprite]:
    """
    Return ``(nml_expression, ground_sprite)`` for a ground sprite.

    When the ground comes from Action 1, a one-sprite ``spriteset`` is emitted
    into *out* and the expression ``{ss_name}(0)`` is returned.
    When it is a base-game sprite the literal number is returned.

    The second element is the original :class:`LayoutSprite` so callers can
    pass it to :func:`_recolour_clause`.
    """
    if gsprite.is_action1:
        sc = sprite_table.get(gsprite.index)
        if sc is None:
            return "0", gsprite   # missing — use blank
        out.append(f"spriteset({ss_name}, \"{pcx_path}\") {{ {_sprite_literal(sc)} }}")
        return f"{ss_name}(0)", gsprite
    return str(gsprite.index), gsprite


def _recolour_clause(bsprite: LayoutSprite) -> str:
    """Return NML recolour clause for a building sprite.

    Checks sprite_type (bits 14-15) and recolour_sprite (bits 16-29)
    to emit the correct palette reference.
    """
    if not bsprite.has_recolour:
        return ""
    rs = bsprite.recolour_sprite
    if rs is not None and rs != 0:
        # Custom recolour sprite — emit as literal sprite number
        return f" recolour_mode: RECOLOUR_REMAP; palette: {rs};"
    return " recolour_mode: RECOLOUR_REMAP; palette: PALETTE_USE_DEFAULT;"


def _bbox_clause(bbox: Optional[BoundingBox]) -> str:
    """Return NML bounding-box properties for a building block, or empty string."""
    if bbox is None:
        return ""
    return (
        f" xoffset: {bbox.xoff}; yoffset: {bbox.yoff};"
        f" xextent: {bbox.xext}; yextent: {bbox.yext}; zextent: {bbox.zext};"
    )


# ---------------------------------------------------------------------------
# Emitting a single climate variant
# ---------------------------------------------------------------------------


def _emit_climate_variant(
    prefix: str,           # e.g. "sl_ttrs_1A_nosnow"
    ss_prefix: str,        # e.g. "ss_ttrs_1A_nosnow"
    cg: ClimateGraphics,
    sprite_table: dict[int, Any],
    pcx_path: str,
    out: list[str],
    fallback_frame_layouts: dict[int, str] | None = None,
) -> tuple[str, dict[int, str]]:
    """
    Emit spriteset / spritelayout / switch blocks for *cg* (one climate).

    Returns ``(top_name, frame_layouts)`` where *top_name* is the top-level
    NML identifier that routes into this climate's construction_state /
    animation switch, and *frame_layouts* maps animation frame indices to
    emitted spritelayout names (empty for non-animated variants).
    """
    # ------------------------------------------------------------------
    # Construction stages
    # ------------------------------------------------------------------
    constr_sprites: list[Any]           = []
    constr_bsprite: Optional[LayoutSprite] = None   # first recoloured building sprite
    constr_ground_expr: Optional[str]   = None
    constr_bbox: Optional[BoundingBox]  = None

    constr_ground_sprite: Optional[LayoutSprite] = None

    # Track which construction stages (0-2) have no building sprite (ground-only)
    ground_only_stages: set[int] = set()
    first_building_stage: int = 0

    valid_constr = [fl for fl in cg.construction_stages if fl is not None]
    if valid_constr:
        # Pick bbox from the first stage that HAS a building sprite
        constr_bbox = next(
            (fl.bbox for fl in valid_constr if fl.building.is_action1), None
        )
        seen_indices: list[int] = []
        found_first_building = False
        for i, fl_raw in enumerate(cg.construction_stages):
            if fl_raw is None or i > 2:
                continue
            bs = fl_raw.building
            if bs.is_action1:
                if not found_first_building:
                    first_building_stage = i
                    found_first_building = True
                idx = bs.index
                if idx not in seen_indices:
                    sc = sprite_table.get(idx)
                    if sc is not None:
                        constr_sprites.append(sc)
                        seen_indices.append(idx)
                        if bs.has_recolour and constr_bsprite is None:
                            constr_bsprite = bs
            else:
                ground_only_stages.add(i)
        # Ground for construction (use first valid stage's ground)
        constr_ground_expr, constr_ground_sprite = _ground_expr(
            valid_constr[0].ground,
            f"{ss_prefix}_gc",
            sprite_table, pcx_path, out,
        )

    # ------------------------------------------------------------------
    # Animation frames
    # ------------------------------------------------------------------
    if cg.animation_frames:
        return _emit_animated_variant(
            prefix, ss_prefix, cg, constr_sprites, constr_bsprite,
            constr_ground_expr, constr_ground_sprite, constr_bbox,
            sprite_table, pcx_path, out,
            fallback_frame_layouts=fallback_frame_layouts,
            ground_only_stages=ground_only_stages,
            first_building_stage=first_building_stage,
        )

    # ------------------------------------------------------------------
    # Completed stage (non-animated)
    # ------------------------------------------------------------------
    done_sprite_sc: Optional[Any] = None
    done_bsprite: Optional[LayoutSprite] = None
    done_ground_expr: Optional[str] = None
    done_bbox: Optional[BoundingBox] = None

    done_ground_sprite: Optional[LayoutSprite] = None

    if cg.completed is not None:
        fl = cg.completed
        done_bbox = fl.bbox
        bs = fl.building
        if bs.is_action1:
            sc = sprite_table.get(bs.index)
            done_sprite_sc = sc
            done_bsprite = bs
        done_ground_expr, done_ground_sprite = _ground_expr(
            fl.ground, f"{ss_prefix}_gd", sprite_table, pcx_path, out,
        )

    # ------------------------------------------------------------------
    # Emit spritesets
    # ------------------------------------------------------------------
    # Construction spriteset
    if constr_sprites:
        rc = _recolour_clause(constr_bsprite) if constr_bsprite is not None else ""
        grc = _recolour_clause(constr_ground_sprite) if constr_ground_sprite is not None else ""
        out.append(f"spriteset({ss_prefix}_c, \"{pcx_path}\") {{")
        for i, sc in enumerate(constr_sprites):
            out.append(f"\t{_sprite_literal(sc)}  /* constr {i} */")
        out.append("}")
        g_c = constr_ground_expr or "0"
        out.append(f"spritelayout {prefix}_constr {{")
        out.append(f"\tground   {{ sprite: {g_c};{grc} }}")
        expr = _build_layout_expr(len(constr_sprites), first_building_stage)
        out.append(f"\tbuilding {{ sprite: {ss_prefix}_c({expr});{rc}{_bbox_clause(constr_bbox)} }}")
        out.append("}")

    # Ground-only construction layout (for stages with no building sprite)
    if ground_only_stages and constr_ground_expr is not None:
        grc_go = _recolour_clause(constr_ground_sprite) if constr_ground_sprite is not None else ""
        out.append(f"spritelayout {prefix}_constr_g {{")
        out.append(f"\tground   {{ sprite: {constr_ground_expr};{grc_go} }}")
        out.append("}")

    # Done spriteset
    if done_sprite_sc is not None:
        rc_d = _recolour_clause(done_bsprite) if done_bsprite is not None else ""
        grc_d = _recolour_clause(done_ground_sprite) if done_ground_sprite is not None else ""
        out.append(f"spriteset({ss_prefix}_done, \"{pcx_path}\") {{")
        out.append(f"\t{_sprite_literal(done_sprite_sc)}  /* completed */")
        out.append("}")
        g_d = done_ground_expr or constr_ground_expr or "0"
        out.append(f"spritelayout {prefix}_done {{")
        out.append(f"\tground   {{ sprite: {g_d};{grc_d} }}")
        out.append(f"\tbuilding {{ sprite: {ss_prefix}_done(0);{rc_d}{_bbox_clause(done_bbox)} }}")
        out.append("}")
    elif done_ground_expr is not None and done_ground_expr != "0":
        # Ground-only completed layout (no building sprite) — e.g. a snow corner tile
        grc_d = _recolour_clause(done_ground_sprite) if done_ground_sprite is not None else ""
        out.append(f"spritelayout {prefix}_done {{")
        out.append(f"\tground   {{ sprite: {done_ground_expr};{grc_d} }}")
        out.append("}")
    elif constr_sprites:
        # Reuse last construction sprite as completed stage
        g_c = constr_ground_expr or "0"
        rc = _recolour_clause(constr_bsprite) if constr_bsprite is not None else ""
        grc = _recolour_clause(constr_ground_sprite) if constr_ground_sprite is not None else ""
        out.append(f"spritelayout {prefix}_done {{")
        out.append(f"\tground   {{ sprite: {g_c};{grc} }}")
        expr = _build_layout_expr(len(constr_sprites), first_building_stage)
        out.append(f"\tbuilding {{ sprite: {ss_prefix}_c({expr});{rc}{_bbox_clause(constr_bbox)} }}")
        out.append("}")

    # ------------------------------------------------------------------
    # Construction_state routing switch
    # ------------------------------------------------------------------
    has_constr = bool(constr_sprites)
    has_ground_only = bool(ground_only_stages) and constr_ground_expr is not None
    has_done   = bool(
        done_sprite_sc
        or (done_ground_expr is not None and done_ground_expr != "0")
        or constr_sprites
    )

    if has_done or has_ground_only:
        # Build explicit routing cases
        cases: list[str] = []
        if has_ground_only:
            for stage in sorted(ground_only_stages):
                cases.append(f"{stage}: {prefix}_constr_g")
        done_target = f"{prefix}_done" if has_done else (
            f"{prefix}_constr" if has_constr else f"{prefix}_constr_g"
        )
        cases.append(f"3: {done_target}")
        cases_str = "; ".join(cases)
        fallback = f"{prefix}_constr" if has_constr else done_target
        out.append(
            f"switch (FEAT_HOUSES, SELF, {prefix}, construction_state) {{"
            f" {cases_str}; return {fallback}; }}"
        )
    elif has_constr:
        # No proper completed sprite — emit plain spritelayout as top-level
        out.append(f"/* NOTE: no completed sprite for {prefix}, using constr */")
        out.append(
            f"switch (FEAT_HOUSES, SELF, {prefix}, construction_state) {{"
            f" return {prefix}_constr; }}"
        )
    else:
        # Completely empty — emit a ground-only fallback so the identifier is valid
        out.append(f"/* WARN: no sprites resolved for {prefix}, using ground-only fallback */")
        out.append(f"spritelayout {prefix} {{")
        out.append(f"\tground {{ sprite: 0; }}")
        out.append("}")
        # Note: done_ground_expr == "0" means the sprite table lookup also failed;
        # in that case there is nothing useful to emit.

    return prefix, {}


def _emit_animated_variant(
    prefix: str,
    ss_prefix: str,
    cg: ClimateGraphics,
    constr_sprites: list[Any],
    constr_bsprite: Optional[LayoutSprite],
    constr_ground_expr: Optional[str],
    constr_ground_sprite: Optional[LayoutSprite],
    constr_bbox: Optional[BoundingBox],
    sprite_table: dict[int, Any],
    pcx_path: str,
    out: list[str],
    fallback_frame_layouts: dict[int, str] | None = None,
    ground_only_stages: set[int] | None = None,
    first_building_stage: int = 0,
) -> tuple[str, dict[int, str]]:
    """
    Emit blocks for an animated climate variant (cg.animation_frames non-empty).

    The animation frame list comes from explicit var-0x46 ranges; the
    *default* branch of that switch (= frame N when no range matches) is
    stored in ``cg.completed``.  We append it to the frame list so all
    frames are included in the ``animation_frame`` switch.

    *fallback_frame_layouts* provides layout names from another climate variant
    (typically temperate) for frames that are ``None`` in the current variant.
    This fills gaps in the animation switch when the NFO uses the same sprite
    regardless of terrain for certain frames.

    Returns ``(top_name, frame_layouts)`` where *frame_layouts* maps frame
    indices to emitted spritelayout names for use as fallback by other variants.
    """
    # Merge explicit frames + default frame (cg.completed)
    all_frames: list[Optional[FrameLayout]] = list(cg.animation_frames)
    if cg.completed is not None:
        all_frames.append(cg.completed)

    if not all_frames or all(f is None for f in all_frames):
        # No frames resolved — emit a stub
        out.append(f"/* WARN: no animation frames resolved for {prefix} */")
        return prefix

    # ---- Ground sprite deduplication ------------------------------------
    # Key: (action1_flag, sprite_index)  →  (emitted expression, LayoutSprite)
    # For base-game ground sprites we use the literal number, no spriteset.
    ground_cache: dict[tuple[bool, int], tuple[str, LayoutSprite]] = {}

    def _get_ground_expr_for_frame(fl: FrameLayout, frame_idx: int) -> tuple[str, LayoutSprite]:
        gs = fl.ground
        cache_key = (gs.is_action1, gs.index)
        if cache_key in ground_cache:
            return ground_cache[cache_key]
        if not gs.is_action1:
            result = (str(gs.index), gs)
            ground_cache[cache_key] = result
            return result
        sc = sprite_table.get(gs.index)
        if sc is None:
            result = ("0", gs)
            ground_cache[cache_key] = result
            return result
        ss_name = f"{ss_prefix}_gf{frame_idx}"
        out.append(f"spriteset({ss_name}, \"{pcx_path}\") {{ {_sprite_literal(sc)} }}")
        result = (f"{ss_name}(0)", gs)
        ground_cache[cache_key] = result
        return result

    # ---- Building spriteset deduplication --------------------------------
    bldg_ss_cache: dict[int, str] = {}   # action1_index → emitted ss name

    def _get_bldg_ss_name(fl: FrameLayout, frame_idx: int) -> Optional[str]:
        bs = fl.building
        if not bs.is_action1:
            return None
        if bs.index in bldg_ss_cache:
            return bldg_ss_cache[bs.index]
        sc = sprite_table.get(bs.index)
        if sc is None:
            return None
        ss_name = f"{ss_prefix}_f{frame_idx}"
        out.append(f"spriteset({ss_name}, \"{pcx_path}\") {{ {_sprite_literal(sc)} }}")
        bldg_ss_cache[bs.index] = ss_name
        return ss_name

    # Pre-pass: emit all unique spriteset declarations
    g_exprs:  list[str]                       = []
    g_sprites: list[Optional[LayoutSprite]]    = []
    ss_names: list[Optional[str]]              = []

    for idx, fl in enumerate(all_frames):
        if fl is None:
            g_exprs.append("0")
            g_sprites.append(None)
            ss_names.append(None)
            continue
        g_expr, g_spr = _get_ground_expr_for_frame(fl, idx)
        g_exprs.append(g_expr)
        g_sprites.append(g_spr)
        ss_names.append(_get_bldg_ss_name(fl, idx))

    # ---- Spritelayout per frame -----------------------------------------
    frame_layout_names: list[Optional[str]] = []
    for idx, (fl, g_expr, g_spr, ss_name) in enumerate(
        zip(all_frames, g_exprs, g_sprites, ss_names)
    ):
        if ss_name is None:
            frame_layout_names.append(None)
            continue
        rc_clause = _recolour_clause(fl.building)
        grc_clause = _recolour_clause(g_spr) if g_spr is not None else ""
        layout_name = f"{prefix}_frame{idx}"
        out.append(f"spritelayout {layout_name} {{")
        out.append(f"\tground   {{ sprite: {g_expr};{grc_clause} }}")
        out.append(f"\tbuilding {{ sprite: {ss_name}(0);{rc_clause}{_bbox_clause(fl.bbox)} }}")
        out.append("}")
        frame_layout_names.append(layout_name)

    # Pick a fallback frame (last valid layout = cg.completed, appended at end)
    fallback_layout = next((n for n in reversed(frame_layout_names) if n is not None), None)
    if fallback_layout is None:
        out.append(f"/* WARN: no valid animation frames for {prefix}, using ground-only fallback */")
        out.append(f"spritelayout {prefix} {{")
        out.append(f"\tground {{ sprite: 0; }}")
        out.append("}")
        return prefix, {}

    # Build frame→layout mapping for this variant (used as fallback by other variants)
    own_frame_layouts: dict[int, str] = {}
    for idx, layout_name in enumerate(frame_layout_names):
        if layout_name is not None:
            own_frame_layouts[idx] = layout_name

    # ---- Animation frame switch ----------------------------------------
    anim_cases: list[str] = []
    for idx, layout_name in enumerate(frame_layout_names):
        if layout_name is not None:
            anim_cases.append(f"{idx}: {layout_name}")
        elif fallback_frame_layouts and idx in fallback_frame_layouts:
            # Frame missing in this variant but available from another climate
            # (e.g. temperate) — reuse that layout to avoid incorrect default.
            anim_cases.append(f"{idx}: {fallback_frame_layouts[idx]}")
    cases_str = "; ".join(anim_cases)
    out.append(
        f"switch (FEAT_HOUSES, SELF, {prefix}_anim, animation_frame) {{"
        f" {cases_str}; return {fallback_layout}; }}"
    )

    # ---- Construction spriteset ----------------------------------------
    if ground_only_stages is None:
        ground_only_stages = set()
    has_ground_only = bool(ground_only_stages) and constr_ground_expr is not None

    if constr_sprites:
        rc = _recolour_clause(constr_bsprite) if constr_bsprite is not None else ""
        grc = _recolour_clause(constr_ground_sprite) if constr_ground_sprite is not None else ""
        g_c = constr_ground_expr or "0"
        out.append(f"spriteset({ss_prefix}_c, \"{pcx_path}\") {{")
        for i, sc in enumerate(constr_sprites):
            out.append(f"\t{_sprite_literal(sc)}  /* constr {i} */")
        out.append("}")
        out.append(f"spritelayout {prefix}_constr {{")
        out.append(f"\tground   {{ sprite: {g_c};{grc} }}")
        expr = _build_layout_expr(len(constr_sprites), first_building_stage)
        out.append(f"\tbuilding {{ sprite: {ss_prefix}_c({expr});{rc}{_bbox_clause(constr_bbox)} }}")
        out.append("}")
        fallback = f"{prefix}_constr"
    else:
        fallback = f"{prefix}_anim"

    # Ground-only construction layout (for stages with no building sprite)
    if has_ground_only:
        grc_go = _recolour_clause(constr_ground_sprite) if constr_ground_sprite is not None else ""
        g_c_go = constr_ground_expr or "0"
        out.append(f"spritelayout {prefix}_constr_g {{")
        out.append(f"\tground   {{ sprite: {g_c_go};{grc_go} }}")
        out.append("}")

    if has_ground_only:
        # Explicit routing for ground-only stages
        cases: list[str] = []
        for stage in sorted(ground_only_stages):
            cases.append(f"{stage}: {prefix}_constr_g")
        cases.append(f"3: {prefix}_anim")
        cases_str = "; ".join(cases)
        out.append(
            f"switch (FEAT_HOUSES, SELF, {prefix}, construction_state) {{"
            f" {cases_str}; return {fallback}; }}"
        )
    else:
        out.append(
            f"switch (FEAT_HOUSES, SELF, {prefix}, construction_state) {{"
            f" 3: {prefix}_anim; return {fallback}; }}"
        )
    return prefix, own_frame_layouts


# ---------------------------------------------------------------------------
# Top-level emitter
# ---------------------------------------------------------------------------


def emit_house_tile_nml(
    house_id_hex: str,
    htg: HouseTileGraphics,
    sprite_table: dict[int, Any],
    pcx_path: str,
) -> tuple[list[str], str, list[int]]:
    """
    Emit NML blocks for all climate variants of one house tile.

    Returns ``(lines, entry_name, colour_values)`` where *entry_name* is always
    ``sl_ttrs_{house_id_hex}`` — the top-level identifier referenced by the
    ``item`` block graphics section, and *colour_values* is a (possibly empty)
    list of 15-bit colour callback results extracted from the CB 0x1E chain.
    Duplicate colour values are preserved to encode probability weights.
    """
    out: list[str] = []
    entry_name = f"sl_ttrs_{house_id_hex}"
    colour_values = list(htg.colour_values)

    has_temp      = htg.temperate   is not None and _cg_has_content(htg.temperate)
    has_snow      = htg.snow        is not None and _cg_has_content(htg.snow)
    has_tropic    = htg.tropic      is not None and _cg_has_content(htg.tropic)
    has_arctic_v2 = htg.arctic_v2   is not None and _cg_has_content(htg.arctic_v2)
    has_random    = any(_rvg_has_content(rvg) for rvg in htg.random_variants)

    if not has_temp and not has_snow and not has_tropic and not has_random:
        out.append(f"/* WARN 0x{house_id_hex}: graph traversal yielded no usable layouts */")
        out.append("")
        return out, entry_name, colour_values

    needs_terrain_switch = has_snow or has_tropic or has_arctic_v2
    climate_suffix_temp  = "_nosnow" if needs_terrain_switch else ""

    # ------------------------------------------------------------------
    # Per-climate blocks
    # ------------------------------------------------------------------
    top_temp_name: str | None = None
    top_snow_name: str | None = None
    top_tropic_name: str | None = None

    # When has_random=True and no terrain switch is needed, the random_switch
    # will be the sole entry point (= entry_name).  Do NOT emit htg.temperate
    # at the entry_name level or it will collide with the random_switch block.
    random_replaces_base = has_random and not needs_terrain_switch

    temp_frame_layouts: dict[int, str] = {}

    if has_temp and not random_replaces_base:
        prefix   = f"sl_ttrs_{house_id_hex}{climate_suffix_temp}"
        ss_pref  = f"ss_ttrs_{house_id_hex}{climate_suffix_temp}"
        _, temp_frame_layouts = _emit_climate_variant(prefix, ss_pref, htg.temperate, sprite_table, pcx_path, out)  # type: ignore[arg-type]
        out.append("")
        top_temp_name = prefix

    if has_snow:
        prefix  = f"sl_ttrs_{house_id_hex}_snow"
        ss_pref = f"ss_ttrs_{house_id_hex}_snow"
        _emit_climate_variant(prefix, ss_pref, htg.snow, sprite_table, pcx_path, out,  # type: ignore[arg-type]
                              fallback_frame_layouts=temp_frame_layouts or None)
        out.append("")
        top_snow_name = prefix

    if has_tropic:
        prefix  = f"sl_ttrs_{house_id_hex}_tropic"
        ss_pref = f"ss_ttrs_{house_id_hex}_tropic"
        _emit_climate_variant(prefix, ss_pref, htg.tropic, sprite_table, pcx_path, out,  # type: ignore[arg-type]
                              fallback_frame_layouts=temp_frame_layouts or None)
        out.append("")
        top_tropic_name = prefix

    # arctic_v2: separate arctic-below-snowline ground sprites
    top_arctic_v2_name: str | None = None
    if has_arctic_v2 and not has_snow:
        # No snow variant — arctic_v2 fills the snow slot
        prefix  = f"sl_ttrs_{house_id_hex}_snow"
        ss_pref = f"ss_ttrs_{house_id_hex}_snow"
        _emit_climate_variant(prefix, ss_pref, htg.arctic_v2, sprite_table, pcx_path, out,  # type: ignore[arg-type]
                              fallback_frame_layouts=temp_frame_layouts or None)
        out.append("")
        top_snow_name = prefix
    elif has_arctic_v2 and has_snow:
        # Both snow AND arctic_v2 exist — emit arctic_v2 as a separate variant
        prefix  = f"sl_ttrs_{house_id_hex}_arctic"
        ss_pref = f"ss_ttrs_{house_id_hex}_arctic"
        _emit_climate_variant(prefix, ss_pref, htg.arctic_v2, sprite_table, pcx_path, out,  # type: ignore[arg-type]
                              fallback_frame_layouts=temp_frame_layouts or None)
        out.append("")
        top_arctic_v2_name = prefix

    # ------------------------------------------------------------------
    # Random variants (append after primary variants, before terrain switch)
    # ------------------------------------------------------------------
    if has_random and not needs_terrain_switch:
        # Emit random_switch routing to each variant's completed sprite
        _emit_random_switch(house_id_hex, htg, sprite_table, pcx_path, out, entry_name)
        out.append("")
        # The random_switch IS the entry — no further terrain switch needed
        return out, entry_name, colour_values

    if has_random and needs_terrain_switch and top_temp_name is None:
        # Random variants exist but we also need a terrain switch (e.g. snow ground-only
        # on one tile of a multi-tile building that has random variants in non-snow climate).
        # Emit random variants under a _nosnow name so they don't collide with the
        # terrain-type switch that will occupy entry_name.
        rand_name = f"{entry_name}_nosnow"
        _emit_random_switch(house_id_hex, htg, sprite_table, pcx_path, out, rand_name)
        out.append("")
        top_temp_name = rand_name

    # ------------------------------------------------------------------
    # Terrain-type routing switch
    # ------------------------------------------------------------------
    if needs_terrain_switch:
        if top_arctic_v2_name and top_snow_name:
            # Both arctic_v2 (below snowline) and snow (above snowline) exist.
            # In arctic climate below snowline, terrain_type is TILETYPE_NORMAL
            # but ground sprites differ from temperate.  Use a runtime
            # var[0x03] (game climate) switch on the non-snow default branch:
            #   terrain_type == SNOW → snow
            #   terrain_type != SNOW → climate==1(arctic) ? arctic_v2 : temperate
            climate_sw = f"sl_ttrs_{house_id_hex}_clisw"
            other_fallback = top_temp_name or top_arctic_v2_name
            out.append(
                f"switch (FEAT_HOUSES, SELF, {climate_sw}, var[0x03, 0, 0xFF]) {{"
                f" 1: {top_arctic_v2_name}; return {other_fallback}; }}"
            )
            cases: list[str] = []
            cases.append(f"TILETYPE_SNOW: {top_snow_name}")
            if top_tropic_name:
                cases.append(f"TILETYPE_DESERT: {top_tropic_name}")
            cases_str = "; ".join(cases)
            out.append(
                f"switch (FEAT_HOUSES, SELF, {entry_name}, terrain_type) {{"
                f" {cases_str}; return {climate_sw}; }}"
            )
        else:
            fallback = top_temp_name or top_snow_name or top_tropic_name
            cases: list[str] = []
            if top_snow_name:
                cases.append(f"TILETYPE_SNOW: {top_snow_name}")
            if top_tropic_name:
                cases.append(f"TILETYPE_DESERT: {top_tropic_name}")
            cases_str = "; ".join(cases)
            out.append(
                f"switch (FEAT_HOUSES, SELF, {entry_name}, terrain_type) {{"
                f" {cases_str}; return {fallback}; }}"
            )
        out.append("")

    return out, entry_name, colour_values


def _emit_random_switch(
    house_id_hex: str,
    htg: HouseTileGraphics,
    sprite_table: dict[int, Any],
    pcx_path: str,
    out: list[str],
    entry_name: str,
) -> None:
    """Emit random_switch selecting among per-climate random variant spritelayouts.

    Each :class:`RandomVariantGraphics` may carry independent sprites for
    temperate, snow, and tropic climates.  When a variant has multiple
    climates, a per-variant ``terrain_type`` switch is emitted so the
    random selection properly routes to the correct climate sprites.
    """
    variant_names: list[str] = []

    for i, rvg in enumerate(htg.random_variants):
        v_entry = f"sl_ttrs_{house_id_hex}_rv{i}"

        has_temp   = rvg.temperate is not None and _cg_has_content(rvg.temperate)
        has_snow   = rvg.snow      is not None and _cg_has_content(rvg.snow)
        has_tropic = rvg.tropic    is not None and _cg_has_content(rvg.tropic)
        has_arctic = rvg.arctic_v2 is not None and _cg_has_content(rvg.arctic_v2)

        if not has_temp and not has_snow and not has_tropic and not has_arctic:
            continue

        needs_terrain = has_snow or has_tropic or has_arctic

        if needs_terrain:
            # --- Per-climate spritelayouts for this variant ----------------
            top_temp: str | None    = None
            top_snow: str | None    = None
            top_tropic: str | None  = None
            rv_temp_frame_layouts: dict[int, str] = {}

            if has_temp:
                prefix  = f"sl_ttrs_{house_id_hex}_rv{i}_nosnow"
                ss_pref = f"ss_ttrs_{house_id_hex}_rv{i}_nosnow"
                _, rv_temp_frame_layouts = _emit_climate_variant(prefix, ss_pref, rvg.temperate, sprite_table, pcx_path, out)  # type: ignore[arg-type]
                out.append("")
                top_temp = prefix

            if has_snow:
                prefix  = f"sl_ttrs_{house_id_hex}_rv{i}_snow"
                ss_pref = f"ss_ttrs_{house_id_hex}_rv{i}_snow"
                _emit_climate_variant(prefix, ss_pref, rvg.snow, sprite_table, pcx_path, out,  # type: ignore[arg-type]
                                      fallback_frame_layouts=rv_temp_frame_layouts or None)
                out.append("")
                top_snow = prefix

            if has_arctic and not has_snow:
                prefix  = f"sl_ttrs_{house_id_hex}_rv{i}_snow"
                ss_pref = f"ss_ttrs_{house_id_hex}_rv{i}_snow"
                _emit_climate_variant(prefix, ss_pref, rvg.arctic_v2, sprite_table, pcx_path, out,  # type: ignore[arg-type]
                                      fallback_frame_layouts=rv_temp_frame_layouts or None)
                out.append("")
                top_snow = prefix

            top_arctic: str | None = None
            if has_arctic and has_snow:
                prefix  = f"sl_ttrs_{house_id_hex}_rv{i}_arctic"
                ss_pref = f"ss_ttrs_{house_id_hex}_rv{i}_arctic"
                _emit_climate_variant(prefix, ss_pref, rvg.arctic_v2, sprite_table, pcx_path, out,  # type: ignore[arg-type]
                                      fallback_frame_layouts=rv_temp_frame_layouts or None)
                out.append("")
                top_arctic = prefix

            if has_tropic:
                prefix  = f"sl_ttrs_{house_id_hex}_rv{i}_tropic"
                ss_pref = f"ss_ttrs_{house_id_hex}_rv{i}_tropic"
                _emit_climate_variant(prefix, ss_pref, rvg.tropic, sprite_table, pcx_path, out,  # type: ignore[arg-type]
                                      fallback_frame_layouts=rv_temp_frame_layouts or None)
                out.append("")
                top_tropic = prefix

            # Terrain-type switch for this variant
            if top_arctic and top_snow:
                # Runtime climate switch for arctic_v2 + snow
                climate_sw = f"sl_ttrs_{house_id_hex}_rv{i}_clisw"
                other_fb = top_temp or top_arctic
                out.append(
                    f"switch (FEAT_HOUSES, SELF, {climate_sw}, var[0x03, 0, 0xFF]) {{"
                    f" 1: {top_arctic}; return {other_fb}; }}"
                )
                cases: list[str] = []
                cases.append(f"TILETYPE_SNOW: {top_snow}")
                if top_tropic:
                    cases.append(f"TILETYPE_DESERT: {top_tropic}")
                cases_str = "; ".join(cases)
                out.append(
                    f"switch (FEAT_HOUSES, SELF, {v_entry}, terrain_type) {{"
                    f" {cases_str}; return {climate_sw}; }}"
                )
            else:
                fallback = top_temp or top_snow or top_tropic
                cases: list[str] = []
                if top_snow:
                    cases.append(f"TILETYPE_SNOW: {top_snow}")
                if top_tropic:
                    cases.append(f"TILETYPE_DESERT: {top_tropic}")
                cases_str = "; ".join(cases)
                out.append(
                    f"switch (FEAT_HOUSES, SELF, {v_entry}, terrain_type) {{"
                    f" {cases_str}; return {fallback}; }}"
                )
            out.append("")
            variant_names.append(v_entry)

        else:
            # Single climate — emit directly
            cg = rvg.temperate
            if cg is not None and _cg_has_content(cg):
                vprefix  = v_entry
                vss_pref = f"ss_ttrs_{house_id_hex}_rv{i}"
                _emit_climate_variant(vprefix, vss_pref, cg, sprite_table, pcx_path, out)  # return value unused
                out.append("")
                variant_names.append(vprefix)

    if not variant_names:
        return

    # Use actual weights from entry repetition counts when available.
    weights = htg.random_weights
    weight_parts: list[str] = []
    for i, name in enumerate(variant_names):
        w = weights[i] if i < len(weights) else 1
        weight_parts.append(f"{w}: {name};")
    counts_str = " ".join(weight_parts)

    # Trigger handling: triggers=0x00 means no rerandomisation trigger
    # (NML default = randomised once on construction).  Non-zero triggers
    # are mapped to NML trigger constants.
    trigger_param = _nfo_trigger_to_nml(htg.random_triggers)
    trigger_sep = f", {trigger_param}" if trigger_param else ""
    out.append(
        f"random_switch (FEAT_HOUSES, SELF, {entry_name}{trigger_sep}) {{"
        f" {counts_str} }}"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# NFO trigger byte → NML trigger constant mapping (houses)
_NFO_HOUSE_TRIGGER_BITS: dict[int, str] = {
    0: "TRIGGER_HOUSE_TILELOOP",       # bit 0
    1: "TRIGGER_HOUSE_TOP_TILELOOP",   # bit 1
}


def _nfo_trigger_to_nml(trigger_byte: int) -> str:
    """Convert an NFO random-trigger byte to an NML trigger expression.

    Returns an empty string when *trigger_byte* is 0x00 (no trigger).
    """
    if trigger_byte == 0:
        return ""
    low7 = trigger_byte & 0x7F
    names: list[str] = []
    for bit, name in _NFO_HOUSE_TRIGGER_BITS.items():
        if low7 & (1 << bit):
            names.append(name)
    if not names:
        # Unknown trigger bits — emit raw value as comment for manual review
        return f"0 /* unknown NFO triggers: 0x{trigger_byte:02X} */"
    if len(names) == 1:
        return names[0]
    return f"bitmask({', '.join(names)})"


def _cg_has_content(cg: ClimateGraphics) -> bool:
    """True when *cg* has at least one non-None frame layout."""
    if cg.completed is not None:
        return True
    if any(fl is not None for fl in cg.construction_stages):
        return True
    if any(fl is not None for fl in cg.animation_frames):
        return True
    return False


def _rvg_has_content(rvg: RandomVariantGraphics) -> bool:
    """True when *rvg* has content in any climate."""
    for cg in (rvg.temperate, rvg.snow, rvg.tropic, rvg.arctic_v2):
        if cg is not None and _cg_has_content(cg):
            return True
    return False


def _build_layout_expr(n: int, offset: int = 0) -> str:
    """NML expression: map construction_state → spriteset index [0, n-1].

    When *offset* > 0 the first building stage is at construction_state==offset,
    so the expression subtracts the offset before indexing into the spriteset.
    """
    if n <= 1:
        return "0"
    base = f"(construction_state - {offset})" if offset > 0 else "construction_state"
    max_idx = n - 1
    return f"{base} < {n} ? {base} : {max_idx}"
