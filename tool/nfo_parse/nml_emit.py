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
    ClimateGraphics, FrameLayout, HouseTileGraphics, LayoutSprite,
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
) -> str:
    """
    Return an NML ground-sprite expression.

    When the ground comes from Action 1, a one-sprite ``spriteset`` is emitted
    into *out* and the expression ``{ss_name}(0)`` is returned.
    When it is a base-game sprite the literal number is returned.
    """
    if gsprite.is_action1:
        sc = sprite_table.get(gsprite.index)
        if sc is None:
            return "0"   # missing — use blank
        out.append(f"spriteset({ss_name}, \"{pcx_path}\") {{ {_sprite_literal(sc)} }}")
        return f"{ss_name}(0)"
    return str(gsprite.index)


def _recolour_clause(bsprite: LayoutSprite) -> str:
    if bsprite.has_recolour:
        return " recolour_mode: RECOLOUR_REMAP; palette: PALETTE_USE_DEFAULT;"
    return ""


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
) -> str:
    """
    Emit spriteset / spritelayout / switch blocks for *cg* (one climate).

    Returns the top-level NML identifier that routes into this climate's
    construction_state / animation switch.
    """
    # ------------------------------------------------------------------
    # Construction stages
    # ------------------------------------------------------------------
    constr_sprites: list[Any]           = []
    constr_recolour: bool               = False
    constr_ground_expr: Optional[str]   = None

    valid_constr = [fl for fl in cg.construction_stages if fl is not None]
    if valid_constr:
        seen_indices: list[int] = []
        for fl in valid_constr:
            bs = fl.building
            if bs.is_action1:
                idx = bs.index
                if idx not in seen_indices:
                    sc = sprite_table.get(idx)
                    if sc is not None:
                        constr_sprites.append(sc)
                        seen_indices.append(idx)
                        if bs.has_recolour:
                            constr_recolour = True
        # Ground for construction (use first valid stage's ground)
        constr_ground_expr = _ground_expr(
            valid_constr[0].ground,
            f"{ss_prefix}_gc",
            sprite_table, pcx_path, out,
        )

    # ------------------------------------------------------------------
    # Animation frames
    # ------------------------------------------------------------------
    if cg.animation_frames:
        return _emit_animated_variant(
            prefix, ss_prefix, cg, constr_sprites, constr_recolour,
            constr_ground_expr, sprite_table, pcx_path, out,
        )

    # ------------------------------------------------------------------
    # Completed stage (non-animated)
    # ------------------------------------------------------------------
    done_sprite_sc: Optional[Any] = None
    done_recolour: bool           = False
    done_ground_expr: Optional[str] = None

    if cg.completed is not None:
        fl = cg.completed
        bs = fl.building
        if bs.is_action1:
            sc = sprite_table.get(bs.index)
            done_sprite_sc = sc
            if bs.has_recolour:
                done_recolour = True
        done_ground_expr = _ground_expr(
            fl.ground, f"{ss_prefix}_gd", sprite_table, pcx_path, out,
        )

    # ------------------------------------------------------------------
    # Emit spritesets
    # ------------------------------------------------------------------
    # Construction spriteset
    if constr_sprites:
        rc = " recolour_mode: RECOLOUR_REMAP; palette: PALETTE_USE_DEFAULT;" if constr_recolour else ""
        out.append(f"spriteset({ss_prefix}_c, \"{pcx_path}\") {{")
        for i, sc in enumerate(constr_sprites):
            out.append(f"\t{_sprite_literal(sc)}  /* constr {i} */")
        out.append("}")
        g_c = constr_ground_expr or "0"
        out.append(f"spritelayout {prefix}_constr {{")
        out.append(f"\tground   {{ sprite: {g_c}; }}")
        expr = _build_layout_expr(len(constr_sprites))
        out.append(f"\tbuilding {{ sprite: {ss_prefix}_c({expr});{rc} }}")
        out.append("}")

    # Done spriteset
    if done_sprite_sc is not None:
        rc_d = " recolour_mode: RECOLOUR_REMAP; palette: PALETTE_USE_DEFAULT;" if done_recolour else ""
        out.append(f"spriteset({ss_prefix}_done, \"{pcx_path}\") {{")
        out.append(f"\t{_sprite_literal(done_sprite_sc)}  /* completed */")
        out.append("}")
        g_d = done_ground_expr or constr_ground_expr or "0"
        out.append(f"spritelayout {prefix}_done {{")
        out.append(f"\tground   {{ sprite: {g_d}; }}")
        out.append(f"\tbuilding {{ sprite: {ss_prefix}_done(0);{rc_d} }}")
        out.append("}")
    elif constr_sprites:
        # Reuse last construction sprite as completed stage
        g_c = constr_ground_expr or "0"
        rc = " recolour_mode: RECOLOUR_REMAP; palette: PALETTE_USE_DEFAULT;" if constr_recolour else ""
        out.append(f"spritelayout {prefix}_done {{")
        out.append(f"\tground   {{ sprite: {g_c}; }}")
        expr = _build_layout_expr(len(constr_sprites))
        out.append(f"\tbuilding {{ sprite: {ss_prefix}_c({expr});{rc} }}")
        out.append("}")

    # ------------------------------------------------------------------
    # Construction_state routing switch
    # ------------------------------------------------------------------
    has_constr = bool(constr_sprites)
    has_done   = bool(done_sprite_sc or (not done_sprite_sc and constr_sprites))

    if has_done:
        fallback = f"{prefix}_constr" if has_constr else f"{prefix}_done"
        out.append(
            f"switch (FEAT_HOUSES, SELF, {prefix}, construction_state) {{"
            f" 3: {prefix}_done; return {fallback}; }}"
        )
    elif has_constr:
        # No proper completed sprite — emit plain spritelayout as top-level
        out.append(f"/* NOTE: no completed sprite for {prefix}, using constr */")
        out.append(
            f"switch (FEAT_HOUSES, SELF, {prefix}, construction_state) {{"
            f" return {prefix}_constr; }}"
        )
    else:
        # Completely empty — emit a trivial fallback
        out.append(f"/* WARN: no sprites resolved for {prefix} */")

    return prefix


def _emit_animated_variant(
    prefix: str,
    ss_prefix: str,
    cg: ClimateGraphics,
    constr_sprites: list[Any],
    constr_recolour: bool,
    constr_ground_expr: Optional[str],
    sprite_table: dict[int, Any],
    pcx_path: str,
    out: list[str],
) -> str:
    """
    Emit blocks for an animated climate variant (cg.animation_frames non-empty).

    The animation frame list comes from explicit var-0x46 ranges; the
    *default* branch of that switch (= frame N when no range matches) is
    stored in ``cg.completed``.  We append it to the frame list so all
    frames are included in the ``animation_frame`` switch.
    """
    # Merge explicit frames + default frame (cg.completed)
    all_frames: list[FrameLayout] = list(cg.animation_frames)
    if cg.completed is not None:
        all_frames.append(cg.completed)

    if not all_frames:
        # No frames resolved — emit a stub
        out.append(f"/* WARN: no animation frames resolved for {prefix} */")
        return prefix

    # ---- Ground sprite deduplication ------------------------------------
    # Key: (action1_flag, sprite_index)  →  emitted spriteset name
    # For base-game ground sprites we use the literal number, no spriteset.
    ground_cache: dict[tuple[bool, int], str] = {}

    def _get_ground_expr_for_frame(fl: FrameLayout, frame_idx: int) -> str:
        gs = fl.ground
        cache_key = (gs.is_action1, gs.index)
        if cache_key in ground_cache:
            return ground_cache[cache_key]
        if not gs.is_action1:
            expr = str(gs.index)
            ground_cache[cache_key] = expr
            return expr
        sc = sprite_table.get(gs.index)
        if sc is None:
            ground_cache[cache_key] = "0"
            return "0"
        ss_name = f"{ss_prefix}_gf{frame_idx}"
        out.append(f"spriteset({ss_name}, \"{pcx_path}\") {{ {_sprite_literal(sc)} }}")
        expr = f"{ss_name}(0)"
        ground_cache[cache_key] = expr
        return expr

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
        rc = " recolour_mode: RECOLOUR_REMAP; palette: PALETTE_USE_DEFAULT;" if bs.has_recolour else ""
        out.append(f"spriteset({ss_name}, \"{pcx_path}\") {{ {_sprite_literal(sc)} }}")
        bldg_ss_cache[bs.index] = ss_name
        return ss_name

    # Pre-pass: emit all unique spriteset declarations
    g_exprs:  list[str]           = []
    ss_names: list[Optional[str]] = []
    rc_flags: list[bool]          = []

    for idx, fl in enumerate(all_frames):
        g_exprs.append(_get_ground_expr_for_frame(fl, idx))
        ss_names.append(_get_bldg_ss_name(fl, idx))
        rc_flags.append(fl.building.has_recolour)

    # ---- Spritelayout per frame -----------------------------------------
    frame_layout_names: list[Optional[str]] = []
    for idx, (fl, g_expr, ss_name, rc) in enumerate(
        zip(all_frames, g_exprs, ss_names, rc_flags)
    ):
        if ss_name is None:
            frame_layout_names.append(None)
            continue
        rc_clause = " recolour_mode: RECOLOUR_REMAP; palette: PALETTE_USE_DEFAULT;" if rc else ""
        layout_name = f"{prefix}_frame{idx}"
        out.append(f"spritelayout {layout_name} {{")
        out.append(f"\tground   {{ sprite: {g_expr}; }}")
        out.append(f"\tbuilding {{ sprite: {ss_name}(0);{rc_clause} }}")
        out.append("}")
        frame_layout_names.append(layout_name)

    # Pick a fallback frame (first valid layout)
    fallback_layout = next((n for n in frame_layout_names if n is not None), None)
    if fallback_layout is None:
        out.append(f"/* WARN: no valid animation frames for {prefix} */")
        return prefix

    # ---- Animation frame switch ----------------------------------------
    anim_cases: list[str] = []
    for idx, layout_name in enumerate(frame_layout_names):
        if layout_name is not None:
            anim_cases.append(f"{idx}: {layout_name}")
    cases_str = "; ".join(anim_cases)
    out.append(
        f"switch (FEAT_HOUSES, SELF, {prefix}_anim, animation_frame) {{"
        f" {cases_str}; return {fallback_layout}; }}"
    )

    # ---- Construction spriteset ----------------------------------------
    if constr_sprites:
        rc = " recolour_mode: RECOLOUR_REMAP; palette: PALETTE_USE_DEFAULT;" if constr_recolour else ""
        g_c = constr_ground_expr or "0"
        out.append(f"spriteset({ss_prefix}_c, \"{pcx_path}\") {{")
        for i, sc in enumerate(constr_sprites):
            out.append(f"\t{_sprite_literal(sc)}  /* constr {i} */")
        out.append("}")
        out.append(f"spritelayout {prefix}_constr {{")
        out.append(f"\tground   {{ sprite: {g_c}; }}")
        expr = _build_layout_expr(len(constr_sprites))
        out.append(f"\tbuilding {{ sprite: {ss_prefix}_c({expr});{rc} }}")
        out.append("}")
        fallback = f"{prefix}_constr"
    else:
        fallback = f"{prefix}_anim"

    out.append(
        f"switch (FEAT_HOUSES, SELF, {prefix}, construction_state) {{"
        f" 3: {prefix}_anim; return {fallback}; }}"
    )
    return prefix


# ---------------------------------------------------------------------------
# Top-level emitter
# ---------------------------------------------------------------------------


def emit_house_tile_nml(
    house_id_hex: str,
    htg: HouseTileGraphics,
    sprite_table: dict[int, Any],
    pcx_path: str,
) -> tuple[list[str], str]:
    """
    Emit NML blocks for all climate variants of one house tile.

    Returns ``(lines, entry_name)`` where *entry_name* is always
    ``sl_ttrs_{house_id_hex}`` — the top-level identifier referenced by the
    ``item`` block graphics section.
    """
    out: list[str] = []
    entry_name = f"sl_ttrs_{house_id_hex}"

    has_temp      = htg.temperate   is not None and _cg_has_content(htg.temperate)
    has_snow      = htg.snow        is not None and _cg_has_content(htg.snow)
    has_tropic    = htg.tropic      is not None and _cg_has_content(htg.tropic)
    has_arctic_v2 = htg.arctic_v2   is not None and _cg_has_content(htg.arctic_v2)
    has_random    = bool(htg.random_variants)

    if not has_temp and not has_snow and not has_tropic and not has_random:
        out.append(f"/* WARN 0x{house_id_hex}: graph traversal yielded no usable layouts */")
        out.append("")
        return out, entry_name

    needs_terrain_switch = has_snow or has_tropic or has_arctic_v2
    climate_suffix_temp  = "_nosnow" if needs_terrain_switch else ""

    # ------------------------------------------------------------------
    # Per-climate blocks
    # ------------------------------------------------------------------
    top_temp_name: str | None = None
    top_snow_name: str | None = None
    top_tropic_name: str | None = None

    if has_temp:
        prefix   = f"sl_ttrs_{house_id_hex}{climate_suffix_temp}"
        ss_pref  = f"ss_ttrs_{house_id_hex}{climate_suffix_temp}"
        _emit_climate_variant(prefix, ss_pref, htg.temperate, sprite_table, pcx_path, out)  # type: ignore[arg-type]
        out.append("")
        top_temp_name = prefix

    if has_snow:
        prefix  = f"sl_ttrs_{house_id_hex}_snow"
        ss_pref = f"ss_ttrs_{house_id_hex}_snow"
        _emit_climate_variant(prefix, ss_pref, htg.snow, sprite_table, pcx_path, out)  # type: ignore[arg-type]
        out.append("")
        top_snow_name = prefix

    if has_tropic:
        prefix  = f"sl_ttrs_{house_id_hex}_tropic"
        ss_pref = f"ss_ttrs_{house_id_hex}_tropic"
        _emit_climate_variant(prefix, ss_pref, htg.tropic, sprite_table, pcx_path, out)  # type: ignore[arg-type]
        out.append("")
        top_tropic_name = prefix

    # arctic_v2: use snow name collision if identical to snow (most cases it is)
    if has_arctic_v2 and not has_snow:
        prefix  = f"sl_ttrs_{house_id_hex}_snow"
        ss_pref = f"ss_ttrs_{house_id_hex}_snow"
        _emit_climate_variant(prefix, ss_pref, htg.arctic_v2, sprite_table, pcx_path, out)  # type: ignore[arg-type]
        out.append("")
        top_snow_name = prefix

    # ------------------------------------------------------------------
    # Random variants (append after primary variants, before terrain switch)
    # ------------------------------------------------------------------
    if has_random and not needs_terrain_switch:
        # Emit random_switch routing to each variant's completed sprite
        _emit_random_switch(house_id_hex, htg, sprite_table, pcx_path, out, entry_name)
        out.append("")
        # The random_switch IS the entry — no further terrain switch needed
        return out, entry_name

    # ------------------------------------------------------------------
    # Terrain-type routing switch
    # ------------------------------------------------------------------
    if needs_terrain_switch:
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

    return out, entry_name


def _emit_random_switch(
    house_id_hex: str,
    htg: HouseTileGraphics,
    sprite_table: dict[int, Any],
    pcx_path: str,
    out: list[str],
    entry_name: str,
) -> None:
    """Emit random_switch selecting among the random variant completed spritelayouts."""
    variant_names: list[str] = []
    for i, cg in enumerate(htg.random_variants):
        if not _cg_has_content(cg):
            continue
        vprefix  = f"sl_ttrs_{house_id_hex}_rv{i}"
        vss_pref = f"ss_ttrs_{house_id_hex}_rv{i}"
        _emit_climate_variant(vprefix, vss_pref, cg, sprite_table, pcx_path, out)
        out.append("")
        variant_names.append(vprefix)

    if not variant_names:
        return

    counts_str = " ".join(f"1: {name};" for name in variant_names)
    out.append(
        f"random_switch (FEAT_HOUSES, SELF, {entry_name}, bitmask(RANDBIT_CASE)) {{"
        f" {counts_str} }}"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cg_has_content(cg: ClimateGraphics) -> bool:
    """True when *cg* has at least one non-None frame layout."""
    if cg.completed is not None:
        return True
    if any(fl is not None for fl in cg.construction_stages):
        return True
    if cg.animation_frames:
        return True
    return False


def _build_layout_expr(n: int) -> str:
    """NML expression: map construction_state → spriteset index [0, n-1]."""
    if n <= 1:
        return "0"
    if n == 2:
        return "construction_state < 2 ? construction_state : 1"
    if n == 3:
        return "construction_state < 3 ? construction_state : 2"
    return "construction_state"
