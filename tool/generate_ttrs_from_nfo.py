#!/usr/bin/env python3
"""
Generate a unified TTRS NML file (spritesets + spritelayouts + item blocks) from ttrs3wmod.nfo.

For every house the output order is strictly:
    spriteset(s)  →  spritelayout(s)  →  anim-switch (if animated)
    →  tile-routing switch (if multi-tile)  →  item block

This ensures no "concurrent switch" ID is ever open for more than a handful
of lines (the 5 global availability switches are the only long-lived ones).

Usage:
    python -m tool.generate_ttrs_from_nfo \\
        --nfo tmp/ttrs3wmod.nfo --out src/ttrs.nml \\
        [--start-id 200] [--pcx-path src/sprites/pcx/ttrs3w.pcx]
"""
import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ============================================================================
# Regex patterns for NFO parsing
# ============================================================================

# House marker comments in NFO: // House XX or // House XX - label
HOUSE_MARKER_RE = re.compile(r"^// House ([0-9A-Fa-f]{2})(?:\s*-\s*(.+))?", re.IGNORECASE)

# Action 3 for houses: maps house IDs to graphics chains
# Format: <sprite> * <len> 03 07 01 <house_id> ...
ACTION3_HOUSE_RE = re.compile(r"^\s*\d+\s+\*\s+\d+\s+03\s+07\s+01\s+([0-9A-Fa-f]{2})\s+")

# Action 0 for house properties
# Format: <sprite> * <len> 00 07 <num_props> <num_houses> <first_id> [<props...>]
ACTION0_HOUSE_RE = re.compile(
    r"^\s*\d+\s+\*\s+\d+\s+00\s+07\s+([0-9A-Fa-f]{2})\s+([0-9A-Fa-f]{2})\s+([0-9A-Fa-f]{2})(?:\s+(.*))?$")

# Action 4 for house names (feature 07 with 0x40 offset = 0x48)
# Format: <sprite> * <len> 04 48 FF 01 <house_id> DC "<name>" 00
# Note: Names may contain escaped quotes like \"
NAME_RE = re.compile(r'\b04\s+48\s+(?:FF|7F|[0-9A-Fa-f]{2})\s+01\s+([0-9A-Fa-f]{2})\s+DC\s+"((?:[^"\\]|\\.)*)"\s+00')

# Hex byte pattern
HEX2_RE = re.compile(r"^[0-9A-Fa-f]{2}$")

# Property value patterns
BACKSLASH_B_RE = re.compile(r"\\b(\d+)")
BACKSLASH_W_RE = re.compile(r"\\w(\d+)")
BACKSLASH_WX_RE = re.compile(r"\\wx([0-9A-Fa-f]+)")

# NFO sprite line header
SPRITE_HEADER_RE = re.compile(r"^\s*\d+\s*\*\s*\d+")

# Action 1 for houses — defines sprite sets
# Format: <sprite> * <len> 01 07 <set_count> ...
ACTION1_HOUSE_RE = re.compile(r"^\s*\d+\s+\*\s+\d+\s+01\s+07\s+([0-9A-Fa-f]{2})\b")

# Action 2 for house type-00 layouts
# Format: <sprite> * <len> 02 07 <set_id> 00 <ground_b0 b1 b2 b3> <bldg_b0 b1 b2 b3> ...
ACTION2_HOUSE_LAYOUT_RE = re.compile(
    r"^\s*\d+\s+\*\s+\d+\s+02\s+07\s+([0-9A-Fa-f]{2})\s+00\s+(.*)$"
)

# NFO sprite coordinate line: index  sprites/pcx/ttrs3w.pcx  xpos ypos 01 ysize xsize xrel yrel
SPRITE_LINE_RE = re.compile(
    r"^\s*(\d+)\s+sprites/pcx/ttrs3w\.pcx\s+(-?\d+)\s+(-?\d+)\s+01\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)"
)

# ============================================================================
# Sprite layout constants (GRF Action2 sprite layout DWORD encoding)
# ============================================================================

SPRITE_ACTION1_FLAG = 0x80000000   # bit 31: sprite from Action1 (not base game)
SPRITE_RECOLOUR_FLAG = 0x00008000  # bit 15: enable recolouring
SPRITE_INDEX_MASK = 0x00003FFF     # bits 0-13: Action1 set index / base-game sprite number

# ============================================================================
# Constants and mappings
# ============================================================================

HOUSE_FLAG_BITS = {
    1: "HOUSE_FLAG_NOT_SLOPED",
    5: "HOUSE_FLAG_ANIMATE",
    6: "HOUSE_FLAG_CHURCH",
    7: "HOUSE_FLAG_STADIUM",
    8: "HOUSE_FLAG_ONLY_SE",
    9: "HOUSE_FLAG_PROTECTED",
    10: "HOUSE_FLAG_SYNC_CALLBACK",
    11: "HOUSE_FLAG_RANDOM_ANIMATION",
}

TOWNZONE_BITS = {
    0: "TOWNZONE_EDGE",
    1: "TOWNZONE_OUTSKIRT",
    2: "TOWNZONE_OUTER_SUBURB",
    3: "TOWNZONE_INNER_SUBURB",
    4: "TOWNZONE_CENTRE",
}

CLIMATE_BITS = {
    0: "CLIMATE_TEMPERATE",
    1: "CLIMATE_ARCTIC",
    2: "CLIMATE_TROPIC",
    3: "CLIMATE_TOYLAND",
}

# Multi-tile house base IDs from nmlc (action0properties.py old_houses dict).
# 2x1 base tiles: {74, 76, 87}  — each paired with base+1
# 1x2 base tiles: {7, 66, 68, 99} — each paired with base+1
# 2x2 base tiles: {20, 32, 40}    — each grouped with base+1, +2, +3
# Map each multi-tile base substitute ID to (NML size constant, num_tiles).
MULTI_TILE_BASES: dict[int, tuple[str, int]] = {}
for _base in (74, 76, 87):
    MULTI_TILE_BASES[_base] = ("HOUSE_SIZE_2X1", 2)
for _base in (7, 66, 68, 99):
    MULTI_TILE_BASES[_base] = ("HOUSE_SIZE_1X2", 2)
for _base in (20, 32, 40):
    MULTI_TILE_BASES[_base] = ("HOUSE_SIZE_2X2", 4)

# Map every multi-tile substitute ID (including secondary tiles) to its base.
SUBSTITUTE_TO_BASE: dict[int, int] = {}
for _base, (_sz, _nt) in MULTI_TILE_BASES.items():
    for _i in range(_nt):
        SUBSTITUTE_TO_BASE[_base + _i] = _base

# Size bits in building_flags that NML sets automatically for multi-tile houses.
SIZE_FLAG_BITS = {2, 3, 4}

# ============================================================================
# File I/O utilities
# ============================================================================


def read_nfo_text(nfo_path: Path) -> str:
    """Read NFO file with fallback encoding."""
    data = nfo_path.read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin1")


# ============================================================================
# Sprite data structures and parsing
# ============================================================================


@dataclass(frozen=True)
class SpriteCoord:
    xpos: int
    ypos: int
    xsize: int
    ysize: int
    xrel: int
    yrel: int


def sprite_literal(sprite: SpriteCoord) -> str:
    """Format sprite coordinates as NML literal."""
    return f"[{sprite.xpos}, {sprite.ypos}, {sprite.xsize}, {sprite.ysize}, {sprite.xrel}, {sprite.yrel}]"


def parse_house_action1_blocks(lines: list[str]) -> list[tuple[int, dict[int, SpriteCoord]]]:
    """Parse all Action 1 blocks for houses; return (line_index, sprite_table) pairs."""
    blocks: list[tuple[int, dict[int, SpriteCoord]]] = []

    for idx, line in enumerate(lines):
        match = ACTION1_HOUSE_RE.match(line)
        if not match:
            continue
        set_count = int(match.group(1), 16)
        if set_count <= 0:
            continue

        table: dict[int, SpriteCoord] = {}
        set_id = 0
        for body_line in lines[idx + 1:]:
            body_match = SPRITE_LINE_RE.match(body_line)
            if not body_match:
                if set_id > 0:
                    break
                continue
            xpos  = int(body_match.group(2))
            ypos  = int(body_match.group(3))
            ysize = int(body_match.group(4))
            xsize = int(body_match.group(5))
            xrel  = int(body_match.group(6))
            yrel  = int(body_match.group(7))
            table[set_id] = SpriteCoord(xpos, ypos, xsize, ysize, xrel, yrel)
            set_id += 1
            if set_id >= set_count:
                break
        if table:
            blocks.append((idx, table))

    return blocks


def sprite_table_for_house(
    house_line_index: int,
    action1_blocks: list[tuple[int, dict[int, SpriteCoord]]],
) -> dict[int, SpriteCoord]:
    """Return the last Action1 block whose definition precedes the given line."""
    chosen: dict[int, SpriteCoord] = {}
    for block_line_index, table in action1_blocks:
        if block_line_index <= house_line_index:
            chosen = table
        else:
            break
    return chosen


def collect_house_sections_from_action3(
    lines: list[str],
) -> list[tuple[str, str, list[str], int]]:
    """Collect per-house lines by scanning backwards from each Action 3 entry.

    Returns list of (house_id_hex, label, lines, start_line_index).
    """
    # Build marker label map
    markers: dict[str, tuple[str, int]] = {}
    for idx, line in enumerate(lines):
        m = HOUSE_MARKER_RE.match(line)
        if m:
            hid = m.group(1).upper()
            if hid not in markers:
                markers[hid] = ((m.group(2) or "").strip(), idx)

    # Locate all Action 3 entries
    action3_positions: list[tuple[str, int]] = []
    for idx, line in enumerate(lines):
        m = ACTION3_HOUSE_RE.match(line)
        if m:
            hid = m.group(1).upper()
            action3_positions.append((hid, idx))

    sections: list[tuple[str, str, list[str], int]] = []
    seen: set[str] = set()

    for house_id_hex, action3_idx in action3_positions:
        if house_id_hex in seen:
            continue
        seen.add(house_id_hex)

        search_start = max(0, action3_idx - 200)
        label = ""
        if house_id_hex in markers:
            label, marker_idx = markers[house_id_hex]
            search_start = max(search_start, marker_idx)
        # Don't include lines from a previous house's Action 3 entry
        for prev_id, prev_idx in action3_positions:
            if prev_idx < action3_idx and prev_idx >= search_start:
                search_start = max(search_start, prev_idx + 1)

        sections.append((
            house_id_hex,
            label,
            lines[search_start : action3_idx + 1],
            search_start,
        ))

    return sections


def collect_house_set_pairs(house_lines: list[str]) -> dict[int, tuple[int, int]]:
    """Return {set_id: (ground_dword, building_dword)} parsed from type-00 Action2 entries.

    DWORD encoding (little-endian 32-bit):
      bit 31 = 1 → sprite from Action1 set index; 0 → base-game sprite
      bit 15 = 1 → enable recolouring
      bits 0-13  → sprite number / Action1 set index

    Note: TTRS reuses low set IDs (0, 1, 3) for per-climate variants within the
    same house block.  The FIRST occurrence (temperate) is kept; later climate
    overrides (tropic reuses the same IDs) are silently ignored so that the
    default/temperate sprite is always preferred.
    """
    pairs: dict[int, tuple[int, int]] = {}
    for line in house_lines:
        match = ACTION2_HOUSE_LAYOUT_RE.match(line)
        if not match:
            continue
        set_id = int(match.group(1), 16)
        # First-wins: keep the temperate (first) definition for each set ID.
        if set_id in pairs:
            continue
        payload = match.group(2)
        raw = re.findall(r"\b[0-9A-Fa-f]{2}\b", payload)
        if len(raw) < 8:
            continue
        ground_dword = (int(raw[0], 16) | (int(raw[1], 16) << 8)
                        | (int(raw[2], 16) << 16) | (int(raw[3], 16) << 24))
        bldg_dword   = (int(raw[4], 16) | (int(raw[5], 16) << 8)
                        | (int(raw[6], 16) << 16) | (int(raw[7], 16) << 24))
        pairs[set_id] = (ground_dword, bldg_dword)
    return pairs


def split_climate_set_pairs(
    set_pairs: dict[int, tuple[int, int]],
) -> tuple[dict[int, tuple[int, int]], dict[int, tuple[int, int]]]:
    """Split a raw set_pairs dict into no-snow and snow sub-dicts.

    Two TTRS snow patterns are supported:

    Pattern A — full snow copy at IDs 0x10-0x1F (offset +0x10 from 0x00-0x0F):
      temp_pairs: IDs 0x00-0x0F (construction + completed, no-snow)
      arct_pairs: IDs 0x10-0x1F remapped back by -0x10

    Pattern B — only completed stage has a snow variant at ID 0x31
    (construction stages 0x00-0x02 are shared; 0x30 = completed no-snow):
      temp_pairs: construction IDs (0x00-0x0F) + { 0x30: set_pairs[0x30] }
      arct_pairs: same construction IDs      + { 0x30: set_pairs[0x31] }

    If arct_pairs is empty the house has no separate snow variant.
    """
    has_pattern_a = any(0x10 <= sid <= 0x1F for sid in set_pairs)
    has_pattern_b = 0x30 in set_pairs and not has_pattern_a
    # Pattern C: only type-00 entries at 0x0A (no-snow) and/or 0x0B (snow),
    # with no constr IDs (0x00-0x02), no Pattern-A/B keys.  Found in the
    # later TTRS additions (houses 0x90-0xA1 etc.) that are permanent tiles
    # with no construction scaffolding.
    has_pattern_c = (
        not has_pattern_a
        and not has_pattern_b
        and 0x0A in set_pairs
        and not any(0x00 <= sid <= 0x09 for sid in set_pairs)
    )

    temp_pairs: dict[int, tuple[int, int]] = {}
    arct_pairs: dict[int, tuple[int, int]] = {}

    if has_pattern_a:
        for sid, val in set_pairs.items():
            if 0x10 <= sid <= 0x1F:
                arct_pairs[sid - 0x10] = val
            else:
                temp_pairs[sid] = val

    elif has_pattern_b:
        # Construction stages are shared between both climate variants.
        constr = {sid: val for sid, val in set_pairs.items() if sid < 0x20}
        temp_pairs = dict(constr)
        temp_pairs[0x30] = set_pairs[0x30]
        if 0x31 in set_pairs:
            arct_pairs = dict(constr)
            arct_pairs[0x30] = set_pairs[0x31]  # remap snow-completed to same key

    elif has_pattern_c:
        # Remap 0x0A → completed no-snow (key 0x30), 0x0B → snow (key 0x30 in
        # arct_pairs) so the existing Case 1 path in _emit_climate_blocks handles
        # them without any further changes.
        temp_pairs[0x30] = set_pairs[0x0A]
        if 0x0B in set_pairs:
            arct_pairs[0x30] = set_pairs[0x0B]

    else:
        temp_pairs = dict(set_pairs)

    return temp_pairs, arct_pairs


def classify_sets(
    set_pairs: dict[int, tuple[int, int]],
    has_animation: bool,
) -> tuple[list[int], list[int], list[int]]:
    """Split set IDs into (constr_ids, completed_ids, anim_ids).

    After split_climate_set_pairs has been applied, each climate variant's
    set_pairs uses these conventions:

      constr_ids   : IDs 0x00-0x02 — construction scaffolding stages 0, 1, 2.
      completed_ids: ID  0x03       — Pattern A completed stage (same climate as
                                      construction sprites); OR
                     ID  0x30       — Pattern B completed stage (after
                                      split_climate_set_pairs remapped 0x31→0x30
                                      for the snow variant).
      anim_ids     : IDs 0x04-0x1F — only present when has_animation is True;
                                      in practice TTRS animation frames are stored
                                      in non-type-00 Action2, so these are usually
                                      empty for type-00 parsed data.
    """
    all_ids = sorted(set_pairs.keys())
    constr_ids    = [sid for sid in all_ids if 0x00 <= sid <= 0x02]
    completed_ids = [sid for sid in all_ids if sid == 0x03 or sid >= 0x20]
    anim_ids      = [sid for sid in all_ids if 0x04 <= sid <= 0x1F] if has_animation else []
    return constr_ids, completed_ids, anim_ids


def _pick_ground_dword(set_ids: list[int], set_pairs: dict[int, tuple[int, int]]) -> int:
    """Pick the best ground sprite DWORD from the given sets.

    Prefers Action1 ground over base-game sprite; among equals prefers the last
    set (usually the completed-stage / last animation frame).
    """
    for sid in reversed(set_ids):
        gd = set_pairs[sid][0]
        if gd & SPRITE_ACTION1_FLAG:
            return gd
    return set_pairs[set_ids[-1]][0]


# ============================================================================
# Sprite-block emitter  (produces lines that go BEFORE the item block)
# ============================================================================


def _build_layout_expr(n: int) -> str:
    """NML index expression mapping construction_state → spriteset index [0, n-1]."""
    if n <= 1:
        return "0"
    if n == 2:
        return "construction_state < 2 ? construction_state : 1"
    if n == 3:
        return "construction_state < 3 ? construction_state : 2"
    return "construction_state"  # n == 4: states 0-3 map 1:1


def _ground_expr_for_set(
    sid: int,
    name_prefix: str,
    set_pairs: dict[int, tuple[int, int]],
    sprite_table: dict[int, SpriteCoord],
    pcx_path: str,
    out: list[str],
) -> Optional[str]:
    """Compute the NML ground-sprite expression for a given type-00 set.

    Emits a one-sprite spriteset with the unique name ``name_prefix`` when the
    ground comes from Action1; otherwise returns a plain integer literal.
    Returns None when the Action1 sprite is missing from the table.
    """
    gd = set_pairs[sid][0]
    if gd & SPRITE_ACTION1_FLAG:
        gidx = gd & SPRITE_INDEX_MASK
        sp = sprite_table.get(gidx)
        if sp is None:
            return None
        out.append(f"spriteset({name_prefix}, \"{pcx_path}\") {{ {sprite_literal(sp)} }}")
        return f"{name_prefix}(0)"
    return str(gd & SPRITE_INDEX_MASK)


def _emit_climate_blocks(
    house_id_hex: str,
    climate_suffix: str,
    has_animation: bool,
    set_pairs: dict[int, tuple[int, int]],
    sprite_table: dict[int, SpriteCoord],
    pcx_path: str,
) -> tuple[list[str], str]:
    """Emit spritesets + layouts for a single climate variant of one house tile.

    ``climate_suffix`` is appended to every identifier for this variant,
    e.g. "" (no arctic variant), "_nosnow", or "_snow".

    Grounds are computed directly from each stage's type-00 entry so that
    construction and completed stages can use different terrain tiles.

    Returns (lines, top_level_name) where top_level_name is the NML identifier
    the caller should reference for this climate path.
    """
    out: list[str] = []
    base   = f"sl_ttrs_{house_id_hex}{climate_suffix}"
    ss_base = f"ss_ttrs_{house_id_hex}{climate_suffix}"

    constr_ids, completed_ids, anim_ids = classify_sets(set_pairs, has_animation)

    # ====================================================================
    # Fallback: no usable sets at all
    # ====================================================================
    if not constr_ids and not completed_ids and not anim_ids:
        out.append(f"/* WARN 0x{house_id_hex}{climate_suffix}: no valid sets after classification */")
        out.append("")
        return out, base

    # ====================================================================
    # Resolve construction building + ground sprites
    # ====================================================================
    constr_sprites: list[SpriteCoord] = []
    constr_has_recolour = False
    if constr_ids:
        seen_constr: list[int] = []
        for sid in constr_ids:
            bd = set_pairs[sid][1]
            if bd & SPRITE_ACTION1_FLAG:
                bidx = bd & SPRITE_INDEX_MASK
                if bd & SPRITE_RECOLOUR_FLAG:
                    constr_has_recolour = True
                if bidx not in seen_constr:
                    sp = sprite_table.get(bidx)
                    if sp is not None:
                        constr_sprites.append(sp)
                        seen_constr.append(bidx)

    # Construction ground: use the first constr set's ground (typically transparent).
    constr_gexpr: Optional[str] = None
    if constr_ids:
        constr_gexpr = _ground_expr_for_set(
            constr_ids[0], f"{ss_base}_gc", set_pairs, sprite_table, pcx_path, out,
        )

    # ====================================================================
    # Resolve completed building + ground sprites
    # ====================================================================
    done_sprite: Optional[SpriteCoord] = None
    done_has_recolour = False
    done_gexpr: Optional[str] = None
    if completed_ids:
        cid = completed_ids[0]
        bd = set_pairs[cid][1]
        if bd & SPRITE_ACTION1_FLAG:
            bidx = bd & SPRITE_INDEX_MASK
            if bd & SPRITE_RECOLOUR_FLAG:
                done_has_recolour = True
            done_sprite = sprite_table.get(bidx)
        done_gexpr = _ground_expr_for_set(
            cid, f"{ss_base}_gd", set_pairs, sprite_table, pcx_path, out,
        )

    # ====================================================================
    # Animation-frame sprites (type-00 only; usually empty for TTRS)
    # ====================================================================
    anim_sprites: list[SpriteCoord] = []
    anim_has_recolour = False
    for sid in anim_ids:
        bd = set_pairs[sid][1]
        if bd & SPRITE_RECOLOUR_FLAG:
            anim_has_recolour = True
        if bd & SPRITE_ACTION1_FLAG:
            bidx = bd & SPRITE_INDEX_MASK
            sp = sprite_table.get(bidx)
            if sp is not None:
                anim_sprites.append(sp)

    # ====================================================================
    # Case 1: has completed_ids  (Pattern A id=0x03, or Pattern B id=0x30)
    # — emit separate constr / done spritelayouts + routing switch
    # ====================================================================
    if completed_ids and (constr_sprites or done_sprite):
        # --- construction spriteset ---
        if constr_sprites:
            recolour_c = " recolour_mode: RECOLOUR_REMAP; palette: PALETTE_USE_DEFAULT;" if constr_has_recolour else ""
            out.append(f"spriteset({ss_base}_c, \"{pcx_path}\") {{")
            for i, sp in enumerate(constr_sprites):
                out.append(f"\t{sprite_literal(sp)}  /* constr {i} */")
            out.append("}")
            g_c = constr_gexpr or "0"
            out.append(f"spritelayout {base}_constr {{")
            out.append(f"\tground   {{ sprite: {g_c}; }}")
            out.append(
                f"\tbuilding {{ sprite: {ss_base}_c"
                f"({_build_layout_expr(len(constr_sprites))});{recolour_c} }}"
            )
            out.append("}")

        # --- completed spriteset ---
        if done_sprite:
            recolour_d = " recolour_mode: RECOLOUR_REMAP; palette: PALETTE_USE_DEFAULT;" if done_has_recolour else ""
            out.append(f"spriteset({ss_base}_done, \"{pcx_path}\") {{")
            out.append(f"\t{sprite_literal(done_sprite)}  /* completed */")
            out.append("}")
            g_d = done_gexpr or constr_gexpr or "0"
            out.append(f"spritelayout {base}_done {{")
            out.append(f"\tground   {{ sprite: {g_d}; }}")
            out.append(f"\tbuilding {{ sprite: {ss_base}_done(0);{recolour_d} }}")
            out.append("}")
        else:
            # No completed building sprite — reuse last constr sprite
            if constr_sprites:
                g_d = constr_gexpr or "0"
                out.append(f"spritelayout {base}_done {{")
                out.append(f"\tground   {{ sprite: {g_d}; }}")
                out.append(
                    f"\tbuilding {{ sprite: {ss_base}_c"
                    f"({_build_layout_expr(len(constr_sprites))});{recolour_c} }}"  # type: ignore[possibly-undefined]
                )
                out.append("}")

        # --- construction_state routing switch ---
        fallback = f"{base}_constr" if constr_sprites else f"{base}_done"
        out.append(
            f"switch (FEAT_HOUSES, SELF, {base}, construction_state) {{"
            f" 3: {base}_done; return {fallback}; }}"
        )
        out.append("")
        return out, base

    # ====================================================================
    # Case 2: animation frames present  (has_animation + anim_ids non-empty)
    # ====================================================================
    if anim_sprites:
        anim_recolour = " recolour_mode: RECOLOUR_REMAP; palette: PALETTE_USE_DEFAULT;" if anim_has_recolour else ""
        out.append(f"spriteset({ss_base}_anim, \"{pcx_path}\") {{")
        for i, sp in enumerate(anim_sprites):
            out.append(f"\t{sprite_literal(sp)}  /* frame {i} */")
        out.append("}")
        g_anim = constr_gexpr or "0"
        if constr_sprites:
            recolour_c = " recolour_mode: RECOLOUR_REMAP; palette: PALETTE_USE_DEFAULT;" if constr_has_recolour else ""
            out.append(f"spriteset({ss_base}_c, \"{pcx_path}\") {{")
            for i, sp in enumerate(constr_sprites):
                out.append(f"\t{sprite_literal(sp)}  /* constr {i} */")
            out.append("}")
            out.append(f"spritelayout {base}_constr {{")
            out.append(f"\tground   {{ sprite: {g_anim}; }}")
            out.append(
                f"\tbuilding {{ sprite: {ss_base}_c"
                f"({_build_layout_expr(len(constr_sprites))});{recolour_c} }}"
            )
            out.append("}")
        out.append(f"spritelayout {base}_done {{")
        out.append(f"\tground   {{ sprite: {g_anim}; }}")
        out.append(f"\tbuilding {{ sprite: {ss_base}_anim(animation_frame);{anim_recolour} }}")
        out.append("}")
        fallback = f"{base}_constr" if constr_sprites else f"{base}_done"
        out.append(
            f"switch (FEAT_HOUSES, SELF, {base}, construction_state) {{"
            f" 3: {base}_done; return {fallback}; }}"
        )
        out.append("")
        return out, base

    # ====================================================================
    # Case 3: only construction stages, no completed, no anim
    # (rare fallback — treat last constr stage as the "done" sprite)
    # ====================================================================
    if constr_sprites:
        recolour_c = " recolour_mode: RECOLOUR_REMAP; palette: PALETTE_USE_DEFAULT;" if constr_has_recolour else ""
        out.append(f"spriteset({ss_base}_b, \"{pcx_path}\") {{")
        for i, sp in enumerate(constr_sprites):
            out.append(f"\t{sprite_literal(sp)}  /* constr/done {i} */")
        out.append("}")
        g_c = constr_gexpr or "0"
        out.append(f"spritelayout {base} {{")
        out.append(f"\tground   {{ sprite: {g_c}; }}")
        out.append(
            f"\tbuilding {{ sprite: {ss_base}_b"
            f"({_build_layout_expr(len(constr_sprites))});{recolour_c} }}"
        )
        out.append("}")
        out.append("")
        return out, base

    out.append(f"/* WARN 0x{house_id_hex}{climate_suffix}: no sprites could be resolved */")
    out.append("")
    return out, base


def emit_sprite_blocks(
    house_id_hex: str,
    props: dict[str, object],
    set_pairs: dict[int, tuple[int, int]],
    sprite_table: dict[int, SpriteCoord],
    pcx_path: str,
) -> tuple[list[str], str]:
    """Emit spriteset(s), spritelayout(s), and routing switches for one house tile.

    Internally splits set_pairs into no-snow and snow sub-dicts (Pattern A or B).
    When arctic sets exist a terrain_type switch named ``sl_ttrs_XX`` routes to
    the appropriate variant.  Ground sprites are computed per stage inside
    _emit_climate_blocks.

    Returns (lines, layout_entry_name) where layout_entry_name is always
    ``sl_ttrs_{house_id_hex}`` regardless of whether arctic variants exist.
    """
    out: list[str] = []
    entry_name = f"sl_ttrs_{house_id_hex}"

    if not set_pairs:
        out.append(f"/* WARN 0x{house_id_hex}: no Action2 type-00 sets found — skipping sprite blocks */")
        out.append("")
        return out, entry_name

    # --- Split no-snow / snow (Pattern A or B) --------------------------
    temp_pairs, arct_pairs = split_climate_set_pairs(set_pairs)
    has_arctic = bool(arct_pairs)

    # --- Detect animation from house properties -------------------------
    anim_info_raw = props.get("1A")
    anim_info_val = token_to_int(anim_info_raw) if isinstance(anim_info_raw, str) else None
    low_raw  = props.get("09")
    high_raw = props.get("19")
    low_val  = token_to_int(low_raw)  if isinstance(low_raw,  str) else 0
    high_val = token_to_int(high_raw) if isinstance(high_raw, str) else 0
    building_flags_mask = (low_val or 0) + ((high_val or 0) << 8)
    has_animation = bool(
        (building_flags_mask & (1 << 5))
        and anim_info_val is not None
    )

    # Quick sanity check using the 3-tuple classify_sets
    temp_constr, temp_completed, temp_anim = classify_sets(temp_pairs, has_animation)
    if not temp_constr and not temp_completed and not temp_anim:
        out.append(f"/* WARN 0x{house_id_hex}: set classification yielded no usable sets */")
        out.append("")
        return out, entry_name

    # --- Emit no-snow variant -------------------------------------------
    temp_suffix = "_nosnow" if has_arctic else ""
    temp_lines, temp_top = _emit_climate_blocks(
        house_id_hex, temp_suffix, has_animation, temp_pairs, sprite_table, pcx_path,
    )
    out.extend(temp_lines)

    # --- Emit snow variant (if any) -------------------------------------
    if has_arctic:
        arct_lines, arct_top = _emit_climate_blocks(
            house_id_hex, "_snow", has_animation, arct_pairs, sprite_table, pcx_path,
        )
        out.extend(arct_lines)

        # terrain_type routing switch — this IS the entry_name the item uses
        out.append(
            f"switch (FEAT_HOUSES, SELF, {entry_name}, terrain_type) {{"
            f" TILETYPE_SNOW: {arct_top}; return {temp_top}; }}"
        )
        out.append("")

    return out, entry_name


# ============================================================================
# String utilities
# ============================================================================


def sanitize_identifier(text: str) -> str:
    ident = text.lower()
    ident = ident.replace("\u201c", "").replace("\u201d", "")
    ident = ident.replace('"', "")
    ident = ident.replace("'", "")
    ident = re.sub(r"[^a-z0-9]+", "_", ident).strip("_")
    if not ident:
        ident = "unnamed"
    return ident


def normalize_name(raw_name: str) -> str:
    """Clean up house name from NFO string."""
    text = raw_name.strip()

    # Unescape NFO-style escaped quotes
    text = text.replace('\\"', '"')

    # Repair common mojibake (UTF-8 bytes decoded as latin1)
    if any(marker in text for marker in ("Ã", "Â")):
        try:
            repaired = text.encode("latin1").decode("utf-8")
            text = repaired
        except UnicodeError:
            pass

    # Remove NewGRF prefix markers
    text = re.sub(r"^[Þþ\x9e\x9f]+", "", text).strip()
    return text or "Unnamed"


# ============================================================================
# Parsing utilities
# ============================================================================


def parse_token_value(token: str) -> Optional[int]:
    """Parse a single token to an integer value."""
    if HEX2_RE.match(token):
        return int(token, 16)

    m = BACKSLASH_B_RE.match(token)
    if m:
        return int(m.group(1), 10)

    m = BACKSLASH_W_RE.match(token)
    if m:
        return int(m.group(1), 10)

    m = BACKSLASH_WX_RE.match(token)
    if m:
        return int(m.group(1), 16)

    return None


def tokenize_line(line: str) -> list[str]:
    """Extract tokens from an NFO line."""
    # Remove comments
    line = re.sub(r'//.*', '', line)
    # Find all tokens: hex bytes, \b###, \w###, \wx###
    return re.findall(r"\\b\d+|\\w\d+|\\wx[0-9A-Fa-f]+|[0-9A-Fa-f]{2}", line)


def parse_names(nfo_text: str) -> dict[int, str]:
    """Extract house names from Action 4 entries."""
    names: dict[int, str] = {}
    for match in NAME_RE.finditer(nfo_text):
        house_id = int(match.group(1), 16)
        raw_name = normalize_name(match.group(2))
        # Only keep the first (English) name for each ID
        if house_id not in names:
            names[house_id] = raw_name
    return names


def extract_house_ids_from_action3(lines: list[str]) -> dict[int, int]:
    """Find all house IDs defined via Action 3, return {house_id: line_number}."""
    house_ids: dict[int, int] = {}
    for line_num, line in enumerate(lines, start=1):
        match = ACTION3_HOUSE_RE.match(line)
        if match:
            house_id = int(match.group(1), 16)
            if house_id not in house_ids:
                house_ids[house_id] = line_num
    return house_ids


def parse_action0_properties(lines: list[str]) -> dict[int, dict[str, object]]:
    """Parse Action 0 property definitions for houses.
    
    Properties from earlier definitions are preserved - later bulk operations
    won't overwrite already-set values (except substitute which is special).
    """
    properties: dict[int, dict[str, object]] = {}

    line_num = 0
    while line_num < len(lines):
        line = lines[line_num]
        match = ACTION0_HOUSE_RE.match(line)
        if not match:
            line_num += 1
            continue

        num_props = int(match.group(1), 16)
        num_houses = int(match.group(2), 16)
        first_id = int(match.group(3), 16)
        rest = match.group(4) or ""

        if num_props == 0 or num_houses == 0:
            line_num += 1
            continue

        # Collect continuation lines (lines that don't start with sprite header)
        content_lines = [rest]
        k = line_num + 1
        while k < len(lines):
            next_line = lines[k]
            stripped = next_line.strip()
            # Stop at empty lines, sprite headers (digits * digits), or comments
            if not stripped or SPRITE_HEADER_RE.match(next_line) or next_line.startswith("//"):
                break
            content_lines.append(stripped)
            k += 1

        full_content = " ".join(content_lines)

        tokens = tokenize_line(full_content)

        # Detect bulk disable operations: single prop 08 with all FF values
        # These are override commands (substitute=255 means disable), not actual house definitions
        is_bulk_disable = False
        if num_props == 1 and len(tokens) >= 1 + num_houses and tokens[0].upper() == "08":
            # Check if all values are FF (disable marker)
            values = tokens[1:1 + num_houses]
            if all(v.upper() == "FF" for v in values):
                is_bulk_disable = True

        # Parse property values for each house
        for h in range(num_houses):
            house_id = first_id + h
            if house_id not in properties:
                properties[house_id] = {}

            props = properties[house_id]

            p = 0
            prop_count = 0
            while p < len(tokens) and prop_count < num_props:
                key = tokens[p]
                p += 1

                if not HEX2_RE.match(key):
                    continue

                prop_count += 1
                key_int = int(key, 16)

                # Property 08: substitute type (1 byte)
                # Keep the first definition - later overrides are typically disables
                # Skip bulk disable operations (substitute=FF)
                if key_int == 0x08:
                    if p < len(tokens):
                        val = parse_token_value(tokens[p])
                        if val is not None:
                            # Skip bulk disable, otherwise keep first definition
                            if not is_bulk_disable and "08" not in props:
                                props["08"] = f"{val:02X}"
                        p += 1

                # Property 09: building_flags low byte
                elif key_int == 0x09:
                    if p < len(tokens):
                        val = parse_token_value(tokens[p])
                        if val is not None and "09" not in props:
                            props["09"] = f"{val:02X}"
                        p += 1

                # Property 0A: years_available (2 words)
                elif key_int == 0x0A:
                    if p + 1 < len(tokens):
                        y0 = parse_token_value(tokens[p])
                        y1 = parse_token_value(tokens[p + 1])
                        if y0 is not None and y1 is not None and "0A" not in props:
                            props["0A"] = [f"\\b{y0}", f"\\b{y1}"]
                        p += 2

                # Property 0B: population
                elif key_int == 0x0B:
                    if p < len(tokens):
                        val = parse_token_value(tokens[p])
                        if val is not None and "0B" not in props:
                            props["0B"] = f"{val:02X}"
                        p += 1

                # Property 0C: mail_multiplier
                elif key_int == 0x0C:
                    if p < len(tokens):
                        val = parse_token_value(tokens[p])
                        if val is not None and "0C" not in props:
                            props["0C"] = f"{val:02X}"
                        p += 1

                # Property 0D: passenger acceptance
                elif key_int == 0x0D:
                    if p < len(tokens):
                        val = parse_token_value(tokens[p])
                        if val is not None and "0D" not in props:
                            props["0D"] = f"{val:02X}"
                        p += 1

                # Property 0E: mail acceptance
                elif key_int == 0x0E:
                    if p < len(tokens):
                        val = parse_token_value(tokens[p])
                        if val is not None and "0E" not in props:
                            props["0E"] = f"{val:02X}"
                        p += 1

                # Property 0F: goods acceptance
                elif key_int == 0x0F:
                    if p < len(tokens):
                        val = parse_token_value(tokens[p])
                        if val is not None and "0F" not in props:
                            props["0F"] = f"{val:02X}"
                        p += 1

                # Property 10: local_authority_impact
                elif key_int == 0x10:
                    if p < len(tokens):
                        val = parse_token_value(tokens[p])
                        if val is not None and "10" not in props:
                            props["10"] = f"{val:02X}"
                        p += 1

                # Property 11: removal_cost_multiplier
                elif key_int == 0x11:
                    if p < len(tokens):
                        val = parse_token_value(tokens[p])
                        if val is not None and "11" not in props:
                            props["11"] = f"{val:02X}"
                        p += 1

                # Property 12: name string ID
                elif key_int == 0x12:
                    if p < len(tokens):
                        val = parse_token_value(tokens[p])
                        if val is not None and "12" not in props:
                            props["12"] = f"{val:02X}"
                        p += 1

                # Property 13: availability_mask (2 bytes)
                elif key_int == 0x13:
                    if p + 1 < len(tokens):
                        zone = parse_token_value(tokens[p])
                        climate = parse_token_value(tokens[p + 1])
                        if zone is not None and climate is not None and "13" not in props:
                            props["13"] = [f"{zone:02X}", f"{climate:02X}"]
                        p += 2

                # Property 16: refresh_multiplier
                elif key_int == 0x16:
                    if p < len(tokens):
                        val = parse_token_value(tokens[p])
                        if val is not None and "16" not in props:
                            props["16"] = f"{val:02X}"
                        p += 1

                # Property 18: probability
                elif key_int == 0x18:
                    if p < len(tokens):
                        val = parse_token_value(tokens[p])
                        if val is not None and "18" not in props:
                            props["18"] = f"{val:02X}"
                        p += 1

                # Property 19: building_flags high byte
                elif key_int == 0x19:
                    if p < len(tokens):
                        val = parse_token_value(tokens[p])
                        if val is not None and "19" not in props:
                            props["19"] = f"{val:02X}"
                        p += 1

                # Property 1A: animation_info
                elif key_int == 0x1A:
                    if p < len(tokens):
                        val = parse_token_value(tokens[p])
                        if val is not None and "1A" not in props:
                            props["1A"] = f"{val:02X}"
                        p += 1

                # Property 1B: animation_speed
                elif key_int == 0x1B:
                    if p < len(tokens):
                        val = parse_token_value(tokens[p])
                        if val is not None and "1B" not in props:
                            props["1B"] = f"{val:02X}"
                        p += 1

                # Property 1C: building_class
                elif key_int == 0x1C:
                    if p < len(tokens):
                        val = parse_token_value(tokens[p])
                        if val is not None and "1C" not in props:
                            props["1C"] = f"{val:02X}"
                        p += 1

                # Property 1F: minimum_lifetime
                elif key_int == 0x1F:
                    if p < len(tokens):
                        val = parse_token_value(tokens[p])
                        if val is not None and "1F" not in props:
                            props["1F"] = f"{val:02X}"
                        p += 1

                # Unknown property - skip 1 byte
                else:
                    p += 1

        line_num += 1

    return properties


def collect_house_action0_blocks(lines: list[str]) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    for idx, line in enumerate(lines):
        m = HOUSE_MARKER_RE.match(line)
        if not m:
            continue
        house_id = m.group(1).upper()

        j = idx + 1
        while j < len(lines) and not lines[j].startswith("// House "):
            if "00 07" in lines[j]:
                break
            j += 1

        if j >= len(lines) or "00 07" not in lines[j]:
            continue

        chunk = [lines[j]]
        k = j + 1
        while k < len(lines):
            stripped = lines[k].strip()
            # Stop at empty lines, house markers, or new sprite headers
            if not stripped or lines[k].startswith("// House ") or SPRITE_HEADER_RE.match(lines[k]):
                break
            chunk.append(lines[k])
            k += 1

        blocks.append((house_id, " ".join(re.sub(r'//.*', '', part).strip() for part in chunk)))
    return blocks


def tokenize_action0(block_text: str) -> list[str]:
    cleaned = re.sub(r'"[^"]*"', ' ', block_text)
    return re.findall(r"\\b\d+|\\w\d+|\\wx[0-9A-Fa-f]+|[0-9A-Fa-f]{2}", cleaned)


def parse_props_from_tokens(tokens: list[str]) -> dict[str, object]:
    props: dict[str, object] = {}

    start = -1
    for i in range(len(tokens) - 4):
        if tokens[i] == "00" and tokens[i + 1] == "07":
            start = i
            break
    if start == -1:
        return props

    # 00 07 <len> 01 <house_id> <props...>
    payload = tokens[start + 5:]
    p = 0
    while p < len(payload):
        key = payload[p]
        p += 1
        if not HEX2_RE.match(key):
            continue

        if key == "0A":
            if p + 1 < len(payload):
                props[key] = [payload[p], payload[p + 1]]
                p += 2
            continue

        if key == "13":
            if p + 1 < len(payload):
                props[key] = [payload[p], payload[p + 1]]
                p += 2
            continue

        if key == "1E":
            vals: list[str] = []
            while p < len(payload) and HEX2_RE.match(payload[p]) and len(vals) < 5:
                vals.append(payload[p])
                p += 1
            props[key] = vals
            continue

        if p < len(payload):
            props[key] = payload[p]
            p += 1

    return props


# ============================================================================
# Output formatting helpers
# ============================================================================


def as_dec(value_hex: str) -> int:
    """Convert hex string to decimal integer."""
    return int(value_hex, 16)


def token_to_int(value_token: str) -> Optional[int]:
    """Parse NFO token to integer. Alias for parse_token_value for backward compatibility."""
    return parse_token_value(value_token)


def bitmask_expr_from_set(names: list[str]) -> str:
    if not names:
        return "0"
    return f"bitmask({', '.join(names)})"


def format_building_flags(mask: int) -> tuple[str, int]:
    names: list[str] = []
    known_mask = 0
    for bit_pos, name in HOUSE_FLAG_BITS.items():
        bit_val = 1 << bit_pos
        if mask & bit_val:
            names.append(name)
            known_mask |= bit_val
    return bitmask_expr_from_set(names), mask & ~known_mask


def format_availability_mask(zone_byte: int, climate_byte: int) -> tuple[str, str, int]:
    packed = zone_byte + (climate_byte << 8)
    zones = packed & 0x1F
    climates = (packed >> 12) & 0x0F
    above_snowline = (packed & 0x0800) != 0

    zone_names = [name for bit, name in TOWNZONE_BITS.items() if zones & (1 << bit)]
    climate_names = [name for bit, name in CLIMATE_BITS.items() if climates & (1 << bit)]
    if above_snowline:
        climate_names.append("ABOVE_SNOWLINE")

    return bitmask_expr_from_set(zone_names), bitmask_expr_from_set(climate_names), packed


def decode_animation_info(value: int) -> tuple[int, int]:
    loop = (value >> 7) & 1
    frames = (value & 0x7F) + 1
    return loop, frames


def class_to_construction_switch(building_class: Optional[int]) -> str:
    """Map building class to appropriate construction check switch."""
    if building_class == 2:
        return "switch_ttrs_flats"
    if building_class == 3:
        return "switch_ttrs_offices"
    if building_class is not None and building_class >= 4:
        return "switch_ttrs_landmark_unique"
    return "switch_ttrs_residential"


# ============================================================================
# NML generation
# ============================================================================


def build_item_block(
    house_id_hex: str,
    item_id: int,
    display_name: str,
    props: dict[str, object],
    layout_name: str,
    house_size: Optional[str] = None,
    tile_layouts: Optional[list[str]] = None,
) -> list[str]:
    """Build NML item block for a house.

    ``layout_name`` is what goes in ``default:`` when tile_layouts is not given.
    """
    ident_suffix = sanitize_identifier(display_name)
    item_ident = f"item_ttrs_{house_id_hex.lower()}_{ident_suffix}"

    building_class: int | None = None
    if isinstance(props.get("1C"), str):
        building_class = as_dec(props["1C"])

    lines: list[str] = []

    # For multi-tile houses, emit a tile-routing switch before the item.
    if tile_layouts and len(tile_layouts) > 1:
        switch_name = f"switch_ttrs_{house_id_hex.lower()}_tile"
        cases = " ".join(f"{i}: {layout};" for i, layout in enumerate(tile_layouts))
        lines.append(f"switch (FEAT_HOUSES, SELF, {switch_name}, house_tile) {{ {cases} }}")
        lines.append("")
        default_graphics = switch_name
    elif tile_layouts:
        default_graphics = tile_layouts[0]
    else:
        default_graphics = layout_name

    if house_size:
        lines.append(f"item (FEAT_HOUSES, {item_ident}, {item_id}, {house_size}) {{")
    else:
        lines.append(f"item (FEAT_HOUSES, {item_ident}, {item_id}) {{")
    lines.append("\tproperty {")

    used_props: set[str] = set()

    sub_raw = props.get("08")
    if isinstance(sub_raw, str):
        sub_val = token_to_int(sub_raw)
        if sub_val is not None:
            lines.append(f"\t\tsubstitute: {sub_val};")
            used_props.add("08")

    low_raw = props.get("09")
    high_raw = props.get("19")
    low = token_to_int(low_raw) if isinstance(low_raw, str) else 0
    high = token_to_int(high_raw) if isinstance(high_raw, str) else 0
    if low is not None and high is not None and (low_raw is not None or high_raw is not None):
        mask = low + (high << 8)
        # Strip size bits (2, 3, 4) — NML sets them automatically for multi-tile items.
        if house_size:
            for sb in SIZE_FLAG_BITS:
                mask &= ~(1 << sb)
        flags_expr, leftover = format_building_flags(mask)
        lines.append(f"\t\tbuilding_flags: {flags_expr};")
        if leftover:
            lines.append(f"\t\t/* NFO building_flags extra bits: {leftover} */")
        if low_raw is not None:
            used_props.add("09")
        if high_raw is not None:
            used_props.add("19")

    if isinstance(props.get("13"), list) and len(props["13"]) == 2:
        zone = token_to_int(str(props["13"][0]))
        climate = token_to_int(str(props["13"][1]))
        if zone is not None and climate is not None:
            zone_expr, climate_expr, packed = format_availability_mask(zone, climate)
            lines.append(f"\t\tavailability_mask: [{zone_expr}, {climate_expr}];")
            lines.append(f"\t\t/* NFO availability packed value: {packed} */")
            used_props.add("13")

    if isinstance(props.get("0A"), list) and len(props["0A"]) == 2:
        y0 = re.sub(r"\\b", "", str(props["0A"][0]))
        y1 = re.sub(r"\\b", "", str(props["0A"][1]))
        if y0.isdigit() and y1.isdigit():
            lines.append(f"\t\tyears_available: [{y0}, {y1}];")
            used_props.add("0A")

    pop_raw = props.get("0B")
    if isinstance(pop_raw, str):
        pop = token_to_int(pop_raw)
        if pop is not None:
            lines.append(f"\t\tpopulation: {pop};")
            used_props.add("0B")

    mm_raw = props.get("0C")
    if isinstance(mm_raw, str):
        mm = token_to_int(mm_raw)
        if mm is not None:
            lines.append(f"\t\tmail_multiplier: {mm};")
            used_props.add("0C")

    lai_raw = props.get("10")
    if isinstance(lai_raw, str):
        lai = token_to_int(lai_raw)
        if lai is not None:
            lines.append(f"\t\tlocal_authority_impact: {lai};")
            used_props.add("10")

    rcm_raw = props.get("11")
    if isinstance(rcm_raw, str):
        rcm = token_to_int(rcm_raw)
        if rcm is not None:
            lines.append(f"\t\tremoval_cost_multiplier: {rcm};")
            used_props.add("11")

    minlife_raw = props.get("1F")
    if isinstance(minlife_raw, str):
        minlife = token_to_int(minlife_raw)
        if minlife is not None:
            lines.append(f"\t\tminimum_lifetime: {minlife};")
            used_props.add("1F")

    if building_class is not None:
        lines.append(f"\t\tbuilding_class: {building_class};")
        used_props.add("1C")

    prob_raw = props.get("18")
    if isinstance(prob_raw, str):
        prob = token_to_int(prob_raw)
        if prob is not None:
            # NFO prop 0x18 uses /16 scaling in NML's user-facing 'probability'.
            lines.append(f"\t\tprobability: {max(1, prob // 16)};")
            used_props.add("18")

    refresh_raw = props.get("16")
    if isinstance(refresh_raw, str):
        refresh = token_to_int(refresh_raw)
        if refresh is not None:
            lines.append(f"\t\trefresh_multiplier: {refresh};")
            used_props.add("16")

    anim_info_raw = props.get("1A")
    if isinstance(anim_info_raw, str):
        anim_info = token_to_int(anim_info_raw)
        if anim_info is not None:
            loop, frames = decode_animation_info(anim_info)
            lines.append(f"\t\tanimation_info: [{loop}, {frames}];")
            used_props.add("1A")

    anim_speed_raw = props.get("1B")
    if isinstance(anim_speed_raw, str):
        anim_speed = token_to_int(anim_speed_raw)
        if anim_speed is not None:
            lines.append(f"\t\tanimation_speed: {anim_speed};")
            used_props.add("1B")

    cargos: list[str] = []
    zero_cargos: list[str] = []
    if isinstance(props.get("0D"), str):
        pass_amt = token_to_int(props["0D"])
        if pass_amt is not None:
            if pass_amt > 0:
                cargos.append(f"[PASS, {pass_amt}]")
            else:
                zero_cargos.append("PASS")
            used_props.add("0D")
    if isinstance(props.get("0E"), str):
        mail_amt = token_to_int(props["0E"])
        if mail_amt is not None:
            if mail_amt > 0:
                cargos.append(f"[MAIL, {mail_amt}]")
            else:
                zero_cargos.append("MAIL")
            used_props.add("0E")
    if isinstance(props.get("0F"), str):
        goods_amt = token_to_int(props["0F"])
        if goods_amt is not None:
            if goods_amt > 0:
                cargos.append(f"[GOOD, {goods_amt}]")
            else:
                zero_cargos.append("GOOD")
            used_props.add("0F")
    if cargos:
        lines.append(f"\t\taccepted_cargos: [{','.join(cargos)}];")
    if zero_cargos:
        lines.append(f"\t\t/* NFO cargo amount is 0 for: {', '.join(zero_cargos)} */")

    name_id_raw = props.get("12")
    if isinstance(name_id_raw, str):
        lines.append(f"\t\t/* NFO name string-id: {name_id_raw}; visible name: {display_name} */")
        used_props.add("12")

    for passthrough_key in ["14", "1D", "1E", "21", "22", "23"]:
        if passthrough_key in props:
            lines.append(f"\t\t/* NFO field {passthrough_key}: {props[passthrough_key]} */")
            used_props.add(passthrough_key)

    # Filter out internal metadata keys (starting with _) and already-used props
    remaining = sorted(k for k in props.keys() if k not in used_props and not k.startswith("_"))
    if remaining:
        pairs = ", ".join(f"{k}={props[k]}" for k in remaining)
        lines.append(f"\t\t/* Unmapped NFO fields retained for review: {pairs} */")

    lines.append("\t}")
    lines.append("\tgraphics {")
    lines.append(f"\t\tdefault: {default_graphics};")
    lines.append(f"\t\tconstruction_check: {class_to_construction_switch(building_class)};")
    lines.append("\t}")
    lines.append("}")
    lines.append("")

    return lines


def generate_ttrs_nml(
    nfo_path: Path,
    output_path: Path,
    start_id: int = 200,
    pcx_path: str = "src/sprites/pcx/ttrs3w.pcx",
) -> None:
    """Generate combined sprites+items TTRS NML file from NFO source."""
    nfo_text = read_nfo_text(nfo_path)
    lines = nfo_text.splitlines()

    # Sprite data
    action1_blocks = parse_house_action1_blocks(lines)
    house_sections = collect_house_sections_from_action3(lines)
    # Build a lookup: house_id_hex -> (set_pairs, sprite_table, start_line_index)
    section_data: dict[str, tuple[dict[int, tuple[int, int]], dict[int, SpriteCoord], int]] = {}
    for house_id_hex, _label, sec_lines, start_idx in house_sections:
        sp = collect_house_set_pairs(sec_lines)
        st = sprite_table_for_house(start_idx, action1_blocks)
        section_data[house_id_hex] = (sp, st, start_idx)

    # House property and name data
    action3_ids = extract_house_ids_from_action3(lines)
    names = parse_names(nfo_text)
    properties = parse_action0_properties(lines)

    out: list[str] = []
    out.append("/* Begin TTRS — sprites and item definitions auto-generated from ttrs3wmod.nfo */")
    out.append("/* Generated by tool/generate_ttrs_from_nfo.py */")
    out.append("")
    out.append("/* --- Global availability/construction switches (always open, 5 slots) --- */")
    out.append("switch (FEAT_HOUSES, SELF, switch_ttrs_residential, CheckValue(1,255) && IsNotDesertTile()) {return;}")
    out.append("switch (FEAT_HOUSES, SELF, switch_ttrs_flats, CheckValue(4,255) && CheckFlatsSprawl()) {return;}")
    out.append(
        "switch (FEAT_HOUSES, SELF, switch_ttrs_offices, "
        "CheckOfficeSprawl(1000) && CheckValue(7,255) && (HasSameClassNearby(2) || IsFirstHouseOfClass())) {return;}"
    )
    out.append("switch (FEAT_HOUSES, SELF, switch_ttrs_landmark, CheckValue(5,255)) {return;}")
    out.append(
        "switch (FEAT_HOUSES, SELF, switch_ttrs_landmark_unique, CheckValue(5,255) && IsUniqueInRadius(10)) {return;}"
    )
    out.append("")

    # --- Per-house blocks (strictly: sprites → layouts/switch → tile-switch → item) ---
    parsed: list[tuple[int, dict[str, object]]] = [
        (house_id, properties.get(house_id, {}))
        for house_id in sorted(action3_ids.keys())
    ]

    skip_ids: set[int] = set()
    for index, (house_id, props) in enumerate(parsed):
        if house_id in skip_ids:
            continue

        house_id_hex = f"{house_id:02X}"
        sub_raw = props.get("08")
        sub_val = token_to_int(sub_raw) if isinstance(sub_raw, str) else None
        display_name = names.get(house_id, f"house_{house_id_hex}")

        if sub_val is not None and sub_val in MULTI_TILE_BASES:
            # ---- Multi-tile primary  ----------------------------------------
            size_name, num_tiles = MULTI_TILE_BASES[sub_val]
            tile_layouts: list[str] = []

            for t in range(num_tiles):
                tile_id = house_id + t
                tile_hex = f"{tile_id:02X}"
                tile_props = properties.get(tile_id, {})
                sp, st, _idx = section_data.get(tile_hex, ({}, {}, 0))
                sprite_lines, layout_name = emit_sprite_blocks(tile_hex, tile_props, sp, st, pcx_path)
                out.extend(sprite_lines)
                tile_layouts.append(layout_name)
                if t > 0:
                    skip_ids.add(tile_id)

            out.extend(
                build_item_block(
                    house_id_hex,
                    start_id + index,
                    display_name,
                    props,
                    layout_name=tile_layouts[0],  # not used directly (tile_layouts takes priority)
                    house_size=size_name,
                    tile_layouts=tile_layouts,
                )
            )

        elif sub_val is not None and sub_val in SUBSTITUTE_TO_BASE and sub_val != SUBSTITUTE_TO_BASE[sub_val]:
            # Secondary tile — already handled via skip_ids, but skip gracefully
            continue

        else:
            # ---- Regular 1×1 house  -----------------------------------------
            sp, st, _idx = section_data.get(house_id_hex, ({}, {}, 0))
            sprite_lines, layout_name = emit_sprite_blocks(house_id_hex, props, sp, st, pcx_path)
            out.extend(sprite_lines)
            out.extend(
                build_item_block(
                    house_id_hex,
                    start_id + index,
                    display_name,
                    props,
                    layout_name=layout_name,
                )
            )

    out.append("/* End TTRS */")
    output_path.write_text("\n".join(out), encoding="utf-8")


# ============================================================================
# Main entry point
# ============================================================================


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate combined TTRS NML (sprites + items) from ttrs3wmod.nfo"
    )
    parser.add_argument("--nfo", required=True, type=Path, help="Path to ttrs3wmod.nfo")
    parser.add_argument("--out", required=True, type=Path, help="Output .nml path")
    parser.add_argument("--start-id", type=int, default=200, help="Starting item ID (default: 200)")
    parser.add_argument(
        "--pcx-path",
        default="src/sprites/pcx/ttrs3w.pcx",
        help="PCX path written into spriteset declarations (default: src/sprites/pcx/ttrs3w.pcx)",
    )
    args = parser.parse_args()

    generate_ttrs_nml(args.nfo, args.out, args.start_id, args.pcx_path)
    print(f"Generated {args.out}")


if __name__ == "__main__":
    main()
