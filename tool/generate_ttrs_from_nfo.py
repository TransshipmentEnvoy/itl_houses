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

# New graph-based NFO parser (tool/nfo_parse/)
from tool.nfo_parse import build_all_house_graphics, emit_house_tile_nml
from tool.nfo_parse.fixups import fixup_food_only_add_pass_mail, fixup_remap_year_1930_to_1870
from tool.nfo_parse.graph import traverse_callback_subgraph

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

# Action 0 for feature 08 (general), property 09 = cargo translation table
# Format: <sprite> * <len> 00 08 01 <count> 00 09 "XXXX" "XXXX" ...
ACTION0_CTT_RE = re.compile(
    r'^\s*\d+\s+\*\s+\d+\s+00\s+08\s+01\s+([0-9A-Fa-f]{2})\s+00\s+09\s+(.*)',
    re.IGNORECASE,
)
CTT_LABEL_RE = re.compile(r'"([A-Z]{4})"')

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

# NFO sprite coordinate line: index  sprites/pcx/ttrs3w.pcx  xpos ypos 01 ysize xsize xrel yrel
SPRITE_LINE_RE = re.compile(
    r"^\s*(\d+)\s+sprites/pcx/ttrs3w\.pcx\s+(-?\d+)\s+(-?\d+)\s+01\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)"
)

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

# NML per-tile graphics callback names for multi-tile houses.
# house_tile variable values: NORTH=0, EAST=1, WEST=2, SOUTH=3
# 2x1 = north + west; 1x2 = north + east; 2x2 = N + E + W + S
TILE_CALLBACK_NAMES: dict[str, list[str]] = {
    "HOUSE_SIZE_2X1": ["graphics_north", "graphics_west"],
    "HOUSE_SIZE_1X2": ["graphics_north", "graphics_east"],
    "HOUSE_SIZE_2X2": ["graphics_north", "graphics_east", "graphics_west", "graphics_south"],
}

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

    sections: list[tuple[str, str, list[str], int, int]] = []
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
            action3_idx,
        ))

    return sections


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


def parse_cargo_translation_table(nfo_text: str) -> dict[int, str]:
    """Extract the Cargo Translation Table (CTT) from Action 0 / Feature 08.

    Returns a mapping from CTT index to NML cargo label, e.g.
    {0: "PASS", 1: "PETR", 2: "MAIL", 3: "TOUR", 4: "FOOD", 5: "GOOD"}.
    """
    ctt_map: dict[int, str] = {}
    for line in nfo_text.splitlines():
        m = ACTION0_CTT_RE.match(line)
        if m:
            labels = CTT_LABEL_RE.findall(m.group(2))
            for idx, label in enumerate(labels):
                ctt_map[idx] = label
            break  # only need the first CTT definition
    return ctt_map


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

        # Parse property values for each house.
        # NFO Action 0 layout: for each property key, num_houses values follow
        # in sequence (outer=props, inner=houses).  The cursor p must advance
        # linearly — never reset between houses.
        for h in range(num_houses):
            house_id = first_id + h
            if house_id not in properties:
                properties[house_id] = {}

        p = 0
        prop_count = 0
        while p < len(tokens) and prop_count < num_props:
            key = tokens[p]
            p += 1

            if not HEX2_RE.match(key):
                continue

            prop_count += 1
            key_int = int(key, 16)

            # For each property key, consume one value per house in order.
            for h in range(num_houses):
                props = properties[first_id + h]

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

                # Property 14: callback flags 1 (1 byte)
                elif key_int == 0x14:
                    if p < len(tokens):
                        val = parse_token_value(tokens[p])
                        if val is not None and "14" not in props:
                            props["14"] = f"{val:02X}"
                        p += 1

                # Property 15: override flag (1 byte)
                elif key_int == 0x15:
                    if p < len(tokens):
                        val = parse_token_value(tokens[p])
                        if val is not None and "15" not in props:
                            props["15"] = f"{val:02X}"
                        p += 1

                # Property 16: refresh_multiplier
                elif key_int == 0x16:
                    if p < len(tokens):
                        val = parse_token_value(tokens[p])
                        if val is not None and "16" not in props:
                            props["16"] = f"{val:02X}"
                        p += 1

                # Property 17: four random colours (4*B)
                elif key_int == 0x17:
                    if p + 3 < len(tokens):
                        vals = [parse_token_value(tokens[p + i]) for i in range(4)]
                        if all(v is not None for v in vals) and "17" not in props:
                            props["17"] = [f"{v:02X}" for v in vals]
                        p += 4

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

                # Property 1D: callback flags 2 (1 byte)
                elif key_int == 0x1D:
                    if p < len(tokens):
                        val = parse_token_value(tokens[p])
                        if val is not None and "1D" not in props:
                            props["1D"] = f"{val:02X}"
                        p += 1

                # Property 1E: accepted cargo types (DWORD = 4 bytes)
                # Format: slot1_ctt_idx slot2_ctt_idx slot3_ctt_idx 0x00
                elif key_int == 0x1E:
                    if p + 3 < len(tokens):
                        vals = [parse_token_value(tokens[p + i]) for i in range(4)]
                        if all(v is not None for v in vals) and "1E" not in props:
                            props["1E"] = [vals[0], vals[1], vals[2]]  # 3 CTT indices
                        p += 4

                # Property 1F: minimum_lifetime
                elif key_int == 0x1F:
                    if p < len(tokens):
                        val = parse_token_value(tokens[p])
                        if val is not None and "1F" not in props:
                            props["1F"] = f"{val:02X}"
                        p += 1

                # Unknown property — use size lookup table to skip correctly.
                # Action0/Houses property sizes (tokens per value):
                #   08:1  09:1  0A:2  0B:1  0C:1  0D:1  0E:1  0F:1
                #   10:2  11:1  12:2  13:2  14:1  15:1  16:1  17:4
                #   18:1  19:1  1A:1  1B:1  1C:1  1D:1  1E:4  1F:1
                #   20:4  21:2  22:4  23:1  24:1
                else:
                    _PROP_SIZES: dict[int, int] = {
                        0x08: 1, 0x09: 1, 0x0A: 2, 0x0B: 1, 0x0C: 1,
                        0x0D: 1, 0x0E: 1, 0x0F: 1, 0x10: 2, 0x11: 1,
                        0x12: 2, 0x13: 2, 0x14: 1, 0x15: 1, 0x16: 1,
                        0x17: 4, 0x18: 1, 0x19: 1, 0x1A: 1, 0x1B: 1,
                        0x1C: 1, 0x1D: 1, 0x1E: 4, 0x1F: 1,
                        0x20: 4, 0x21: 2, 0x22: 4, 0x23: 1, 0x24: 1,
                    }
                    skip = _PROP_SIZES.get(key_int, 1)
                    p += skip

        line_num += 1

    return properties


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


# ============================================================================
# House classification for construction checks and probability
# ============================================================================

# Substitute IDs that are clearly office buildings in vanilla OpenTTD
OFFICE_SUBSTITUTES = {13, 19, 30, 31, 36}
# Substitute IDs that are tall residential / flats
FLAT_SUBSTITUTES = {15, 16, 17, 18}
# Substitute IDs for landmark / special buildings
LANDMARK_SPECIAL_SUBSTITUTES = {9, 20, 40, 54, 87}

# Name-based keyword sets for classification
_OFFICE_NAME_KW = {'office', 'z_office', 'z block'}
_FLAT_NAME_KW = {'flat', 'apartment', 'endless'}
_LANDMARK_UNIQUE_NAME_KW = {
    'cathedral', 'statue', 'stock exchange', 'world trade',
    'museum', 'old town',
}
_LANDMARK_NAME_KW = {
    'hospital', 'fire station', 'police', 'prison', 'library',
    'planetarium', 'observatorium', 'hotel', 'water tower',
}
# Probability caps per category
_PROB_CAPS = {
    'residential': 1,
    'flats': 1,
    'offices': 1,
    'landmark': 3,
    'landmark_unique': 3,
}
# Default probability per category (when NFO does not provide one)
_PROB_DEFAULTS = {
    'residential': 1,
    'flats': 1,
    'offices': 1,
    'landmark': 2,
    'landmark_unique': 2,
}


def classify_ttrs_house(
    substitute: Optional[int],
    building_class: Optional[int],
    display_name: str,
    building_flags_mask: int,
) -> tuple[str, str, int]:
    """Classify a TTRS house and return (category, construction_switch, default_probability).

    Classification priority:
      1. Church flag → landmark_unique
      2. Name-based landmark_unique keywords
      3. Name-based landmark keywords
      4. Name-based office / flat keywords
      5. Protected + special substitute → landmark_unique
      6. Substitute-based office / flat classification
      7. Default → residential
    """
    name_lower = display_name.lower().replace('"', '').replace("'", '')
    is_protected = bool(building_flags_mask & (1 << 9))  # HOUSE_FLAG_PROTECTED
    is_church = bool(building_flags_mask & (1 << 6))     # HOUSE_FLAG_CHURCH

    # 1) Churches (by flag)
    if is_church:
        return 'landmark_unique', 'switch_ttrs_landmark_unique', _PROB_DEFAULTS['landmark_unique']

    # 2) Unique landmark buildings (by name)
    if any(kw in name_lower for kw in _LANDMARK_UNIQUE_NAME_KW):
        return 'landmark_unique', 'switch_ttrs_landmark_unique', _PROB_DEFAULTS['landmark_unique']

    # 3) Regular landmark / public buildings (by name)
    if any(kw in name_lower for kw in _LANDMARK_NAME_KW):
        return 'landmark', 'switch_ttrs_landmark', _PROB_DEFAULTS['landmark']

    # 4a) Offices (by name)
    if any(kw in name_lower for kw in _OFFICE_NAME_KW):
        return 'offices', 'switch_ttrs_offices', _PROB_DEFAULTS['offices']

    # 4b) Flats (by name)
    if any(kw in name_lower for kw in _FLAT_NAME_KW):
        return 'flats', 'switch_ttrs_flats', _PROB_DEFAULTS['flats']

    # 5) Protected buildings with special substitute → landmark_unique
    if is_protected and substitute in LANDMARK_SPECIAL_SUBSTITUTES:
        return 'landmark_unique', 'switch_ttrs_landmark_unique', _PROB_DEFAULTS['landmark_unique']

    # 6a) Offices (by substitute type)
    if substitute in OFFICE_SUBSTITUTES:
        return 'offices', 'switch_ttrs_offices', _PROB_DEFAULTS['offices']

    # 6b) Flats (by substitute type)
    if substitute in FLAT_SUBSTITUTES:
        return 'flats', 'switch_ttrs_flats', _PROB_DEFAULTS['flats']

    # 7) Default → residential
    return 'residential', 'switch_ttrs_residential', _PROB_DEFAULTS['residential']


# ============================================================================
# NFO Callback ID → NML graphics{} field name mapping
# ============================================================================

_CB_NML_FIELD: dict[int, str] = {
    # 0x17 (construction_check) is handled separately — not included here
    0x1A: "anim_next_frame",
    0x1B: "anim_control",
    0x1C: "construction_anim",
    0x1F: "cargo_amount_accept",
    0x20: "anim_speed",
    0x21: "destruction",
    0x2A: "cargo_type_accept",
    0x2E: "cargo_production",
    0x143: "protection",
    0x148: "watched_cargo_accepted",
    0x14D: "name",
    0x14E: "foundations",
    0x14F: "autoslope",
}


def _translate_callback_handlers(
    htg_or_htg_list: object,
    house_id_hex: str,
    house_graph: dict,
    final_graph: dict,
) -> tuple[list[str], dict[str, str]]:
    """Translate callback handler sub-graphs into NML switch blocks.

    Parameters
    ----------
    htg_or_htg_list : HouseTileGraphics
        The tile graphics object with ``callback_handlers`` populated.
    house_id_hex : str
    house_graph : per-house Action2Graph (for node lookup).
    final_graph : global fallback graph.

    Returns
    -------
    (nml_lines, cb_switches)
        *nml_lines* are NML switch block definitions to be emitted before the
        item block.  *cb_switches* maps NML graphics-field names (e.g.
        ``"protection"``) to the top-level switch name for that callback.
    """
    from tool.nfo_parse.nodes import HouseTileGraphics as _HTG
    htg = htg_or_htg_list
    if not isinstance(htg, _HTG):
        return [], {}

    all_lines: list[str] = []
    cb_switches: dict[str, str] = {}

    for cb_id, target_id in sorted(htg.callback_handlers.items()):
        nml_field = _CB_NML_FIELD.get(cb_id)
        if nml_field is None:
            # CB 0x17 = construction_check — handled by dedicated logic;
            # skip silently.  Other unknown CBs get a note.
            if cb_id != 0x17:
                all_lines.append(
                    f"/* NOTE: NFO callback 0x{cb_id:X} has no NML "
                    f"graphics-block mapping, skipped */"
                )
            continue

        cb_lines, cb_name = traverse_callback_subgraph(
            target_id, house_graph, final_graph,
            house_id_hex, cb_id,
            cb_label=nml_field,
        )

        # Check if any branch fell through to a layout node (= CB_FAILED).
        # For callbacks in _SUPPRESS_ON_FALLTHROUGH, this means the generated
        # switch would lose information (CB_FAILED cannot be expressed as a
        # return value), so we suppress the entire callback with a warning.
        has_fallthrough = any(
            "fell through to layout node" in line for line in cb_lines
        )
        if has_fallthrough and nml_field in _SUPPRESS_ON_FALLTHROUGH:
            all_lines.append(
                f"/* WARN: {nml_field} (CB 0x{cb_id:X}) sub-graph has branches "
                f"that fall through to layout nodes (CB_FAILED semantics lost); "
                f"callback omitted */"
            )
            continue

        all_lines.extend(cb_lines)

        if cb_name is not None:
            cb_switches[nml_field] = cb_name
        else:
            all_lines.append(
                f"/* NOTE: {nml_field} (0x{cb_id:X}) sub-graph could not "
                f"be fully translated, omitted */"
            )

    return all_lines, cb_switches


# Callbacks where fallthrough to a layout node (= CB_FAILED semantics) cannot
# be safely approximated by any fixed return value.  When the sub-graph for
# one of these callbacks contains a fallthrough branch, the entire callback is
# suppressed and a warning comment is emitted instead.
_SUPPRESS_ON_FALLTHROUGH: set[str] = {
    "cargo_amount_accept",
    "cargo_type_accept",
}


def _compute_colour_weights(colour_values: list[int]) -> tuple[list[int], list[int]]:
    """Deduplicate colour values and compute per-value weights from repetition counts.

    Returns ``(unique_values, weights)``.
    """
    order: list[int] = []
    counts: dict[int, int] = {}
    for v in colour_values:
        if v not in counts:
            order.append(v)
            counts[v] = 1
        else:
            counts[v] += 1
    weights = [counts[v] for v in order]
    return order, weights


def emit_colour_switch(
    house_id_hex: str,
    colour_values: list[int],
    colour_triggers: int = 0,
) -> tuple[list[str], str]:
    """Emit NML colour callback block and return (lines, switch_name).

    For a single colour value, emits an inline ``return <value>;`` expression.
    For multiple values, emits a ``random_switch`` with weights derived from
    entry repetition counts.  Trigger semantics are preserved: triggers=0x00
    means no rerandomisation trigger (omit the bitmask clause); non-zero
    triggers are emitted as a comment for manual review.
    """
    switch_name = f"switch_ttrs_{house_id_hex.lower()}_colour"
    lines: list[str] = []

    unique_vals, weights = _compute_colour_weights(colour_values)

    if len(unique_vals) == 1:
        # Single colour — no need for a random_switch, just inline it.
        # We still emit a switch so the identifier exists.
        lines.append(
            f"switch (FEAT_HOUSES, SELF, {switch_name}, 0) "
            f"{{ return {unique_vals[0]}; }}"
        )
    else:
        # Multiple colours — random_switch with actual weights
        entries = " ".join(f"{w}: return {v};" for v, w in zip(unique_vals, weights))
        # Trigger handling
        trigger_comment = ""
        if colour_triggers != 0:
            trigger_comment = f" /* NFO triggers: 0x{colour_triggers:02X} */"
        lines.append(
            f"random_switch (FEAT_HOUSES, SELF, {switch_name}) "
            f"{{ {entries} }}{trigger_comment}"
        )

    return lines, switch_name


# ============================================================================
# Language string generation
# ============================================================================


def sanitize_string_id(text: str) -> str:
    """Convert a house name to an NML string identifier component.

    E.g. 'Fire station' -> 'FIRE_STATION',
         '"Z" office block' -> 'Z_OFFICE_BLOCK'
    """
    ident = text.upper()
    ident = ident.replace('\u201c', '').replace('\u201d', '')
    ident = ident.replace('"', '').replace("'", '')
    ident = re.sub(r'[^A-Z0-9]+', '_', ident).strip('_')
    # Collapse multiple underscores
    ident = re.sub(r'_+', '_', ident)
    return ident or 'UNNAMED'


def resolve_house_names(
    dc_names: dict[int, str],
    properties: dict[int, dict[str, object]],
) -> dict[int, str]:
    """Map house IDs to display names via Action 0 property 0x12 (DC string ref).

    Action 4 ``04 48`` entries define names indexed by *DC string slot*
    (0x00, 0x01, ...) — these are NOT house IDs.  The actual house→name
    binding comes from Action 0 property ``0x12``, which stores a word
    value like ``\\wxDCxx`` pointing into the DC00 string range.

    This function resolves:  house_id  →  prop 0x12  →  DC slot  →  name.
    """
    resolved: dict[int, str] = {}
    for house_id, props in properties.items():
        name_ref_raw = props.get("12")
        if not isinstance(name_ref_raw, str):
            continue
        name_ref = int(name_ref_raw, 16)
        # DC string references are in the 0xDC00-0xDCFF range
        if 0xDC00 <= name_ref <= 0xDCFF:
            dc_slot = name_ref - 0xDC00
            if dc_slot in dc_names:
                resolved[house_id] = dc_names[dc_slot]
    return resolved


def generate_lang_strings(
    names: dict[int, str],
    action3_ids: dict[int, int],
) -> dict[int, tuple[str, str]]:
    """Build a mapping {house_id: (string_id, english_value)} for named houses.

    Only houses that appear in both *names* (have a resolved name) **and**
    *action3_ids* (will get an item block) are included.

    Houses that share the same name (e.g. multiple Hospital tiles) reuse
    the same string ID so only one lang entry is emitted per unique name.
    """
    result: dict[int, tuple[str, str]] = {}
    # Map each unique name to a single string ID
    name_to_string_id: dict[str, str] = {}
    used_ids: set[str] = set()

    for house_id in sorted(action3_ids.keys()):
        if house_id not in names:
            continue
        english_name = names[house_id]

        if english_name in name_to_string_id:
            # Reuse existing string ID for this name
            string_id = name_to_string_id[english_name]
        else:
            base_id = f"STR_TTRS_NAME_{sanitize_string_id(english_name)}"
            string_id = base_id
            if string_id in used_ids:
                string_id = f"{base_id}_{house_id:02X}"
            used_ids.add(string_id)
            name_to_string_id[english_name] = string_id

        result[house_id] = (string_id, english_name)

    return result


def write_lang_strings(
    lang_dir: Path,
    lang_strings: dict[int, tuple[str, str]],
) -> None:
    """Update src/lang/english.lng with STR_TTRS_NAME_* entries.

    Existing STR_TTRS_NAME_* lines are removed first (idempotent).
    New entries are inserted after the last existing STR_NAME_* line,
    or appended at the end of the file.
    """
    lng_path = lang_dir / 'english.lng'
    if not lng_path.exists():
        return

    existing_lines = lng_path.read_text(encoding='utf-8').splitlines()

    # Remove any previous STR_TTRS_NAME_* lines
    cleaned = [l for l in existing_lines if not l.startswith('STR_TTRS_NAME_')]

    # Remove trailing blank lines to avoid accumulation
    while cleaned and cleaned[-1].strip() == '':
        cleaned.pop()

    # Find insertion point: after last STR_NAME_* line
    insert_idx = len(cleaned)
    for i, line in enumerate(cleaned):
        if line.startswith('STR_NAME_'):
            insert_idx = i + 1

    # Build new lines (deduplicate: one entry per unique string_id)
    new_lines: list[str] = ['']
    written_ids: set[str] = set()
    for house_id in sorted(lang_strings.keys()):
        string_id, english_value = lang_strings[house_id]
        if string_id in written_ids:
            continue
        written_ids.add(string_id)
        # Tab-align the colon to match existing convention (~6 tabs)
        tab_count = max(1, 6 - len(string_id) // 4)
        tabs = '\t' * tab_count
        new_lines.append(f'{string_id}{tabs}:{english_value}')

    # Insert
    for j, nl in enumerate(new_lines):
        cleaned.insert(insert_idx + j, nl)

    # Ensure file ends with newline
    cleaned.append('')

    lng_path.write_text('\n'.join(cleaned), encoding='utf-8')


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
    colour_switch: Optional[str] = None,
    name_string_id: Optional[str] = None,
    ctt_map: Optional[dict[int, str]] = None,
    callback_switches: Optional[dict[str, str]] = None,
) -> list[str]:
    """Build NML item block for a house.

    ``layout_name`` is what goes in ``default:`` when tile_layouts is not given.
    ``colour_switch`` if set, is the NML identifier for the colour callback.
    ``callback_switches`` if set, maps NML callback field names to switch names.
    ``name_string_id`` if set, emits a ``name: string(...)`` property.
    """
    ident_suffix = sanitize_identifier(display_name)
    item_ident = f"item_ttrs_{house_id_hex.lower()}_{ident_suffix}"

    building_class: int | None = None
    if isinstance(props.get("1C"), str):
        building_class = as_dec(props["1C"])

    # --- Compute building flags mask for classification ---
    _low_raw = props.get("09")
    _high_raw = props.get("19")
    _low_val = token_to_int(_low_raw) if isinstance(_low_raw, str) else 0
    _high_val = token_to_int(_high_raw) if isinstance(_high_raw, str) else 0
    building_flags_mask = (_low_val or 0) + ((_high_val or 0) << 8)
    has_animate_flag = bool(building_flags_mask & (1 << 5))  # HOUSE_FLAG_ANIMATE

    # --- Classify house ---
    sub_for_class = token_to_int(props["08"]) if isinstance(props.get("08"), str) else None
    category, construction_switch, default_prob = classify_ttrs_house(
        sub_for_class, building_class, display_name, building_flags_mask,
    )

    lines: list[str] = []

    # For multi-tile houses, use per-tile graphics callbacks instead of
    # a house_tile routing switch.  NML provides graphics_north,
    # graphics_east, graphics_west, graphics_south that map directly to
    # the correct tile positions.
    per_tile_callbacks: list[tuple[str, str]] | None = None
    if tile_layouts and len(tile_layouts) > 1 and house_size and house_size in TILE_CALLBACK_NAMES:
        cb_names = TILE_CALLBACK_NAMES[house_size]
        per_tile_callbacks = list(zip(cb_names, tile_layouts))
        default_graphics = tile_layouts[0]
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

    if name_string_id is not None:
        lines.append(f"\t\tname: string({name_string_id});")

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
            y0_int, y1_int, year_fixup_applied = fixup_remap_year_1930_to_1870(int(y0), int(y1))
            y1_fmt = "0xFFFF" if y1_int >= 2170 else str(y1_int)
            lines.append(f"\t\tyears_available: [{y0_int}, {y1_fmt}];")
            if year_fixup_applied:
                lines.append(f"\t\t/* CUSTOM: years_available remapped 1930 → 1870 (orig: [{y0}, {y1}]) */")
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

    # --- Probability: use NFO value (capped by category) or category default ---
    prob_raw = props.get("18")
    if isinstance(prob_raw, str):
        prob = token_to_int(prob_raw)
        if prob is not None:
            nml_prob = max(1, min(15, round(prob / 17)))
            cap = _PROB_CAPS.get(category, 15)
            nml_prob = min(nml_prob, cap)
            lines.append(f"\t\tprobability: {nml_prob};")
            used_props.add("18")
        else:
            lines.append(f"\t\tprobability: {default_prob};")
    else:
        lines.append(f"\t\tprobability: {default_prob};")

    refresh_raw = props.get("16")
    if isinstance(refresh_raw, str):
        refresh = token_to_int(refresh_raw)
        if refresh is not None:
            lines.append(f"\t\trefresh_multiplier: {refresh};")
            used_props.add("16")

    # --- Animation properties (ensure consistency when HOUSE_FLAG_ANIMATE is set) ---
    anim_info_raw = props.get("1A")
    has_anim_info = False
    if isinstance(anim_info_raw, str):
        anim_info = token_to_int(anim_info_raw)
        if anim_info is not None:
            loop, frames = decode_animation_info(anim_info)
            lines.append(f"\t\tanimation_info: [{loop}, {frames}];")
            used_props.add("1A")
            has_anim_info = True
    # If ANIMATE flag is set but no animation_info, add a minimal default
    if has_animate_flag and not has_anim_info:
        lines.append("\t\tanimation_info: [1, 1];")
        lines.append("\t\t/* NOTE: ANIMATE flag set but NFO had no animation_info; using minimal default */")

    anim_speed_raw = props.get("1B")
    has_anim_speed = False
    if isinstance(anim_speed_raw, str):
        anim_speed = token_to_int(anim_speed_raw)
        if anim_speed is not None:
            lines.append(f"\t\tanimation_speed: {anim_speed};")
            used_props.add("1B")
            has_anim_speed = True
    # If ANIMATE flag is set but no animation_speed, add a safe default
    if has_animate_flag and not has_anim_speed:
        lines.append("\t\tanimation_speed: 2;")
        lines.append("\t\t/* NOTE: ANIMATE flag set but NFO had no animation_speed; defaulting to 2 */")

    # Resolve cargo labels for each acceptance slot.
    # Default: slot1=PASS, slot2=MAIL, slot3=GOOD.
    # Property 1E overrides the cargo types via the Cargo Translation Table.
    slot_labels = ["PASS", "MAIL", "GOOD"]
    if isinstance(props.get("1E"), list) and ctt_map:
        ctt_indices: list[int] = props["1E"]
        for i, idx in enumerate(ctt_indices):
            if idx in ctt_map:
                slot_labels[i] = ctt_map[idx]
        used_props.add("1E")

    cargo_pairs: list[tuple[str, int]] = []
    zero_cargos: list[str] = []
    for slot_idx, prop_key in enumerate(["0D", "0E", "0F"]):
        if isinstance(props.get(prop_key), str):
            amt = token_to_int(props[prop_key])
            if amt is not None:
                if amt > 0:
                    cargo_pairs.append((slot_labels[slot_idx], amt))
                else:
                    zero_cargos.append(slot_labels[slot_idx])
                used_props.add(prop_key)

    # --- CUSTOM fixup: FOOD-only houses also accept PASS and MAIL ---
    cargo_pairs, food_fixup_applied = fixup_food_only_add_pass_mail(cargo_pairs)

    cargos = [f"[{label}, {amt}]" for label, amt in cargo_pairs]
    if cargos:
        lines.append(f"\t\taccepted_cargos: [{','.join(cargos)}];")
    if food_fixup_applied:
        lines.append(f"\t\t/* CUSTOM: FOOD-only house — added equal PASS and MAIL acceptance */")
    if zero_cargos:
        lines.append(f"\t\t/* NFO cargo amount is 0 for: {', '.join(zero_cargos)} */")

    name_id_raw = props.get("12")
    if isinstance(name_id_raw, str):
        lines.append(f"\t\t/* NFO name string-id: {name_id_raw}; visible name: {display_name} */")
        used_props.add("12")

    for passthrough_key in ["14", "1D", "21", "22", "23"]:
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
    if per_tile_callbacks:
        for cb_name, cb_layout in per_tile_callbacks:
            lines.append(f"\t\t{cb_name}: {cb_layout};")
    else:
        lines.append(f"\t\tdefault: {default_graphics};")
    lines.append(f"\t\tconstruction_check: {construction_switch};")
    lines.append(f"\t\t/* classification: {category} */")
    if callback_switches:
        for cb_field, cb_switch in sorted(callback_switches.items()):
            lines.append(f"\t\t{cb_field}: {cb_switch};")
    if colour_switch is not None:
        lines.append(f"\t\tcolour: {colour_switch};")
    lines.append("\t}")
    lines.append("}")
    lines.append("")

    return lines


def generate_ttrs_nml(
    nfo_path: Path,
    output_path: Path,
    start_id: int = 200,
    pcx_path: str = "src/sprites/pcx/ttrs3w.pcx",
    lang_dir: Optional[Path] = None,
) -> None:
    """Generate combined sprites+items TTRS NML file from NFO source."""
    nfo_text = read_nfo_text(nfo_path)
    lines = nfo_text.splitlines()

    # Build new graph-based house graphics (tropic / animation / random support)
    _house_graphics, _house_graphs = build_all_house_graphics(lines)

    # Sprite data
    action1_blocks = parse_house_action1_blocks(lines)
    house_sections = collect_house_sections_from_action3(lines)
    # Build a lookup: house_id_hex -> sprite_table
    # Use the Action 3 line index (not section start) for sprite table lookup.
    # The correct Action 1 block is always the last one *before* the house's
    # Action 3 entry.  Using start_idx was wrong when an Action 1 block fell
    # between the previous house's Action 3 and the current section start.
    section_data: dict[str, dict[int, SpriteCoord]] = {}
    for house_id_hex, _label, _sec_lines, _start_idx, action3_idx in house_sections:
        section_data[house_id_hex] = sprite_table_for_house(action3_idx, action1_blocks)

    # Cargo translation table and house property/name data
    ctt_map = parse_cargo_translation_table(nfo_text)
    action3_ids = extract_house_ids_from_action3(lines)
    dc_names = parse_names(nfo_text)  # DC string slot index -> name
    properties = parse_action0_properties(lines)
    # Resolve actual house_id -> name via Action 0 prop 0x12 -> DC slot
    names = resolve_house_names(dc_names, properties)

    # Generate lang strings and write to english.lng
    lang_strings = generate_lang_strings(names, action3_ids)
    if lang_dir is not None:
        write_lang_strings(lang_dir, lang_strings)

    # Build a merged global graph for cross-section callback lookups.
    _global_graph: dict = {}
    for hg in _house_graphs.values():
        _global_graph.update(hg)

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
        name_str_id = lang_strings[house_id][0] if house_id in lang_strings else None

        if sub_val is not None and sub_val in MULTI_TILE_BASES:
            # ---- Multi-tile primary  ----------------------------------------
            size_name, num_tiles = MULTI_TILE_BASES[sub_val]
            tile_layouts: list[str] = []
            multi_colour_values: list[int] = []
            multi_colour_triggers: int = 0
            multi_cb_switches: dict[str, str] = {}

            for t in range(num_tiles):
                tile_id = house_id + t
                tile_hex = f"{tile_id:02X}"
                st = section_data.get(tile_hex, {})
                htg = _house_graphics[tile_id]
                sprite_lines, layout_name, tile_colours = emit_house_tile_nml(tile_hex, htg, st, pcx_path)
                out.extend(sprite_lines)
                tile_layouts.append(layout_name)
                if t == 0 and tile_colours:
                    multi_colour_values = tile_colours
                    multi_colour_triggers = htg.colour_triggers
                # Translate callback handlers for the primary tile
                if t == 0 and htg.callback_handlers:
                    house_g = _house_graphs.get(tile_id, {})
                    cb_lines, cb_sw = _translate_callback_handlers(
                        htg, tile_hex, house_g, _global_graph,
                    )
                    out.extend(cb_lines)
                    multi_cb_switches.update(cb_sw)
                if t > 0:
                    skip_ids.add(tile_id)

            colour_switch_name = None
            if multi_colour_values:
                colour_lines, colour_switch_name = emit_colour_switch(
                    house_id_hex, multi_colour_values, multi_colour_triggers
                )
                out.extend(colour_lines)

            out.extend(
                build_item_block(
                    house_id_hex,
                    start_id + index,
                    display_name,
                    props,
                    layout_name=tile_layouts[0],  # not used directly (tile_layouts takes priority)
                    house_size=size_name,
                    tile_layouts=tile_layouts,
                    colour_switch=colour_switch_name,
                    name_string_id=name_str_id,
                    ctt_map=ctt_map,
                    callback_switches=multi_cb_switches or None,
                )
            )

        elif sub_val is not None and sub_val in SUBSTITUTE_TO_BASE and sub_val != SUBSTITUTE_TO_BASE[sub_val]:
            # Secondary tile — already handled via skip_ids, but skip gracefully
            continue

        else:
            # ---- Regular 1×1 house  -----------------------------------------
            st = section_data.get(house_id_hex, {})
            htg = _house_graphics[house_id]
            sprite_lines, layout_name, house_colours = emit_house_tile_nml(house_id_hex, htg, st, pcx_path)
            out.extend(sprite_lines)

            colour_switch_name = None
            if house_colours:
                colour_lines, colour_switch_name = emit_colour_switch(
                    house_id_hex, house_colours, htg.colour_triggers
                )
                out.extend(colour_lines)

            # Translate callback handlers
            house_cb_switches: dict[str, str] = {}
            if htg.callback_handlers:
                house_g = _house_graphs.get(house_id, {})
                cb_lines, house_cb_switches = _translate_callback_handlers(
                    htg, house_id_hex, house_g, _global_graph,
                )
                out.extend(cb_lines)

            out.extend(
                build_item_block(
                    house_id_hex,
                    start_id + index,
                    display_name,
                    props,
                    layout_name=layout_name,
                    colour_switch=colour_switch_name,
                    name_string_id=name_str_id,
                    ctt_map=ctt_map,
                    callback_switches=house_cb_switches or None,
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
    parser.add_argument(
        "--lang-dir",
        type=Path,
        default=Path("src/lang"),
        help="Directory containing .lng language files (default: src/lang)",
    )
    args = parser.parse_args()

    generate_ttrs_nml(args.nfo, args.out, args.start_id, args.pcx_path, args.lang_dir)
    print(f"Generated {args.out}")


if __name__ == "__main__":
    main()
