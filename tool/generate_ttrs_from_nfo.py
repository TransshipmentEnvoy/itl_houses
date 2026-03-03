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

                # Property 1F: minimum_lifetime
                elif key_int == 0x1F:
                    if p < len(tokens):
                        val = parse_token_value(tokens[p])
                        if val is not None and "1F" not in props:
                            props["1F"] = f"{val:02X}"
                        p += 1

                # Unknown property — use size lookup table to skip correctly.
                # Action0/Houses property sizes (bytes per value):
                #   08:1  09:1  0A:2  0B:1  0C:1  0D:1  0E:1  0F:1
                #   10:2  11:1  12:2  13:2  14:1  15:1  16:1  17:2
                #   18:1  19:1  1A:2  1B:1  1C:1  1D:1  1E:2  1F:2
                #   20:4  21:2  22:4  23:1  24:1
                else:
                    _PROP_SIZES: dict[int, int] = {
                        0x08: 1, 0x09: 1, 0x0A: 2, 0x0B: 1, 0x0C: 1,
                        0x0D: 1, 0x0E: 1, 0x0F: 1, 0x10: 2, 0x11: 1,
                        0x12: 2, 0x13: 2, 0x14: 1, 0x15: 1, 0x16: 1,
                        0x17: 2, 0x18: 1, 0x19: 1, 0x1A: 2, 0x1B: 1,
                        0x1C: 1, 0x1D: 1, 0x1E: 2, 0x1F: 2,
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


def class_to_construction_switch(building_class: Optional[int]) -> str:
    """Map building class to appropriate construction check switch."""
    if building_class == 2:
        return "switch_ttrs_flats"
    if building_class == 3:
        return "switch_ttrs_offices"
    if building_class is not None and building_class >= 4:
        return "switch_ttrs_landmark_unique"
    return "switch_ttrs_residential"


def emit_colour_switch(
    house_id_hex: str,
    colour_values: list[int],
) -> tuple[list[str], str]:
    """Emit NML colour callback block and return (lines, switch_name).

    For a single colour value, emits an inline ``return <value>;`` expression.
    For multiple values, emits a ``random_switch`` that picks among them with
    equal probability (triggered on tile loop, matching NFO semantics).
    """
    switch_name = f"switch_ttrs_{house_id_hex.lower()}_colour"
    lines: list[str] = []

    if len(colour_values) == 1:
        # Single colour — no need for a random_switch, just inline it.
        # We still emit a switch so the identifier exists.
        lines.append(
            f"switch (FEAT_HOUSES, SELF, {switch_name}, 0) "
            f"{{ return {colour_values[0]}; }}"
        )
    else:
        # Multiple colours — random_switch with equal weights
        entries = " ".join(f"1: return {v};" for v in colour_values)
        lines.append(
            f"random_switch (FEAT_HOUSES, SELF, {switch_name}, "
            f"bitmask(TRIGGER_HOUSE_TILELOOP)) {{ {entries} }}"
        )

    return lines, switch_name


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
) -> list[str]:
    """Build NML item block for a house.

    ``layout_name`` is what goes in ``default:`` when tile_layouts is not given.
    ``colour_switch`` if set, is the NML identifier for the colour callback.
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
            # NFO prop 0x18 (0-255) → NML probability (1-15).
            # Use round(prob/17) for a more uniform distribution
            # (17 ≈ 255/15) instead of the coarser prob//16.
            lines.append(f"\t\tprobability: {max(1, min(15, round(prob / 17)))};")
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
) -> None:
    """Generate combined sprites+items TTRS NML file from NFO source."""
    nfo_text = read_nfo_text(nfo_path)
    lines = nfo_text.splitlines()

    # Build new graph-based house graphics (tropic / animation / random support)
    _house_graphics, _ = build_all_house_graphics(lines)

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
            multi_colour_values: list[int] = []

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
                if t > 0:
                    skip_ids.add(tile_id)

            colour_switch_name = None
            if multi_colour_values:
                colour_lines, colour_switch_name = emit_colour_switch(
                    house_id_hex, multi_colour_values
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
                    house_id_hex, house_colours
                )
                out.extend(colour_lines)

            out.extend(
                build_item_block(
                    house_id_hex,
                    start_id + index,
                    display_name,
                    props,
                    layout_name=layout_name,
                    colour_switch=colour_switch_name,
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
