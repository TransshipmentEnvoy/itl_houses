#!/usr/bin/env python3
"""
Generate graphics_ttrs-style NML sprites from a ttrs3wmod.nfo file.

This script parses the NFO to extract Action 1 sprite definitions and
Action 2 sprite layouts, then generates NML spriteset/spritelayout blocks
with proper construction stage handling.

Usage:
    python generate_graphics_ttrs_from_nfo.py --nfo path/to/ttrs3wmod.nfo --out graphics_ttrs.nml [--pcx-path src/sprites/pcx/ttrs3w.pcx]
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
ACTION3_HOUSE_RE = re.compile(
    r"^\s*\d+\s+\*\s+\d+\s+03\s+07\s+01\s+([0-9A-Fa-f]{2})\s+"
)

# NFO sprite line header
SPRITE_HEADER_RE = re.compile(r"^\s*\d+\s*\*\s*\d+")

# Action 4 for house names (feature 07 with 0x40 offset = 0x48)
# Format: <sprite> * <len> 04 48 FF 01 <house_id> DC "<name>" 00
# Note: Names may contain escaped quotes like \"
NAME_RE = re.compile(
    r'\b04\s+48\s+(?:FF|7F|[0-9A-Fa-f]{2})\s+01\s+([0-9A-Fa-f]{2})\s+DC\s+"((?:[^"\\]|\\.)*)"\s+00'
)

# NFO sprite line format:
# <index> sprites/pcx/ttrs3w.pcx xpos ypos comp ysize xsize xrel yrel
SPRITE_LINE_RE = re.compile(
    r"^\s*(\d+)\s+sprites/pcx/ttrs3w\.pcx\s+(-?\d+)\s+(-?\d+)\s+01\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)")

# Action 1 for houses: defines sprite sets
# Format: <sprite> * <len> 01 07 <set_count> ...
ACTION1_HOUSE_RE = re.compile(r"^\s*\d+\s+\*\s+\d+\s+01\s+07\s+([0-9A-Fa-f]{2})\b")

# Action 2 for house layouts (type 00)
# Format: <sprite> * <len> 02 07 <set_id> 00 ...
ACTION2_HOUSE_LAYOUT_RE = re.compile(r"^\s*\d+\s+\*\s+\d+\s+02\s+07\s+([0-9A-Fa-f]{2})\s+00\s+(.*)$")

# ============================================================================
# Constants
# ============================================================================

# Sprite DWORD flag/mask constants (GRF Action2 sprite layout format)
SPRITE_ACTION1_FLAG = 0x80000000  # Bit 31: sprite from Action1 set (vs base game)
SPRITE_RECOLOUR_FLAG = 0x00008000  # Bit 15: enable recolouring
SPRITE_INDEX_MASK = 0x00003FFF  # Bits 0-13: sprite number / Action1 set index


# ============================================================================
# Data structures
# ============================================================================


@dataclass(frozen=True)
class SpriteCoord:
    xpos: int
    ypos: int
    xsize: int
    ysize: int
    xrel: int
    yrel: int


def read_nfo_text(nfo_path: Path) -> str:
    data = nfo_path.read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin1")


def normalize_house_name(raw_name: str) -> str:
    text = raw_name.strip()
    # Unescape NFO-style escaped quotes
    text = text.replace('\\"', '"')
    # Repair common mojibake (UTF-8 bytes decoded as latin1), e.g. "Ã" -> "Þ".
    if any(marker in text for marker in ("Ã", "Â")):
        try:
            repaired = text.encode("latin1").decode("utf-8")
            text = repaired
        except UnicodeError:
            pass

    # Remove NewGRF prefix marker(s) often used in name strings.
    text = re.sub(r"^[Þþ\x9e\x9f]+", "", text).strip()
    return text or "Unnamed"


def parse_names(nfo_text: str) -> dict[str, str]:
    names: dict[str, str] = {}
    for match in NAME_RE.finditer(nfo_text):
        house_id = match.group(1).upper()
        raw_name = normalize_house_name(match.group(2))
        if house_id not in names:
            names[house_id] = raw_name
    return names


def extract_house_ids_from_action3(lines: list[str]) -> dict[str, int]:
    """Find all house IDs defined via Action 3, return {house_id_hex: line_number}."""
    house_ids: dict[str, int] = {}
    for line_num, line in enumerate(lines, start=1):
        match = ACTION3_HOUSE_RE.match(line)
        if match:
            house_id = match.group(1).upper()
            if house_id not in house_ids:
                house_ids[house_id] = line_num
    return house_ids


def parse_house_action1_set_table(lines: list[str]) -> dict[int, SpriteCoord]:
    """Parse the first Action 1 block to build a sprite coordinate table."""
    table: dict[int, SpriteCoord] = {}

    start_line = -1
    set_count = 0
    for idx, line in enumerate(lines):
        match = ACTION1_HOUSE_RE.match(line)
        if not match:
            continue
        start_line = idx + 1
        set_count = int(match.group(1), 16)
        break

    if start_line == -1 or set_count <= 0:
        return table

    set_id = 0
    for line in lines[start_line:]:
        match = SPRITE_LINE_RE.match(line)
        if not match:
            if set_id > 0:
                break
            continue

        xpos = int(match.group(2), 10)
        ypos = int(match.group(3), 10)
        ysize = int(match.group(4), 10)
        xsize = int(match.group(5), 10)
        xrel = int(match.group(6), 10)
        yrel = int(match.group(7), 10)

        table[set_id] = SpriteCoord(
            xpos=xpos,
            ypos=ypos,
            xsize=xsize,
            ysize=ysize,
            xrel=xrel,
            yrel=yrel,
        )
        set_id += 1

        if set_id >= set_count:
            break

    return table


def parse_house_action1_blocks(lines: list[str]) -> list[tuple[int, dict[int, SpriteCoord]]]:
    """Parse all Action 1 blocks for houses, returning (line_index, sprite_table) pairs."""
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

            xpos = int(body_match.group(2), 10)
            ypos = int(body_match.group(3), 10)
            ysize = int(body_match.group(4), 10)
            xsize = int(body_match.group(5), 10)
            xrel = int(body_match.group(6), 10)
            yrel = int(body_match.group(7), 10)

            table[set_id] = SpriteCoord(
                xpos=xpos,
                ypos=ypos,
                xsize=xsize,
                ysize=ysize,
                xrel=xrel,
                yrel=yrel,
            )
            set_id += 1
            if set_id >= set_count:
                break

        if table:
            blocks.append((idx, table))

    return blocks


def sprite_table_for_house(
    house_line_index: int,
    action1_blocks: list[tuple[int, dict[int, SpriteCoord]]]
) -> dict[int, SpriteCoord]:
    """Get the appropriate sprite table for a house based on its line position."""
    chosen: dict[int, SpriteCoord] = {}
    for block_line_index, table in action1_blocks:
        if block_line_index <= house_line_index:
            chosen = table
        else:
            break
    return chosen


def collect_house_sections(lines: list[str]) -> list[tuple[str, str, list[str], int]]:
    """Collect house sections delimited by marker comments.
    
    Returns list of (house_id_hex, marker_label, lines, start_line_index).
    """
    sections: list[tuple[str, str, list[str], int]] = []
    current_house_id: Optional[str] = None
    current_house_label: str = ""
    current_lines: list[str] = []
    current_start_idx = -1

    for idx, line in enumerate(lines):
        marker = HOUSE_MARKER_RE.match(line)
        if marker:
            if current_house_id is not None:
                sections.append((current_house_id, current_house_label, current_lines, current_start_idx))
            current_house_id = marker.group(1).upper()
            current_house_label = (marker.group(2) or "").strip()
            current_lines = []
            current_start_idx = idx
            continue

        if current_house_id is not None:
            current_lines.append(line)

    if current_house_id is not None:
        sections.append((current_house_id, current_house_label, current_lines, current_start_idx))

    return sections


def collect_house_sections_from_action3(lines: list[str]) -> list[tuple[str, str, list[str], int]]:
    """Collect house sections based on Action 3 entries (more reliable than markers).
    
    For each Action 3 house definition, scan backwards to find associated Action 2 
    layout definitions. Falls back to marker info for labels when available.
    
    Returns list of (house_id_hex, label, relevant_lines, start_line_index).
    """
    # First, collect marker info for labels
    markers: dict[str, tuple[str, int]] = {}
    for idx, line in enumerate(lines):
        marker = HOUSE_MARKER_RE.match(line)
        if marker:
            house_id = marker.group(1).upper()
            label = (marker.group(2) or "").strip()
            if house_id not in markers:
                markers[house_id] = (label, idx)
    
    # Find all Action 3 entries and their positions
    action3_positions: list[tuple[str, int]] = []
    for idx, line in enumerate(lines):
        match = ACTION3_HOUSE_RE.match(line)
        if match:
            house_id = match.group(1).upper()
            action3_positions.append((house_id, idx))
    
    # For each Action 3, find preceding Action 2 definitions
    sections: list[tuple[str, str, list[str], int]] = []
    seen_ids: set[str] = set()
    
    for house_id, action3_idx in action3_positions:
        if house_id in seen_ids:
            continue
        seen_ids.add(house_id)
        
        # Determine search range for Action 2 entries
        # Look backwards from Action 3 position until we hit:
        # - A marker comment for a different house
        # - Another Action 3 entry
        # - Start of file
        # - 200 lines (safety limit)
        
        search_start = max(0, action3_idx - 200)
        
        # Check for marker for this house
        if house_id in markers:
            marker_label, marker_idx = markers[house_id]
            search_start = max(search_start, marker_idx)
        else:
            marker_label = ""
        
        # Find where previous house ends
        for prev_id, prev_idx in action3_positions:
            if prev_idx < action3_idx and prev_idx >= search_start:
                # There's another Action 3 between our search_start and current house
                search_start = max(search_start, prev_idx + 1)
        
        # Collect lines from search_start to action3_idx (inclusive)
        relevant_lines = lines[search_start:action3_idx + 1]
        
        sections.append((house_id, marker_label, relevant_lines, search_start))
    
    return sections


def collect_house_set_pairs(house_lines: list[str]) -> dict[int, tuple[int, int]]:
    """Return {set_id: (ground_dword, building_dword)} with full 32-bit LE values.

    Each DWORD encodes the sprite reference for an Action2 type-00 layout:
      - Bit 31: 1 = Action1 set index, 0 = base game sprite
      - Bit 15: 1 = enable recolouring
      - Bits 0-13: sprite number / set index
    """
    pairs: dict[int, tuple[int, int]] = {}
    for line in house_lines:
        match = ACTION2_HOUSE_LAYOUT_RE.match(line)
        if not match:
            continue

        set_id = int(match.group(1), 16)
        payload = match.group(2)
        bytes_payload = re.findall(r"\b[0-9A-Fa-f]{2}\b", payload)
        if len(bytes_payload) < 8:
            continue

        # Parse full 4-byte little-endian DWORDs for ground and building sprites.
        ground_dword = (int(bytes_payload[0], 16) | (int(bytes_payload[1], 16) << 8) |
                        (int(bytes_payload[2], 16) << 16) | (int(bytes_payload[3], 16) << 24))
        building_dword = (int(bytes_payload[4], 16) | (int(bytes_payload[5], 16) << 8) |
                          (int(bytes_payload[6], 16) << 16) | (int(bytes_payload[7], 16) << 24))
        pairs[set_id] = (ground_dword, building_dword)
    return pairs


def pick_representative_set_ids(set_pairs: dict[int, tuple[int, int]]) -> list[int]:
    """Pick representative set IDs for construction stage sprites."""
    if not set_pairs:
        return []

    preferred_order = [0x00, 0x01, 0x02, 0x03, 0x30, 0x31, 0x32]
    ordered: list[int] = []

    for set_id in preferred_order:
        if set_id in set_pairs and set_id not in ordered:
            ordered.append(set_id)

    for set_id in sorted(set_pairs.keys()):
        if set_id not in ordered:
            ordered.append(set_id)

    return ordered[:4]


# ============================================================================
# Output formatting helpers
# ============================================================================

def sprite_literal(sprite: SpriteCoord) -> str:
    """Format sprite coordinates as NML literal."""
    return f"[{sprite.xpos}, {sprite.ypos}, {sprite.xsize}, {sprite.ysize}, {sprite.xrel}, {sprite.yrel}]"


def build_layout_expr(stage_count: int) -> str:
    """Build construction_state expression for sprite selection."""
    if stage_count <= 1:
        return "0"
    if stage_count == 2:
        return "construction_state < 3 ? construction_state : 1"
    if stage_count == 3:
        return "construction_state < 3 ? construction_state : 2"
    return "construction_state"


def _pick_ground_dword(selected_set_ids: list[int], set_pairs: dict[int, tuple[int, int]]) -> int:
    """Pick the best ground sprite DWORD, preferring the completed-stage (last) set.

    Construction-stage sets often use bare ground (e.g. sprite 3924).
    The completed stage (last selected set) usually has the real ground.
    """
    # Try sets in reverse order; prefer Action1 ground over base-game bare ground.
    for set_id in reversed(selected_set_ids):
        ground_dword = set_pairs[set_id][0]
        if ground_dword & SPRITE_ACTION1_FLAG:
            return ground_dword
    # Fall back to the last set's ground (may be base-game sprite).
    return set_pairs[selected_set_ids[-1]][0]


# ============================================================================
# NML generation
# ============================================================================

def emit_house_block(
    house_id_hex: str,
    title: str,
    selected_set_ids: list[int],
    set_pairs: dict[int, tuple[int, int]],
    sprite_table: dict[int, SpriteCoord],
    pcx_path: str,
) -> list[str]:
    """Emit NML spriteset and spritelayout blocks for a single house."""
    lines: list[str] = []

    set_id_note = ", ".join(f"0x{value:02X}" for value in selected_set_ids)
    lines.append(f"/* --- 0x{house_id_hex}  {title} (sets: {set_id_note}) --- */")

    # --- Ground sprite --------------------------------------------------
    ground_dword = _pick_ground_dword(selected_set_ids, set_pairs)
    ground_is_action1 = bool(ground_dword & SPRITE_ACTION1_FLAG)
    ground_index = ground_dword & SPRITE_INDEX_MASK

    ground_pcx: SpriteCoord | None = None
    if ground_is_action1:
        ground_pcx = sprite_table.get(ground_index)
        if ground_pcx is None:
            lines.append(f"/* skipped: missing ground sprite Action1 index 0x{ground_index:02X} */")
            lines.append("")
            return lines

    # --- Building sprites -----------------------------------------------
    building_pcx_list: list[SpriteCoord] = []
    seen_building_indices: list[int] = []
    has_recolour = False
    for set_id in selected_set_ids:
        _, building_dword = set_pairs[set_id]
        if not (building_dword & SPRITE_ACTION1_FLAG):
            continue  # skip base-game / empty building sprites
        building_index = building_dword & SPRITE_INDEX_MASK
        if building_dword & SPRITE_RECOLOUR_FLAG:
            has_recolour = True
        if building_index not in seen_building_indices:
            sprite = sprite_table.get(building_index)
            if sprite is not None:
                building_pcx_list.append(sprite)
                seen_building_indices.append(building_index)

    if not building_pcx_list:
        lines.append("/* skipped: no valid building sprites */")
        lines.append("")
        return lines

    # --- Emit spritesets -------------------------------------------------
    if ground_is_action1 and ground_pcx is not None:
        lines.append(f"spriteset(ss_ttrs_{house_id_hex}_g, \"{pcx_path}\") {{ {sprite_literal(ground_pcx)} }}")

    lines.append(f"spriteset(ss_ttrs_{house_id_hex}_b, \"{pcx_path}\") {{")
    for idx, sprite in enumerate(building_pcx_list, start=1):
        suffix = "completed" if idx == len(building_pcx_list) else f"constr {idx}"
        lines.append(f"\t{sprite_literal(sprite)}  /* {suffix} */")
    lines.append("}")

    # --- Emit spritelayout -----------------------------------------------
    lines.append(f"spritelayout sl_ttrs_{house_id_hex} {{")
    if ground_is_action1:
        lines.append(f"\tground   {{ sprite: ss_ttrs_{house_id_hex}_g(0); }}")
    else:
        lines.append(f"\tground   {{ sprite: {ground_index}; }}")

    recolour_suffix = " recolour_mode: RECOLOUR_REMAP; palette: PALETTE_USE_DEFAULT;" if has_recolour else ""
    lines.append(
        f"\tbuilding {{ sprite: ss_ttrs_{house_id_hex}_b({build_layout_expr(len(building_pcx_list))});{recolour_suffix} }}"
    )
    lines.append("}")
    lines.append("")

    return lines


def generate_graphics_ttrs_nml(nfo_path: Path, output_path: Path, pcx_path: str) -> None:
    """Generate complete graphics_ttrs NML file from NFO source."""
    nfo_text = read_nfo_text(nfo_path)
    lines = nfo_text.splitlines()

    names = parse_names(nfo_text)
    action1_blocks = parse_house_action1_blocks(lines)
    # Use Action 3-based collection for more reliable house detection
    house_sections = collect_house_sections_from_action3(lines)

    out: list[str] = []
    out.append("/* Auto-generated from ttrs3wmod.nfo */")
    out.append("/* Generated by tool/generate_graphics_ttrs_from_nfo.py */")
    out.append("/* NFO format source: xpos ypos comp YSIZE XSIZE xrel yrel */")
    out.append("/* NML sprite format: [xpos, ypos, xsize, ysize, xoffset, yoffset] */")
    out.append("")

    for house_id_hex, marker_label, house_lines, house_start_idx in house_sections:
        set_pairs = collect_house_set_pairs(house_lines)
        if not set_pairs:
            continue

        sprite_table = sprite_table_for_house(house_start_idx, action1_blocks)
        if not sprite_table:
            continue

        title = names.get(house_id_hex) or marker_label or f"House {house_id_hex}"
        selected_set_ids = pick_representative_set_ids(set_pairs)
        if not selected_set_ids:
            continue

        out.extend(
            emit_house_block(
                house_id_hex=house_id_hex,
                title=title,
                selected_set_ids=selected_set_ids,
                set_pairs=set_pairs,
                sprite_table=sprite_table,
                pcx_path=pcx_path,
            ))

    output_path.write_text("\n".join(out), encoding="utf-8")


# ============================================================================
# Main entry point
# ============================================================================

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate graphics_ttrs-style NML sprites from ttrs3wmod.nfo"
    )
    parser.add_argument("--nfo", required=True, type=Path, help="Path to ttrs3wmod.nfo")
    parser.add_argument("--out", required=True, type=Path, help="Output .nml path")
    parser.add_argument(
        "--pcx-path",
        default="src/sprites/pcx/ttrs3w.pcx",
        help="PCX path to write in spriteset declarations (default: src/sprites/pcx/ttrs3w.pcx)",
    )
    args = parser.parse_args()

    generate_graphics_ttrs_nml(args.nfo, args.out, args.pcx_path)
    print(f"Generated {args.out}")


if __name__ == "__main__":
    main()
