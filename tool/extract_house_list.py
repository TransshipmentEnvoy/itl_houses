#!/usr/bin/env python3
"""
Extract the complete list of houses from a TTRS NFO file.

This script parses the NFO to find all house definitions via Action 3 entries,
extracts names from Action 4, and properties from Action 0. It outputs a
comprehensive building list that can be used for verification and debugging.

Usage:
    python extract_house_list.py --nfo path/to/ttrs3wmod.nfo [--output houses.txt]
"""
import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ============================================================================
# Regex patterns for NFO parsing
# ============================================================================

# Action 3 for houses: maps house IDs to graphics chains
# Format: <sprite> * <len> 03 07 01 <house_id> ...
ACTION3_HOUSE_RE = re.compile(
    r"^\s*\d+\s+\*\s+\d+\s+03\s+07\s+01\s+([0-9A-Fa-f]{2})\s+"
)

# Action 4 for house names (feature 07 with 0x40 offset = 0x48)
# Format: <sprite> * <len> 04 48 FF 01 <house_id> DC "<name>" 00
# Note: Names may contain escaped quotes like \"
NAME_RE = re.compile(
    r'\b04\s+48\s+(?:FF|7F|[0-9A-Fa-f]{2})\s+01\s+([0-9A-Fa-f]{2})\s+DC\s+"((?:[^"\\]|\\.)*)"\s+00'
)

# Action 0 for house properties
# Format: <sprite> * <len> 00 07 <num_props> 01 <house_id> <props...>
# Or multi-house: 00 07 <num_props> <num_houses> <first_id> <props...>
ACTION0_HOUSE_RE = re.compile(
    r"^\s*\d+\s+\*\s+\d+\s+00\s+07\s+([0-9A-Fa-f]{2})\s+([0-9A-Fa-f]{2})\s+([0-9A-Fa-f]{2})\s*(.*)"
)

# House marker comments (only present for some houses)
HOUSE_MARKER_RE = re.compile(r"^//\s*House\s+([0-9A-Fa-f]{2})(?:\s*-\s*(.+))?", re.IGNORECASE)

# Property value patterns
HEX2_RE = re.compile(r"^[0-9A-Fa-f]{2}$")
BACKSLASH_B_RE = re.compile(r"\\b(\d+)")
BACKSLASH_W_RE = re.compile(r"\\w(\d+)")
BACKSLASH_WX_RE = re.compile(r"\\wx([0-9A-Fa-f]+)")

# Sprite header pattern (identifies start of new sprite)
SPRITE_HEADER_RE = re.compile(r"^\s*\d+\s*\*\s*\d+")


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class HouseInfo:
    """Information about a single house."""
    house_id: int
    name: str = ""
    substitute: Optional[int] = None
    years_available: Optional[tuple[int, int]] = None
    population: Optional[int] = None
    mail_multiplier: Optional[int] = None
    building_class: Optional[int] = None
    probability: Optional[int] = None
    availability_mask: Optional[tuple[int, int]] = None  # (zone, climate)
    cargo_acceptance: dict = field(default_factory=dict)  # {cargo_type: amount}
    building_flags_low: Optional[int] = None
    building_flags_high: Optional[int] = None
    has_action3: bool = False
    marker_label: str = ""
    line_numbers: list = field(default_factory=list)
    
    @property
    def hex_id(self) -> str:
        return f"{self.house_id:02X}"
    
    def __str__(self) -> str:
        parts = [f"0x{self.hex_id}"]
        if self.name:
            parts.append(f'"{self.name}"')
        if self.substitute is not None:
            parts.append(f"sub={self.substitute}")
        if self.years_available:
            parts.append(f"years={self.years_available[0]}-{self.years_available[1]}")
        if self.building_class is not None:
            parts.append(f"class={self.building_class}")
        if self.population is not None:
            parts.append(f"pop={self.population}")
        return " ".join(parts)


# ============================================================================
# Parsing utilities
# ============================================================================

def read_nfo_text(nfo_path: Path) -> str:
    """Read NFO file with fallback encoding."""
    data = nfo_path.read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin1")


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


# ============================================================================
# Main parsing functions
# ============================================================================

def extract_house_names(nfo_text: str) -> dict[int, str]:
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


def extract_house_markers(lines: list[str]) -> dict[int, tuple[str, int]]:
    """Find house marker comments, return {house_id: (label, line_number)}."""
    markers: dict[int, tuple[str, int]] = {}
    for line_num, line in enumerate(lines, start=1):
        match = HOUSE_MARKER_RE.match(line)
        if match:
            house_id = int(match.group(1), 16)
            label = (match.group(2) or "").strip()
            if house_id not in markers:
                markers[house_id] = (label, line_num)
    return markers


def parse_action0_properties(lines: list[str]) -> dict[int, dict[str, object]]:
    """Parse Action 0 property definitions for houses.
    
    Properties from earlier definitions are preserved - later bulk operations
    won't overwrite already-set values (except substitute which is special).
    """
    properties: dict[int, dict[str, object]] = {}
    
    line_idx = 0
    while line_idx < len(lines):
        line = lines[line_idx]
        match = ACTION0_HOUSE_RE.match(line)
        if not match:
            line_idx += 1
            continue
        
        line_num = line_idx + 1  # 1-based line number for debugging
        num_props = int(match.group(1), 16)
        num_houses = int(match.group(2), 16)
        first_id = int(match.group(3), 16)
        rest = match.group(4)
        
        if num_props == 0 or num_houses == 0:
            line_idx += 1
            continue
        
        # Collect continuation lines (lines that don't start with sprite header)
        content_lines = [rest] if rest else []
        k = line_idx + 1
        while k < len(lines):
            next_line = lines[k]
            stripped = next_line.strip()
            # Stop at empty lines, sprite headers (digits * digits), or comments
            if not stripped or SPRITE_HEADER_RE.match(next_line) or next_line.startswith("//"):
                break
            content_lines.append(stripped)
            k += 1
        
        full_content = " ".join(content_lines)
        
        # Tokenize the content
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
            if "_line" not in props:
                props["_line"] = line_num
            
            # Simple property parsing - just grab key properties
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
                # Special handling: only set if this looks like a real definition
                if key_int == 0x08:
                    if p < len(tokens):
                        val = parse_token_value(tokens[p])
                        if val is not None:
                            # Only set substitute if not already set, or if this is a specific definition
                            if "substitute" not in props or not is_bulk_disable:
                                props["substitute"] = val
                        p += 1
                
                # Property 09: building_flags low byte
                elif key_int == 0x09:
                    if p < len(tokens):
                        val = parse_token_value(tokens[p])
                        if val is not None and "building_flags_low" not in props:
                            props["building_flags_low"] = val
                        p += 1
                
                # Property 0A: years_available (2 words)
                elif key_int == 0x0A:
                    if p + 1 < len(tokens):
                        y0 = parse_token_value(tokens[p])
                        y1 = parse_token_value(tokens[p + 1])
                        if y0 is not None and y1 is not None and "years_available" not in props:
                            props["years_available"] = (y0, y1)
                        p += 2
                
                # Property 0B: population
                elif key_int == 0x0B:
                    if p < len(tokens):
                        val = parse_token_value(tokens[p])
                        if val is not None and "population" not in props:
                            props["population"] = val
                        p += 1
                
                # Property 0C: mail_multiplier
                elif key_int == 0x0C:
                    if p < len(tokens):
                        val = parse_token_value(tokens[p])
                        if val is not None and "mail_multiplier" not in props:
                            props["mail_multiplier"] = val
                        p += 1
                
                # Property 0D: passenger acceptance
                elif key_int == 0x0D:
                    if p < len(tokens):
                        val = parse_token_value(tokens[p])
                        if val is not None and "cargo_pass" not in props:
                            props["cargo_pass"] = val
                        p += 1
                
                # Property 0E: mail acceptance
                elif key_int == 0x0E:
                    if p < len(tokens):
                        val = parse_token_value(tokens[p])
                        if val is not None and "cargo_mail" not in props:
                            props["cargo_mail"] = val
                        p += 1
                
                # Property 0F: goods acceptance
                elif key_int == 0x0F:
                    if p < len(tokens):
                        val = parse_token_value(tokens[p])
                        if val is not None and "cargo_goods" not in props:
                            props["cargo_goods"] = val
                        p += 1
                
                # Property 13: availability_mask (2 bytes)
                elif key_int == 0x13:
                    if p + 1 < len(tokens):
                        zone = parse_token_value(tokens[p])
                        climate = parse_token_value(tokens[p + 1])
                        if zone is not None and climate is not None and "availability_mask" not in props:
                            props["availability_mask"] = (zone, climate)
                        p += 2
                
                # Property 18: probability
                elif key_int == 0x18:
                    if p < len(tokens):
                        val = parse_token_value(tokens[p])
                        if val is not None and "probability" not in props:
                            props["probability"] = val
                        p += 1
                
                # Property 19: building_flags high byte
                elif key_int == 0x19:
                    if p < len(tokens):
                        val = parse_token_value(tokens[p])
                        if val is not None and "building_flags_high" not in props:
                            props["building_flags_high"] = val
                        p += 1
                
                # Property 1C: building_class
                elif key_int == 0x1C:
                    if p < len(tokens):
                        val = parse_token_value(tokens[p])
                        if val is not None and "building_class" not in props:
                            props["building_class"] = val
                        p += 1
                
                # Skip unknown properties (assume 1 byte)
                else:
                    p += 1
        
        line_idx += 1
    
    return properties


def build_house_list(nfo_path: Path) -> list[HouseInfo]:
    """Build a complete list of houses from the NFO file."""
    nfo_text = read_nfo_text(nfo_path)
    lines = nfo_text.splitlines()
    
    # Extract all information
    names = extract_house_names(nfo_text)
    action3_ids = extract_house_ids_from_action3(lines)
    markers = extract_house_markers(lines)
    properties = parse_action0_properties(lines)
    
    # Get all house IDs (union of all sources)
    all_ids = set(action3_ids.keys()) | set(properties.keys())
    
    # Build house info objects
    houses: list[HouseInfo] = []
    for house_id in sorted(all_ids):
        info = HouseInfo(house_id=house_id)
        
        # Name from Action 4
        if house_id in names:
            info.name = names[house_id]
        
        # Marker comment
        if house_id in markers:
            info.marker_label = markers[house_id][0]
            info.line_numbers.append(markers[house_id][1])
        
        # Action 3 presence
        if house_id in action3_ids:
            info.has_action3 = True
            info.line_numbers.append(action3_ids[house_id])
        
        # Properties from Action 0
        if house_id in properties:
            props = properties[house_id]
            if "substitute" in props:
                info.substitute = props["substitute"]
            if "years_available" in props:
                info.years_available = props["years_available"]
            if "population" in props:
                info.population = props["population"]
            if "mail_multiplier" in props:
                info.mail_multiplier = props["mail_multiplier"]
            if "building_class" in props:
                info.building_class = props["building_class"]
            if "probability" in props:
                info.probability = props["probability"]
            if "availability_mask" in props:
                info.availability_mask = props["availability_mask"]
            if "building_flags_low" in props:
                info.building_flags_low = props["building_flags_low"]
            if "building_flags_high" in props:
                info.building_flags_high = props["building_flags_high"]
            if "cargo_pass" in props:
                info.cargo_acceptance["PASS"] = props["cargo_pass"]
            if "cargo_mail" in props:
                info.cargo_acceptance["MAIL"] = props["cargo_mail"]
            if "cargo_goods" in props:
                info.cargo_acceptance["GOOD"] = props["cargo_goods"]
            if "_line" in props:
                info.line_numbers.append(props["_line"])
        
        houses.append(info)
    
    return houses


def format_house_list(houses: list[HouseInfo], verbose: bool = False) -> str:
    """Format the house list for output."""
    lines = []
    lines.append(f"Total houses found: {len(houses)}")
    lines.append(f"Houses with Action 3: {sum(1 for h in houses if h.has_action3)}")
    lines.append(f"Houses with names: {sum(1 for h in houses if h.name)}")
    lines.append("")
    lines.append("=" * 80)
    lines.append("")
    
    # Group by ranges for better readability
    current_range_start = None
    for house in houses:
        range_start = (house.house_id // 16) * 16
        if range_start != current_range_start:
            if current_range_start is not None:
                lines.append("")
            lines.append(f"--- Houses 0x{range_start:02X} - 0x{range_start + 15:02X} ---")
            current_range_start = range_start
        
        # Basic info line
        flags = []
        if house.has_action3:
            flags.append("A3")
        if house.name:
            flags.append("named")
        if house.marker_label:
            flags.append(f"marker:{house.marker_label}")
        
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        
        if verbose:
            lines.append(f"  {house}{flag_str}")
        else:
            name_part = f' "{house.name}"' if house.name else ""
            sub_part = f" sub={house.substitute}" if house.substitute is not None else ""
            lines.append(f"  0x{house.hex_id}{name_part}{sub_part}{flag_str}")
    
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract the complete list of houses from a TTRS NFO file"
    )
    parser.add_argument("--nfo", required=True, type=Path, help="Path to ttrs3wmod.nfo")
    parser.add_argument("--output", "-o", type=Path, help="Output file (default: stdout)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed info")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()
    
    houses = build_house_list(args.nfo)
    
    if args.json:
        import json
        data = []
        for h in houses:
            entry = {
                "id": h.house_id,
                "hex_id": h.hex_id,
                "name": h.name,
                "substitute": h.substitute,
                "years_available": h.years_available,
                "population": h.population,
                "building_class": h.building_class,
                "has_action3": h.has_action3,
            }
            data.append(entry)
        output = json.dumps(data, indent=2)
    else:
        output = format_house_list(houses, verbose=args.verbose)
    
    if args.output:
        args.output.write_text(output, encoding="utf-8")
        print(f"Output written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
