"""
parse_raw.py — Multi-line NFO sprite byte collector.

NFO "pseudo-sprites" can span multiple physical lines.  The header line has
the form::

    <num> * <byte_count>  <hex_byte> <hex_byte> ...

and continuation bytes appear on subsequent lines that do NOT start with a
new sprite header (``\d+ * \d+``).  Comments (``// …``) are stripped before
scanning.

Public API
----------
    collect_action2_house_entries(lines)
        → list[ RawSprite ]

    RawSprite.bytes   : list[int]   — raw byte payload
    RawSprite.line_no : int         — 0-based index of the header line
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

# Matches the start of any pseudo-sprite: ``NNN * LEN  <optional bytes>``
# Groups: (1) byte_count  (2) rest-of-line bytes
_SPRITE_HEADER_RE = re.compile(
    r"^\s*\d+\s+\*\s+(\d+)\s+(.*)"
)

# Matches any 2-hex-char token.
_HEX_TOKEN_RE = re.compile(r"\b[0-9A-Fa-f]{2}\b")

# GRFCodec \2<op> escape sequences used in type-89 Action 2 computation chains.
# Each escape encodes a single operator byte in the binary stream.
_NFO_ESCAPE_RE = re.compile(
    r"\\2(?:u>>|u<<|u<|u>|u/|u%|<<|>>|sto|rst|psto|ror|cmp|ucmp"
    r"|[+\-*/<>%&|^])"
)
_NFO_ESCAPE_MAP: dict[str, int] = {
    r"\2+": 0x00,   r"\2-":  0x01,  r"\2<":    0x04,  r"\2>": 0x05,
    r"\2u<": 0x06,  r"\2u>": 0x07,  r"\2/":    0x02,  r"\2%": 0x03,
    r"\2u/": 0x08,  r"\2u%": 0x09,  r"\2*":    0x10,  r"\2&": 0x11,
    r"\2|": 0x12,   r"\2^": 0x13,   r"\2sto":  0x0A,  r"\2rst": 0x0D,
    r"\2psto": 0x0E, r"\2ror": 0x0B, r"\2cmp":  0x0C, r"\2ucmp": 0x0F,
    r"\2<<": 0x14,  r"\2u>>": 0x15, r"\2>>": 0x16,
}

# A line that starts a new pseudo-sprite (used to detect end of continuation).
_NEW_SPRITE_RE = re.compile(r"^\s*\d+\s+\*\s+\d+")


def _strip_comment(line: str) -> str:
    """Remove trailing NFO comment (// …)."""
    idx = line.find("//")
    return line[:idx] if idx >= 0 else line


# Combined pattern: match either an NFO escape or a hex token, in order.
_TOKEN_RE = re.compile(
    r"(?:" + _NFO_ESCAPE_RE.pattern + r")"
    r"|(?:\b[0-9A-Fa-f]{2}\b)"
)


def _hex_bytes_from(text: str) -> list[int]:
    """Extract all hex tokens and NFO escape sequences from *text* as bytes."""
    result: list[int] = []
    for m in _TOKEN_RE.finditer(text):
        tok = m.group()
        if tok.startswith("\\"):
            val = _NFO_ESCAPE_MAP.get(tok)
            if val is not None:
                result.append(val)
        else:
            result.append(int(tok, 16))
    return result


# ---------------------------------------------------------------------------
# Public data structure
# ---------------------------------------------------------------------------


@dataclass
class RawSprite:
    """The raw byte payload of one pseudo-sprite, with its source line number."""
    bytes:   list[int]   = field(default_factory=list)
    line_no: int         = 0   # 0-based index of the header line in the NFO


# ---------------------------------------------------------------------------
# Core collector
# ---------------------------------------------------------------------------


def collect_action2_house_entries(lines: list[str]) -> list[RawSprite]:
    """
    Return every Action 2 pseudo-sprite for house feature (``02 07 …``).

    We collect the declared number of bytes by combining the header line with
    as many continuation lines as necessary.  We stop early and emit a warning
    when we run out of lines.
    """
    results: list[RawSprite] = []

    i = 0
    while i < len(lines):
        m = _SPRITE_HEADER_RE.match(lines[i])
        if not m:
            i += 1
            continue

        byte_count = int(m.group(1))
        rest_text  = _strip_comment(m.group(2))
        data_bytes = _hex_bytes_from(rest_text)

        # Fast path: not enough bytes yet to check feature; also skip non-Action-2
        # We need at least 4 bytes: 02 07 <set_id> <type>
        # Collect continuation until we have at least 4 or run out
        j = i + 1
        while len(data_bytes) < min(byte_count, 4) and j < len(lines):
            if _NEW_SPRITE_RE.match(lines[j]):
                break
            data_bytes.extend(_hex_bytes_from(_strip_comment(lines[j])))
            j += 1

        # Skip if not ``02 07`` (Action 2 for houses)
        if len(data_bytes) < 2 or data_bytes[0] != 0x02 or data_bytes[1] != 0x07:
            i += 1
            continue

        # Collect remaining bytes up to byte_count
        while len(data_bytes) < byte_count and j < len(lines):
            if _NEW_SPRITE_RE.match(lines[j]):
                break
            data_bytes.extend(_hex_bytes_from(_strip_comment(lines[j])))
            j += 1

        if len(data_bytes) < byte_count:
            # Truncated — emit what we have so callers can at least see the type
            pass  # still useful for type-detection

        results.append(RawSprite(bytes=data_bytes[:byte_count], line_no=i))
        i += 1

    return results


# ---------------------------------------------------------------------------
# Scoped collector: only bytes in a defined range of lines
# ---------------------------------------------------------------------------


def collect_action2_in_range(
    lines: list[str],
    start: int,
    end: int,
) -> list[RawSprite]:
    """
    Like :func:`collect_action2_house_entries` but restricted to
    ``lines[start:end]``.  Returned ``line_no`` values are absolute (relative
    to the full ``lines`` list).
    """
    sub = collect_action2_house_entries(lines[start:end])
    for rs in sub:
        rs.line_no += start
    return sub


# ---------------------------------------------------------------------------
# Action 3 entry scanner
# ---------------------------------------------------------------------------

_ACTION3_HOUSE_RE = re.compile(
    r"^\s*\d+\s+\*\s+\d+\s+03\s+07\s+01\s+([0-9A-Fa-f]{2})\s+00\s+([0-9A-Fa-f]{2})\s+([0-9A-Fa-f]{2})"
)


def collect_action3_entries(lines: list[str]) -> dict[int, int]:
    """
    Return ``{house_id: root_action2_id}`` by scanning all
    ``03 07 01 <house_id> 00 <group_lo> <group_hi>`` entries.

    Note: the group ID is a little-endian word.
    """
    result: dict[int, int] = {}
    for line in lines:
        m = _ACTION3_HOUSE_RE.match(line)
        if m:
            house_id = int(m.group(1), 16)
            group_id = int(m.group(2), 16) | (int(m.group(3), 16) << 8)
            if house_id not in result:
                result[house_id] = group_id
    return result


# ---------------------------------------------------------------------------
# Per-house section boundaries
# ---------------------------------------------------------------------------


def collect_house_section_ranges(
    lines: list[str],
) -> list[tuple[int, int, int, int]]:
    """
    Return one ``(house_id, start_idx, end_idx, root_action2_id)`` tuple per
    house defined via Action 3, ordered by their position in the NFO.

    *start_idx* is the first line to include in the house's own Action 2 graph
    (exclusive of any Action 3 entries from the previous house).
    *end_idx* is the Action 3 line index itself (inclusive).

    The window is at most 400 lines prior to the Action 3 entry (large enough
    to capture all Action 2 entries emitted per house in TTRS, which has up to
    ~200 type-00 entries for heavily animated houses).
    """
    WINDOW = 400

    # Find all Action 3 positions in order
    action3_positions: list[tuple[int, int, int]] = []  # (house_id, line_idx, root_id)
    for idx, line in enumerate(lines):
        m = _ACTION3_HOUSE_RE.match(line)
        if m:
            house_id = int(m.group(1), 16)
            root_id  = int(m.group(2), 16) | (int(m.group(3), 16) << 8)
            action3_positions.append((house_id, idx, root_id))

    # Deduplicate: first Action 3 per house ID wins
    seen: set[int] = set()
    deduped: list[tuple[int, int, int]] = []
    for house_id, idx, root_id in action3_positions:
        if house_id not in seen:
            seen.add(house_id)
            deduped.append((house_id, idx, root_id))

    results: list[tuple[int, int, int, int]] = []
    prev_end: int = 0   # exclusive end of the previous house's section

    for i, (house_id, action3_idx, root_id) in enumerate(deduped):
        start_idx = max(prev_end, action3_idx - WINDOW)
        # Push start past any previous house's Action 3 line
        for _, prev_a3_idx, _ in deduped[:i]:
            if prev_a3_idx >= start_idx:
                start_idx = max(start_idx, prev_a3_idx + 1)
        results.append((house_id, start_idx, action3_idx, root_id))
        prev_end = action3_idx + 1

    return results
