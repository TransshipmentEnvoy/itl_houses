"""
parse_layout.py — Parse type-00 Action 2 (basic sprite layout) entries.

Type-00 format
--------------
    02 07 <set_id> 00
    <ground_b0 b1 b2 b3>          ← 32-bit LE DWORD
    <bldg_b0   b1 b2 b3>          ← 32-bit LE DWORD
    <xoff> <yoff>                  ← signed offsets
    <xext> <yext> <zext>           ← bounding-box extents

Total: 14+ bytes (the ``02 07``  prefix + 3 type+content bytes + 13 payload bytes = 17 bytes typical).

DWORD encoding
~~~~~~~~~~~~~~
    bit 31 (0x80000000)  → sprite from Action 1 set (use Action1 index)
    bit 15 (0x00008000)  → enable recolour remap
    bits 0–13            → sprite index / base-game sprite number
"""

from __future__ import annotations

from .nodes import BoundingBox, LayoutNode, LayoutSprite
from .parse_raw import RawSprite


# ---------------------------------------------------------------------------
# Type-00 parser
# ---------------------------------------------------------------------------


def parse_layout_node(rs: RawSprite) -> LayoutNode | None:
    """
    Try to parse a :class:`~nodes.LayoutNode` from *rs*.

    Returns ``None`` when *rs* is not a type-00 entry or has too few bytes.

    Expected byte layout of ``rs.bytes`` (basic format)::

        [0]  02   — Action 2
        [1]  07   — feature: houses
        [2]  <set_id>
        [3]  00   — type: basic sprite layout
        [4]  ground_b0
        [5]  ground_b1
        [6]  ground_b2
        [7]  ground_b3
        [8]  bldg_b0
        [9]  bldg_b1
        [10] bldg_b2
        [11] bldg_b3
        [12] xoff
        [13] yoff
        [14] xext
        [15] yext
        [16] zext

    .. note::

       NFO also defines **extended** sprite layouts (num-sprites > 0,
       multiple building sprites) and **advanced** layouts (with register
       modifier flags after each sprite entry).  These are detected and
       logged as warnings but not fully parsed — the basic 1-ground +
       1-building format is the only layout emitted.
    """
    b = rs.bytes
    if len(b) < 4:
        return None
    if b[0] != 0x02 or b[1] != 0x07:
        return None

    type_byte = b[3]

    # --- Detect extended / advanced layout formats -------------------------
    # type_byte < 0x40: sprite layout; 0x00 = basic (1 building sprite).
    # type_byte >= 0x40 and < 0x80: advanced sprite layout (register flags).
    # type_byte >= 0x80: not a layout (random / variational / computation).
    if type_byte >= 0x80:
        return None  # not a sprite layout at all
    if type_byte >= 0x40:
        import logging
        logging.warning(
            "set_id 0x%02X: advanced sprite layout (type byte 0x%02X) "
            "— not supported, skipping.",
            b[2], type_byte,
        )
        return None
    if type_byte != 0x00:
        import logging
        logging.warning(
            "set_id 0x%02X: extended sprite layout with %d building "
            "sprites — only basic format (1 building) is supported, "
            "skipping.",
            b[2], type_byte,
        )
        return None

    set_id = b[2]

    if len(b) < 12:
        return None   # need at least two DWORDs

    def le32(offset: int) -> int:
        return (b[offset]
                | (b[offset + 1] << 8)
                | (b[offset + 2] << 16)
                | (b[offset + 3] << 24))

    ground_dword = le32(4)
    bldg_dword   = le32(8)

    bbox: BoundingBox | None = None
    if len(b) >= 17:
        xoff = _signed_byte(b[12])
        yoff = _signed_byte(b[13])
        xext = b[14]
        yext = b[15]
        zext = b[16]
        bbox = BoundingBox(xoff, yoff, xext, yext, zext)

    return LayoutNode(
        node_id  = set_id,
        ground   = LayoutSprite(ground_dword),
        building = LayoutSprite(bldg_dword),
        bbox     = bbox,
    )


def parse_all_layout_nodes(sprites: list[RawSprite]) -> dict[int, LayoutNode]:
    """
    Parse every type-00 entry from *sprites*.

    When multiple entries share the same set ID the **first** one wins
    (i.e. the temperate / no-snow definition takes priority over later
    climate overrides that reuse the same low set IDs).
    """
    nodes: dict[int, LayoutNode] = {}
    for rs in sprites:
        node = parse_layout_node(rs)
        if node is not None and node.node_id not in nodes:
            nodes[node.node_id] = node
    return nodes


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _signed_byte(value: int) -> int:
    """Interpret an unsigned 0-255 value as a signed byte (-128 … 127)."""
    return value if value < 128 else value - 256
