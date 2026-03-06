"""
parse_variational.py — Parse type-81/82 and type-85/86 Action 2 variational entries.

Type-81 format (byte-range variational)
----------------------------------------
    02 07 <set_id> 81
    <variable>           (1 byte)
    <shift>              (1 byte — shift-and-and, bit 5-7 encode operation)
    <mask>               (1 byte — 8-bit AND mask)
    <num_ranges>         (1 byte)
    [ <result_lo> <result_hi>  <range_lo> <range_hi> ]  × num_ranges
    <default_lo> <default_hi>

Type-85 format (word-range variational)
----------------------------------------
    02 07 <set_id> 85
    <variable>           (1 byte)
    <shift>              (1 byte)
    <mask_lo> <mask_hi>  (2 bytes — 16-bit AND mask)
    <num_ranges>         (1 byte)
    [ <result_lo> <result_hi>  <lo_lo> <lo_hi>  <hi_lo> <hi_hi> ]  × num_ranges
    <default_lo> <default_hi>

In both cases *result_id*, *range_lo*, *range_hi* values that have bit 15 set
are callback return values (not node IDs).  Use :func:`~nodes.is_callback_result`
to distinguish them.

The parsed ``shift`` byte encodes:
    bits 0-4  → actual right-shift amount
    bits 5-7  → operation type (00 = add, 01 = sub, 02 = unsigned div, …)
Only bits 0-4 are used for simple routing variables; the operation bits matter
for advanced var-61/var-7B uses which we don't need for houses.
"""

from __future__ import annotations

from .nodes import VariationalNode, VariationalRange
from .parse_raw import RawSprite


# ---------------------------------------------------------------------------
# Type-81 parser (byte-range)
# ---------------------------------------------------------------------------


def parse_variational_node_81(rs: RawSprite) -> VariationalNode | None:
    """
    Parse a type-81/82 variational node from *rs*.

    Type-82 is structurally identical to type-81 (byte-range) but accesses
    variables from the related object scope.

    Returns ``None`` when *rs* is not type-81/82 or has too few bytes.

    Minimum required: ``02 07 set_id 81 var shift mask 0``
    = 8 bytes + 2-byte default = 10 bytes.
    """
    b = rs.bytes
    if len(b) < 10:
        return None
    if b[0] != 0x02 or b[1] != 0x07:
        return None
    if b[3] not in (0x81, 0x82):
        return None

    set_id     = b[2]
    variable   = b[4]
    shift      = b[5]
    mask       = b[6]

    # Decompose shift byte: bits 0-4 = shift count, bit 5 = chain, bit 6 = sign
    has_chain = bool(shift & 0x20)
    if has_chain:
        import logging
        logging.warning(
            "set_id 0x%02X: type-81/82 variational with shift-and-add-divmod "
            "chain (shift byte 0x%02X) — chain not fully parsed, "
            "ranges may be inaccurate.",
            set_id, shift,
        )

    num_ranges = b[7]

    # Minimum byte check for ranges + default
    # Each range: 4 bytes (2 result + 2 range bounds).  Default: 2 bytes.
    needed = 8 + num_ranges * 4 + 2
    if len(b) < needed:
        # Try to parse what we have — emit a partial node
        num_ranges = max(0, (len(b) - 10) // 4)
        needed     = 8 + num_ranges * 4 + 2

    ranges: list[VariationalRange] = []
    p = 8
    for _ in range(num_ranges):
        if p + 3 >= len(b):
            break
        result_id = b[p] | (b[p + 1] << 8)
        range_lo  = b[p + 2]
        range_hi  = b[p + 3]
        ranges.append(VariationalRange(result_id, range_lo, range_hi))
        p += 4

    default_id = 0
    if p + 1 < len(b):
        default_id = b[p] | (b[p + 1] << 8)

    return VariationalNode(
        node_id  = set_id,
        var_type = b[3],
        variable = variable,
        shift    = shift,
        mask     = mask,
        ranges   = ranges,
        default  = default_id,
    )


# ---------------------------------------------------------------------------
# Type-85 parser (word-range)
# ---------------------------------------------------------------------------


def parse_variational_node_85(rs: RawSprite) -> VariationalNode | None:
    """
    Parse a type-85/86 variational node from *rs*.

    Returns ``None`` when *rs* is not type-85/86 or has too few bytes.
    Type-86 is structurally identical to type-85 (word-range variational)
    but accesses variables from the related object scope.

    Minimum required: ``02 07 set_id 85 var shift mask_lo mask_hi 0``
    = 9 bytes + 2-byte default = 11 bytes.
    """
    b = rs.bytes
    if len(b) < 11:
        return None
    if b[0] != 0x02 or b[1] != 0x07:
        return None
    if b[3] not in (0x85, 0x86):
        return None

    set_id     = b[2]
    var_type   = b[3]
    variable   = b[4]
    shift      = b[5]
    mask       = b[6] | (b[7] << 8)   # 16-bit mask

    # Decompose shift byte: bits 0-4 = shift count, bit 5 = chain, bit 6 = sign
    has_chain = bool(shift & 0x20)
    if has_chain:
        import logging
        logging.warning(
            "set_id 0x%02X: type-85/86 variational with shift-and-add-divmod "
            "chain (shift byte 0x%02X) — chain not fully parsed, "
            "ranges may be inaccurate.",
            set_id, shift,
        )

    num_ranges = b[8]

    # Each word range: 6 bytes (2 result + 4 range bounds).  Default: 2 bytes.
    needed = 9 + num_ranges * 6 + 2
    if len(b) < needed:
        num_ranges = max(0, (len(b) - 11) // 6)
        needed     = 9 + num_ranges * 6 + 2

    ranges: list[VariationalRange] = []
    p = 9
    for _ in range(num_ranges):
        if p + 5 >= len(b):
            break
        result_id = b[p] | (b[p + 1] << 8)
        range_lo  = b[p + 2] | (b[p + 3] << 8)
        range_hi  = b[p + 4] | (b[p + 5] << 8)
        ranges.append(VariationalRange(result_id, range_lo, range_hi))
        p += 6

    default_id = 0
    if p + 1 < len(b):
        default_id = b[p] | (b[p + 1] << 8)

    return VariationalNode(
        node_id  = set_id,
        var_type = var_type,
        variable = variable,
        shift    = shift,
        mask     = mask,
        ranges   = ranges,
        default  = default_id,
    )


# ---------------------------------------------------------------------------
# Combined dispatcher
# ---------------------------------------------------------------------------


def parse_variational_node(rs: RawSprite) -> VariationalNode | None:
    """
    Dispatch to the correct variational parser based on the type byte.

    Returns a :class:`~nodes.VariationalNode` or ``None``.
    """
    if len(rs.bytes) < 4:
        return None
    t = rs.bytes[3]
    if t in (0x81, 0x82):
        return parse_variational_node_81(rs)
    if t in (0x85, 0x86):
        return parse_variational_node_85(rs)
    return None


def parse_all_variational_nodes(sprites: list[RawSprite]) -> dict[int, VariationalNode]:
    """
    Parse every type-81 / type-82 / type-85 / type-86 entry from *sprites*.

    When multiple entries have the same set ID the **first** wins.
    """
    nodes: dict[int, VariationalNode] = {}
    for rs in sprites:
        node = parse_variational_node(rs)
        if node is not None and node.node_id not in nodes:
            nodes[node.node_id] = node
    return nodes
