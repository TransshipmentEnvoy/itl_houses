"""
parse_random.py — Parse type-80 and type-82 Action 2 random-selection entries.

Type-80 format (simple random)
--------------------------------
    02 07 <set_id> 80
    <triggers>               (1 byte)
    <rand_bit_start>         (1 byte)
    <num_entries_power2>     (1 byte — entry count = 2 ^ value)
    [ <entry_lo> <entry_hi> ]  × (2 ^ num_entries_power2)

Type-82 format (random with re-randomisation)
----------------------------------------------
    02 07 <set_id> 82
    <variable>               (1 byte — variable to check for trigger)
    <shift_and_triggers>     (1 byte — bits 0-4: shift; bits 5-7: trigger bits)
    <mask>                   (1 byte)
    <num_ranges>             (1 byte — as in type-81, not a power-of-2)
    [ <result_lo> <result_hi>  <lo> <hi> ]  × num_ranges
    <default_lo> <default_hi>

Note: for type-82 the entry list is structured like a variational node (with
explicit ranges) rather than the flat power-of-2 list used in type-80.  The
key semantic difference is that type-82 *triggers re-randomisation* when a
condition is met, rather than just selecting among a fixed list.

For sprite-generation purposes the simplest correct behaviour is:
  type-80 → pick any one entry (preferably first or most-common)
  type-82 → follow the default (= the group used when no trigger fires)
"""

from __future__ import annotations

from .nodes import RandomNode, VariationalNode, VariationalRange
from .parse_raw import RawSprite


# ---------------------------------------------------------------------------
# Type-80 parser
# ---------------------------------------------------------------------------


def parse_random_node_80(rs: RawSprite) -> RandomNode | None:
    """
    Parse a type-80 random node from *rs*.

    Returns ``None`` when *rs* is not type-80 or has too few bytes.
    """
    b = rs.bytes
    if len(b) < 7:
        return None
    if b[0] != 0x02 or b[1] != 0x07:
        return None
    if b[3] != 0x80:
        return None

    set_id         = b[2]
    triggers       = b[4]
    rand_bit_start = b[5]
    power          = b[6]

    count = 1 << power   # number of entries

    entries: list[int] = []
    p = 7
    for _ in range(count):
        if p + 1 >= len(b):
            break
        entry_id = b[p] | (b[p + 1] << 8)
        entries.append(entry_id)
        p += 2

    return RandomNode(
        node_id        = set_id,
        rand_type      = 0x80,
        triggers       = triggers,
        rand_bit_start = rand_bit_start,
        count          = count,
        entries        = entries,
    )


# ---------------------------------------------------------------------------
# Type-82 parser
# ---------------------------------------------------------------------------


def parse_random_node_82(rs: RawSprite) -> VariationalNode | None:
    """
    Parse a type-82 "random with re-randomise" node.

    Because type-82 has a variational-style range list (not a power-of-2
    flat list), we reuse :class:`~nodes.VariationalNode` to represent it.
    The ``rand_type`` field of the parent :class:`~nodes.RandomNode` is ``0x82``
    but the *structure* is identical to type-81, so we return a VariationalNode
    with ``var_type=0x82``.
    """
    b = rs.bytes
    if len(b) < 10:
        return None
    if b[0] != 0x02 or b[1] != 0x07:
        return None
    if b[3] != 0x82:
        return None

    set_id     = b[2]
    variable   = b[4]
    shift      = b[5]
    mask       = b[6]
    num_ranges = b[7]

    needed = 8 + num_ranges * 4 + 2
    if len(b) < needed:
        num_ranges = max(0, (len(b) - 10) // 4)

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
        var_type = 0x82,   # re-random marker
        variable = variable,
        shift    = shift,
        mask     = mask,
        ranges   = ranges,
        default  = default_id,
    )


# ---------------------------------------------------------------------------
# Combined dispatcher
# ---------------------------------------------------------------------------


def parse_random_node(rs: RawSprite) -> RandomNode | VariationalNode | None:
    """
    Dispatch to the correct random parser based on the type byte.

    - Type 0x80 → :class:`~nodes.RandomNode`
    - Type 0x82 → :class:`~nodes.VariationalNode` (re-random, range-based)
    """
    if len(rs.bytes) < 4:
        return None
    t = rs.bytes[3]
    if t == 0x80:
        return parse_random_node_80(rs)
    if t == 0x82:
        return parse_random_node_82(rs)
    return None


def parse_all_random_nodes(
    sprites: list[RawSprite],
) -> tuple[dict[int, RandomNode], dict[int, VariationalNode]]:
    """
    Parse every type-80 and type-82 entry from *sprites*.

    Returns a pair ``(random_nodes, rerand_nodes)`` where:
        random_nodes : type-80 → :class:`~nodes.RandomNode`
        rerand_nodes : type-82 → :class:`~nodes.VariationalNode` (var_type=0x82)

    In both dicts the key is the set ID.  First-wins on duplicates.
    """
    random_nodes:  dict[int, RandomNode]      = {}
    rerand_nodes:  dict[int, VariationalNode] = {}

    for rs in sprites:
        node = parse_random_node(rs)
        if node is None:
            continue
        if isinstance(node, RandomNode) and node.node_id not in random_nodes:
            random_nodes[node.node_id] = node
        elif isinstance(node, VariationalNode) and node.node_id not in rerand_nodes:
            rerand_nodes[node.node_id] = node

    return random_nodes, rerand_nodes
