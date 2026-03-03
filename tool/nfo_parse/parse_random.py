"""
parse_random.py — Parse type-80/83 Action 2 random-selection entries.

Type-80 format (self-scope random)
------------------------------------
    02 07 <set_id> 80
    <triggers>               (1 byte)
    <rand_bit_start>         (1 byte)
    <nrand>                  (1 byte — number of entries, must be power of 2)
    [ <entry_lo> <entry_hi> ]  × nrand

Type-83 (related-object random) has the same byte layout as type-80.

Note: type-82 (related-object byte-range variational) is structurally
identical to type-81 and is handled by parse_variational.py.
"""

from __future__ import annotations

from .nodes import RandomNode
from .parse_raw import RawSprite


# ---------------------------------------------------------------------------
# Type-80 parser
# ---------------------------------------------------------------------------


def parse_random_node_80(rs: RawSprite) -> RandomNode | None:
    """
    Parse a type-80 or type-83 random node from *rs*.

    Both types share the same byte layout; type-83 uses the related
    object scope instead of the self scope.

    Returns ``None`` when *rs* is not type-80/83 or has too few bytes.
    """
    b = rs.bytes
    if len(b) < 7:
        return None
    if b[0] != 0x02 or b[1] != 0x07:
        return None
    if b[3] not in (0x80, 0x83):
        return None

    set_id         = b[2]
    rand_type      = b[3]
    triggers       = b[4]
    rand_bit_start = b[5]
    nrand          = b[6]

    # nrand IS the count directly (must be a power of 2 per spec)
    count = nrand
    assert count > 0 and (count & (count - 1)) == 0, \
        f"nrand must be a power of 2, got {count}"

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
        rand_type      = rand_type,
        triggers       = triggers,
        rand_bit_start = rand_bit_start,
        count          = count,
        entries        = entries,
    )


# ---------------------------------------------------------------------------
# Combined dispatcher
# ---------------------------------------------------------------------------


def parse_random_node(rs: RawSprite) -> RandomNode | None:
    """
    Parse a type-80 or type-83 random node from *rs*.

    Returns a :class:`~nodes.RandomNode` or ``None``.
    """
    if len(rs.bytes) < 4:
        return None
    if rs.bytes[3] in (0x80, 0x83):
        return parse_random_node_80(rs)
    return None


def parse_all_random_nodes(
    sprites: list[RawSprite],
) -> dict[int, RandomNode]:
    """
    Parse every type-80 random entry from *sprites*.

    Returns ``{set_id: RandomNode}``.  First-wins on duplicates.
    """
    random_nodes: dict[int, RandomNode] = {}

    for rs in sprites:
        node = parse_random_node(rs)
        if node is not None and node.node_id not in random_nodes:
            random_nodes[node.node_id] = node

    return random_nodes
