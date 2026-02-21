"""
parse_computation.py — Parse type-89 Action 2 advanced computation entries.

Type-89 format
--------------
    02 07 <set_id> 89
    <computation_chain>          ← variable-length; terminated by \\2 escape
    <num_ranges>                 (1 byte)
    [ <result_lo> <result_hi>  <lo_lo> <lo_hi>  <hi_lo> <hi_hi> ]  × num_ranges
    <default_lo> <default_hi>

The computation chain consists of steps separated by ``\\2`` *operation* bytes
(in the raw NFO byte stream these appear literally as ``02`` bytes mid-payload).
Each step has the form:

    <var>  <shift>  <and_mask_b0 b1 b2 b3>  <add_val_b0 b1 b2 b3>

Followed by an operation byte:
    ``\\2+``  (``0x02``, ``+`` char — add to accumulator)
    ``\\2-``  (subtract)
    ``\\2*``  (multiply)
    ``\\2/``  (unsigned divide)
    ``\\2%``  (modulo)
    or another variable byte starting the next step.

In practice the operation byte is *between* steps (immediately after the
per-step data), and the chain ends with ``<num_ranges>`` (distinguishable
because it is followed by range data).

Because the binary encoding is ambiguous (an ``02`` byte could be part of an
AND mask), we rely on the ``\\2<op>`` notation used in the NFO text source.
The binary byte stream stores the operation byte literally.  We interpret:

    *first byte of each step* = variable selector:
        ``0x1A``   → load immediate constant (and_mask = constant, add = 0)
        ``0x7E``   → call subroutine (add_val = sub procedure node ID)
        other      → load named variable

The step _terminates_ when the 10-byte step block is consumed.  The parser
peeks at the 11th byte: if it is one of ``02 + - * / %`` treat it as an
inter-step operator byte; otherwise assume the chain is done.

This is a best-effort parser: the TTRS NFO is the only corpus being targeted.

Notes
-----
- Only ~7 houses in TTRS use type-89 (sections §10 and §15 of nfo_patterns.md).
- For NML generation we don't need to fully evaluate type-89; we just need to
  know which subroutines it calls so we can record the dependency and fall
  back gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .nodes import ComputationNode, ComputationStep, VariationalRange
from .parse_raw import RawSprite


# ---------------------------------------------------------------------------
# Inter-step operation byte values (come AFTER each step's 10 bytes)
# ---------------------------------------------------------------------------

_OP_ADD = ord("+")   # 0x2B
_OP_SUB = ord("-")   # 0x2D
_OP_MUL = ord("*")   # 0x2A
_OP_DIV = ord("/")   # 0x2F
_OP_MOD = ord("%")   # 0x25

_OP_NAMES = {
    _OP_ADD: "add",
    _OP_SUB: "sub",
    _OP_MUL: "mul",
    _OP_DIV: "div",
    _OP_MOD: "mod",
}

# In raw binary the operation byte shows up as the ASCII character.
# Type 89 itself is signalled by byte 0x89 at position [3] in the sprite.


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _le32(b: list[int], offset: int) -> int:
    return (b[offset]
            | (b[offset + 1] << 8)
            | (b[offset + 2] << 16)
            | (b[offset + 3] << 24))


def _parse_steps(b: list[int], start: int) -> tuple[list[ComputationStep], int]:
    """
    Parse the computation chain starting at *start*.

    Returns ``(steps, cursor)`` where *cursor* is the byte position just
    after the last consumed byte of the chain.
    """
    steps: list[ComputationStep] = []
    p = start

    while True:
        # Each step: var(1) shift(1) and_mask(4) add_val(4) = 10 bytes
        if p + 9 >= len(b):
            break

        var      = b[p]
        shift    = b[p + 1]
        and_mask = _le32(b, p + 2)
        add_val  = _le32(b, p + 6)
        p += 10

        # Determine operation (comes after the 10-byte step block)
        op = "var"
        if p < len(b) and b[p] in _OP_NAMES:
            op = _OP_NAMES[b[p]]
            p += 1   # consume the operation byte
        elif p < len(b) and b[p] == 0x7E:
            # Next step starts with 0x7E (subroutine call) — no explicit op
            op = "call"
        elif p < len(b) and b[p] == 0x1A:
            op = "add"   # implicit add when loading a constant next

        steps.append(ComputationStep(
            operation = op,
            var       = var,
            shift     = shift,
            and_mask  = and_mask,
            add_val   = add_val,
        ))

        # Stop if the next byte looks like num_ranges (i.e. the chain has ended).
        # We detect this heuristically: if the byte is small (0-32 ranges is
        # plausible) AND it is followed by a coherent range payload, stop.
        # A simpler heuristic: stop when the next byte is NOT a known var or op.
        if p >= len(b):
            break
        next_byte = b[p]
        if next_byte not in _OP_NAMES and next_byte not in (0x7E, 0x1A):
            # Looks like num_ranges — end of chain
            break

    return steps, p


# ---------------------------------------------------------------------------
# Public parser
# ---------------------------------------------------------------------------


def parse_computation_node(rs: RawSprite) -> ComputationNode | None:
    """
    Try to parse a type-89 computation node from *rs*.

    Returns ``None`` when *rs* is not type-89 or has too few bytes.
    """
    b = rs.bytes
    if len(b) < 5:
        return None
    if b[0] != 0x02 or b[1] != 0x07:
        return None
    if b[3] != 0x89:
        return None

    set_id = b[2]

    # Chain starts at byte 4
    steps, p = _parse_steps(b, 4)

    # Now parse num_ranges + range table
    if p >= len(b):
        return ComputationNode(node_id=set_id, steps=steps)

    num_ranges = b[p]
    p += 1

    ranges: list[VariationalRange] = []
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

    return ComputationNode(
        node_id  = set_id,
        steps    = steps,
        ranges   = ranges,
        default  = default_id,
    )


def parse_all_computation_nodes(sprites: list[RawSprite]) -> dict[int, ComputationNode]:
    """
    Parse every type-89 entry from *sprites*.  First-wins on duplicates.
    """
    nodes: dict[int, ComputationNode] = {}
    for rs in sprites:
        node = parse_computation_node(rs)
        if node is not None and node.node_id not in nodes:
            nodes[node.node_id] = node
    return nodes
