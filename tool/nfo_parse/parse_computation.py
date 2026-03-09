"""
parse_computation.py — Parse type-89 Action 2 advanced computation entries.

Type-89 format
--------------
    02 07 <set_id> 89
    <computation_chain>          ← variable-length; terminated by \\2 escape
    <num_ranges>                 (1 byte)
    [ <result_lo> <result_hi>  <lo_b0..b3>  <hi_b0..b3> ]  × num_ranges  (DWORD ranges)
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

# GRFCodec binary operator byte values for type-89 inter-step operations.
# These are the actual bytes in the binary stream after escape resolution
# (\2+ → 0x00, \2- → 0x01, \2* → 0x10, etc.).
_OP_ADD = 0x00
_OP_SUB = 0x01
_OP_MUL = 0x10
_OP_DIV = 0x02
_OP_MOD = 0x03

_OP_NAMES = {
    _OP_ADD: "add",
    _OP_SUB: "sub",
    _OP_MUL: "mul",
    _OP_DIV: "div",
    _OP_MOD: "mod",
}

# All known operator byte values (used for step-termination heuristic).
_ALL_OP_BYTES = frozenset(_OP_NAMES) | {
    0x04, 0x05, 0x06, 0x07, 0x08, 0x09,  # <, >, u<, u>, u/, u%
    0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,  # sto, ror, cmp, rst, psto, ucmp
    0x11, 0x12, 0x13, 0x14, 0x15, 0x16,   # &, |, ^, <<, u>>, >>
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _le32(b: list[int], offset: int) -> int:
    return (b[offset]
            | (b[offset + 1] << 8)
            | (b[offset + 2] << 16)
            | (b[offset + 3] << 24))


def _parse_steps(b: list[int], start: int, chain_end: int) -> list[ComputationStep]:
    """
    Parse the computation chain between ``b[start]`` and ``b[chain_end]``.

    Type-89 computation chain format::

        <var_access_1> <op_1> <var_access_2> <op_2> … <var_access_N>

    Each variable access: ``var(1) [param(1) for 0x60-0x7F] shift(1) and_mask(4)``
    Each operator: single byte (``0x00``=add, ``0x01``=sub, ``0x10``=mul, …).
    The operator follows the step it applies to (i.e. between step and next).
    The last step has no trailing operator.
    """
    steps: list[ComputationStep] = []
    p = start

    while p < chain_end:
        # --- Read variable access ---
        var = b[p]
        p += 1

        # 60+x variables carry an extra parameter byte
        param: int | None = None
        if 0x60 <= var <= 0x7F:
            if p >= chain_end:
                break
            param = b[p]
            p += 1

        # Fixed part: shift(1) + and_mask(4) = 5 bytes
        if p + 4 >= chain_end + 1:  # need exactly 5 bytes
            break

        shift    = b[p]
        and_mask = _le32(b, p + 1)
        p += 5

        # --- Read operator (or mark as last step) ---
        op = "var"  # sentinel: last step in chain
        if p < chain_end:
            op_byte = b[p]
            op = _OP_NAMES.get(op_byte, f"op_{op_byte:02x}")
            p += 1

        # For var 0x7E calls, store param as add_val (subroutine set-id)
        add_val = param if param is not None else 0

        steps.append(ComputationStep(
            operation = op,
            var       = var,
            shift     = shift,
            and_mask  = and_mask,
            add_val   = add_val,
        ))

    return steps


# ---------------------------------------------------------------------------
# Public parser
# ---------------------------------------------------------------------------


def parse_computation_node(rs: RawSprite) -> ComputationNode | None:
    """
    Try to parse a type-89 or type-8A computation node from *rs*.

    Type-8A uses the related-object scope but has the same byte layout
    as type-89.

    Returns ``None`` when *rs* is not type-89/8A or has too few bytes.
    """
    b = rs.bytes
    if len(b) < 5:
        return None
    if b[0] != 0x02 or b[1] != 0x07:
        return None
    if b[3] not in (0x89, 0x8A):
        return None

    set_id = b[2]

    # ---- Locate the chain/range boundary by scanning from the end ----
    # Layout: … <chain> <num_ranges(1)> [result(2)+range_lo(4)+range_hi(4)]*N <default(2)>
    # Solve: chain_end = len(b) - 2 - 10*N - 1, where b[chain_end] == N.
    chain_end: int | None = None
    for nr in range(256):
        pos = len(b) - 2 - 10 * nr - 1
        if pos < 4:
            break
        if b[pos] == nr:
            chain_end = pos
            break

    if chain_end is None:
        # Fallback: assume entire payload after header is chain (no ranges)
        chain_end = len(b)

    # ---- Parse computation chain ----
    steps = _parse_steps(b, 4, chain_end)

    # ---- Parse ranges & default ----
    p = chain_end
    if p >= len(b):
        return ComputationNode(node_id=set_id, steps=steps)

    num_ranges = b[p]
    p += 1

    # Type-89/8A uses DWORD ranges: result(W) range_lo(DW) range_hi(DW) = 10 bytes
    ranges: list[VariationalRange] = []
    for _ in range(num_ranges):
        if p + 9 >= len(b):
            break
        result_id = b[p] | (b[p + 1] << 8)
        range_lo  = _le32(b, p + 2)
        range_hi  = _le32(b, p + 6)
        ranges.append(VariationalRange(result_id, range_lo, range_hi))
        p += 10

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
