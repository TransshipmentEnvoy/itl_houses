"""
Data classes representing every Action 2 node type for TTRS/OpenTTD houses.

Node hierarchy
==============

Action2Node (abstract base) — identified by node_id (== set_id in NFO byte)
  ├─ LayoutNode       type-00  basic sprite layout
  ├─ VariationalNode  type-81  byte-mask variational (var-based routing)
  │                   type-85  word-mask variational
  ├─ RandomNode       type-80  random selection
  │                   type-82  random with re-randomise trigger
  └─ ComputationNode  type-89  multi-step arithmetic computation

Special result IDs
==================
Results whose high byte is 0x80 are *callback return values*, not node IDs.
    0x8000 → return false / no graphics
    0x8001 → return true / accept
    0x00FF → return 0xFF (no colour change)
    etc.

Use `is_callback_result(result_id)` to distinguish these from real node IDs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Union


# ============================================================================
# Enums
# ============================================================================


class NodeType(Enum):
    LAYOUT = auto()        # type-00 sprite layout
    VARIATIONAL = auto()   # type-81 / type-85
    RANDOM = auto()        # type-80 / type-82
    COMPUTATION = auto()   # type-89


# Variable constants (Action 2 variational variable bytes)
VAR_CONSTRUCTION_STATE = 0x40   # build progress 0-3
VAR_BUILDING_AGE       = 0x41   # age in years
VAR_TOWN_ZONE          = 0x42   # town zone 0-4
VAR_TERRAIN_TYPE       = 0x43   # terrain / snow check
VAR_ANIMATION_FRAME    = 0x44   # animation frame info (also used as counter)
VAR_ANIMATION_COUNTER  = 0x46   # animation frame by counter (sprite select)
VAR_CALLBACK_ID        = 0x0C   # which callback is being invoked
VAR_CLIMATE            = 0x03   # game climate (0=temp, 1=arctic, 2=tropic, 3=toy)

# Terrain type values (for VAR_TERRAIN_TYPE)
TERRAIN_SNOW    = 0x04
TERRAIN_DESERT  = 0x02
TERRAIN_ARCTIC  = 0x01  # arctic ground (not snow per se — used in 3-way checks)

# Climate values (for VAR_CLIMATE)
CLIMATE_TEMPERATE = 0
CLIMATE_ARCTIC    = 1
CLIMATE_TROPIC    = 2
CLIMATE_TOYLAND   = 3

# Sentinel for callback return values
CALLBACK_RESULT_MASK = 0x8000   # result_id with this bit = callback return


def is_callback_result(result_id: int) -> bool:
    """True when result_id is a callback return value, not a node reference."""
    return bool(result_id & CALLBACK_RESULT_MASK)


# ============================================================================
# Sprite helpers
# ============================================================================


@dataclass(frozen=True)
class LayoutSprite:
    """Single sprite DWORD from a type-00 layout entry."""
    dword: int

    # bit 31 = sprite from Action 1 set (not base-game sprite)
    SPRITE_ACTION1_FLAG  = 0x80000000
    # bit 15 = enable recolour remap
    SPRITE_RECOLOUR_FLAG = 0x00008000
    # bits 0-13 = sprite index
    SPRITE_INDEX_MASK    = 0x00003FFF

    @property
    def is_action1(self) -> bool:
        return bool(self.dword & self.SPRITE_ACTION1_FLAG)

    @property
    def has_recolour(self) -> bool:
        return bool(self.dword & self.SPRITE_RECOLOUR_FLAG)

    @property
    def index(self) -> int:
        return self.dword & self.SPRITE_INDEX_MASK

    @property
    def is_zero(self) -> bool:
        """True for a blank/transparent placeholder (all zero)."""
        return self.dword == 0


@dataclass(frozen=True)
class BoundingBox:
    """Bounding box from a type-00 layout entry."""
    xoff: int
    yoff: int
    xext: int
    yext: int
    zext: int


# ============================================================================
# Action 2 node types
# ============================================================================


@dataclass
class LayoutNode:
    """Type-00: Basic sprite layout — terminal node, references actual sprites."""
    node_id: int
    node_type: NodeType = field(default=NodeType.LAYOUT, init=False, repr=False)

    ground:   LayoutSprite
    building: LayoutSprite
    bbox:     Optional[BoundingBox] = None


@dataclass
class VariationalRange:
    """One range entry inside a variational (type-81/85) node."""
    result_id: int   # target group ID (or callback-result sentinel)
    range_lo:  int   # inclusive lower bound
    range_hi:  int   # inclusive upper bound


@dataclass
class VariationalNode:
    """Type-81 / type-85: conditional routing based on a variable."""
    node_id:  int
    node_type: NodeType = field(default=NodeType.VARIATIONAL, init=False, repr=False)

    var_type: int               # 0x81 (byte ranges) or 0x85 (word ranges)
    variable: int               # e.g. VAR_TERRAIN_TYPE, VAR_ANIMATION_COUNTER …
    shift:    int               # shift amount (bits 0-4) + and/or flags (bits 5-7)
    mask:     int               # AND mask applied after shift

    ranges:  list[VariationalRange] = field(default_factory=list)
    default: int = 0            # default result when no range matches


@dataclass
class RandomNode:
    """Type-80 / type-82: random selection among multiple groups."""
    node_id:  int
    node_type: NodeType = field(default=NodeType.RANDOM, init=False, repr=False)

    rand_type:      int          # 0x80 (simple) or 0x82 (with re-randomisation)
    triggers:       int          # trigger-bits byte
    rand_bit_start: int          # which random bit to start reading from
    count:          int          # number of entries (already resolved, not the power)

    entries: list[int] = field(default_factory=list)   # list of result group IDs


@dataclass
class ComputationStep:
    """One arithmetic step in a type-89 computation chain."""
    operation: str   # "call", "add", "sub", "mul", "div", "mod", "var"
    var:       int   # variable byte (0x7E = call subroutine, 0x1A = constant …)
    shift:     int   # shift/flags byte
    and_mask:  int   # AND mask (4 bytes, LE)
    add_val:   int   # additional value (4 bytes, LE) — also used as const


@dataclass
class ComputationNode:
    """Type-89: multi-step arithmetic computation (advanced variational)."""
    node_id:  int
    node_type: NodeType = field(default=NodeType.COMPUTATION, init=False, repr=False)

    steps:   list[ComputationStep]         = field(default_factory=list)
    ranges:  list[VariationalRange]        = field(default_factory=list)
    default: int = 0


# Union type for any Action 2 node
Action2Node = Union[LayoutNode, VariationalNode, RandomNode, ComputationNode]

# Graph: node_id → node
Action2Graph = dict[int, Action2Node]


# ============================================================================
# Traversal result types
# ============================================================================


@dataclass
class FrameLayout:
    """
    A resolved, renderable sprite layout: one animation frame, one climate,
    one construction stage — whatever the context is.
    """
    ground:   LayoutSprite
    building: LayoutSprite
    bbox:     Optional[BoundingBox] = None


@dataclass
class ClimateGraphics:
    """All graphics for a single climate variant of one house tile."""
    # Each entry is None when the stage is absent (e.g. no scaffolding).
    construction_stages: list[Optional[FrameLayout]] = field(default_factory=list)
    completed:           Optional[FrameLayout]        = None
    # When the completed stage is animated, frames replaces completed.
    animation_frames:    list[FrameLayout]            = field(default_factory=list)


@dataclass
class HouseTileGraphics:
    """
    Fully resolved graphics for one physical tile (even in a multi-tile house).

    climate variants — any may be None if the house doesn't have that variant:
      temperate : normal temperate / all-weather
      snow      : above snowline / arctic
      tropic    : sub-tropical desert / tropic
      arctic_v2 : second arctic variant (Pattern B ID 0x33) — rare
    random_variants : list of temperate-climate FrameLayouts for random_switch
    """
    temperate:      Optional[ClimateGraphics]      = None
    snow:           Optional[ClimateGraphics]      = None
    tropic:         Optional[ClimateGraphics]      = None
    arctic_v2:      Optional[ClimateGraphics]      = None
    random_variants: list[ClimateGraphics]         = field(default_factory=list)
