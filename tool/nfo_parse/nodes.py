"""
Data classes representing every Action 2 node type for TTRS/OpenTTD houses.

Node hierarchy
==============

Action2Node (abstract base) — identified by node_id (== set_id in NFO byte)
  ├─ LayoutNode       type-00  basic sprite layout
  ├─ VariationalNode  type-81  byte-mask variational (var-based routing)
  │                   type-82  byte-mask variational (related object)
  │                   type-85  word-mask variational
  │                   type-86  word-mask variational (related object)
  ├─ RandomNode       type-80  random selection
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
VAR_BUILDING_COUNTS    = 0x44   # building counts in town/map (DWORD LLllCCcc)
VAR_ANIMATION_FRAME    = 0x46   # current animation frame being displayed
VAR_CALLBACK_ID        = 0x0C   # which callback is being invoked
VAR_CLIMATE            = 0x03   # game climate (0=temp, 1=arctic, 2=tropic, 3=toy)

# Terrain type values (for VAR_TERRAIN_TYPE)
# Per NFO spec — variable 43 returns: 0=normal, 1=desert, 2=rainforest, 4=snow
TERRAIN_NORMAL     = 0x00
TERRAIN_DESERT     = 0x01
TERRAIN_RAINFOREST = 0x02
TERRAIN_SNOW       = 0x04

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
    """Single sprite DWORD from a type-00 layout entry.

    DWORD bit layout (per NFO Action2/Sprite_Layout spec):
        bit 31         : custom sprite from Action 1 set
        bits 16-29     : recolour sprite number (when sprite_type != 0)
        bits 14-15     : sprite type (0=normal, 1=use recolour sprite)
        bits 0-13      : sprite index / base-game sprite number
    """
    dword: int

    # bit 31 = sprite from Action 1 set (not base-game sprite)
    SPRITE_ACTION1_FLAG  = 0x80000000
    # bits 14-15 = sprite type (2-bit field)
    SPRITE_TYPE_MASK     = 0x0000C000
    SPRITE_TYPE_SHIFT    = 14
    # bit 15 = enable recolour remap (legacy single-bit check)
    SPRITE_RECOLOUR_FLAG = 0x00008000
    # bits 16-29 = recolour sprite number
    RECOLOUR_SPRITE_MASK  = 0x3FFF0000
    RECOLOUR_SPRITE_SHIFT = 16
    # bits 0-13 = sprite index
    SPRITE_INDEX_MASK    = 0x00003FFF

    @property
    def is_action1(self) -> bool:
        return bool(self.dword & self.SPRITE_ACTION1_FLAG)

    @property
    def has_recolour(self) -> bool:
        return bool(self.dword & self.SPRITE_RECOLOUR_FLAG)

    @property
    def sprite_type(self) -> int:
        """2-bit sprite type field (bits 14-15): 0=normal, 1=recolour sprite."""
        return (self.dword & self.SPRITE_TYPE_MASK) >> self.SPRITE_TYPE_SHIFT

    @property
    def recolour_sprite(self) -> int | None:
        """Recolour sprite number (bits 16-29), or None if sprite_type==0."""
        if self.sprite_type == 0:
            return None
        return (self.dword & self.RECOLOUR_SPRITE_MASK) >> self.RECOLOUR_SPRITE_SHIFT

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
    shift:    int               # raw shift byte (kept for backward compat)
    mask:     int               # AND mask applied after shift
    param:    Optional[int] = None  # extra parameter byte for vars 0x60-0x7F

    ranges:  list[VariationalRange] = field(default_factory=list)
    default: int = 0            # default result when no range matches

    @property
    def shift_count(self) -> int:
        """Right-shift amount (bits 0-4 of the shift byte)."""
        return self.shift & 0x1F

    @property
    def has_chain(self) -> bool:
        """Whether shift-and-add-divmod chain follows (bit 5)."""
        return bool(self.shift & 0x20)

    @property
    def sign_extend(self) -> bool:
        """Whether the variable should be sign-extended (bit 6)."""
        return bool(self.shift & 0x40)


@dataclass
class RandomNode:
    """Type-80 / type-83: random selection among multiple groups."""
    node_id:  int
    node_type: NodeType = field(default=NodeType.RANDOM, init=False, repr=False)

    rand_type:      int          # 0x80 (self) or 0x83 (related-object)
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
    # Entries may be None for sparse frame indices (padding).
    animation_frames:    list[Optional[FrameLayout]]  = field(default_factory=list)


@dataclass
class RandomVariantGraphics:
    """All graphics for one random variant, with per-climate sub-variants.

    Each random selection entry (one "branch" of a type-80 random node)
    may have independent sprites for temperate, snow, and tropic climates.
    """
    temperate: Optional[ClimateGraphics] = None
    snow:      Optional[ClimateGraphics] = None
    tropic:    Optional[ClimateGraphics] = None
    arctic_v2: Optional[ClimateGraphics] = None


@dataclass
class HouseTileGraphics:
    """
    Fully resolved graphics for one physical tile (even in a multi-tile house).

    climate variants — any may be None if the house doesn't have that variant:
      temperate : normal temperate / all-weather
      snow      : above snowline / arctic
      tropic    : sub-tropical desert / tropic
      arctic_v2 : second arctic variant (Pattern B ID 0x33) — rare
    random_variants : per-variant graphics with per-climate sub-variants
    """
    temperate:      Optional[ClimateGraphics]           = None
    snow:           Optional[ClimateGraphics]           = None
    tropic:         Optional[ClimateGraphics]           = None
    arctic_v2:      Optional[ClimateGraphics]           = None
    random_variants: list[RandomVariantGraphics]        = field(default_factory=list)

    # Colour callback values extracted from CB 0x1E random results.
    # Each int is a 15-bit colour callback result (0–0x7FFF).
    # Duplicates are preserved to encode probability weights.
    colour_values:   list[int]                          = field(default_factory=list)

    # Callback handlers extracted from the var 0x0C callback router.
    # Maps NFO callback ID (e.g. 0x17, 0x1B, 0x2E, 0x143) → target node ID.
    callback_handlers: dict[int, int]                   = field(default_factory=dict)

    # Random variant weights derived from entry repetition counts.
    # Parallel to random_variants — random_weights[i] is the weight for variant i.
    # Empty when there are no random variants or weights were not resolved.
    random_weights:  list[int]                          = field(default_factory=list)

    # NFO triggers byte from the outermost RandomNode (0x00 = no trigger).
    random_triggers: int                                = 0

    # NFO triggers byte specifically for the colour random node.
    colour_triggers: int                                = 0