"""
tool.nfo_parse — NFO Action 2 parser and NML emitter for OpenTTD house GRFs.

This package parses every Action 2 node type defined in TTRS (and compatible
house GRFs) into a directed graph, traverses the graph to resolve per-tile
sprite information for every climate / construction stage / animation frame,
and emits NML code that is a drop-in replacement for the heuristic pipeline
in ``tool/generate_ttrs_from_nfo.py``.

Module layout
=============
    nodes.py              Data classes for all node types and traversal results
    parse_raw.py          Multi-line NFO byte collector
    parse_layout.py       Type-00 sprite layout parser
    parse_variational.py  Type-81 / type-82 / type-85 / type-86 variational parser
    parse_random.py       Type-80 random-selection parser
    parse_computation.py  Type-89 advanced computation parser
    graph.py              Graph builder + recursive traversal → HouseTileGraphics
    nml_emit.py           NML code emitter (spritesets, spritelayouts, switches)

Quick-start
===========
    from tool.nfo_parse import build_all_house_graphics, emit_house_tile_nml

    house_graphics, graph = build_all_house_graphics(nfo_lines)
    lines, entry = emit_house_tile_nml(
        house_id_hex  = "1A",
        htg           = house_graphics[0x1A],
        sprite_table  = action1_sprite_table,   # dict[int, SpriteCoord]
        pcx_path      = "src/sprites/pcx/ttrs3w.pcx",
    )
"""

from .nodes import (
    # Node types
    LayoutNode,
    VariationalNode,
    RandomNode,
    ComputationNode,
    # Node type enum
    NodeType,
    # Variable constants
    VAR_BUILDING_COUNTS,
    VAR_ANIMATION_FRAME,
    VAR_BUILDING_AGE,
    VAR_CALLBACK_ID,
    VAR_CLIMATE,
    VAR_CONSTRUCTION_STATE,
    VAR_TERRAIN_TYPE,
    VAR_TOWN_ZONE,
    # Traversal result types
    ClimateGraphics,
    FrameLayout,
    HouseTileGraphics,
    RandomVariantGraphics,
    # Helpers
    is_callback_result,
)

from .parse_raw import (
    RawSprite,
    collect_action2_house_entries,
    collect_action2_in_range,
    collect_action3_entries,
    collect_house_section_ranges,
)

from .parse_layout import (
    parse_layout_node,
    parse_all_layout_nodes,
)

from .parse_variational import (
    parse_variational_node,
    parse_all_variational_nodes,
)

from .parse_random import (
    parse_random_node,
    parse_all_random_nodes,
)

from .parse_computation import (
    parse_computation_node,
    parse_all_computation_nodes,
)

from .graph import (
    build_graph,
    build_graph_from_lines,
    build_house_tile_graphics,
    build_all_house_graphics,
    traverse_callback_subgraph,
)

from .nml_emit import (
    emit_house_tile_nml,
)

from .fixups import (
    fixup_food_only_add_pass_mail,
)

__all__ = [
    # Data classes
    "LayoutNode",
    "VariationalNode",
    "RandomNode",
    "ComputationNode",
    "NodeType",
    # Variable constants
    "VAR_BUILDING_COUNTS",
    "VAR_ANIMATION_FRAME",
    "VAR_BUILDING_AGE",
    "VAR_CALLBACK_ID",
    "VAR_CLIMATE",
    "VAR_CONSTRUCTION_STATE",
    "VAR_TERRAIN_TYPE",
    "VAR_TOWN_ZONE",
    # Traversal result types
    "ClimateGraphics",
    "FrameLayout",
    "HouseTileGraphics",
    "RandomVariantGraphics",
    "is_callback_result",
    # Raw parsing
    "RawSprite",
    "collect_action2_house_entries",
    "collect_action2_in_range",
    "collect_action3_entries",
    "collect_house_section_ranges",
    # Individual parsers
    "parse_layout_node",
    "parse_all_layout_nodes",
    "parse_variational_node",
    "parse_all_variational_nodes",
    "parse_random_node",
    "parse_all_random_nodes",
    "parse_computation_node",
    "parse_all_computation_nodes",
    # Graph
    "build_graph",
    "build_graph_from_lines",
    "build_house_tile_graphics",
    "build_all_house_graphics",
    "traverse_callback_subgraph",
    # NML emission
    "emit_house_tile_nml",
    # Fixups
    "fixup_food_only_add_pass_mail",
]
