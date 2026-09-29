"""Kaggriculture Farm Grid Renderer (10x10).

High-fidelity HTML5/CSS grid rendering for farm tiles, crops, livestock, workers,
Kuhn-Munkres priority heatmaps, and assignment routing vectors.
"""

from __future__ import annotations

import html
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from dashboard.data_loader import TurnData, LaborCostAnalysis


def render_farm_grid_html(
    turn_data: TurnData,
    labor_analysis: Optional[LaborCostAnalysis] = None,
    mode: str = "normal",  # 'normal', 'heatmap', 'assignments'
    selected_coord: Optional[Tuple[int, int]] = None,
) -> str:
    """Generate high-performance standalone HTML & CSS for the 10x10 farm grid."""
    tiles = turn_data.tiles
    day = turn_data.day
    farmer_pos = turn_data.farmer_pos
    hands_pos = turn_data.hands_pos
    shed_coords = {(4, 4), (5, 4), (4, 5), (5, 5)}

    # Map worker positions
    worker_map: Dict[Tuple[int, int], List[str]] = {}
    worker_map.setdefault(farmer_pos, []).append("🧑‍🌾 Farmer")
    for idx, h_pos in enumerate(hands_pos):
        worker_map.setdefault(h_pos, []).append(f"👷 Hand {idx}")

    # Assignment vectors map: {worker_pos: target_pos}
    assignment_map: Dict[Tuple[int, int], Tuple[int, int]] = {}
    target_pos_assigned = set()
    if labor_analysis:
        for w_idx, t_idx in labor_analysis.assignments:
            if w_idx < len(labor_analysis.workers) and t_idx < len(labor_analysis.targets):
                w_pos = labor_analysis.workers[w_idx]["pos"]
                t_obj = labor_analysis.targets[t_idx]
                t_pos = (t_obj.x, t_obj.y)
                assignment_map[w_pos] = t_pos
                target_pos_assigned.add(t_pos)

    # CSS styles
    css = """
    <style>
    .farm-grid-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin: 0 auto;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .farm-grid {
        display: grid;
        grid-template-columns: repeat(10, 56px);
        grid-template-rows: repeat(10, 56px);
        gap: 3px;
        background: #0d1117;
        padding: 8px;
        border-radius: 10px;
        border: 2px solid #30363d;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
        user-select: none;
    }
    .farm-tile {
        position: relative;
        width: 56px;
        height: 56px;
        border-radius: 6px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        box-sizing: border-box;
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: transform 0.1s ease, border-color 0.1s ease;
        overflow: hidden;
        cursor: pointer;
    }
    .farm-tile:hover {
        transform: scale(1.06);
        z-index: 10;
        border-color: #58a6ff !important;
        box-shadow: 0 4px 12px rgba(88, 166, 255, 0.4);
    }
    .farm-tile.selected {
        border: 2px solid #f0883e !important;
        box-shadow: 0 0 10px #f0883e;
    }
    /* Quadrant dividers */
    .farm-tile[data-x="4"] { border-right: 2px dashed #8b949e; }
    .farm-tile[data-y="4"] { border-bottom: 2px dashed #8b949e; }

    /* Tile Backgrounds */
    .tile-locked {
        background: #161b22;
        color: #484f58;
    }
    .tile-empty {
        background: #1a271d;
        color: #7ee787;
    }
    .tile-shed {
        background: #21262d;
        border: 1px solid #6e7681;
    }
    .tile-weed {
        background: #2b1d14;
        border-color: #8b3a1a;
    }
    .tile-plant {
        background: #132d20;
    }
    .tile-pasture {
        background: #1b2d38;
    }

    /* Badges */
    .tile-icon {
        font-size: 20px;
        line-height: 20px;
        margin-bottom: 1px;
    }
    .tile-label {
        font-size: 9px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.3px;
        max-width: 50px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .worker-badge {
        position: absolute;
        top: 2px;
        left: 2px;
        font-size: 11px;
        background: rgba(0, 0, 0, 0.75);
        padding: 1px 3px;
        border-radius: 4px;
        border: 1px solid #e3b341;
        box-shadow: 0 0 4px #e3b341;
        z-index: 5;
    }
    .worker-hand {
        border-color: #58a6ff;
        box-shadow: 0 0 4px #58a6ff;
    }
    .status-dot {
        position: absolute;
        bottom: 2px;
        right: 2px;
        font-size: 10px;
        line-height: 10px;
    }
    .status-left {
        position: absolute;
        bottom: 2px;
        left: 2px;
        font-size: 9px;
        color: #58a6ff;
        font-weight: bold;
    }
    .quadrant-legend {
        display: flex;
        justify-content: space-between;
        width: 580px;
        margin-bottom: 6px;
        font-size: 11px;
        color: #8b949e;
        font-weight: 600;
    }
    .coord-badge {
        position: absolute;
        top: 2px;
        right: 2px;
        font-size: 7.5px;
        color: rgba(255, 255, 255, 0.35);
        font-family: monospace;
    }
    .assignment-target {
        box-shadow: inset 0 0 0 2px #3fb950 !important;
    }
    .assignment-marker {
        position: absolute;
        bottom: 1px;
        left: 1px;
        background: #238636;
        color: #ffffff;
        font-size: 8px;
        padding: 0 3px;
        border-radius: 3px;
        font-weight: bold;
        z-index: 6;
    }
    </style>
    """

    grid_cells = []
    for y in range(10):
        for x in range(10):
            pos = (x, y)
            is_selected = (selected_coord == pos)
            selected_cls = " selected" if is_selected else ""
            tile_raw = tiles[y][x] if y < len(tiles) and x < len(tiles[y]) else None

            # Determine Heatmap color if active
            style_override = ""
            if mode == "heatmap" and labor_analysis is not None:
                val = labor_analysis.heatmap_grid[y, x]
                if val < 0:
                    style_override = "background-color: #121417; color: #30363d;"
                else:
                    # Color map from deep navy to cyan to bright emerald
                    # val is between 0.1 and 1.0
                    r = int(np.clip(255 * (1.0 - val) * 0.4, 15, 60))
                    g = int(np.clip(255 * val, 30, 230))
                    b = int(np.clip(255 * (1.0 - (val - 0.5)**2), 40, 200))
                    style_override = f"background-color: rgb({r}, {g}, {b}); color: #ffffff;"

            # Check workers at this cell
            workers_here = worker_map.get(pos, [])
            worker_html = ""
            if workers_here:
                has_farmer = any("Farmer" in w for w in workers_here)
                hand_count = sum(1 for w in workers_here if "Hand" in w)
                badge_cls = "worker-badge" if has_farmer else "worker-badge worker-hand"
                if has_farmer and hand_count > 0:
                    worker_html = f'<div class="{badge_cls}" title="{html.escape(", ".join(workers_here))}">🧑‍🌾+{hand_count}</div>'
                elif has_farmer:
                    worker_html = f'<div class="{badge_cls}" title="Farmer">🧑‍🌾</div>'
                elif hand_count > 1:
                    worker_html = f'<div class="{badge_cls}" title="{html.escape(", ".join(workers_here))}">👷x{hand_count}</div>'
                else:
                    worker_html = f'<div class="{badge_cls}" title="{html.escape(workers_here[0])}">👷</div>'

            # Assignment Target indicator
            assignment_html = ""
            is_target = pos in target_pos_assigned
            target_cls = " assignment-target" if (is_target and mode in ("heatmap", "assignments")) else ""
            if is_target and mode in ("heatmap", "assignments"):
                assignment_html = '<div class="assignment-marker" title="Assigned Target">TARGET</div>'

            coord_html = f'<div class="coord-badge">{x},{y}</div>'

            # Locked tile
            if tile_raw == "LOCKED":
                cell = f"""
                <div class="farm-tile tile-locked{selected_cls}" data-x="{x}" data-y="{y}" style="{style_override}">
                    {coord_html}
                    <div class="tile-icon">🔒</div>
                    <div class="tile-label" style="color: #6e7681;">Locked</div>
                    {worker_html}
                </div>
                """
                grid_cells.append(cell)
                continue

            # Shed tile
            if pos in shed_coords:
                shed_inv = turn_data.shed_inventory
                inv_summary = ", ".join(f"{k}:{v}" for k, v in list(shed_inv.items())[:3] if v > 0)
                cell = f"""
                <div class="farm-tile tile-shed{selected_cls}{target_cls}" data-x="{x}" data-y="{y}" style="{style_override}" title="Shed Storage ({inv_summary})">
                    {coord_html}
                    <div class="tile-icon">🏚️</div>
                    <div class="tile-label" style="color: #e3b341;">Shed</div>
                    {worker_html}
                    {assignment_html}
                </div>
                """
                grid_cells.append(cell)
                continue

            # Empty arable land
            if tile_raw is None:
                cell = f"""
                <div class="farm-tile tile-empty{selected_cls}{target_cls}" data-x="{x}" data-y="{y}" style="{style_override}" title="Empty Arable Plot">
                    {coord_html}
                    <div class="tile-icon" style="opacity: 0.25;">🌱</div>
                    <div class="tile-label" style="color: #484f58;">Soil</div>
                    {worker_html}
                    {assignment_html}
                </div>
                """
                grid_cells.append(cell)
                continue

            # Dict tile
            if isinstance(tile_raw, dict):
                kind = tile_raw.get("kind", "")

                if kind == "WEED":
                    cell = f"""
                    <div class="farm-tile tile-weed{selected_cls}{target_cls}" data-x="{x}" data-y="{y}" style="{style_override}" title="Wild Invasive Weed">
                        {coord_html}
                        <div class="tile-icon">🌿</div>
                        <div class="tile-label" style="color: #f85149;">Weed</div>
                        {worker_html}
                        {assignment_html}
                    </div>
                    """
                    grid_cells.append(cell)
                    continue

                elif kind == "PLANT":
                    crop = str(tile_raw.get("crop", "CROP"))
                    y_units = int(tile_raw.get("yield_units", 0))
                    watered = bool(tile_raw.get("watered_today", False))
                    fertilized = int(tile_raw.get("fertilized_until_day", -1)) >= day
                    crop_icons = {
                        "STRAWBERRY": "🍓",
                        "MELON": "🍈",
                        "WHEAT": "🌾",
                        "CARROT": "🥕",
                        "TOMATO": "🍅",
                    }
                    icon = crop_icons.get(crop, "🌱")
                    status_icons = ("💧" if watered else "🥀") + ("✨" if fertilized else "")
                    yield_txt = f"x{y_units}" if y_units > 0 else ""
                    crop_color = "#ff7b72" if crop == "STRAWBERRY" else ("#7ee787" if crop == "MELON" else "#d29922")

                    cell = f"""
                    <div class="farm-tile tile-plant{selected_cls}{target_cls}" data-x="{x}" data-y="{y}" style="{style_override}" title="{crop} (Yield: {y_units}, {'Watered' if watered else 'Thirsty'})">
                        {coord_html}
                        <div class="tile-icon">{icon}</div>
                        <div class="tile-label" style="color: {crop_color};">{crop[:6]}</div>
                        <div class="status-dot">{status_icons}</div>
                        <div class="status-left">{yield_txt}</div>
                        {worker_html}
                        {assignment_html}
                    </div>
                    """
                    grid_cells.append(cell)
                    continue

                elif kind == "PASTURE":
                    animal = str(tile_raw.get("animal", "ANIMAL"))
                    fed = bool(tile_raw.get("fed_today", False))
                    cared = bool(tile_raw.get("cared_today", False))
                    fert_ready = bool(tile_raw.get("fertilizer_available", False))
                    y_units = int(tile_raw.get("yield_units", 0))

                    animal_icons = {"COW": "🐄", "SHEEP": "🐑", "GOOSE": "🪿"}
                    icon = animal_icons.get(animal, "🐾")
                    status_icons = ("🌾" if fed else "⚠️") + ("❤️" if cared else "") + ("💩" if fert_ready else "")
                    yield_txt = f"x{y_units}" if y_units > 0 else ""

                    cell = f"""
                    <div class="farm-tile tile-pasture{selected_cls}{target_cls}" data-x="{x}" data-y="{y}" style="{style_override}" title="{animal} ({'Fed' if fed else 'Unfed'}, {'Cared' if cared else 'Uncared'})">
                        {coord_html}
                        <div class="tile-icon">{icon}</div>
                        <div class="tile-label" style="color: #79c0ff;">{animal}</div>
                        <div class="status-dot">{status_icons}</div>
                        <div class="status-left">{yield_txt}</div>
                        {worker_html}
                        {assignment_html}
                    </div>
                    """
                    grid_cells.append(cell)
                    continue

            # Fallback unknown tile
            cell = f"""
            <div class="farm-tile tile-empty{selected_cls}" data-x="{x}" data-y="{y}" style="{style_override}">
                {coord_html}
                <div class="tile-icon">❓</div>
                <div class="tile-label">Unknown</div>
                {worker_html}
            </div>
            """
            grid_cells.append(cell)

    legend_html = """
    <div class="quadrant-legend">
        <span>NW Quadrant (Base)</span>
        <span>NE Quadrant ($1,000)</span>
    </div>
    """

    bottom_legend_html = """
    <div class="quadrant-legend" style="margin-top: 6px;">
        <span>SW Quadrant ($2,000)</span>
        <span>SE Quadrant ($4,000)</span>
    </div>
    """

    return f"""
    <div class="farm-grid-container">
        {css}
        {legend_html}
        <div class="farm-grid">
            {"".join(grid_cells)}
        </div>
        {bottom_legend_html}
    </div>
    """
