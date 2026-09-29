"""Diagnostic Panels for Kaggriculture Streamlit Dashboard.

Contains the 3 critical panels required:
1. Value Panel: IQL Predicted Return-To-Go vs Actual Replay Cash & Divergence Turn.
2. Search Panel: 24-Step Beam Search Candidates Tree-Graph & Scores.
3. Labor Panel: Kuhn-Munkres Cost Matrix & Strawberry vs Weed Assignment Debugger.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.data_loader import LaborCostAnalysis, MatchDataset, TurnData


# =============================================================================
# PANEL 1: VALUE PANEL (IQL RTG vs Replay Terminal Cash & Divergence)
# =============================================================================

def render_value_panel(dataset: MatchDataset, current_turn: int) -> None:
    """Render the IQL Value Net diagnostic panel with interactive divergence tracking."""
    st.markdown("### 📈 Value Panel: IQL Value Net vs Replay Terminal Cash")
    st.caption("Tracks the IQL Value Net's predicted Return-To-Go vs actual terminal cash and flags exact divergence points.")

    turn_data = dataset.turns[min(len(dataset.turns) - 1, max(0, current_turn))]
    df = dataset.df_values
    div_turn = dataset.divergence_turn
    div_obj = dataset.turns[div_turn]

    # KPI Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Liquid Cash", f"${turn_data.cash:,.1f}", delta=f"Turn {turn_data.turn}")
    with m2:
        st.metric("IQL Predicted RTG", f"${turn_data.predicted_rtg:,.0f}", delta=f"Day {turn_data.day}, Hr {turn_data.hour}")
    with m3:
        st.metric("Actual RTG Remaining", f"${turn_data.actual_rtg:,.0f}")
    with m4:
        st.metric("Pred. Final Worth", f"${turn_data.predicted_terminal_cash:,.0f}", delta=f"Actual: ${dataset.final_cash:,.0f}")
    with m5:
        st.metric("Divergence Error", f"${turn_data.divergence_error:,.0f}", delta_color="inverse")

    # Divergence Alert Callout
    st.info(
        f"🚨 **Exact Prediction Divergence Detected at Turn {div_turn} (Day {div_obj.day}, Hour {div_obj.hour})**\n\n"
        f"{dataset.divergence_reason}"
    )

    # Plotly Dual Line Chart
    fig = go.Figure()

    # 1. Predicted Return-To-Go
    fig.add_trace(go.Scatter(
        x=df["turn"],
        y=df["pred_rtg"],
        name="IQL Predicted Return-To-Go",
        line=dict(color="#58a6ff", width=2.5),
        hovertemplate="Turn %{x}<br>IQL Pred RTG: $%{y:,.0f}<extra></extra>",
    ))

    # 2. Actual Return-To-Go
    fig.add_trace(go.Scatter(
        x=df["turn"],
        y=df["actual_rtg"],
        name="Actual Replay Return-To-Go",
        line=dict(color="#3fb950", width=2.5),
        hovertemplate="Turn %{x}<br>Actual RTG: $%{y:,.0f}<extra></extra>",
    ))

    # 3. Predicted Terminal Cash (Cash + RTG)
    fig.add_trace(go.Scatter(
        x=df["turn"],
        y=df["pred_final"],
        name="IQL Predicted Terminal Cash",
        line=dict(color="#d29922", width=1.5, dash="dot"),
        hovertemplate="Turn %{x}<br>Pred Final Cash: $%{y:,.0f}<extra></extra>",
    ))

    # 4. Actual Replay Final Cash Baseline
    fig.add_trace(go.Scatter(
        x=df["turn"],
        y=[dataset.final_cash] * len(df),
        name=f"Actual Final Cash (${dataset.final_cash:,.0f})",
        line=dict(color="#f85149", width=1.5, dash="dash"),
        hoverinfo="skip",
    ))

    # Add Divergence Vertical Line
    fig.add_vline(
        x=div_turn,
        line_width=2,
        line_dash="dash",
        line_color="#ff7b72",
        annotation_text=f"Divergence Turn {div_turn}",
        annotation_position="top left",
        annotation_font=dict(color="#ff7b72", size=11),
    )

    # Add Current Turn Cursor
    fig.add_vline(
        x=current_turn,
        line_width=2,
        line_color="#f0883e",
        annotation_text=f"Current: T{current_turn}",
        annotation_position="bottom right",
        annotation_font=dict(color="#f0883e", size=11),
    )

    fig.update_layout(
        title=f"Trajectory Value Curves: IQL Return-To-Go vs Replay Reality (Episode {dataset.episode_id})",
        xaxis_title="Simulation Turn (0 to 719)",
        yaxis_title="Cash / Return Value ($)",
        template="plotly_dark",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PANEL 2: SEARCH PANEL (24-Step Beam Search Candidates & Tree)
# =============================================================================

def render_search_panel(dataset: MatchDataset, current_turn: int) -> None:
    """Render the 24-step Beam Search candidates, scores, and decision tree."""
    st.markdown("### 🌲 Search Panel: 24-Step Beam Search Candidates & Tree")
    st.caption("Displays the candidate 24-step macro-action trajectories considered on this day and their mathematical scores.")

    candidates = dataset.get_beam_search_candidates(current_turn)
    turn_data = dataset.turns[min(len(dataset.turns) - 1, max(0, current_turn))]
    day = turn_data.day

    st.markdown(f"**Temporal Context:** Day {day} (Turns {day * 24} to {day * 24 + 23}) | Active Candidates: **{len(candidates)}**")

    # 1. Candidate Summary Cards
    cols = st.columns(len(candidates))
    for idx, (col, cand) in enumerate(zip(cols, candidates)):
        with col:
            badge = "🏆 WINNER" if cand.is_winner else f"❌ PRUNED #{cand.rank}"
            border_color = "#238636" if cand.is_winner else "#da3633"
            st.markdown(
                f"""
                <div style="border: 1px solid {border_color}; border-radius: 8px; padding: 12px; background: #161b22; margin-bottom: 12px;">
                    <div style="font-weight: bold; color: {'#3fb950' if cand.is_winner else '#8b949e'}; font-size: 13px;">{badge}</div>
                    <div style="font-size: 14px; font-weight: 600; color: #f0f6fc; margin: 4px 0;">{cand.name}</div>
                    <div style="font-size: 20px; font-weight: bold; color: #58a6ff;">${cand.score:,.1f}</div>
                    <div style="font-size: 11px; color: #8b949e; margin-top: 4px;">Leaf Value: ${cand.leaf_iql_value:,.1f} | Penalty: -${cand.holding_penalty:,.1f}</div>
                    <div style="font-size: 11px; color: #c9d1d9; margin-top: 8px;">{cand.summary}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 2. Interactive Beam Tree Visualization (Plotly Tree Graph)
    st.markdown("#### 🌿 24-Step Beam Trajectory Tree")
    
    # Construct tree nodes for the 3 candidates across key hours: 0, 6, 12, 18, 24
    anchor_steps = [0, 6, 12, 18, 23]
    edge_x = []
    edge_y = []
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []

    # Root Node
    root_x, root_y = 0, 0
    node_x.append(root_x)
    node_y.append(root_y)
    node_text.append(f"Root: Day {day} Hr 0")
    node_color.append("#58a6ff")
    node_size.append(18)

    y_offsets = [1.2, 0.0, -1.2]
    colors = ["#3fb950", "#d29922", "#f85149"]

    for c_idx, cand in enumerate(candidates):
        prev_x, prev_y = root_x, root_y
        for step_i in anchor_steps:
            macro_action = cand.actions[min(step_i, len(cand.actions) - 1)]
            x_pos = step_i + 1
            y_pos = y_offsets[c_idx] * (1.0 + (step_i / 24.0) * 0.5)

            # Edge
            edge_x.extend([prev_x, x_pos, None])
            edge_y.extend([prev_y, y_pos, None])

            # Node
            node_x.append(x_pos)
            node_y.append(y_pos)
            node_text.append(f"Hr {step_i}: {macro_action}")
            node_color.append(colors[c_idx])
            node_size.append(14 if step_i == 23 else 10)

            prev_x, prev_y = x_pos, y_pos

    fig_tree = go.Figure()

    # Draw Edges
    fig_tree.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(color="#30363d", width=2),
        hoverinfo="none",
    ))

    # Draw Nodes
    fig_tree.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(size=node_size, color=node_color, line=dict(color="#ffffff", width=1)),
        text=[t.split(":")[-1] if ":" in t else t for t in node_text],
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
    ))

    fig_tree.update_layout(
        title=f"Beam Search 24-Step Rollout Branches (Width = 3, Horizon = 24)",
        xaxis=dict(title="Hour of Day (0 to 24)", showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        template="plotly_dark",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig_tree, use_container_width=True)

    # 3. Step-by-Step Trajectory Breakdown Table
    with st.expander("📋 View Full 24-Hour Macro Action Schedule Table", expanded=False):
        sched_rows = []
        for h in range(24):
            sched_rows.append({
                "Hour": f"Hour {h:02d}",
                "Candidate 1 (Winner)": candidates[0].actions[h] if h < len(candidates[0].actions) else "PASS",
                "Candidate 2": candidates[1].actions[h] if h < len(candidates[1].actions) else "PASS",
                "Candidate 3": candidates[2].actions[h] if h < len(candidates[2].actions) else "PASS",
            })
        st.dataframe(pd.DataFrame(sched_rows), use_container_width=True)


# =============================================================================
# PANEL 3: LABOR PANEL (Kuhn-Munkres Routing & Strawberry vs Weed Debugger)
# =============================================================================

def render_labor_panel(dataset: MatchDataset, current_turn: int) -> None:
    """Render Kuhn-Munkres cost matrix, assignment weights, and Strawberry vs Weed debugger."""
    st.markdown("### 🚜 Labor Panel: Kuhn-Munkres Cost Matrix & Assignment Debugger")
    st.caption("Inspects the bipartite matching weights for worker-to-target routing and explains task prioritization.")

    labor = dataset.get_labor_cost_matrix(current_turn)
    n_workers = len(labor.workers)
    n_targets = len(labor.targets)

    # 1. Strawberry vs Weed Debugger (Prompt Requirement)
    st.markdown("#### 🔍 Strawberry vs Weed Decision Forensic")
    if labor.strawberry_weed_debug:
        dbg = labor.strawberry_weed_debug
        st.success(
            f"**Diagnostic Case: Why did `{dbg['worker_name']}` ignore Strawberry for Weed?**\n\n"
            f"- **Worker Location:** `{dbg['worker_pos']}`\n"
            f"- **Weed Location:** `{dbg['weed_pos']}` (Manhattan Distance: **{dbg['dist_weed']} steps** | Cost: **{dbg['cost_weed']:.1f}**)\n"
            f"- **Strawberry Location:** `{dbg['strawberry_pos']}` (Manhattan Distance: **{dbg['dist_strawberry']} steps** | Cost: **{dbg['cost_strawberry']:.1f}**)\n"
            f"- **Mathematical Delta:** $\\Delta C = C(\\text{{Weed}}) - C(\\text{{Strawberry}}) = {dbg['delta_cost']:.1f}$\n\n"
            f"**Verdict:** {dbg['reason']}"
        )
    else:
        st.info("No active Strawberry vs Weed conflict detected on this specific turn. (Either no weeds or no strawberries on board).")

    # 2. Kuhn-Munkres Cost Matrix Heatmap
    st.markdown(f"#### 🔢 Kuhn-Munkres Cost Matrix ($N={n_workers}$ Workers $\\times$ $M={n_targets}$ Targets)")
    st.caption("Green border cells denote the optimal matching chosen by the Hungarian algorithm.")

    # Create readable matrix labels
    worker_labels = [f"{w['name']} @ {w['pos']}" for w in labor.workers]
    target_labels = [f"{t.label}" for t in labor.targets]

    # Build cost DataFrame
    df_cost = pd.DataFrame(labor.cost_matrix, index=worker_labels, columns=target_labels)

    # Plotly Matrix Heatmap
    fig_mat = px.imshow(
        labor.cost_matrix,
        x=target_labels,
        y=worker_labels,
        labels=dict(x="Assigned Target", y="Farm Worker", color="Cost"),
        color_continuous_scale="Viridis_r",  # Low cost = bright yellow/green
        aspect="auto",
    )

    # Overlay markers on assigned cells
    for w_idx, t_idx in labor.assignments:
        fig_mat.add_annotation(
            x=t_idx,
            y=w_idx,
            text="✓ ASSIGNED",
            showarrow=False,
            font=dict(color="#00ff00", size=11, family="monospace"),
            bgcolor="rgba(0, 0, 0, 0.75)",
            bordercolor="#00ff00",
            borderwidth=1.5,
            borderpad=3,
        )

    fig_mat.update_layout(
        template="plotly_dark",
        height=max(280, 45 * n_workers),
        margin=dict(l=20, r=20, t=30, b=50),
        xaxis=dict(tickangle=-35),
    )
    st.plotly_chart(fig_mat, use_container_width=True)

    # 3. Assigned Pairs Summary Table
    with st.expander("📊 View Assigned Worker-to-Target Routing Details", expanded=False):
        pairs_data = []
        for w_idx, t_idx in labor.assignments:
            w = labor.workers[w_idx]
            t = labor.targets[t_idx]
            dist = abs(w["pos"][0] - t.x) + abs(w["pos"][1] - t.y)
            cost = labor.cost_matrix[w_idx, t_idx]
            pairs_data.append({
                "Worker": w["name"],
                "Worker Role": w["role"],
                "Start Pos": str(w["pos"]),
                "Assigned Target": t.label,
                "Target Kind": t.kind,
                "Manhattan Dist": dist,
                "KM Cost Weight": round(cost, 1),
            })
        st.dataframe(pd.DataFrame(pairs_data), use_container_width=True)
