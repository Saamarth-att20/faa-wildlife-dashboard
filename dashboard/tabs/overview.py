"""
Tab 1 - Overview: KPI tiles, US map, damage-level donut, and a feature explorer
(histogram + descriptive statistics for numeric features, bar chart + frequency table for categorical ones).

Author: Saamarth Attray
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

from dashboard import theme
from dashboard.components import card, control, controls, grid, kpi, notice, table
from dashboard.data import (CATEGORICAL_EXPLORER, NUMERIC_EXPLORER, US_STATES, damage_rate, get_df)
from backend.contract import DAMAGE_LEVEL_ORDER

MAP_METRICS = {"strikes": "Number of strikes", "rate": "Damage rate", "cost": "Reported cost"}


# ----------------------------------------------------------------------------------------------
# layout
# ----------------------------------------------------------------------------------------------
def _damage_donut(df: pd.DataFrame) -> go.Figure:
    counts = df["DAMAGE_LEVEL"].astype(str).value_counts()
    levels = [lvl for lvl in DAMAGE_LEVEL_ORDER if lvl in counts.index]
    fig = go.Figure(go.Pie(
        labels=[f"{lvl} ({counts[lvl]:,})" for lvl in levels], values=[int(counts[lvl]) for lvl in levels],
        hole=0.62, sort=False, direction="clockwise", textinfo="none",
        marker=dict(colors=[theme.SEVERITY_COLORS[lvl] for lvl in levels], line=dict(color=theme.SURFACE, width=2)),
        hovertemplate="%{label}<br>%{percent} of known outcomes<extra></extra>"))
    fig.add_annotation(text=f"<b>{theme.pct(damage_rate(df))}</b><br>caused damage", x=0.5, y=0.5,
                       showarrow=False, font=dict(size=15, color=theme.INK))
    theme.style(fig, height=330, legend=True)
    fig.update_layout(legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.0),
                      margin=dict(l=8, r=8, t=8, b=8))
    return fig


def layout():
    df = get_df()
    known = df["STRIKE_CAUSED_DAMAGE"].notna()
    options = (
        [{"label": f"Numeric - {k}", "value": f"num:{v[0]}"} for k, v in NUMERIC_EXPLORER.items()]
        + [{"label": f"Categorical - {k}", "value": f"cat:{v}"} for k, v in CATEGORICAL_EXPLORER.items()]
    )
    return html.Div([
        html.Div(className="kpi-row", children=[
            kpi("Total strikes", f"{len(df):,}", f"{df['INCIDENT_YEAR'].min()}-2024 (2025 has only 16 rows)"),
            kpi("Caused damage", theme.pct(damage_rate(df)), f"of {int(known.sum()):,} strikes with a known outcome",
                accent=True),
            kpi("Reported cost", theme.compact(df["TOTAL_COST"].sum(), "$"),
                f"repairs + other, only {int(df['TOTAL_COST'].notna().sum()):,} strikes report a cost"),
            kpi("Outcome unknown", theme.pct(1 - known.mean()), "damage level not reported"),
        ]),
        grid([
            card("Strikes by state", [
                controls([control("Color states by", dcc.RadioItems(
                    id="ov-map-metric", className="radio",
                    options=[{"label": v, "value": k} for k, v in MAP_METRICS.items()], value="strikes",
                    inline=True))]),
                dcc.Graph(id="ov-map", config={"displaylogo": False, "responsive": True}),
            ], subtitle="50 states + DC. Territories and non-US locations are not drawn."),
            card("Damage level", [dcc.Graph(figure=_damage_donut(df), config={"displaylogo": False})],
                 subtitle="Strikes with a reported outcome. Gray = no damage; orange shades = damage, darker = more severe."),
            card("Feature explorer", [
                controls([
                    control("Feature", dcc.Dropdown(id="ov-feature", options=options, value="num:HEIGHT",
                                                    clearable=False), width="360px"),
                    control("Scale", dcc.Checklist(id="ov-log", className="radio",
                                                   options=[{"label": "Log scale (numeric)", "value": "log"}],
                                                   value=["log"])),
                ]),
                html.Div(className="split", children=[
                    dcc.Graph(id="ov-explorer-graph", config={"displaylogo": False, "responsive": True}),
                    html.Div(id="ov-explorer-stats", className="stats"),
                ]),
            ], subtitle="Histogram and descriptive statistics for numeric features; bar chart and frequency table "
                        "for categorical ones.", span=2),
        ]),
    ])


# ----------------------------------------------------------------------------------------------
# callbacks
# ----------------------------------------------------------------------------------------------
def _map_figure(metric: str) -> go.Figure:
    df = get_df()
    df = df[df["STATE"].isin(US_STATES)]
    g = df.groupby("STATE", observed=True)
    agg = pd.DataFrame({
        "strikes": g.size(),
        "known": g["STRIKE_CAUSED_DAMAGE"].count(),
        "damaged": g["STRIKE_CAUSED_DAMAGE"].sum(),
        "cost": g["TOTAL_COST"].sum(min_count=1),
    })
    agg["rate"] = np.where(agg["known"] >= 30, agg["damaged"] / agg["known"].where(agg["known"] > 0), np.nan)
    z = agg[metric]
    fmt = {"strikes": ",d", "rate": ".1%", "cost": "$,.0f"}[metric]
    custom = np.stack([agg["strikes"], agg["rate"].fillna(-1), agg["cost"].fillna(0)], axis=-1)
    fig = go.Figure(go.Choropleth(
        locations=agg.index.astype(str), z=z, locationmode="USA-states", colorscale=theme.BLUE_SCALE,
        marker=dict(line=dict(color=theme.SURFACE, width=1)), customdata=custom,
        hovertemplate=("<b>%{location}</b><br>Strikes: %{customdata[0]:,.0f}<br>"
                       "Damage rate: %{customdata[1]:.1%}<br>Reported cost: $%{customdata[2]:,.0f}<extra></extra>"),
        colorbar=dict(title=dict(text=MAP_METRICS[metric], font=dict(size=11, color=theme.INK_2)),
                      tickformat={"strikes": "~s", "rate": ".0%", "cost": "$~s"}[metric], thickness=10, len=0.8,
                      tickfont=dict(size=11, color=theme.INK_2), outlinewidth=0)))
    fig.update_geos(scope="usa", showlakes=False, bgcolor="rgba(0,0,0,0)", landcolor=theme.GRID)
    theme.style(fig, height=360)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    if metric == "rate":
        fig.add_annotation(text="States with fewer than 30 known outcomes are left blank.", x=0, y=0,
                           xref="paper", yref="paper", showarrow=False, font=dict(size=11, color=theme.MUTED))
    return fig


def _numeric_view(col: str, unit: str, log: bool):
    s = get_df()[col]
    vals = s.dropna().astype(float)
    n_zero = int((vals <= 0).sum())
    note = ""
    if vals.nunique() <= 12:  # discrete (engines, mass class): plain counts
        counts = vals.value_counts().sort_index()
        fig = go.Figure(go.Bar(x=[f"{v:g}" for v in counts.index], y=counts.values, marker_color=theme.BLUE,
                               hovertemplate="%{x}: %{y:,}<extra></extra>"))
        theme.style(fig, height=330, x_title=col, y_title="Strikes")
    elif log:
        pos = vals[vals > 0]
        hist, edges = np.histogram(np.log10(pos), bins=50)
        mids = (edges[:-1] + edges[1:]) / 2
        fig = go.Figure(go.Bar(x=mids, y=hist, marker_color=theme.BLUE, width=np.diff(edges) * 0.92,
                               customdata=np.stack([10 ** edges[:-1], 10 ** edges[1:]], axis=-1),
                               hovertemplate="%{customdata[0]:,.3g} to %{customdata[1]:,.3g}<br>%{y:,} strikes<extra></extra>"))
        lo, hi = int(np.floor(edges[0])), int(np.ceil(edges[-1]))
        ticks = list(range(lo, hi + 1))
        fig.update_xaxes(tickvals=ticks, ticktext=[theme.compact(10.0 ** t, "$" if unit == "$" else "") for t in ticks])
        theme.style(fig, height=330, x_title=f"{col} (log scale)", y_title="Strikes")
        if n_zero:
            note = f"{n_zero:,} zero values are not shown on the log scale."
    else:
        hi = float(vals.quantile(0.995))
        clipped = vals[vals <= hi]
        hist, edges = np.histogram(clipped, bins=50)
        mids = (edges[:-1] + edges[1:]) / 2
        fig = go.Figure(go.Bar(x=mids, y=hist, marker_color=theme.BLUE, width=np.diff(edges) * 0.92,
                               customdata=np.stack([edges[:-1], edges[1:]], axis=-1),
                               hovertemplate="%{customdata[0]:,.3g} to %{customdata[1]:,.3g}<br>%{y:,} strikes<extra></extra>"))
        theme.style(fig, height=330, x_title=col, y_title="Strikes")
        note = f"x-axis clipped at the 99.5th percentile ({hi:,.0f}); {int((vals > hi).sum()):,} larger values not drawn."

    desc = vals.describe(percentiles=[0.25, 0.5, 0.75])
    rows = [["Valid values", f"{len(vals):,}"], ["Missing", theme.pct(1 - len(vals) / len(s))],
            ["Mean", f"{desc['mean']:,.2f}"], ["Std. deviation", f"{desc['std']:,.2f}"], ["Minimum", f"{desc['min']:,.2f}"],
            ["25th percentile", f"{desc['25%']:,.2f}"], ["Median", f"{desc['50%']:,.2f}"],
            ["75th percentile", f"{desc['75%']:,.2f}"], ["Maximum", f"{desc['max']:,.2f}"],
            ["Skewness", f"{vals.skew():,.2f}"]]
    stats = [table(["Statistic", "Value"], rows)]
    if note:
        stats.append(html.P(note, className="fine-print"))
    return fig, stats


def _categorical_view(col: str):
    s = get_df()[col].astype("object")
    counts = s.value_counts()
    top = counts.head(12)
    fig = go.Figure(go.Bar(y=[str(i) for i in top.index], x=top.values, orientation="h", marker_color=theme.BLUE,
                           text=[f"{v:,}" for v in top.values], textposition="outside", cliponaxis=False,
                           textfont=dict(size=11, color=theme.INK_2),
                           hovertemplate="%{y}: %{x:,}<extra></extra>"))
    theme.style(fig, height=330, x_title="Strikes")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(range=[0, top.max() * 1.18])
    total = int(counts.sum())
    rows = [[str(k), f"{v:,}", theme.pct(v / total)] for k, v in counts.head(10).items()]
    summary = [["Distinct values", f"{counts.size:,}"], ["Most common (mode)", str(counts.index[0])],
               ["Missing", theme.pct(s.isna().mean())]]
    stats = [table(["Summary", ""], summary, align=["l", "r"]),
             table(["Category", "Strikes", "Share"], rows, caption="Top 10 (share of reported values)")]
    return fig, stats


def register(app):
    @app.callback(Output("ov-map", "figure"), Input("ov-map-metric", "value"))
    def _map(metric):
        return _map_figure(metric)

    @app.callback(Output("ov-explorer-graph", "figure"), Output("ov-explorer-stats", "children"),
                  Input("ov-feature", "value"), Input("ov-log", "value"))
    def _explorer(feature, log_value):
        kind, col = feature.split(":", 1)
        if kind == "num":
            unit = next(u for c, u in NUMERIC_EXPLORER.values() if c == col)
            return _numeric_view(col, unit, "log" in (log_value or []))
        return _categorical_view(col)
