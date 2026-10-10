"""
Tab 3 - Risk Factors: which conditions go with damage.
Damage rate by phase of flight, species group and aircraft type; distribution of height / speed / cost
by species group (box plot); speed vs. height scatter colored by outcome.

Author: Saamarth Attray

"Damage rate" always means: damaging strikes / strikes with a KNOWN outcome. Strikes whose outcome was not
reported are left out (never counted as "no damage").
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

from backend.contract import PHASE_ORDER
from dashboard import theme
from dashboard.components import card, control, controls, grid
from dashboard.data import damage_rate, get_df

BOX_METRICS = {
    "Height above ground (ft)": "HEIGHT", "Speed (knots)": "SPEED", "Distance from airport (nm)": "DISTANCE",
    "Repair cost, inflation-adj. ($)": "COST_REPAIRS_INFL_ADJ",
}


def layout():
    return html.Div([
        controls([
            control("Minimum known outcomes per category", dcc.Slider(
                id="rk-min", min=50, max=1000, step=50, value=200,
                marks={50: "50", 250: "250", 500: "500", 750: "750", 1000: "1000"})),
            control("Aircraft types shown", dcc.Slider(id="rk-top", min=5, max=30, step=1, value=15,
                                                       marks={5: "5", 15: "15", 30: "30"})),
        ]),
        grid([
            card("Damage rate by phase of flight", [dcc.Graph(id="rk-phase", config={"displaylogo": False})],
                 subtitle="Phases in flight order. The vertical line is the overall rate."),
            card("Damage rate by species group", [dcc.Graph(id="rk-group", config={"displaylogo": False})],
                 subtitle="Large-bodied animals (raptors, waterfowl, mammals) damage aircraft far more often than songbirds."),
            card("Damage rate by aircraft type", [dcc.Graph(id="rk-aircraft", config={"displaylogo": False})],
                 subtitle="Highest rates among aircraft with enough known outcomes (slider above)."),
            card("Distribution by species group", [
                controls([
                    control("Measure", dcc.Dropdown(id="rk-metric", options=[{"label": k, "value": v} for k, v in
                                                                              BOX_METRICS.items()],
                                                    value="HEIGHT", clearable=False), width="300px"),
                    control("Scale", dcc.Checklist(id="rk-log", className="radio",
                                                   options=[{"label": "Log scale", "value": "log"}], value=["log"])),
                ]),
                dcc.Graph(id="rk-box", config={"displaylogo": False}),
            ], subtitle="Box = 25th to 75th percentile, line = median, whiskers = 5th to 95th percentile."),
            card("Speed vs. height", [
                controls([control("Height axis", dcc.Checklist(id="rk-scatter-log", className="radio",
                                                               options=[{"label": "Log scale", "value": "log"}],
                                                               value=["log"]))]),
                dcc.Graph(id="rk-scatter", config={"displaylogo": False}),
            ], subtitle="5,000 randomly sampled strikes with a reported speed, height and outcome.", span=2),
        ]),
    ])


def _rates(df: pd.DataFrame, col: str, min_known: int) -> pd.DataFrame:
    g = df.groupby(col, observed=True)["STRIKE_CAUSED_DAMAGE"]
    out = pd.DataFrame({"known": g.count(), "damaged": g.sum()})
    out["rate"] = out["damaged"] / out["known"].where(out["known"] > 0)
    out = out[out["known"] >= min_known].dropna(subset=["rate"])
    out.index = out.index.astype(str)
    return out


def _rate_bars(rates: pd.DataFrame, overall: float, height: int, ref_line: bool = True) -> go.Figure:
    fig = go.Figure(go.Bar(
        y=rates.index, x=rates["rate"], orientation="h", marker_color=theme.DAMAGE,
        text=[theme.pct(v) for v in rates["rate"]], textposition="outside", cliponaxis=False,
        textfont=dict(size=11, color=theme.INK_2), customdata=np.stack([rates["known"], rates["damaged"]], axis=-1),
        hovertemplate="%{y}<br>Damage rate: %{x:.1%}<br>%{customdata[1]:,.0f} of %{customdata[0]:,.0f} known outcomes"
                      "<extra></extra>"))
    theme.style(fig, height=height, x_title="Strikes that caused damage")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(tickformat=".0%", range=[0, max(rates["rate"].max() * 1.2, overall * 1.3)])
    if ref_line:
        fig.add_vline(x=overall, line=dict(color=theme.MUTED, width=1.5), annotation_text=f"All: {theme.pct(overall)}",
                      annotation_position="top", annotation_font=dict(size=11, color=theme.INK_2))
    return fig


def register(app):
    @app.callback(Output("rk-phase", "figure"), Output("rk-group", "figure"), Output("rk-aircraft", "figure"),
                  Input("rk-min", "value"), Input("rk-top", "value"))
    def _bars(min_known, top_n):
        df = get_df()
        overall = damage_rate(df)

        phase = _rates(df, "PHASE_OF_FLIGHT", min_known)
        phase = phase.reindex([p for p in PHASE_ORDER if p in phase.index])
        fig_phase = (_rate_bars(phase, overall, 360) if len(phase)
                     else theme.empty_figure("No phase has enough known outcomes", 360))

        grp = _rates(df, "SPECIES_GROUP", min_known).sort_values("rate", ascending=False)
        fig_group = (_rate_bars(grp, overall, 360) if len(grp)
                     else theme.empty_figure("No species group has enough known outcomes", 360))

        ac = _rates(df, "AIRCRAFT", min_known).sort_values("rate", ascending=False).head(top_n)
        fig_air = (_rate_bars(ac, overall, max(360, 22 * len(ac) + 90)) if len(ac)
                   else theme.empty_figure("No aircraft type has enough known outcomes", 360))
        return fig_phase, fig_group, fig_air

    @app.callback(Output("rk-box", "figure"), Input("rk-metric", "value"), Input("rk-log", "value"))
    def _box(metric, log_value):
        log = "log" in (log_value or [])
        df = get_df()[["SPECIES_GROUP", metric]].dropna()
        df["SPECIES_GROUP"] = df["SPECIES_GROUP"].astype(str)
        stats = df.groupby("SPECIES_GROUP")[metric].agg(
            n="size", q1=lambda s: s.quantile(.25), med="median", q3=lambda s: s.quantile(.75),
            lo=lambda s: s.quantile(.05), hi=lambda s: s.quantile(.95))
        stats = stats[stats["n"] >= 30].sort_values("med")
        if stats.empty:
            return theme.empty_figure("Not enough data for this measure", 420)
        floor = 1.0 if log else 0.0
        fig = go.Figure(go.Box(
            y=stats.index, q1=np.maximum(stats["q1"], floor), median=np.maximum(stats["med"], floor),
            q3=np.maximum(stats["q3"], floor), lowerfence=np.maximum(stats["lo"], floor),
            upperfence=np.maximum(stats["hi"], floor), orientation="h", boxpoints=False,
            marker_color=theme.BLUE, line=dict(color=theme.BLUE, width=1.5), fillcolor="rgba(42,120,214,0.18)",
            customdata=np.stack([stats["n"], stats["med"]], axis=-1),
            hovertemplate="%{y}<br>Median: %{customdata[1]:,.0f}<br>%{customdata[0]:,.0f} strikes<extra></extra>"))
        theme.style(fig, height=420, x_title=next(k for k, v in BOX_METRICS.items() if v == metric))
        fig.update_yaxes(showgrid=False)
        fig.update_xaxes(type="log" if log else "linear")
        return fig

    @app.callback(Output("rk-scatter", "figure"), Input("rk-scatter-log", "value"))
    def _scatter(log_value):
        df = get_df()
        d = df.dropna(subset=["SPEED", "HEIGHT", "STRIKE_CAUSED_DAMAGE"])
        d = d[(d["SPEED"] > 0) & (d["HEIGHT"] >= 0)].sample(n=5000, random_state=7)
        fig = go.Figure()
        for flag, name, color in [(0.0, "No damage", theme.NO_DAMAGE), (1.0, "Caused damage", theme.DAMAGE)]:
            s = d[d["STRIKE_CAUSED_DAMAGE"] == flag]
            fig.add_trace(go.Scatter(
                x=s["SPEED"], y=s["HEIGHT"] + (1 if "log" in (log_value or []) else 0), mode="markers", name=name,
                marker=dict(color=color, size=6, opacity=0.5, line=dict(color=theme.SURFACE, width=0.5)),
                hovertemplate=f"{name}<br>Speed: %{{x:,.0f}} kt<br>Height: %{{customdata:,.0f}} ft<extra></extra>",
                customdata=s["HEIGHT"]))
        theme.style(fig, height=420, legend=True, x_title="Speed (knots)", y_title="Height above ground (ft)")
        fig.update_yaxes(type="log" if "log" in (log_value or []) else "linear")
        return fig
