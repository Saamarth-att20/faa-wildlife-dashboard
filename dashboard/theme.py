"""
Visual theme for the dashboard: colors, Plotly styling helpers, number formatting.

Author: Saamarth Attray

Color rules (from the data-viz guidelines the project follows):
  * categorical hues are assigned in a FIXED order and never cycled
  * single-series charts use one color (blue); category bars are never colored by value
  * damage vs. no damage always use the same two colors (orange = damage, blue = no damage)
  * magnitude (map, heatmap) uses ONE hue, light -> dark; signed loadings use blue <-> gray <-> red
  * marks are thin, grids are hairline + solid, text wears text tokens (never the series color)
"""
import math

import plotly.graph_objects as go

# ---- surfaces and ink ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
PAGE_BG = "#f3f2ee"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e0"
NEUTRAL = "#c9c8c1"   # "everything else" gray in emphasis charts

# ---- categorical slots (fixed order) --------------------------------------------------------------
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

NO_DAMAGE = BLUE
DAMAGE = ORANGE

# ---- sequential (blue, light -> dark) ---------------------------------------------------------------
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUE_SCALE = [[i / (len(BLUE_RAMP) - 1), c] for i, c in enumerate(BLUE_RAMP)]
# damage severity (ordinal, light -> dark orange); "None" is gray so damage stands out
SEVERITY_COLORS = {
    "None": NEUTRAL, "Minor (uncertain)": "#f6b79c", "Minor": "#f08d65",
    "Substantial": "#d4521f", "Destroyed": "#8f2f0e",
}
# diverging for signed values (PCA loadings): blue <-> neutral gray <-> red
DIVERGING = [[0.0, "#2a78d6"], [0.5, "#f0efec"], [1.0, "#e34948"]]

FONT = "Inter, 'Segoe UI', system-ui, -apple-system, Roboto, Arial, sans-serif"


# ---- number formatting ---------------------------------------------------------------------------
def compact(value, prefix: str = "", digits: int = 1) -> str:
    """1,284 -> '1,284'; 12,900 -> '12.9K'; 4_200_000 -> '4.2M'."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    a = abs(value)
    if a >= 1e9:
        return f"{prefix}{value / 1e9:.{digits}f}B"
    if a >= 1e6:
        return f"{prefix}{value / 1e6:.{digits}f}M"
    if a >= 1e4:
        return f"{prefix}{value / 1e3:.{digits}f}K"
    return f"{prefix}{value:,.0f}"


def pct(value, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value * 100:.{digits}f}%"


# ---- plotly styling -------------------------------------------------------------------------------
def style(fig: go.Figure, height: int = 340, title: str | None = None, legend: bool = False,
          x_title: str | None = None, y_title: str | None = None) -> go.Figure:
    """Apply the shared look: transparent surface, hairline solid grid, recessive axes, text tokens."""
    fig.update_layout(
        height=height,
        title=dict(text=title, x=0, xanchor="left", font=dict(size=14, color=INK), pad=dict(l=0)) if title else None,
        font=dict(family=FONT, size=12, color=INK_2),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=52, r=18, t=44 if title else 16, b=44),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0,
                    font=dict(size=12, color=INK_2), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=GRID, font=dict(family=FONT, size=12, color=INK)),
        bargap=0.45,
    )
    try:  # rounded data-end on bars (needs a recent plotly.js)
        fig.update_layout(barcornerradius=4)
    except ValueError:
        pass
    axis = dict(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False, linecolor=GRID,
                ticks="", tickfont=dict(size=11, color=INK_2), automargin=True)
    fig.update_xaxes(**axis, title=dict(text=x_title, font=dict(size=12, color=INK_2)) if x_title else None)
    fig.update_yaxes(**axis, title=dict(text=y_title, font=dict(size=12, color=INK_2)) if y_title else None)
    return fig


def empty_figure(message: str, height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(size=14, color=MUTED))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style(fig, height=height)
