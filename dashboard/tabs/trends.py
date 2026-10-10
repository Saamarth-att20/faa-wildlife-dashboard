"""
Tab 2 - Trends: strikes and damage rate over time, seasonality, and severity by time of day.
Filterable by state, species group and year range.

Author: Saamarth Attray

Note: the source file ends in 2024 (2025 has only 16 strikes), so the year slider defaults to 2000-2024.
"""
import calendar

import plotly.graph_objects as go
from dash import Input, Output, dcc, html

from backend.contract import DAMAGE_LEVEL_ORDER, TIME_OF_DAY_ORDER
from dashboard import theme
from dashboard.components import card, control, controls, grid, kpi
from dashboard.data import damage_rate, filter_df, get_df, species_group_options, state_options

DEFAULT_YEARS = [2000, 2024]


def layout():
    df = get_df()
    lo, hi = int(df["INCIDENT_YEAR"].min()), int(df["INCIDENT_YEAR"].max())
    return html.Div([
        controls([
            control("State", dcc.Dropdown(id="tr-states", options=state_options(), multi=True,
                                          placeholder="All states"), width="300px"),
            control("Species group", dcc.Dropdown(id="tr-groups", options=species_group_options(), multi=True,
                                                  placeholder="All species groups"), width="300px"),
            control("Years", dcc.RangeSlider(id="tr-years", min=lo, max=hi, step=1, value=DEFAULT_YEARS,
                                             marks={y: str(y) for y in range(lo, hi + 1, 5)},
                                             tooltip={"placement": "bottom", "always_visible": False})),
        ]),
        html.Div(id="tr-summary", className="kpi-row"),
        grid([
            card("Strikes per year", [dcc.Graph(id="tr-year-count", config={"displaylogo": False})],
                 subtitle="Count of reported strikes. The 2020 dip is the COVID-19 drop in air traffic."),
            card("Share of strikes that caused damage", [dcc.Graph(id="tr-year-rate", config={"displaylogo": False})],
                 subtitle="Among strikes with a known outcome. The share of strikes with a reported outcome also fell "
                          "from about 80% to 60% over this period."),
            card("Strikes by month", [dcc.Graph(id="tr-month", config={"displaylogo": False})],
                 subtitle="Strikes peak in late summer and fall, when birds migrate."),
            card("Damaging strikes by time of day", [dcc.Graph(id="tr-tod", config={"displaylogo": False})],
                 subtitle="Strikes that caused damage, split by severity. Excludes no-damage and unknown outcomes."),
        ]),
        html.P("Source file ends in 2024: 2025 holds only 16 strikes, so it is left out of the default range.",
               className="fine-print"),
    ])


def _line(x, y, name, color, y_title, fmt, extra=None, y_range=None):
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, width=2, shape="linear"), name=name,
                               customdata=extra, hovertemplate=f"%{{x}}: %{{y:{fmt}}}" +
                               ("<br>%{customdata:,} known outcomes" if extra is not None else "") + "<extra></extra>"))
    if len(x):  # end dot with a surface ring so it stays legible
        fig.add_trace(go.Scatter(x=[x[-1]], y=[y[-1]], mode="markers", hoverinfo="skip",
                                 marker=dict(color=color, size=10, line=dict(color=theme.SURFACE, width=2))))
    theme.style(fig, height=320, y_title=y_title)
    fig.update_xaxes(dtick=2, showgrid=False)
    fig.update_yaxes(rangemode="tozero", tickformat=fmt if "%" in fmt else "~s")
    if y_range:
        fig.update_yaxes(range=y_range)
    return fig


def register(app):
    @app.callback(
        Output("tr-summary", "children"), Output("tr-year-count", "figure"), Output("tr-year-rate", "figure"),
        Output("tr-month", "figure"), Output("tr-tod", "figure"),
        Input("tr-states", "value"), Input("tr-groups", "value"), Input("tr-years", "value"))
    def _update(states, groups, years):
        f = filter_df(states, groups, years)
        if f.empty:
            empty = theme.empty_figure("No strikes match these filters", 320)
            return [kpi("Strikes in selection", "0")], empty, empty, empty, empty

        known_n = int(f["STRIKE_CAUSED_DAMAGE"].notna().sum())
        summary = [
            kpi("Strikes in selection", f"{len(f):,}", f"{years[0]}-{years[1]}"),
            kpi("Caused damage", theme.pct(damage_rate(f)), f"of {known_n:,} with a known outcome", accent=True),
            kpi("States / territories", f"{f['STATE'].nunique():,}"),
            kpi("Species groups", f"{f['SPECIES_GROUP'].nunique():,}"),
        ]

        by_year = f.groupby("INCIDENT_YEAR", observed=True).agg(
            n=("INDEX_NR", "size"), known=("STRIKE_CAUSED_DAMAGE", "count"), damaged=("STRIKE_CAUSED_DAMAGE", "sum"))
        by_year["rate"] = by_year["damaged"] / by_year["known"].where(by_year["known"] > 0)
        by_year = by_year[by_year["known"] >= 0]
        years_x = [int(y) for y in by_year.index]

        fig_count = _line(years_x, by_year["n"].tolist(), "Strikes", theme.BLUE, "Strikes", ",d")
        if 2020 in years_x and len(years_x) > 3:
            fig_count.add_annotation(x=2020, y=int(by_year.loc[2020, "n"]), text="COVID-19", showarrow=True,
                                     arrowhead=0, arrowcolor=theme.MUTED, ax=0, ay=-34,
                                     font=dict(size=11, color=theme.INK_2))
        rate_years = by_year[by_year["known"] >= 30]  # avoid rates built on a handful of strikes
        fig_rate = _line([int(y) for y in rate_years.index], rate_years["rate"].tolist(), "Damage rate", theme.DAMAGE,
                         "Caused damage", ".0%", extra=rate_years["known"].tolist())

        by_month = f.groupby("INCIDENT_MONTH", observed=True).size().reindex(range(1, 13), fill_value=0)
        fig_month = go.Figure(go.Bar(x=[calendar.month_abbr[m] for m in by_month.index], y=by_month.values,
                                     marker_color=theme.BLUE, hovertemplate="%{x}: %{y:,}<extra></extra>"))
        theme.style(fig_month, height=320, y_title="Strikes")
        fig_month.update_xaxes(showgrid=False)
        fig_month.update_yaxes(tickformat="~s")

        dmg = f[f["DAMAGE_LEVEL"].isin(["Minor (uncertain)", "Minor", "Substantial", "Destroyed"])]
        fig_tod = go.Figure()
        if dmg.empty:
            fig_tod = theme.empty_figure("No damaging strikes match these filters", 320)
        else:
            ct = (dmg.assign(TIME_OF_DAY=dmg["TIME_OF_DAY"].astype("object").fillna("Not reported"),
                             DAMAGE_LEVEL=dmg["DAMAGE_LEVEL"].astype(str))
                  .groupby(["TIME_OF_DAY", "DAMAGE_LEVEL"]).size().unstack(fill_value=0))
            order = [t for t in TIME_OF_DAY_ORDER + ["Not reported"] if t in ct.index]
            ct = ct.reindex(order)
            for level in [lv for lv in DAMAGE_LEVEL_ORDER if lv in ct.columns]:
                fig_tod.add_trace(go.Bar(x=ct.index, y=ct[level], name=level,
                                         marker=dict(color=theme.SEVERITY_COLORS[level],
                                                     line=dict(color=theme.SURFACE, width=2)),
                                         hovertemplate=f"{level}<br>%{{x}}: %{{y:,}}<extra></extra>"))
            theme.style(fig_tod, height=320, legend=True, y_title="Damaging strikes")
            fig_tod.update_layout(barmode="stack")
            fig_tod.update_xaxes(showgrid=False)
            fig_tod.update_yaxes(tickformat="~s")
        return summary, fig_count, fig_rate, fig_month, fig_tod
