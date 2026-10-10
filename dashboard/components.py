"""
Small reusable layout pieces (cards, KPI tiles, tables, notices).

Author: Saamarth Attray
"""
from dash import dcc, html


def kpi(label: str, value: str, note: str | None = None, accent: bool = False):
    """Stat tile: label (sentence case), big value, optional one-line note."""
    return html.Div(
        className="kpi" + (" kpi-accent" if accent else ""),
        children=[
            html.Div(label, className="kpi-label"),
            html.Div(value, className="kpi-value"),
            html.Div(note, className="kpi-note") if note else None,
        ],
    )


def card(title: str | None, children, subtitle: str | None = None, span: int = 1, graph_id: str | None = None):
    """White card. `span=2` makes it full width in the 2-column grid."""
    header = []
    if title:
        header.append(html.H3(title, className="card-title"))
    if subtitle:
        header.append(html.P(subtitle, className="card-sub"))
    if graph_id:  # convenience: a card that holds one graph
        body = [dcc.Graph(id=graph_id, config={"displaylogo": False, "responsive": True})]
    elif isinstance(children, list):
        body = children
    else:
        body = [children]
    return html.Section(className=f"card span-{span}", children=header + body)


def grid(children):
    return html.Div(className="grid", children=children)


def controls(children):
    return html.Div(className="controls", children=children)


def control(label: str, component, width: str | None = None):
    style = {"flex": f"0 0 {width}"} if width else None
    return html.Div(className="control", style=style, children=[html.Label(label, className="control-label"), component])


def notice(title: str, body: str, kind: str = "info"):
    return html.Div(className=f"notice notice-{kind}", children=[html.Strong(title), html.Span(" " + body)])


def table(headers, rows, align=None, caption: str | None = None):
    """Plain HTML table. `rows` is a list of lists of strings / components. `align` = list of 'l' / 'r'."""
    align = align or ["l"] + ["r"] * (len(headers) - 1)
    head = html.Thead(html.Tr([html.Th(h, className=f"a-{a}") for h, a in zip(headers, align)]))
    body = html.Tbody([html.Tr([html.Td(c, className=f"a-{a}") for c, a in zip(row, align)]) for row in rows])
    children = [head, body]
    if caption:
        children.insert(0, html.Caption(caption))
    return html.Div(className="table-wrap", children=html.Table(className="table", children=children))
