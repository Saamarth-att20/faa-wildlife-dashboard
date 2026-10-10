"""
FAA Wildlife Strikes dashboard - entry point.
CS/STAT 3654 term project (Front End: Saamarth Attray, Back End: Zareena Yussouff).

Run locally:   python app.py            (http://127.0.0.1:8050)
Production:    gunicorn app:server      (see Dockerfile)
"""
import os

from dash import Dash, Input, Output, dcc, html

from dashboard.tabs import modeling, overview, risk, trends

app = Dash(__name__, title="Wildlife strikes on aircraft", suppress_callback_exceptions=True,
           meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
server = app.server  # gunicorn entry point

TABS = {
    "overview": ("Overview", overview),
    "trends": ("Trends", trends),
    "risk": ("Risk factors", risk),
    "modeling": ("Modeling results", modeling),
}

app.layout = html.Div(className="page", children=[
    html.Header(className="header", children=[
        html.H1("Wildlife strikes on aircraft"),
        html.P("FAA National Wildlife Strike Database, 2000-2024. Which conditions turn a bird strike into damage?",
               className="header-sub"),
    ]),
    dcc.Tabs(id="main-tabs", value="overview", className="tabs", parent_className="tabs-parent", children=[
        dcc.Tab(label=label, value=key, className="tab", selected_className="tab-on")
        for key, (label, _) in TABS.items()]),
    html.Main(id="tab-body", className="tab-body"),
    html.Footer("Data: FAA Wildlife Strike Database. Damage rate = damaging strikes / strikes with a known outcome.",
                className="footer"),
])


@app.callback(Output("tab-body", "children"), Input("main-tabs", "value"))
def _render(tab):
    return TABS[tab][1].layout()


for _, module in TABS.values():
    module.register(app)

if __name__ == "__main__":
    app.run(debug=os.environ.get("DEBUG") == "1", host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))
