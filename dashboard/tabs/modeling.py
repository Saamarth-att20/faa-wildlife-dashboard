"""
Tab 4 - Modeling Results: everything the back end computes, read from data/processed/model_results.json
and projection_sample.parquet (see backend/contract.py for the exact shapes).

Author: Saamarth Attray

Sub-tabs:  Classification | PCA, LDA and clusters | Randomization test | Normality and outliers
If the back end has not produced its files yet, the tab shows a notice instead of failing.
"""
import math

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, dcc, html
from plotly.subplots import make_subplots

from dashboard import theme
from dashboard.components import card, control, controls, grid, kpi, notice, table
from dashboard.data import get_projection, get_results


def _p(value) -> str:
    if value is None:
        return "n/a"
    if value < 0.001:
        return "< 0.001"
    return f"{value:.3f}"


def _p_capped(value) -> str:
    """Anderson-Darling p-values from scipy are interpolated and floor at 0.01."""
    if value is None:
        return "n/a"
    return "<= 0.01" if value <= 0.0101 else f"{value:.3f}"


# ----------------------------------------------------------------------------------------------
# layout
# ----------------------------------------------------------------------------------------------
def layout():
    res = get_results()
    if res is None:
        return notice("Waiting for the back end.",
                      "model_results.json was not found in data/processed/. Run `python -m backend.pipeline` "
                      "(Zareena's pipeline) and reload this page.", kind="warn")
    return html.Div([
        html.P(f"Pipeline {res['meta'].get('pipeline_version', '?')} - generated {res['meta'].get('generated_utc', '?')[:10]}"
               f" - {res['meta']['n_rows']:,} strikes", className="fine-print"),
        dcc.Tabs(id="md-subtabs", value="clf", className="subtabs", parent_className="subtabs-parent", children=[
            dcc.Tab(label="Classification", value="clf", className="subtab", selected_className="subtab-on",
                    children=_classification_layout(res)),
            dcc.Tab(label="PCA, LDA and clusters", value="pca", className="subtab", selected_className="subtab-on",
                    children=_pca_layout(res)),
            dcc.Tab(label="Randomization test", value="rand", className="subtab", selected_className="subtab-on",
                    children=_randomization_layout(res)),
            dcc.Tab(label="Normality and outliers", value="norm", className="subtab", selected_className="subtab-on",
                    children=_normality_layout(res)),
        ]),
    ])


# ---- classification --------------------------------------------------------------------------------
def _classification_layout(res):
    c = res["classification"]
    names = [m["name"] for m in c["models"]]
    rows = [[m["name"] + ("  (best ROC-AUC)" if m["name"] == c["best_model"] else "")] +
            [f"{m['metrics'][k]:.3f}" for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")]
            for m in c["models"]]
    imp = c["feature_importance"][:12]
    fig = go.Figure(go.Bar(y=[d["feature"] for d in imp], x=[d["importance"] for d in imp], orientation="h",
                           marker_color=theme.BLUE, text=[f"{d['importance']:.0%}" for d in imp],
                           textposition="outside", cliponaxis=False, textfont=dict(size=11, color=theme.INK_2),
                           hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    theme.style(fig, height=380, x_title="Share of the random forest's total importance")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(tickformat=".0%", range=[0, max(d["importance"] for d in imp) * 1.2])
    return html.Div([
        html.P(f"Goal: predict whether a strike causes damage. {c['n_rows']:,} strikes with a known outcome "
               f"({theme.pct(c['positive_rate'])} caused damage), split {c['split']}; tested on {c['n_test']:,}.",
               className="lede"),
        controls([control("Model", dcc.RadioItems(id="md-model", className="radio", inline=True,
                                                  options=[{"label": n, "value": n} for n in names],
                                                  value=c["best_model"]))]),
        html.Div(id="md-metric-tiles", className="kpi-row"),
        grid([
            card("Confusion matrix", [dcc.Graph(id="md-confusion", config={"displaylogo": False})],
                 subtitle="Rows = what happened, columns = what the model predicted. Color shows the share of each row."),
            card("What the model relies on", [dcc.Graph(figure=fig, config={"displaylogo": False})],
                 subtitle="Random forest feature importance, one-hot categories added back together."),
            card("Model comparison", [table(["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"],
                                            rows)],
                 subtitle="Damage is rare, so accuracy alone is misleading: always predicting 'no damage' scores "
                          f"{theme.pct(1 - c['positive_rate'])}. Compare ROC-AUC and PR-AUC.", span=2),
            card("Method notes", [html.Ul(className="notes", children=[
                html.Li(f"Features used: {', '.join(c['features_used'])}."),
                html.Li("Left out to avoid leaking the answer: " + ", ".join(c["excluded_for_leakage"]) + "."),
                html.Li(c.get("note", "")),
            ])], span=2),
        ]),
    ])


# ---- PCA / LDA / clusters ---------------------------------------------------------------------------
def _scree(res):
    ev = res["pca"]["explained_variance_ratio"]
    cum = res["pca"]["cumulative_variance"]
    pcs = [f"PC{i + 1}" for i in range(len(ev))]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=pcs, y=ev, marker_color=theme.BLUE, name="Each component",
                         hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=pcs, y=cum, mode="lines+markers", name="Cumulative",
                             line=dict(color=theme.ORANGE, width=2),
                             marker=dict(size=8, color=theme.ORANGE, line=dict(color=theme.SURFACE, width=2)),
                             hovertemplate="%{x}: %{y:.1%} cumulative<extra></extra>"))
    theme.style(fig, height=320, legend=True, y_title="Share of variance")
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
    fig.update_xaxes(showgrid=False)
    return fig


def _loadings(res):
    load = res["pca"]["loadings"]
    feats = list(load)
    n_pc = min(4, len(next(iter(load.values()))))
    z = np.array([[load[f][i] for i in range(n_pc)] for f in feats])
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"PC{i + 1}" for i in range(n_pc)], y=feats, colorscale=theme.DIVERGING, zmin=-1, zmax=1,
        text=[[f"{v:+.2f}" for v in row] for row in z], texttemplate="%{text}", textfont=dict(size=12, color=theme.INK),
        xgap=2, ygap=2, hovertemplate="%{y} on %{x}: %{z:+.2f}<extra></extra>",
        colorbar=dict(thickness=10, outlinewidth=0, tickfont=dict(size=11, color=theme.INK_2))))
    theme.style(fig, height=320)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="top")
    return fig


def _k_selection(res):
    scan = res["clustering"]["k_scan"]
    best = res["clustering"]["k"]
    ks = [d["k"] for d in scan]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Silhouette (higher is better)", "Inertia (look for the elbow)"),
                        horizontal_spacing=0.14)
    for col, key in ((1, "silhouette"), (2, "inertia")):
        fig.add_trace(go.Scatter(
            x=ks, y=[d[key] for d in scan], mode="lines+markers", line=dict(color=theme.BLUE, width=2),
            marker=dict(size=[12 if k == best else 8 for k in ks], color=[theme.ORANGE if k == best else theme.BLUE for k in ks],
                        line=dict(color=theme.SURFACE, width=2)),
            hovertemplate=f"k = %{{x}}<br>{key}: %{{y:,.3f}}<extra></extra>"), row=1, col=col)
    theme.style(fig, height=300)
    fig.update_xaxes(title_text="Number of clusters (k)", dtick=1, showgrid=False)
    fig.update_annotations(font=dict(size=12, color=theme.INK_2))
    return fig


def _lda_figure(res):
    proj = get_projection()
    if proj is None or "LD1" not in proj:
        return theme.empty_figure("LD1 scores not available", 300)
    d = proj.dropna(subset=["LD1", "STRIKE_CAUSED_DAMAGE"])
    lo, hi = np.percentile(d["LD1"], [0.5, 99.5])
    bins = np.linspace(lo, hi, 41)
    fig = go.Figure()
    for flag, name, color in [(0.0, "No damage", theme.NO_DAMAGE), (1.0, "Caused damage", theme.DAMAGE)]:
        vals = d.loc[d["STRIKE_CAUSED_DAMAGE"] == flag, "LD1"]
        hist, edges = np.histogram(vals.clip(lo, hi), bins=bins)
        fig.add_trace(go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=hist / hist.sum(), name=name, marker_color=color,
                             opacity=0.65, hovertemplate=f"{name}<br>LD1 %{{x:.2f}}: %{{y:.1%}} of group<extra></extra>"))
    theme.style(fig, height=300, legend=True, x_title="LD1 score", y_title="Share of group")
    fig.update_layout(barmode="overlay", bargap=0.05)
    fig.update_yaxes(tickformat=".0%")
    return fig


def _pca_layout(res):
    pca, clu, lda = res["pca"], res["clustering"], res["lda"]
    feats = pca["features"]
    prof_rows = []
    for p in clu["profiles"]:
        m = p["means"]
        prof_rows.append([p["cluster"], f"{p['n']:,} ({theme.pct(p['share'], 0)})", theme.pct(p["damage_rate"]),
                          p.get("top_phase") or "n/a", p["top_species_group"],
                          f"{m.get('HEIGHT', float('nan')):,.0f}", f"{m.get('SPEED', float('nan')):,.0f}",
                          f"{m.get('DISTANCE', float('nan')):,.1f}", f"{m.get('NUM_ENGS', float('nan')):,.1f}",
                          f"{m.get('AC_MASS', float('nan')):,.1f}", f"{m.get('NUM_STRUCK', float('nan')):,.1f}"])
    coef_rows = [[f, f"{v:+.3f}"] for f, v in sorted(lda["coefficients"].items(), key=lambda kv: -abs(kv[1]))]
    return html.Div([
        html.P(f"{pca['n_rows']:,} strikes have every one of {', '.join(feats)} reported. {pca['note']}", className="lede"),
        grid([
            card("PCA projection", [
                controls([control("Show", dcc.RadioItems(
                    id="md-pca-view", className="radio", inline=True, value="outcome",
                    options=[{"label": "By outcome", "value": "outcome"},
                             {"label": "By cluster (one panel each)", "value": "cluster"},
                             {"label": "Multivariate outliers", "value": "outlier"}]))]),
                dcc.Graph(id="md-pca-scatter", config={"displaylogo": False}),
            ], subtitle="Each dot is a strike, placed by its first two principal components (random sample of 20,000).",
                 span=2),
            card("How much each component explains", [dcc.Graph(figure=_scree(res), config={"displaylogo": False})]),
            card("What each component is made of", [dcc.Graph(figure=_loadings(res), config={"displaylogo": False})],
                 subtitle="Loadings: blue = negative, red = positive. Read down a column to name the component."),
            card(f"K-means chose k = {clu['k']}", [dcc.Graph(figure=_k_selection(res), config={"displaylogo": False})],
                 subtitle=f"Best silhouette: {clu['silhouette']:.2f}. The orange dot marks the chosen k.", span=2),
            card("Cluster profiles", [table(
                ["Cluster", "Strikes", "Damage rate", "Typical phase", "Top species group", "Height (ft)",
                 "Speed (kt)", "Dist. (nm)", "Engines", "Mass class", "Animals"], prof_rows,
                align=["l", "r", "r", "l", "l", "r", "r", "r", "r", "r", "r"])],
                 subtitle="Average of each feature inside the cluster, in original units.", span=2),
            card("Linear discriminant analysis", [dcc.Graph(figure=_lda_figure(res), config={"displaylogo": False})],
                 subtitle=f"One discriminant axis (two classes). Training accuracy {theme.pct(lda['training_accuracy'])} "
                          f"vs. {theme.pct(lda['majority_class_baseline'])} for always guessing 'no damage', so read the "
                          "overlap of the two distributions, not the accuracy."),
            card("LDA coefficients", [table(["Feature (standardized)", "Coefficient"], coef_rows)],
                 subtitle="Larger magnitude = pulls harder on the discriminant axis."),
        ]),
    ])


# ---- randomization ------------------------------------------------------------------------------------
def _randomization_layout(res):
    tests = res["randomization"]["tests"]
    return html.Div([
        html.P(res["randomization"]["method"] + ".", className="lede"),
        controls([control("Test", dcc.Dropdown(id="md-rand-test", clearable=False,
                                               options=[{"label": t["name"], "value": i} for i, t in enumerate(tests)],
                                               value=0), width="360px")]),
        html.Div(id="md-rand-tiles", className="kpi-row"),
        grid([card("Permutation distribution", [dcc.Graph(id="md-rand-graph", config={"displaylogo": False})],
                   subtitle="Gray bars: the difference in damage rate we would see by chance if the label did not "
                            "matter. Orange line: what we actually observed.", span=2)]),
    ])


# ---- normality + outliers ------------------------------------------------------------------------------
def _normality_layout(res):
    nm, out = res["normality"], res["outliers"]
    feats = nm["features"]
    rows = []
    for name, entry in feats.items():
        r = entry["raw"]
        rows.append([name, f"{r['n']:,}", f"{r['skew']:.2f}", _p(r["shapiro_p"]), _p(r["dagostino_p"]),
                     _p_capped(r["anderson_p"]), "Normal" if r["normal_by_all_three"] else "Rejected"])
    out_rows = [[name, f"{v['n_valid']:,}", f"{v['iqr_outliers']:,}", f"{v['z_outliers']:,}",
                 f"{v['iqr_upper_bound']:,.0f}", f"{v['max_value']:,.0f}"] for name, v in out["features"].items()]
    top = [[str(e["year"]), e["state"] or "n/a", e["aircraft"] or "n/a", e["species"] or "n/a",
            theme.compact(e["total_cost"], "$", 2)] for e in out["top_cost_events"]]
    iso = out.get("isolation_forest")
    return html.Div([
        html.P("Normality is tested three ways (" + "; ".join(nm["methods"]) + f") at alpha = {nm['alpha']}. "
               + nm.get("note", ""), className="lede"),
        grid([
            card("Normality tests", [table(
                ["Feature", "n", "Skew", "Shapiro-Wilk p", "D'Agostino p", "Anderson-Darling p", "Verdict"], rows,
                align=["l", "r", "r", "r", "r", "r", "l"])],
                 subtitle="Verdict is 'Normal' only if none of the three tests rejects.", span=2),
            card("Q-Q plot", [
                controls([
                    control("Feature", dcc.Dropdown(id="md-qq-feature", clearable=False,
                                                    options=[{"label": k, "value": k} for k in feats],
                                                    value=next(iter(feats))), width="300px"),
                    control("Scale", dcc.RadioItems(id="md-qq-scale", className="radio", inline=True, value="raw",
                                                    options=[{"label": "Original", "value": "raw"},
                                                             {"label": "log(1 + x)", "value": "log1p"}])),
                ]),
                dcc.Graph(id="md-qq", config={"displaylogo": False}),
            ], subtitle="Points on the line = normal. A curve away from it = skewed or heavy-tailed.", span=2),
            card("Outlier detection", [table(
                ["Feature", "Valid values", "IQR outliers", "|z| outliers", "IQR upper fence", "Maximum"], out_rows),
                html.P("HEIGHT, DISTANCE and the cost columns are tested on a log(1 + x) scale because they are heavily "
                       "skewed, so their fences are shown converted back to original units (a very large fence just "
                       "means almost nothing counts as unusually high on the log scale). DISTANCE is 0 for most strikes, "
                       "so its IQR is 0 and any nonzero distance is flagged.", className="fine-print")],
                 subtitle=out.get("policy", ""), span=2),
            card("Isolation forest", [
                html.Div(className="kpi-row", children=[
                    kpi("Flagged as multivariate outliers", f"{iso['n_flagged']:,}",
                        f"of {iso['n_rows']:,} complete rows ({theme.pct(iso['contamination'], 0)} contamination)")
                ]) if iso else html.P("Not computed.")]),
            card("Costliest strikes", [table(["Year", "State", "Aircraft", "Species", "Cost"], top,
                                             align=["l", "l", "l", "l", "r"])],
                 subtitle="Kept in the data: these are real, high-severity events."),
        ]),
    ])


# ----------------------------------------------------------------------------------------------
# callbacks
# ----------------------------------------------------------------------------------------------
def _confusion_figure(model):
    cm = np.array(model["confusion_matrix"]["matrix"], dtype=float)
    labels = model["confusion_matrix"]["labels"]
    row_pct = cm / cm.sum(axis=1, keepdims=True)
    text = [[f"{int(cm[i, j]):,}<br>{row_pct[i, j]:.1%}" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(
        z=row_pct, x=[f"Predicted: {l}" for l in labels], y=[f"Actual: {l}" for l in labels], zmin=0, zmax=1,
        colorscale=theme.BLUE_SCALE, showscale=False, xgap=3, ygap=3, text=text, texttemplate="%{text}",
        textfont=dict(size=14), hovertemplate="%{y}, %{x}<br>%{text}<extra></extra>"))
    # white text on dark cells, ink on light cells
    for i in range(2):
        for j in range(2):
            fig.add_annotation(x=j, y=i, text=text[i][j], showarrow=False,
                               font=dict(size=14, color="white" if row_pct[i, j] > 0.5 else theme.INK))
    fig.update_traces(texttemplate="")
    theme.style(fig, height=380)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False)
    return fig


def _qq_figure(res, feature, scale):
    entry = res["normality"]["features"][feature]
    block = entry.get(scale) or entry["raw"]
    qq = block["qq"]
    x, y = np.array(qq["theoretical"]), np.array(qq["sample"])
    q1, q3 = np.quantile(y, 0.25), np.quantile(y, 0.75)
    t1, t3 = -0.6745, 0.6745
    slope = (q3 - q1) / (t3 - t1)
    intercept = q1 - slope * t1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[x.min(), x.max()], y=[intercept + slope * x.min(), intercept + slope * x.max()],
                             mode="lines", line=dict(color=theme.MUTED, width=1.5), name="Normal reference",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Data quantiles",
                             marker=dict(color=theme.BLUE, size=7, line=dict(color=theme.SURFACE, width=1)),
                             hovertemplate="Normal quantile %{x:.2f}<br>Data quantile %{y:,.2f}<extra></extra>"))
    theme.style(fig, height=380, legend=True, x_title="Theoretical normal quantile",
                y_title=f"{feature} quantile" + (" (log(1 + x))" if block is entry.get("log1p") else ""))
    return fig


def register(app):
    @app.callback(Output("md-metric-tiles", "children"), Output("md-confusion", "figure"), Input("md-model", "value"))
    def _model(name):
        res = get_results()
        model = next(m for m in res["classification"]["models"] if m["name"] == name)
        m = model["metrics"]
        pos = res["classification"]["positive_rate"]
        tiles = [
            kpi("Accuracy", theme.pct(m["accuracy"]), "share of all predictions that were right"),
            kpi("Precision", theme.pct(m["precision"]), "of predicted damage, share that was real"),
            kpi("Recall", theme.pct(m["recall"]), "of real damage, share the model caught", accent=True),
            kpi("F1", f"{m['f1']:.3f}", "balance of precision and recall"),
            kpi("ROC-AUC", f"{m['roc_auc']:.3f}", "0.5 = coin flip, 1.0 = perfect"),
            kpi("PR-AUC", f"{m['pr_auc']:.3f}", f"chance level here = {pos:.3f}"),
        ]
        return tiles, _confusion_figure(model)

    @app.callback(Output("md-pca-scatter", "figure"), Input("md-pca-view", "value"))
    def _pca(view):
        proj = get_projection()
        res = get_results()
        if proj is None:
            return theme.empty_figure("projection_sample.parquet not found", 520)
        ev = res["pca"]["explained_variance_ratio"]
        x_title, y_title = f"PC1 ({ev[0]:.0%} of variance)", f"PC2 ({ev[1]:.0%} of variance)"
        hover = "%{customdata[0]}<br>%{customdata[1]}<br>PC1 %{x:.2f}, PC2 %{y:.2f}<extra></extra>"
        custom = np.stack([proj["SPECIES_GROUP"].astype(str), proj["PHASE_OF_FLIGHT"].astype(str)], axis=-1)

        if view == "cluster":
            names = sorted(proj["CLUSTER"].unique())
            cols = min(3, len(names))
            rows = math.ceil(len(names) / cols)
            fig = make_subplots(rows=rows, cols=cols, subplot_titles=names, shared_xaxes=True, shared_yaxes=True,
                                horizontal_spacing=0.03, vertical_spacing=0.12)
            backdrop = proj.sample(n=min(1500, len(proj)), random_state=3)
            for i, name in enumerate(names):
                r, c = divmod(i, cols)
                sel = proj["CLUSTER"] == name
                fig.add_trace(go.Scatter(x=backdrop["PC1"], y=backdrop["PC2"], mode="markers", hoverinfo="skip",
                                         marker=dict(color=theme.NEUTRAL, size=3, opacity=0.5)), row=r + 1, col=c + 1)
                fig.add_trace(go.Scatter(x=proj.loc[sel, "PC1"], y=proj.loc[sel, "PC2"], mode="markers",
                                         marker=dict(color=theme.BLUE, size=4, opacity=0.6),
                                         customdata=custom[sel.to_numpy()], hovertemplate=hover), row=r + 1, col=c + 1)
            theme.style(fig, height=300 * rows + 60)
            fig.update_annotations(font=dict(size=12, color=theme.INK_2))
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)
            fig.update_xaxes(title_text=x_title, row=rows)
            fig.update_yaxes(title_text=y_title, col=1)
            return fig

        fig = go.Figure()
        if view == "outcome":
            groups = [("Outcome not reported", proj["STRIKE_CAUSED_DAMAGE"].isna(), theme.NEUTRAL, 4, 0.4),
                      ("No damage", proj["STRIKE_CAUSED_DAMAGE"] == 0, theme.NO_DAMAGE, 5, 0.5),
                      ("Caused damage", proj["STRIKE_CAUSED_DAMAGE"] == 1, theme.DAMAGE, 5, 0.6)]
        else:
            flag = proj["ISO_OUTLIER"].astype(bool) if "ISO_OUTLIER" in proj else proj["PC1"] != proj["PC1"]
            groups = [("Typical", ~flag, theme.NEUTRAL, 4, 0.4), ("Isolation-forest outlier", flag, theme.RED, 6, 0.8)]
        for name, mask, color, size, opacity in groups:
            m = mask.to_numpy()
            fig.add_trace(go.Scatter(x=proj.loc[m, "PC1"], y=proj.loc[m, "PC2"], mode="markers", name=name,
                                     marker=dict(color=color, size=size, opacity=opacity),
                                     customdata=custom[m], hovertemplate=hover))
        theme.style(fig, height=520, legend=True, x_title=x_title, y_title=y_title)
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        return fig

    @app.callback(Output("md-rand-tiles", "children"), Output("md-rand-graph", "figure"), Input("md-rand-test", "value"))
    def _rand(idx):
        t = get_results()["randomization"]["tests"][idx]
        null = np.array(t["null_distribution"])
        obs = t["observed_diff"]
        hist, edges = np.histogram(null, bins=45)
        fig = go.Figure(go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=hist, marker_color=theme.NEUTRAL,
                               width=np.diff(edges) * 0.95, hovertemplate="Difference %{x:+.2%}<br>%{y:,} shuffles<extra></extra>"))
        fig.add_vline(x=obs, line=dict(color=theme.ORANGE, width=3),
                      annotation_text=f"Observed {obs:+.2%}", annotation_position="top",
                      annotation_font=dict(size=12, color=theme.INK))
        theme.style(fig, height=380, x_title=f"Difference in damage rate ({t['group_a']['label']} minus {t['group_b']['label']})",
                    y_title=f"Shuffles (of {len(null):,} shown)")
        fig.update_xaxes(tickformat="+.1%")
        verdict = ("Unlikely to be chance (p < 0.05)." if t["p_value"] < 0.05 else "Consistent with chance (p >= 0.05).")
        a, b = t["group_a"], t["group_b"]
        tiles = [
            kpi(a["label"], theme.pct(a["damage_rate"]), f"damage rate, {a['n']:,} strikes"),
            kpi(b["label"], theme.pct(b["damage_rate"]), f"damage rate, {b['n']:,} strikes"),
            kpi("Difference", f"{obs:+.2%}", t["question"]),
            kpi("p-value", f"{t['p_value']:.4f}", f"{verdict} {t['n_permutations']:,} permutations, {t['alternative']}.",
                accent=True),
        ]
        return tiles, fig

    @app.callback(Output("md-qq", "figure"), Input("md-qq-feature", "value"), Input("md-qq-scale", "value"))
    def _qq(feature, scale):
        return _qq_figure(get_results(), feature, scale)
