"""TrackCast dashboard.  Run:  streamlit run app/streamlit_app.py

Six tabs, kept to what a 10-minute demo needs: overview, live route forecast (the hurdle model), the delay
cascade, the model's test-set results, results per train class / corridor with the top bottleneck junctions,
and the data-science concepts the project uses. The full detail is in reports/ and docs/REPORT.md.
All logic lives in src/trackcast (forecast.py builds features with the same code as training; serving.py
finds, checks and loads the artifacts; concepts.py holds the concept list); this file is layout, caching and
charts.

Artifacts: full-year and dev-sample builds when present, else the demo bundle (python -m trackcast.serving demo);
TRACKCAST_DEMO=1 serves the bundle only, as on Streamlit Community Cloud.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

try:
    import trackcast  # noqa: F401  (installed with pip install -e .)
except ImportError:      # hosted runs (Streamlit Community Cloud) install requirements only, not the package
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trackcast import concepts as K  # noqa: E402
from trackcast import config as C  # noqa: E402
from trackcast import forecast as FC  # noqa: E402
from trackcast import profiles as P  # noqa: E402
from trackcast import serving as S  # noqa: E402

st.set_page_config(page_title="TrackCast: train delay forecasts", page_icon="🚆", layout="wide")

# Reference data-viz palette: categorical slots 1-3 (validated all-pairs), blue ordinal ramp, muted chrome.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
ORDINAL = ["#86b6ef", "#3987e5", "#256abf", "#184f95"]   # cascade buckets, light -> dark = small -> large
MUTED = "#898781"
LATE = C.SIGNIFICANT_DELAY_MIN


# --------------------------------------------------------------------------------------------------
# Artifacts. Every loader takes the files' fingerprint (mtime, size) as an argument, so it is part of the
# cache key: a rebuilt artifact is reloaded on the next rerun instead of being served stale until restart.
# Shared, read-only objects (model, timetable, tables) use cache_resource: one copy per server, not a
# deserialised copy per rerun (cache_data cost ~50 ms a rerun for the full profiles). Never mutate them in
# place: derive with .copy(), .assign() or a filter.
# --------------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading profiles...", max_entries=4)
def load_profiles(directory: str, fp: tuple) -> dict[str, pd.DataFrame]:
    return S.load_tables(directory, P.TABLES)


@st.cache_resource(show_spinner="Loading results...", max_entries=4)
def load_results(directory: str, fp: tuple) -> dict[str, pd.DataFrame] | None:
    try:
        return S.load_tables(directory, C.RESULTS_TABLES)
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner="Loading and checking the model...", max_entries=2)
def load_model(path: str, fp: tuple) -> S.LoadedModel:
    return S.load_model(path)   # never raises: a broken model is cached as an error until its file changes


@st.cache_resource(show_spinner="Loading timetable...", max_entries=2)
def load_timetable(_source: S.Source, fp: tuple):
    return _source.load_timetable()


@st.cache_data(max_entries=8)
def load_report(path: str, fp: tuple) -> dict | None:
    return S.load_json(path)


@st.cache_data(show_spinner="Forecasting the rest of the route...", max_entries=256)
def cached_forecast(model_key: tuple, tt_key: tuple, train_no: int, date: dt.date,
                    observed: tuple[tuple[int, float], ...], _model, _tt) -> pd.DataFrame:
    """One recursive route forecast (~0.2 s a stop). Keyed on the inputs and the artifacts' fingerprints:
    Streamlit reruns the whole script on any widget change, in any tab, so without this every slider
    move elsewhere would recompute the forecast."""
    return FC.forecast_route(_model, _tt, train_no, date, dict(observed))


def model_report() -> dict | None:
    return load_report(str(C.MODELS_JSON), S.fingerprint(C.MODELS_JSON))


# --------------------------------------------------------------------------------------------------
# Chart helpers (thin 2px lines, 4px rounded bar ends, recessive axes, hover tooltips on every mark)
# --------------------------------------------------------------------------------------------------
def _legend():
    return alt.Legend(orient="top", title=None, symbolStrokeWidth=2, labelLimit=260)


def hover_lines(df: pd.DataFrame, x: str, y: str, series: str | None = None, domain=None, colors=None,
                tooltip=None, x_title=None, y_title=None, height: int = 280, dashed: list[str] | None = None):
    """Line chart with a crosshair: the nearest x is highlighted and its values shown in a tooltip."""
    xf = x.split(":")[0]
    color = (alt.Color(series, scale=alt.Scale(domain=domain, range=colors), legend=_legend())
             if series else alt.value(BLUE))
    base = alt.Chart(df).encode(x=alt.X(x, title=x_title), y=alt.Y(y, title=y_title))
    line_enc = {"color": color}
    if series and dashed:
        line_enc["strokeDash"] = alt.condition(alt.FieldOneOfPredicate(series.split(":")[0], dashed),
                                               alt.value([5, 4]), alt.value([1, 0]))
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=[xf], empty=False)
    lines = base.mark_line(strokeWidth=2).encode(**line_enc)
    points = base.mark_point(size=70, filled=True).encode(
        color=color, opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
        tooltip=tooltip or [x, y]).add_params(nearest)
    rule = alt.Chart(df).mark_rule(color=MUTED, strokeWidth=1).encode(x=x).transform_filter(nearest)
    return (lines + rule + points).properties(height=height)


def bars(df: pd.DataFrame, x: str, y: str, tooltip=None, x_title=None, y_title=None, height: int = 220,
         sort=None, horizontal: bool = False):
    """Single-series bar chart (blue), rounded data end, per-bar tooltip."""
    if horizontal:
        mark = alt.Chart(df).mark_bar(color=BLUE, cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        enc = {"y": alt.Y(x, title=x_title, sort=sort), "x": alt.X(y, title=y_title)}
    else:
        mark = alt.Chart(df).mark_bar(color=BLUE, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        enc = {"x": alt.X(x, title=x_title, sort=sort), "y": alt.Y(y, title=y_title)}
    return mark.encode(**enc, tooltip=tooltip or [x, y]).properties(height=height)


def show(chart):
    st.altair_chart(chart, width="stretch")


@st.cache_resource(max_entries=4)
def train_labels(directory: str, fp: tuple, _trains: pd.DataFrame) -> dict[int, str]:
    """{train_no: "12723 · TELANGANA EXP"}, most-run first. Built once per profiles file, vectorised: building
    it per rerun with iterrows took ~0.4 s of every interaction (7k trains, two tabs)."""
    t = _trains.sort_values("n_runs", ascending=False)
    no = t["train_no"].astype("int64").astype("string")
    name = t["train_name"].astype("string").fillna("").str.strip() if "train_name" in t else no.str.slice(0, 0)
    return dict(zip(t["train_no"].astype("int64").tolist(), no.where(name == "", no + " · " + name).tolist()))


# --------------------------------------------------------------------------------------------------
# Tabs (each kept to its key chart and numbers; the full tables are in reports/)
# --------------------------------------------------------------------------------------------------
FAMILY = {"rf": "Random forest", "logreg": "Logistic regression", "hgb": "Gradient boosting", "ridge": "Ridge"}


def _weighted(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Mean delay and % late per `by`, weighted by arrivals."""
    g = df.assign(s=df["mean_delay"] * df["n_obs"], l=df["late_pct"] * df["n_obs"]).groupby(by)[
        ["n_obs", "s", "l"]].sum().reset_index()
    return g.assign(mean_delay=g["s"] / g["n_obs"], late_pct=g["l"] / g["n_obs"])


def tab_overview(pr: dict, report: dict | None):
    d = pr["by_date"]
    n_obs = d["n_obs"].sum()
    st.markdown("**The network** (one year, Feb 2025 to Feb 2026)")
    c = st.columns(4)
    c[0].metric("Mean arrival delay", f"{(d['mean_delay'] * d['n_obs']).sum() / n_obs:.1f} min")
    c[1].metric(f"Arrivals late > {LATE} min", f"{(d['late_pct'] * d['n_obs']).sum() / n_obs:.1f}%")
    c[2].metric("Trains profiled", f"{len(pr['trains']):,}")
    c[3].metric("Stations profiled", f"{len(pr['stations']):,}")
    if report:
        t = report["test"]
        sel, pers = t["overall"][t["selected"]], t["overall"]["persistence"]
        f1 = [v["f1"] for k, v in t["stage1"].items() if k.startswith(report["cv"]["selected"]["classifier"])
              and "calibrated" in k]
        st.markdown("**The model on unseen winter months** (vs persistence: \"the delay carries over\")")
        c = st.columns(4)
        c[0].metric("Error reduction vs persistence", f"{sel['skill_vs_persistence_pct']:.0f}%",
                    help="How much lower the forecast error (RMSLE) is than persistence's.")
        c[1].metric("Mean error", f"{sel['mae_min']:.1f} min", delta=f"{sel['mae_min'] - pers['mae_min']:+.1f} min",
                    delta_color="inverse")
        c[2].metric(f"F1, late > {LATE} min", f"{f1[0]:.3f}" if f1 else "n/a",
                    help="How well the model flags trains that will be significantly late (1 = perfect).")
        c[3].metric("Forecast error (RMSLE)", f"{sel['rmsle']:.3f}", delta=f"{sel['rmsle'] - pers['rmsle']:+.3f}",
                    delta_color="inverse")

    st.subheader("When are trains late?")
    month = _weighted(pr["by_month_zone"], "month")
    month["month_name"] = [dt.date(2000, int(m), 1).strftime("%b") for m in month["month"]]
    tip = lambda k, t: [alt.Tooltip(k, title=t), alt.Tooltip("mean_delay:Q", title="Mean delay (min)", format=".1f"),
                        alt.Tooltip("late_pct:Q", title="% late", format=".1f")]
    c = st.columns(2)
    with c[0]:
        st.markdown("**By month**: worst in the fog season (Dec-Feb)")
        show(bars(month, "month_name:N", "mean_delay:Q", tip("month_name:N", "Month"), y_title="Mean delay (min)",
                  sort=list(month["month_name"])))
    with c[1]:
        st.markdown("**By hour of scheduled arrival**: worst overnight")
        show(bars(pr["by_hour"], "hour:O", "mean_delay:Q", tip("hour:O", "Hour"), y_title="Mean delay (min)"))


def tab_forecast(pr: dict, lm: S.LoadedModel, tt, tt_key: tuple):
    if lm.model is None:
        st.error(f"The model cannot be served, so forecasts are off (the other tabs work): {lm.error}")
        return
    st.markdown("Pick a train, the last stop it passed and how late it was there. TrackCast forecasts the delay at "
                "every later stop.")
    if tt is None:
        st.warning("The timetable is missing. Run `python -m trackcast.download`.")
        return
    in_tt = set(tt.schedule["train_no"].unique().tolist())
    labels = {k: v for k, v in pr["train_labels"].items() if k in in_tt}
    c = st.columns([2, 1])
    train_no = c[0].selectbox("Train (type to search)", list(labels), format_func=labels.get, key="fc_train")
    date = c[1].date_input("Run start date", value=dt.date.today(), key="fc_date")
    route = tt.route(train_no)
    stops = route[~route["is_terminus"]]
    stop_label = {int(r.station_no): f"{r.station_name} · {r.station_full_name or ''} "
                                     f"(sched {FC.sched_clock(r.sched_dep_offset_min)})" for r in stops.itertuples()}
    c = st.columns([2, 1])
    current = c[0].selectbox("Last stop passed", list(stop_label), format_func=stop_label.get,
                             index=min(2, len(stop_label) - 1), key="fc_stop")
    delay_now = c[1].number_input("Delay there (min)", 0, 1440, 30, step=5, key="fc_delay")

    out = cached_forecast((str(lm.path), S.fingerprint(lm.path)), tt_key, int(train_no), date,
                          ((int(current), float(delay_now)),), lm.model, tt)
    fc = out[out["kind"] == "forecast"]
    if fc.empty:
        st.info("No stops left to forecast.")
        return
    nxt, term = fc.iloc[0], fc.iloc[-1]
    c = st.columns(3)
    c[0].metric(f"Next stop: {nxt['station_name']}", f"{nxt['pred_delay_min']:.0f} min late")
    c[1].metric(f"Chance of > {LATE} min late at {nxt['station_name']}", f"{100 * nxt['p_late']:.0f}%")
    c[2].metric(f"Terminus: {term['station_name']}", f"{term['pred_delay_min']:.0f} min late")

    hist = pr["train_stops"]
    hist = hist[hist["train_no"] == train_no][["station_no", "median_delay"]]
    out = out.merge(hist, on="station_no", how="left")
    seen = out[out["kind"] == "observed"].assign(series="Delay you entered")
    path = pd.concat([seen.tail(1), out[out["kind"] == "forecast"]]).assign(series="Forecast")
    typical = out[out["median_delay"].notna()].assign(series="Typical (historic median)",
                                                      delay=lambda d: d["median_delay"])
    long = pd.concat([typical, seen, path], ignore_index=True)
    long["sched"] = long["sched_arr_offset_min"].fillna(long["sched_dep_offset_min"]).map(FC.sched_clock)
    long["p_late_pct"] = 100 * long["p_late"]
    domain = ["Delay you entered", "Forecast", "Typical (historic median)"]
    show(hover_lines(long, "distance_from_origin:Q", "delay:Q", "series:N", domain, [BLUE, ORANGE, AQUA],
                     tooltip=[alt.Tooltip("station_name:N", title="Station"), alt.Tooltip("sched:N", title="Scheduled"),
                              alt.Tooltip("series:N", title="Series"), alt.Tooltip("delay:Q", title="Delay (min)", format=".0f"),
                              alt.Tooltip("p_late_pct:Q", title=f"% chance > {LATE} min", format=".0f")],
                     x_title="Distance from origin (km)", y_title="Arrival delay (min)", height=340,
                     dashed=["Typical (historic median)"]))

    sched = fc["sched_arr_offset_min"].fillna(fc["sched_dep_offset_min"])
    table = pd.DataFrame({"Station": fc["station_name"] + " · " + fc["station_full_name"].fillna(""),
                          "Scheduled": sched.map(FC.sched_clock),
                          "Forecast delay (min)": fc["pred_delay_min"].round(0),
                          f"Chance > {LATE} min": (100 * fc["p_late"]).round(0),
                          "Expected arrival": (sched + fc["pred_delay_min"]).map(FC.sched_clock)})
    st.dataframe(table, hide_index=True,
                 column_config={f"Chance > {LATE} min": st.column_config.ProgressColumn(
                     format="%d%%", min_value=0, max_value=100)})


def tab_cascade(pr: dict):
    st.markdown("**Delays cascade: a late train stays late.** For every stop of every run, the share of trains "
                f"still more than {LATE} min late *h* stops later, by how late they were at the start.")
    cas = pr["cascade"].sort_values(["bucket", "h"])
    labels = list(dict.fromkeys(cas["initial_delay"]))
    tip = [alt.Tooltip("initial_delay:N", title="Initial delay (min)"), alt.Tooltip("h:Q", title="Stops later"),
           alt.Tooltip("late_pct_h:Q", title="% still late", format=".1f"),
           alt.Tooltip("mean_change:Q", title="Mean change (min)", format="+.1f")]
    show(hover_lines(cas, "h:Q", "late_pct_h:Q", "initial_delay:N", labels, ORDINAL[: len(labels)], tip,
                     x_title="Stops later", y_title=f"% still late > {LATE} min", height=360))
    st.caption("This is why the naive forecast \"the delay carries over\" is strong, and why the model is judged "
               "by how much it beats it.")


def tab_model(report: dict | None):
    if not report:
        st.info("No model report yet. Run `python -m trackcast.train`.")
        return
    cv, t = report["cv"], report["test"]
    clf, reg = (FAMILY.get(cv["selected"][k], cv["selected"][k]) for k in ("classifier", "regressor"))
    st.markdown(f"**Two-stage hurdle model.** Stage 1 ({clf.lower()}) estimates the chance a train is more than "
                f"{LATE} min late. Stage 2 ({reg.lower()}) estimates how late, once for late and once for on-time "
                "trains. The forecast mixes the two, weighted by that chance. Trained on Feb-Nov 2025, tested once "
                "on Dec 2025-Feb 2026.")
    rows = {"persistence": "Persistence (the delay carries over)",
            f"direct_{cv['selected']['regressor']}": f"Single-stage {reg.lower()}",
            t["selected"]: "TrackCast hurdle (used)"}
    overall = pd.DataFrame([{"Model": name, "RMSLE": t["overall"][k]["rmsle"], "Mean error (min)": t["overall"][k]["mae_min"],
                             "Error reduction vs persistence (%)": t["overall"][k]["skill_vs_persistence_pct"]}
                            for k, name in rows.items() if k in t["overall"]])
    c = st.columns([3, 2])
    with c[0]:
        st.markdown("**Test results** (lower error is better)")
        st.dataframe(overall, hide_index=True,
                     column_config={"RMSLE": st.column_config.NumberColumn(format="%.3f"),
                                    "Mean error (min)": st.column_config.NumberColumn(format="%.2f"),
                                    "Error reduction vs persistence (%)": st.column_config.NumberColumn(format="%.1f")})
        st.caption("The single-stage model is slightly better on RMSLE; the hurdle is kept because it also gives the "
                   "chance of being late and has the lower mean error in minutes.")
    with c[1]:
        st.markdown("**What the model relies on** (error increase when a feature group is shuffled)")
        imp = pd.DataFrame(t["importance"])
        show(bars(imp, "group:N", "rmsle_increase:Q",
                  [alt.Tooltip("group:N", title="Group"), alt.Tooltip("rmsle_increase:Q", title="RMSLE increase", format=".3f")],
                  y_title="RMSLE increase", sort="-x", horizontal=True, height=260))


def skill_dots(df: pd.DataFrame, label: str, height: int):
    """Skill over persistence per group: dot + 95% CI rule, sorted, one blue series, hover tooltip."""
    order = list(df.sort_values("skill_pct", ascending=False)[label])
    y = alt.Y(f"{label}:N", sort=order, title=None)
    tip = [alt.Tooltip(f"{label}:N", title="Group"), alt.Tooltip("skill_pct:Q", title="Error reduction (%)", format=".1f"),
           alt.Tooltip("skill_lo:Q", title="95% CI low", format=".1f"),
           alt.Tooltip("skill_hi:Q", title="95% CI high", format=".1f"),
           alt.Tooltip("rows:Q", title="Test stops", format=",")]
    base = alt.Chart(df).encode(y=y)
    rule = base.mark_rule(color=ORDINAL[0], strokeWidth=2).encode(
        x=alt.X("skill_lo:Q", title="Error reduction vs persistence (%)"), x2="skill_hi:Q", tooltip=tip)
    dots = base.mark_point(filled=True, size=90, color=BLUE, opacity=1).encode(x="skill_pct:Q", tooltip=tip)
    return (rule + dots).properties(height=height)


def tab_results(res: dict | None, full: bool):
    if res is None:
        st.info(f"No results for this data yet. Run `python -m trackcast.results{' --full' if full else ''}`.")
        return
    c = st.columns(2)
    with c[0]:
        st.markdown("**Does the model help every kind of train?** Error reduction vs persistence, with a 95% interval")
        by = st.radio("Group by", ["Train class", "Corridor"], horizontal=True, key="res_by",
                      label_visibility="collapsed")
        df = res["scores_class" if by == "Train class" else "scores_corridor"].copy()
        label = "label" if "label" in df.columns else "group"
        df = df.head(10).dropna(subset=["skill_pct"])
        show(skill_dots(df, label, height=max(160, 30 * len(df))))
    with c[1]:
        st.markdown("**Bottleneck junctions**: delay-minutes added per day by trains passing through")
        thr = res["through"]
        cand = thr[thr["station_class"].isin(C.BOTTLENECK_CLASSES)
                   & (thr["n_through"] >= C.PROFILE_MIN_OBS["station"][1 if full else 0])]
        top = cand.nlargest(10, "minutes_per_day")
        top = top.assign(label=top["station_name"] + " · " + top["station_full_name"].fillna(""))
        show(bars(top, "label:N", "minutes_per_day:Q",
                  [alt.Tooltip("label:N", title="Station"),
                   alt.Tooltip("minutes_per_day:Q", title="Minutes added per day", format=",.0f"),
                   alt.Tooltip("through_change:Q", title="Added per train (min)", format="+.1f"),
                   alt.Tooltip("through_per_day:Q", title="Trains through per day", format=".0f")],
                  y_title="Minutes added per day", height=340, sort="-x", horizontal=True))


def tab_concepts():
    st.markdown("The data-science concepts this project uses, where, and why.")
    df = pd.DataFrame(K.CONCEPTS, columns=["Stage", "Concept", "Where in TrackCast", "Why"])
    for stage in K.STAGES:
        st.markdown(f"**{stage}**")
        st.dataframe(df[df["Stage"] == stage].drop(columns="Stage"), hide_index=True, width="stretch",
                     column_config={"Concept": st.column_config.TextColumn(width="small"),
                                    "Where in TrackCast": st.column_config.TextColumn(width="medium"),
                                    "Why": st.column_config.TextColumn(width="large")})


# --------------------------------------------------------------------------------------------------
TABS = ["Overview", "Forecast a train", "Delay cascade", "Model", "Classes & bottlenecks", "DSF concepts"]


def main():
    st.title("🚆 TrackCast")
    st.caption("Forecasting cascading train delays on Indian Railways: DSF project, NIT Warangal.")
    sources = {s.name: s for s in S.available_sources()}
    if not sources:
        st.error("No artifacts found. Run `python -m trackcast.train` and `python -m trackcast.profiles` (or build "
                 "the demo bundle: `python -m trackcast.serving demo`).")
        st.stop()
    name = st.sidebar.radio("Data", list(sources))
    src = sources[name]
    st.sidebar.markdown(f"Delay > **{LATE} min** counts as late.\n\nDataset: Kaggle "
                        "*Indian Railways Delay Dataset* (MIT), Feb 2025 to Feb 2026.")
    pr_fp = S.fingerprint(src.profiles)
    pr = load_profiles(str(src.profiles), pr_fp)
    pr = {**pr, "train_labels": train_labels(str(src.profiles), pr_fp, pr["trains"])}
    lm = load_model(str(src.model), S.fingerprint(src.model))
    if lm.model is None:   # only a real problem is shown; version notes are in `python -m trackcast.serving check`
        st.sidebar.error(f"Model not served: {lm.error}")
    tt_key = S.fingerprint(*src.timetable_files())
    tabs = st.tabs(TABS)
    with tabs[0]:
        tab_overview(pr, model_report())
    with tabs[1]:
        tab_forecast(pr, lm, load_timetable(src, tt_key), tt_key)
    with tabs[2]:
        tab_cascade(pr)
    with tabs[3]:
        tab_model(model_report())
    with tabs[4]:
        tab_results(load_results(str(src.results), S.fingerprint(src.results)), src.full)
    with tabs[5]:
        tab_concepts()


main()
