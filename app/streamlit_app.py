"""TrackCast dashboard.  Run:  streamlit run app/streamlit_app.py

Tabs: network overview, live route forecast (Phase 3 hurdle model), a train's historic route profile,
stations & segments, the delay cascade, the model's test-set evaluation, and results per train class and
route with bottleneck junctions (Phase 7). All logic lives in
src/trackcast (forecast.py builds features with the same code as training); this file is layout and charts.
"""
from __future__ import annotations

import datetime as dt
import json

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from trackcast import config as C
from trackcast import forecast as FC
from trackcast import profiles as P
from trackcast import results as R

st.set_page_config(page_title="TrackCast: train delay forecasts", page_icon="🚆", layout="wide")

# Reference data-viz palette: categorical slots 1-3 (validated all-pairs), blue ordinal ramp, muted chrome.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
ORDINAL = ["#86b6ef", "#3987e5", "#256abf", "#184f95"]   # cascade buckets, light -> dark = small -> large
MUTED = "#898781"
LATE = C.SIGNIFICANT_DELAY_MIN


# --------------------------------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------------------------------
SOURCES = {
    "Full year": {"profiles": C.PROFILES_DIR, "model": C.MODEL_PATH, "results": C.RESULTS_DIR},
    "Dev sample": {"profiles": C.PROFILES_SAMPLE_DIR, "model": C.MODEL_SAMPLE_PATH, "results": C.RESULTS_SAMPLE_DIR},
}


def source_ready(name: str) -> bool:
    s = SOURCES[name]
    return s["model"].exists() and all((s["profiles"] / f"{t}.parquet").exists() for t in P.TABLES)


@st.cache_data(show_spinner="Loading profiles...")
def load_profiles(directory: str) -> dict[str, pd.DataFrame]:
    return {t: pd.read_parquet(f"{directory}/{t}.parquet") for t in P.TABLES}


@st.cache_resource(show_spinner="Loading model...")
def load_model(path: str):
    return joblib.load(path)


@st.cache_resource(show_spinner="Loading timetable...")
def load_timetable():
    try:
        return FC.Timetable.load()
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner="Loading results...")
def load_results(directory: str) -> dict[str, pd.DataFrame] | None:
    try:
        return {t: pd.read_parquet(f"{directory}/{t}.parquet") for t in R.TABLES}
    except FileNotFoundError:
        return None


@st.cache_data
def load_model_report() -> dict | None:
    return json.loads(C.MODELS_JSON.read_text(encoding="utf-8")) if C.MODELS_JSON.exists() else None


@st.cache_data
def load_route_report() -> dict | None:
    p = C.ROUTE_FORECAST_JSON
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


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


def fmt_train(row) -> str:
    name = row.get("train_name")
    return f"{int(row['train_no'])} · {name}" if isinstance(name, str) and name else str(int(row["train_no"]))


# --------------------------------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------------------------------
def tab_overview(pr: dict, report: dict | None):
    d = pr["by_date"].copy()
    d["date"] = pd.to_datetime(d["date"])
    n_obs = d["n_obs"].sum()
    c = st.columns(4)
    c[0].metric("Mean arrival delay", f"{(d['mean_delay'] * d['n_obs']).sum() / n_obs:.1f} min")
    c[1].metric(f"Arrivals late > {LATE} min", f"{(d['late_pct'] * d['n_obs']).sum() / n_obs:.1f}%")
    c[2].metric("Trains profiled", f"{len(pr['trains']):,}")
    c[3].metric("Stations profiled", f"{len(pr['stations']):,}")
    if report:
        t = report["test"]
        sel, pers = t["overall"][t["selected"]], t["overall"]["persistence"]
        c = st.columns(4)
        c[0].metric("Forecast error (RMSLE)", f"{sel['rmsle']:.3f}",
                    delta=f"{sel['rmsle'] - pers['rmsle']:+.3f} vs persistence", delta_color="inverse",
                    help="Root mean squared log error on the held-out test months; lower is better.")
        c[1].metric("Skill vs persistence", f"{sel['skill_vs_persistence_pct']:.1f}%",
                    help="RMSLE reduction against 'the delay stays what it was at the previous stop'.")
        f1 = [v["f1"] for k, v in t["stage1"].items() if k.startswith(report["cv"]["selected"]["classifier"])
              and "calibrated" in k]
        c[2].metric(f"F1, late > {LATE} min", f"{f1[0]:.3f}" if f1 else "n/a")
        c[3].metric("Mean abs. error", f"{sel['mae_min']:.1f} min", delta=f"{sel['mae_min'] - pers['mae_min']:+.1f} vs persistence",
                    delta_color="inverse")
        m = report["meta"]
        st.caption(f"Model metrics: latest training run (`{m['source']}`), scored on {m['test_rows']:,} held-out "
                   f"stops of runs starting {m['test_dates'][0]} to {m['test_dates'][1]}.")

    st.subheader("Network delay by day")
    st.caption("Mean arrival delay of all observed stops of runs starting that day (hover for values).")
    show(hover_lines(d, "date:T", "mean_delay:Q", x_title=None, y_title="Mean delay (min)",
                     tooltip=[alt.Tooltip("date:T", title="Date"),
                              alt.Tooltip("mean_delay:Q", title="Mean delay (min)", format=".1f"),
                              alt.Tooltip("late_pct:Q", title="% late", format=".1f"),
                              alt.Tooltip("n_obs:Q", title="Arrivals", format=",")]))

    mz = pr["by_month_zone"]
    month = mz.assign(s=mz["mean_delay"] * mz["n_obs"], l=mz["late_pct"] * mz["n_obs"]).groupby("month")[
        ["n_obs", "s", "l"]].sum().reset_index()
    month["mean_delay"], month["late_pct"] = month["s"] / month["n_obs"], month["l"] / month["n_obs"]
    month["month_name"] = [dt.date(2000, int(m), 1).strftime("%b") for m in month["month"]]
    tip = lambda k, t: [alt.Tooltip(k, title=t), alt.Tooltip("mean_delay:Q", title="Mean delay (min)", format=".1f"),
                        alt.Tooltip("late_pct:Q", title="% late", format=".1f")]
    c = st.columns(3)
    with c[0]:
        st.markdown("**By month**")
        show(bars(month, "month_name:N", "mean_delay:Q", tip("month_name:N", "Month"), y_title="Mean delay (min)",
                  sort=list(month["month_name"])))
    with c[1]:
        st.markdown("**By day of week**")
        show(bars(pr["by_dow"], "day:N", "mean_delay:Q", tip("day:N", "Day"), y_title="Mean delay (min)",
                  sort=P.DOW_NAMES))
    with c[2]:
        st.markdown("**By hour of scheduled arrival**")
        show(bars(pr["by_hour"], "hour:O", "mean_delay:Q", tip("hour:O", "Hour"), y_title="Mean delay (min)"))

    st.subheader("By railway zone")
    zone = mz.assign(s=mz["mean_delay"] * mz["n_obs"], l=mz["late_pct"] * mz["n_obs"]).groupby("station_zone")[
        ["n_obs", "s", "l"]].sum()
    zone = zone[zone["n_obs"] >= max(50, zone["n_obs"].sum() / 2000)]
    zone = zone.assign(mean_delay=zone["s"] / zone["n_obs"], late_pct=zone["l"] / zone["n_obs"]).reset_index()
    show(bars(zone, "station_zone:N", "mean_delay:Q",
              [alt.Tooltip("station_zone:N", title="Zone"), alt.Tooltip("mean_delay:Q", title="Mean delay (min)", format=".1f"),
               alt.Tooltip("late_pct:Q", title="% late", format=".1f"), alt.Tooltip("n_obs:Q", title="Arrivals", format=",")],
              y_title="Mean delay (min)", sort="-y"))


def tab_forecast(pr: dict, model, tt):
    st.markdown("Enter where a train is and how late it is; the model forecasts the delay at every later stop. "
                "Each forecast is fed back in as the next stop's upstream delay (the model forecasts one stop "
                "ahead), so stops far ahead show a typical path, not a guarantee.")
    if tt is None:
        st.warning("The timetable (data/raw/combined_schedule.csv etc.) is missing. Run `python -m trackcast.download`.")
        return
    trains = pr["trains"].sort_values("n_runs", ascending=False)
    trains = trains[trains["train_no"].isin(tt.schedule["train_no"].unique())]
    labels = {int(r["train_no"]): fmt_train(r) for _, r in trains.iterrows()}
    c = st.columns([2, 1])
    train_no = c[0].selectbox("Train (most-run first; type to search)", list(labels), format_func=labels.get,
                              key="fc_train")
    date = c[1].date_input("Run start date", value=dt.date.today(), key="fc_date")
    route = tt.route(train_no)
    stops = route[~route["is_terminus"]]
    stop_label = {int(r.station_no): f"{int(r.station_no)} · {r.station_name} · {r.station_full_name or ''} "
                                     f"(sched {FC.sched_clock(r.sched_dep_offset_min)})" for r in stops.itertuples()}
    c = st.columns([2, 1, 1])
    current = c[0].selectbox("Last stop passed", list(stop_label), format_func=stop_label.get,
                             index=min(2, len(stop_label) - 1), key="fc_stop")
    delay_now = c[1].number_input("Delay there (min)", 0, 1440, 30, step=5, key="fc_delay")
    prev_stops = route[route["station_no"] < current]
    delay_prev = None
    if len(prev_stops):
        use_prev = c[2].checkbox("I know the stop before", value=False, key="fc_use_prev")
        if use_prev:
            delay_prev = c[2].number_input(f"Delay at {prev_stops.iloc[-1]['station_name']} (min)", 0, 1440, 20,
                                           step=5, key="fc_prev")
    observed = {int(current): float(delay_now)}
    if delay_prev is not None:
        observed[int(prev_stops.iloc[-1]["station_no"])] = float(delay_prev)

    with st.spinner("Forecasting the rest of the route..."):
        out = FC.forecast_route(model, tt, train_no, date, observed)
    fc = out[out["kind"] == "forecast"]
    if fc.empty:
        st.info("No stops left to forecast.")
        return
    nxt, term = fc.iloc[0], fc.iloc[-1]
    c = st.columns(3)
    c[0].metric(f"Next stop: {nxt['station_name']}", f"{nxt['pred_delay_min']:.0f} min late",
                help="Forecast arrival delay at the next scheduled stop.")
    c[1].metric(f"Chance of > {LATE} min late at {nxt['station_name']}", f"{100 * nxt['p_late']:.0f}%")
    c[2].metric(f"Terminus: {term['station_name']}", f"{term['pred_delay_min']:.0f} min late")

    hist = pr["train_stops"]
    hist = hist[hist["train_no"] == train_no][["station_no", "median_delay", "p90_delay", "n_obs"]]
    out = out.merge(hist, on="station_no", how="left")
    seen = out[out["kind"] == "observed"].assign(series="Observed (you entered)")
    path = pd.concat([seen.tail(1), out[out["kind"] == "forecast"]]).assign(series="Forecast")
    typical = out[out["median_delay"].notna()].assign(series="Typical: historic median", delay=lambda d: d["median_delay"])
    long = pd.concat([typical, seen, path], ignore_index=True)
    long["sched"] = long["sched_arr_offset_min"].fillna(long["sched_dep_offset_min"]).map(FC.sched_clock)
    long["p_late_pct"] = 100 * long["p_late"]
    domain = ["Observed (you entered)", "Forecast", "Typical: historic median"]
    show(hover_lines(long, "distance_from_origin:Q", "delay:Q", "series:N", domain, [BLUE, ORANGE, AQUA],
                     tooltip=[alt.Tooltip("station_name:N", title="Station"), alt.Tooltip("sched:N", title="Scheduled"),
                              alt.Tooltip("series:N", title="Series"), alt.Tooltip("delay:Q", title="Delay (min)", format=".0f"),
                              alt.Tooltip("p_late_pct:Q", title=f"% chance > {LATE} min", format=".0f")],
                     x_title="Distance from origin (km)", y_title="Arrival delay (min)", height=340,
                     dashed=["Typical: historic median"]))

    table = out[out["kind"] == "forecast"].copy()
    sched = table["sched_arr_offset_min"].fillna(table["sched_dep_offset_min"])
    table = pd.DataFrame({
        "Stop": table["station_no"], "Station": table["station_name"], "Name": table["station_full_name"],
        "Km": table["distance_from_origin"], "Scheduled": sched.map(FC.sched_clock),
        "Forecast delay (min)": table["pred_delay_min"].round(0),
        f"P(> {LATE} min)": (100 * table["p_late"]).round(0),
        "Expected arrival": (sched + table["pred_delay_min"]).map(FC.sched_clock),
        "Slack before stop (min)": table["slack_since_lag"].round(0),
        "Historic median (min)": table["median_delay"]})
    st.dataframe(table, hide_index=True,
                 column_config={f"P(> {LATE} min)": st.column_config.ProgressColumn(
                     format="%d%%", min_value=0, max_value=100)})
    rr = load_route_report()
    if rr:
        by_h = {row["h"]: row for row in rr["by_horizon"]}
        bits = [f"{by_h[h]['mean']['mae_min']:.0f} min {h} stop{'s' if h > 1 else ''} ahead"
                for h in (1, 5, 10, 20) if h in by_h]
        st.caption(f"How accurate is this? Simulated on {rr['meta']['runs']:,} test-period runs, the typical "
                   f"error was {', '.join(bits)}, and far-ahead forecasts tend to run low; see the Model tab. "
                   "Entering the stop before improves the next-stop forecast.")


def tab_route(pr: dict):
    trains = pr["trains"].sort_values("n_runs", ascending=False)
    labels = {int(r["train_no"]): fmt_train(r) for _, r in trains.iterrows()}
    train_no = st.selectbox("Train", list(labels), format_func=labels.get, key="rt_train")
    t = trains[trains["train_no"] == train_no].iloc[0]
    c = st.columns(5)
    c[0].metric("Runs observed", f"{int(t['n_runs']):,}")
    c[1].metric("Route", f"{t['origin']} → {t['terminus']}", help=f"{t['route_km']:.0f} km, {int(t['n_sched_stops'])} stops")
    c[2].metric("Mean delay leaving origin", f"{t['mean_origin_delay']:.0f} min" if pd.notna(t["mean_origin_delay"]) else "n/a")
    c[3].metric("Mean delay at terminus", f"{t['mean_terminus_delay']:.0f} min" if pd.notna(t["mean_terminus_delay"]) else "n/a")
    c[4].metric(f"Late > {LATE} min at terminus", f"{t['terminus_late_pct']:.0f}%" if pd.notna(t["terminus_late_pct"]) else "n/a")

    s = pr["train_stops"]
    s = s[(s["train_no"] == train_no) & s["n_obs"].notna()].copy()
    s["sched"] = s["sched_arr_offset_min"].fillna(s["sched_dep_offset_min"]).map(FC.sched_clock)
    st.markdown("**How delay builds up along the route**: median arrival delay per stop, the band reaches the "
                "90th percentile (1 run in 10 is later than its top edge).")
    tip = [alt.Tooltip("station_name:N", title="Station"), alt.Tooltip("sched:N", title="Scheduled"),
           alt.Tooltip("median_delay:Q", title="Median (min)", format=".0f"),
           alt.Tooltip("p90_delay:Q", title="p90 (min)", format=".0f"),
           alt.Tooltip("late_pct:Q", title="% late", format=".0f"), alt.Tooltip("n_obs:Q", title="Runs observed")]
    band = alt.Chart(s).mark_area(color=BLUE, opacity=0.15).encode(
        x=alt.X("distance_from_origin:Q", title="Distance from origin (km)"), y="median_delay:Q", y2="p90_delay:Q")
    show(band + hover_lines(s, "distance_from_origin:Q", "median_delay:Q", tooltip=tip,
                            y_title="Arrival delay (min)", height=320))
    st.dataframe(s[["station_no", "station_name", "sched", "distance_from_origin", "n_obs", "median_delay", "p90_delay",
                    "late_pct", "mean_change", "recovery_pct_when_late", "missing_pct"]].rename(columns={
        "station_no": "Stop", "station_name": "Station", "sched": "Scheduled", "distance_from_origin": "Km",
        "n_obs": "Runs", "median_delay": "Median", "p90_delay": "p90", "late_pct": "% late",
        "mean_change": "Mean change", "recovery_pct_when_late": "% late that recover", "missing_pct": "% missing"}),
        hide_index=True)


def tab_stations(pr: dict, full: bool):
    k = 1 if full else 0
    min_st = st.slider("Minimum arrivals for a station to be ranked", 10, 20_000,
                       C.PROFILE_MIN_OBS["station"][k], step=10, key="st_min")
    stn = pr["stations"][pr["stations"]["n_obs"] >= min_st]
    metric = st.radio("Rank stations by", ["Mean arrival delay", "Delay gained on arrival", "% late arrivals that recover"],
                      horizontal=True, key="st_metric")
    col = {"Mean arrival delay": "mean_delay", "Delay gained on arrival": "mean_change",
           "% late arrivals that recover": "recovery_pct_when_late"}[metric]
    top = stn.nlargest(20, col).assign(label=lambda d: d["station_name"] + " · " + d["station_full_name"].fillna(""))
    show(bars(top, "label:N", f"{col}:Q",
              [alt.Tooltip("label:N", title="Station"), alt.Tooltip(f"{col}:Q", title=metric, format=".1f"),
               alt.Tooltip("n_obs:Q", title="Arrivals", format=","), alt.Tooltip("station_zone:N", title="Zone")],
              y_title=metric, height=460, sort="-x", horizontal=True))
    st.dataframe(stn.sort_values(col, ascending=False), hide_index=True)

    st.subheader("Segments: where late trains lose or recover time")
    min_seg = st.slider("Minimum late arrivals on a segment", 5, 5_000, C.PROFILE_MIN_OBS["segment"][k], step=5, key="seg_min")
    seg = pr["segments"][pr["segments"]["n_late_upstream"] >= min_seg].copy()
    seg["segment"] = seg["from_station"] + " → " + seg["to_station"]
    cols = ["segment", "n_obs", "mean_sched_min", "mean_change", "mean_change_when_late", "recovery_pct_when_late"]
    names = {"segment": "Segment", "n_obs": "Pairs", "mean_sched_min": "Scheduled min", "mean_change": "Mean change",
             "mean_change_when_late": "Mean change if late", "recovery_pct_when_late": "% late that recover"}
    c = st.columns(2)
    c[0].markdown("**Late trains lose the most time**")
    c[0].dataframe(seg.nlargest(15, "mean_change_when_late")[cols].rename(columns=names), hide_index=True)
    c[1].markdown("**Late trains recover the most time**")
    c[1].dataframe(seg.nsmallest(15, "mean_change_when_late")[cols].rename(columns=names), hide_index=True)


def tab_cascade(pr: dict):
    st.markdown("Take every stop of every run, note the delay there, and look at the same train *h* observed "
                "stops later. Lines are grouped by the delay at the start.")
    cas = pr["cascade"].sort_values(["bucket", "h"])
    labels = list(dict.fromkeys(cas["initial_delay"]))
    metric = st.radio("Show", ["% still late after h stops", "Mean change in delay (min)"], horizontal=True, key="cas_m")
    y, title = (("late_pct_h:Q", f"% late > {LATE} min") if metric.startswith("%")
                else ("mean_change:Q", "Change since start (min)"))
    tip = [alt.Tooltip("initial_delay:N", title="Initial delay (min)"), alt.Tooltip("h:Q", title="Stops later"),
           alt.Tooltip("late_pct_h:Q", title="% late", format=".1f"),
           alt.Tooltip("mean_change:Q", title="Mean change (min)", format=".1f"),
           alt.Tooltip("n_pairs:Q", title="Pairs", format=",")]
    show(hover_lines(cas, "h:Q", y, "initial_delay:N", labels, ORDINAL[: len(labels)], tip,
                     x_title="Observed stops later (h)", y_title=title, height=340))
    st.dataframe(cas.drop(columns=["bucket"]), hide_index=True)


def tab_model(report: dict | None):
    if not report:
        st.info("No model report yet. Run `python -m trackcast.train`.")
        return
    m, cv, t = report["meta"], report["cv"], report["test"]
    st.markdown(f"Two-stage hurdle model trained on runs starting {m['train_dates'][0]} to {m['train_dates'][1]}, "
                f"tested once on {m['test_dates'][0]} to {m['test_dates'][1]} ({m['test_rows']:,} stops). Stage 1 "
                f"= **{cv['selected']['classifier']}** estimates P(delay > {LATE} min); stage 2 = "
                f"**{cv['selected']['regressor']}** estimates the delay for each case; the forecast mixes the two.")
    overall = pd.DataFrame(t["overall"]).T.reset_index(names="model")
    st.dataframe(overall[["model", "rmsle", "r2_log", "mae_min", "skill_vs_persistence_pct"]].rename(columns={
        "model": "Model", "rmsle": "RMSLE", "r2_log": "R² (log)", "mae_min": "MAE (min)",
        "skill_vs_persistence_pct": "Skill vs persistence (%)"}), hide_index=True,
        column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ["RMSLE", "R² (log)"]})
    for k, v in t["bootstrap"].items():
        st.caption(f"{k}: {v['diff']:+.4f} RMSLE (95% day-block bootstrap CI {v['ci_low']:+.4f} to {v['ci_high']:+.4f}).")

    sel = t["selected"]
    for name, title in [("RMSLE by upstream delay (lag1, min)", "RMSLE"),
                        ("MAE (minutes) by upstream delay (lag1, min)", "MAE (min)")]:
        rows = pd.DataFrame(t["breakdowns"][name])
        long = rows.melt(id_vars=["segment", "rows", "late_pct"], var_name="model", value_name="value")
        domain = ["persistence", f"direct_{cv['selected']['regressor']}", "selected"]
        st.markdown(f"**{title} by delay at the previous stop** (lower is better)")
        show(alt.Chart(long).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("segment:N", title="Delay at previous stop (min)", sort=list(rows["segment"])),
            xOffset=alt.XOffset("model:N", sort=domain),
            y=alt.Y("value:Q", title=title),
            color=alt.Color("model:N", scale=alt.Scale(domain=domain, range=[BLUE, ORANGE, AQUA]), legend=_legend()),
            tooltip=[alt.Tooltip("segment:N", title="Upstream delay"), alt.Tooltip("model:N", title="Model"),
                     alt.Tooltip("value:Q", title=title, format=".3f"), alt.Tooltip("rows:Q", title="Test stops", format=",")]
        ).properties(height=260))
    st.caption(f"'selected' = {sel}. Trees on a log target give up minutes for trains already hours late, where "
               "the delay simply persists; persistence stays best in minutes there.")

    rr = load_route_report()
    if rr:
        st.markdown(f"**Multi-stop forecasts** (the Forecast tab), simulated on {rr['meta']['runs']:,} test-period "
                    "runs knowing only the start delay: mean absolute error by stops ahead (lower is better)")
        names = {"mean": "TrackCast (expected minutes, used in the app)", "persistence": "Persistence",
                 "median": "Chained median forecast"}
        long = pd.DataFrame([{"h": row["h"], "strategy": names[k], "mae": row[k]["mae_min"],
                              "bias": row[k]["bias_min"]} for row in rr["by_horizon"] for k in names if k in row])
        show(hover_lines(long, "h:Q", "mae:Q", "strategy:N", list(names.values()), [BLUE, ORANGE, AQUA],
                         [alt.Tooltip("strategy:N", title="Method"), alt.Tooltip("h:Q", title="Stops ahead"),
                          alt.Tooltip("mae:Q", title="MAE (min)", format=".1f"),
                          alt.Tooltip("bias:Q", title="Bias (min)", format="+.1f")],
                         x_title="Stops ahead", y_title="MAE (min)", height=260))
        st.caption("Chaining the 1-stop median forecast shrinks delays towards zero; feeding forward the expected "
                   "minutes avoids most of that and beats persistence from 2 stops ahead.")

    c = st.columns(2)
    with c[0]:
        st.markdown("**Stage 1: late > 15 min**")
        st.dataframe(pd.DataFrame(t["stage1"]).T[["f1", "precision", "recall", "roc_auc", "brier"]].round(4))
    with c[1]:
        st.markdown("**Which features matter** (RMSLE increase when a group is shuffled)")
        imp = pd.DataFrame(t["importance"])
        show(bars(imp, "group:N", "rmsle_increase:Q",
                  [alt.Tooltip("group:N", title="Group"), alt.Tooltip("rmsle_increase:Q", title="RMSLE increase", format=".4f")],
                  y_title="RMSLE increase", sort="-x", horizontal=True, height=260))


def skill_dots(df: pd.DataFrame, label: str, height: int):
    """Skill over persistence per group: dot + 95% CI rule, sorted, one blue series, hover tooltip."""
    order = list(df.sort_values("skill_pct", ascending=False)[label])
    y = alt.Y(f"{label}:N", sort=order, title=None)
    tip = [alt.Tooltip(f"{label}:N", title="Group"), alt.Tooltip("skill_pct:Q", title="Skill (%)", format=".1f"),
           alt.Tooltip("skill_lo:Q", title="95% CI low", format=".1f"),
           alt.Tooltip("skill_hi:Q", title="95% CI high", format=".1f"),
           alt.Tooltip("rmsle_selected:Q", title="RMSLE hurdle", format=".3f"),
           alt.Tooltip("rmsle_persistence:Q", title="RMSLE persistence", format=".3f"),
           alt.Tooltip("rows:Q", title="Test stops", format=",")]
    base = alt.Chart(df).encode(y=y)
    rule = base.mark_rule(color=ORDINAL[0], strokeWidth=2).encode(
        x=alt.X("skill_lo:Q", title="Skill over persistence (%)"), x2="skill_hi:Q", tooltip=tip)
    dots = base.mark_point(filled=True, size=90, color=BLUE, opacity=1).encode(x="skill_pct:Q", tooltip=tip)
    return (rule + dots).properties(height=height)


def _segment_text(a, b, ch) -> str:
    return "" if pd.isna(a) or pd.isna(ch) else f"{a} → {b} ({ch:+.1f})"


def tab_results(res: dict | None, full: bool):
    if res is None:
        st.info(f"No Phase 7 results for this data yet. Run `python -m trackcast.results{' --full' if full else ''}`.")
        return
    st.markdown("**How well does the model do for each kind of train and route?** *Skill* = % lower RMSLE than "
                "persistence (\"the delay at the previous stop carries over\") on the test months, with a 95% "
                "day-block bootstrap interval.")
    by = st.radio("Group test stops by", ["Train class", "Corridor", "Station group", "Train"], horizontal=True,
                  key="res_by")
    names = {"rows": "Test stops", "late_pct": "% late", "rmsle_persistence": "RMSLE persistence",
             "rmsle_selected": "RMSLE hurdle", "skill_pct": "Skill %", "skill_lo": "CI low", "skill_hi": "CI high",
             "mae_persistence": "MAE persistence", "mae_selected": "MAE hurdle", "f1_persistence": "F1 persistence",
             "f1_selected": "F1 hurdle", "mean_delay": "Year mean delay", "recovery_pct_when_late": "Year % recover",
             "congestion_slope_per10": "Min per +10 trains/h", "mean_gained_en_route": "Gained en route"}
    if by == "Train":
        tr = res["scores_train"].assign(label=lambda d: d["train_no"] + " · " + d["train_name"].fillna(""))
        st.caption(f"{len(tr):,} trains with enough test stops, sorted by RMSLE gain over persistence "
                   "(the table is searchable).")
        st.dataframe(tr.sort_values("rmsle_gain", ascending=False)[
            ["label", "type_code", "corridor", "rmsle_gain", *[c for c in names if c in tr.columns]]].rename(
            columns={"label": "Train", "type_code": "Type", "corridor": "Corridor", "rmsle_gain": "RMSLE gain",
                     **names}), hide_index=True)
    else:
        key = {"Train class": "scores_class", "Corridor": "scores_corridor", "Station group": "scores_station_group"}[by]
        df = res[key].copy()
        label = "label" if "label" in df.columns else "group"
        if by == "Corridor":
            df = df.head(C.RESULTS_TOP_CORRIDORS)
        df = df.dropna(subset=["skill_pct"])
        show(skill_dots(df, label, height=max(160, 28 * len(df))))
        st.dataframe(df[[label, *[c for c in names if c in df.columns]]].rename(columns={label: by, **names}),
                     hide_index=True)
        if by == "Station group":
            st.caption("Bottlenecks / absorbers = the junctions and major stations ranked below.")

    st.subheader("Bottleneck junctions")
    st.markdown("For trains passing straight through a station, the change in delay from the stop before to the "
                "stop after (approach, platform dwell and exit together). Arrival-only change misleads at big "
                "stations: timetables pad the approach, and the time is lost again on the way out.")
    k = 1 if full else 0
    thr = res["through"]
    cand = thr[thr["station_class"].isin(C.BOTTLENECK_CLASSES)]
    c = st.columns([2, 3])
    side = c[0].radio("Show", ["Adds delay (bottlenecks)", "Recovers delay (absorbers)"], key="bn_side")
    min_thr = c[1].slider("Minimum trains through over the year", 10, 50_000, C.PROFILE_MIN_OBS["station"][k],
                          step=10, key="bn_min")
    cand = cand[cand["n_through"] >= min_thr]
    adds = side.startswith("Adds")
    top = (cand.nlargest(20, "minutes_per_day") if adds
           else cand.nsmallest(20, "minutes_per_day").assign(minutes_per_day=lambda d: -d["minutes_per_day"]))
    top = top.assign(label=top["station_name"] + " · " + top["station_full_name"].fillna(""))
    metric = "Minutes added per day" if adds else "Minutes recovered per day"
    show(bars(top, "label:N", "minutes_per_day:Q",
              [alt.Tooltip("label:N", title="Station"), alt.Tooltip("minutes_per_day:Q", title=metric, format=",.0f"),
               alt.Tooltip("through_change:Q", title="Mean change per train (min)", format="+.1f"),
               alt.Tooltip("arrival_change:Q", title="On approach", format="+.1f"),
               alt.Tooltip("departure_change:Q", title="Dwell + exit", format="+.1f"),
               alt.Tooltip("recovery_pct:Q", title="% late that recover", format=".1f"),
               alt.Tooltip("through_per_day:Q", title="Trains through per day", format=".1f")],
              y_title=metric, height=480, sort="-x", horizontal=True))
    top = top.assign(
        approach=[_segment_text(a, b, ch) for a, b, ch in zip(top["approach_from"], top["station_name"],
                                                              top["approach_change"])],
        exit=[_segment_text(a, b, ch) for a, b, ch in zip(top["station_name"], top["exit_to"], top["exit_change"])])
    st.dataframe(top[["station_name", "station_full_name", "station_zone", "station_class", "through_per_day",
                      "arrival_change", "departure_change", "through_change", "recovery_pct", "made_late_pct",
                      "minutes_per_day", "approach", "exit"]].rename(columns={
        "station_name": "Code", "station_full_name": "Station", "station_zone": "Zone", "station_class": "Class",
        "through_per_day": "Through/day", "arrival_change": "Approach", "departure_change": "Dwell + exit",
        "through_change": "Through", "recovery_pct": "% late that recover", "made_late_pct": "% on time made late",
        "minutes_per_day": metric, "approach": "Worst approach segment", "exit": "Worst exit segment"}),
        hide_index=True)


# --------------------------------------------------------------------------------------------------
def main():
    st.title("🚆 TrackCast")
    st.caption("Forecasting cascading train delays on Indian Railways: DSF project, NIT Warangal.")
    ready = [s for s in SOURCES if source_ready(s)]
    if not ready:
        st.error("No artifacts found. Run `python -m trackcast.train` and `python -m trackcast.profiles` first.")
        st.stop()
    source = st.sidebar.radio("Data", ready, help="Full year needs `--full` runs; the dev sample is ~100k stops.")
    st.sidebar.markdown(f"Delay > **{LATE} min** counts as late.\n\nDataset: Kaggle "
                        "*Indian Railways Delay Dataset* (MIT), Feb 2025 to Feb 2026.")
    pr = load_profiles(str(SOURCES[source]["profiles"]))
    model = load_model(str(SOURCES[source]["model"]))
    report = load_model_report()
    tabs = st.tabs(["Overview", "Forecast a train", "Route profile", "Stations & segments", "Delay cascade", "Model",
                    "Classes & bottlenecks"])
    with tabs[0]:
        tab_overview(pr, report)
    with tabs[1]:
        tab_forecast(pr, model, load_timetable())
    with tabs[2]:
        tab_route(pr)
    with tabs[3]:
        tab_stations(pr, source == "Full year")
    with tabs[4]:
        tab_cascade(pr)
    with tabs[5]:
        tab_model(report)
    with tabs[6]:
        tab_results(load_results(str(SOURCES[source]["results"])), source == "Full year")


main()
