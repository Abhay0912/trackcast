"""TrackCast dashboard.  Run:  streamlit run app/streamlit_app.py

Five tabs in plain language, kept to what a 10-minute class demo needs: the dataset and when trains are late,
a live forecast for one train, how delays carry on down the line, how the model works and how well, and
results per kind of train with the stations that add the most delay. The full detail is in reports/ and
docs/REPORT.md. All logic lives in src/trackcast (forecast.py builds features with the same code as training;
serving.py finds, checks and loads the artifacts); this file is layout, caching and charts.

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
# Tabs (plain language for a class demo; the full tables are in reports/)
# --------------------------------------------------------------------------------------------------
ZONES = {"CR": "Central", "ER": "Eastern", "ECR": "East Central", "ECOR": "East Coast", "NR": "Northern",
         "NCR": "North Central", "NER": "North Eastern", "NFR": "Northeast Frontier", "NWR": "North Western",
         "SR": "Southern", "SCR": "South Central", "SER": "South Eastern", "SECR": "South East Central",
         "SWR": "South Western", "WR": "Western", "WCR": "West Central", "KR": "Konkan"}
START_DELAY = {"<= 15": "On time (up to 15 min late)", "(15, 60]": "15 min to 1 hour late",
               "(60, 180]": "1 to 3 hours late", "> 180": "More than 3 hours late"}
FEATURE_GROUP = {"cascade": "Delay at earlier stations", "slack": "Spare time in the timetable",
                 "station": "Which station", "route": "Position along the route",
                 "congestion": "How busy the station is", "train": "Which train",
                 "time": "Time of day, week and year", "season": "Fog or monsoon season"}


def corridor_name(code: str) -> str:
    """'within SR' -> 'Southern zone only'; 'NR-NWR' -> 'Northern - North Western'."""
    if code.startswith("within "):
        z = code.split(" ", 1)[1]
        return f"{ZONES.get(z, z)} zone only"
    return " - ".join(ZONES.get(z, z) for z in code.split("-"))


def _weighted(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Mean delay and % late per `by`, weighted by arrivals."""
    g = df.assign(s=df["mean_delay"] * df["n_obs"], l=df["late_pct"] * df["n_obs"]).groupby(by)[
        ["n_obs", "s", "l"]].sum().reset_index()
    return g.assign(mean_delay=g["s"] / g["n_obs"], late_pct=g["l"] / g["n_obs"])


def _millions(n: int) -> str:
    return f"{n / 1e6:.1f} million"


def tab_overview(pr: dict, dq: dict | None, prof: dict | None):
    st.markdown("#### Predicting how late Indian Railways trains will be at their next stations.")
    st.subheader("The dataset")
    st.markdown("Real delay records of Indian Railways passenger trains for one year, from the Kaggle "
                "*Indian Railways Delay Dataset* (free to use, MIT licence). For every train, every day and every "
                "station it stops at, the data says how many minutes late the train arrived.")
    if dq:
        cl, wf = dq["clean"], dict(dq["waterfall"])
        c = st.columns(4)
        c[0].metric("Station stops", _millions(cl["rows"]), help="One record = one train arriving at one station.")
        c[1].metric("Train journeys", _millions(cl["runs"]), help="One journey = one train on one day.")
        c[2].metric("Trains", f"{cl['trains']:,}")
        c[3].metric("Stations", f"{cl['stations']:,}")
        start = dt.date.fromisoformat(cl["date_min"]).strftime("%d %b %Y")
        end = dt.date.fromisoformat(cl["date_max"]).strftime("%d %b %Y")
        raw = wf.get("raw rows", cl["rows"])
        st.markdown(f"**Period:** {start} to {end}.  \n"
                    f"**Cleaning:** the original file has {_millions(raw)} records. We kept {_millions(cl['rows'])} "
                    "after removing records that could not be matched to the timetable and journeys with fewer than "
                    "3 recorded stations. Early arrivals count as 0 minutes late, and impossible values (more than "
                    "a day late) are treated as missing.")
    st.dataframe(pd.DataFrame(
        [("Delays", "Train number, date, station and how many minutes late the train arrived"),
         ("Timetable", "Each train's stations in order, scheduled arrival and departure times, distance"),
         ("Trains", "Train name and type (Passenger, Express, Superfast, Rajdhani, Vande Bharat, ...)"),
         ("Stations", "Station code, full name and railway zone")],
        columns=["File", "What it contains"]), hide_index=True, width="stretch")
    st.caption("The data has no weather and no actual arrival times; delays are given directly in minutes.")

    st.subheader("How late are trains?")
    if prof:
        net = prof["network"]
        mean_delay, late_pct = net["mean_delay"], net["late_pct"]
        month, hour = pd.DataFrame(prof["by_month"]), pd.DataFrame(prof["by_hour"])
    else:   # no committed report: use the served source's own tables
        d = pr["by_date"]
        mean_delay = (d["mean_delay"] * d["n_obs"]).sum() / d["n_obs"].sum()
        late_pct = (d["late_pct"] * d["n_obs"]).sum() / d["n_obs"].sum()
        month, hour = _weighted(pr["by_month_zone"], "month"), pr["by_hour"]
    c = st.columns(2)
    c[0].metric("Average delay at a station", f"{mean_delay:.0f} min")
    c[1].metric(f"Arrivals more than {LATE} min late", f"{late_pct:.0f}%", help="About half of all arrivals.")
    month["month_name"] = [dt.date(2000, int(m), 1).strftime("%b") for m in month["month"]]
    tip = lambda k, t: [alt.Tooltip(k, title=t), alt.Tooltip("mean_delay:Q", title="Average delay (min)", format=".0f"),
                        alt.Tooltip("late_pct:Q", title=f"% more than {LATE} min late", format=".0f")]
    c = st.columns(2)
    with c[0]:
        st.markdown("**By month**: worst in winter, when fog slows trains in the north (Dec to Feb)")
        show(bars(month, "month_name:N", "mean_delay:Q", tip("month_name:N", "Month"), x_title=None,
                  y_title="Average delay (min)", sort=list(month["month_name"])))
    with c[1]:
        st.markdown("**By hour of the day**: worst for trains due at night")
        show(bars(hour, "hour:O", "mean_delay:Q", tip("hour:O", "Hour"), x_title="Scheduled arrival hour",
                  y_title="Average delay (min)"))


def tab_forecast(pr: dict, lm: S.LoadedModel, tt, tt_key: tuple):
    if lm.model is None:
        st.error(f"The model cannot be served, so forecasts are off (the other tabs work): {lm.error}")
        return
    st.markdown("Pick a train, the last station it passed and how late it was there. The app predicts how late it "
                "will be at every station after that.")
    if tt is None:
        st.warning("The timetable is missing. Run `python -m trackcast.download`.")
        return
    in_tt = set(tt.schedule["train_no"].unique().tolist())
    labels = {k: v for k, v in pr["train_labels"].items() if k in in_tt}
    c = st.columns([2, 1])
    train_no = c[0].selectbox("Train (type to search)", list(labels), format_func=labels.get, key="fc_train")
    date = c[1].date_input("Date of the journey", value=dt.date.today(), key="fc_date")
    route = tt.route(train_no)
    stops = route[~route["is_terminus"]]
    stop_label = {int(r.station_no): f"{r.station_name} · {r.station_full_name or ''} "
                                     f"(due {FC.sched_clock(r.sched_dep_offset_min)})" for r in stops.itertuples()}
    c = st.columns([2, 1])
    current = c[0].selectbox("Last station passed", list(stop_label), format_func=stop_label.get,
                             index=min(2, len(stop_label) - 1), key="fc_stop")
    delay_now = c[1].number_input("Minutes late there", 0, 1440, 30, step=5, key="fc_delay")

    out = cached_forecast((str(lm.path), S.fingerprint(lm.path)), tt_key, int(train_no), date,
                          ((int(current), float(delay_now)),), lm.model, tt)
    fc = out[out["kind"] == "forecast"]
    if fc.empty:
        st.info("No stations left to predict.")
        return
    nxt, term = fc.iloc[0], fc.iloc[-1]
    c = st.columns(3)
    c[0].metric(f"Next station: {nxt['station_name']}", f"{nxt['pred_delay_min']:.0f} min late")
    c[1].metric(f"Chance of being more than {LATE} min late there", f"{100 * nxt['p_late']:.0f}%")
    c[2].metric(f"Last station: {term['station_name']}", f"{term['pred_delay_min']:.0f} min late")

    hist = pr["train_stops"]
    hist = hist[hist["train_no"] == train_no][["station_no", "median_delay"]]
    out = out.merge(hist, on="station_no", how="left")
    seen = out[out["kind"] == "observed"].assign(series="Delay you entered")
    path = pd.concat([seen.tail(1), out[out["kind"] == "forecast"]]).assign(series="Predicted delay")
    typical = out[out["median_delay"].notna()].assign(series="Usual delay of this train",
                                                      delay=lambda d: d["median_delay"])
    long = pd.concat([typical, seen, path], ignore_index=True)
    long["sched"] = long["sched_arr_offset_min"].fillna(long["sched_dep_offset_min"]).map(FC.sched_clock)
    long["p_late_pct"] = 100 * long["p_late"]
    domain = ["Delay you entered", "Predicted delay", "Usual delay of this train"]
    show(hover_lines(long, "distance_from_origin:Q", "delay:Q", "series:N", domain, [BLUE, ORANGE, AQUA],
                     tooltip=[alt.Tooltip("station_name:N", title="Station"), alt.Tooltip("sched:N", title="Due at"),
                              alt.Tooltip("series:N", title="Line"), alt.Tooltip("delay:Q", title="Minutes late", format=".0f"),
                              alt.Tooltip("p_late_pct:Q", title=f"Chance of > {LATE} min late (%)", format=".0f")],
                     x_title="Distance from the starting station (km)", y_title="Minutes late", height=340,
                     dashed=["Usual delay of this train"]))
    st.caption("\"Usual delay\" is the middle value over this train's journeys in the past year.")

    sched = fc["sched_arr_offset_min"].fillna(fc["sched_dep_offset_min"])
    table = pd.DataFrame({"Station": fc["station_name"] + " · " + fc["station_full_name"].fillna(""),
                          "Due at": sched.map(FC.sched_clock),
                          "Predicted minutes late": fc["pred_delay_min"].round(0),
                          f"Chance of > {LATE} min late": (100 * fc["p_late"]).round(0),
                          "Expected arrival": (sched + fc["pred_delay_min"]).map(FC.sched_clock)})
    st.dataframe(table, hide_index=True,
                 column_config={f"Chance of > {LATE} min late": st.column_config.ProgressColumn(
                     format="%d%%", min_value=0, max_value=100)})


def tab_cascade(pr: dict):
    st.markdown("**Once a train is late, it usually stays late.** We took every station stop and looked at the same "
                f"train a few stations later. Each line is a group of trains, by how late they were at the start; "
                f"it shows the share still more than {LATE} minutes late.")
    cas = pr["cascade"].sort_values(["bucket", "h"])
    cas = cas.assign(start=cas["initial_delay"].map(START_DELAY).fillna(cas["initial_delay"]))
    labels = list(dict.fromkeys(cas["start"]))
    tip = [alt.Tooltip("start:N", title="Late at the start"), alt.Tooltip("h:Q", title="Stations later"),
           alt.Tooltip("late_pct_h:Q", title="% still late", format=".0f"),
           alt.Tooltip("mean_change:Q", title="Minutes gained (+) or made up (-)", format="+.0f")]
    show(hover_lines(cas, "h:Q", "late_pct_h:Q", "start:N", labels, ORDINAL[: len(labels)], tip,
                     x_title="Stations later", y_title=f"% still more than {LATE} min late", height=360))
    st.caption("So the simplest guess, \"the train will be as late as it is now\", is already a good guess. "
               "Our model has to beat it.")


MODELS_TABLE = """| Model | Used in | How it works, in simple words | What happened |
|---|---|---|---|
| **Logistic regression** | Step 1 (tried) | Adds up the inputs with a weight for each (e.g. "last delay counts a lot"), then turns the total into a chance between 0% and 100%. | Simple and fast, but it can only draw one straight boundary between late and on time, so it made more mistakes. |
| **Random forest** | Step 1 (**used**) | {trees} decision trees. Each tree asks a chain of yes/no questions ("Was the train more than 30 min late at the last station?", "Is there spare time before the next one?"). Each tree learns from a different random sample of past stops, then they vote: the share of "late" votes is the chance. | Best at telling late from on time. We then adjust its chances so that "80%" really means late about 8 times in 10 (calibration). |
| **Ridge regression** | Step 2 (tried) | One straight-line formula: minutes late = a x last delay + b x spare time + ... "Ridge" keeps the weights small, so it does not overreact to any one input. | Too simple: real delays do not follow one straight line. |
| **Gradient boosting** | Step 2 (**used**) | Builds up to {rounds} small trees one after another. Each new tree looks at the mistakes the trees before it still make and corrects them a little. | Most accurate for the minutes. |
"""


def tab_model(report: dict | None):
    if not report:
        st.info("No model report yet. Run `python -m trackcast.train`.")
        return
    cv, t, m = report["cv"], report["test"], report["meta"]
    st.markdown(f"""**How the prediction works: two steps**

1. **Will the train be more than {LATE} minutes late?** A *random forest* gives the chance, e.g. 80%.
2. **How many minutes late?** A *gradient boosting* model predicts the minutes, once assuming the train is late
   and once assuming it is on time. It starts from the simplest guess (the delay stays the same) and predicts
   only the *change*: how many minutes the train will gain or make up.

The final prediction mixes the two answers using the chance from step 1. The models look at how late the train
was at the last stations, the spare time in the timetable ahead, the station, the time of day and the season.""")

    st.markdown("**The models we tried**")
    st.markdown(MODELS_TABLE.format(trees=C.RF_TREES, rounds=500))
    st.markdown(f"**How we picked them:** each model learned from about {m['train_rows'] / 1e6:.0f} million past "
                "station stops (Feb to Nov 2025). We first compared them on a practice test (Sep, Oct and Nov, "
                "each predicted from the months before it) and kept the best one for each step. **How we tested "
                "the final model:** once, on Dec 2025 to Feb 2026, months it had never seen. We compare it with "
                "the simplest guess: *the train will be as late as it is now*.")

    rows = {"persistence": "Simplest guess (delay stays the same)",
            f"direct_{cv['selected']['regressor']}": "One-step model (gradient boosting only)",
            t["selected"]: "TrackCast (two steps)"}
    overall = pd.DataFrame([{"Method": name, "RMSLE (lower is better)": t["overall"][k]["rmsle"],
                             "Average error (min)": t["overall"][k]["mae_min"],
                             "Better than the simplest guess (%)": t["overall"][k]["skill_vs_persistence_pct"]}
                            for k, name in rows.items() if k in t["overall"]])
    sel, pers = t["overall"][t["selected"]]["rmsle"], t["overall"]["persistence"]["rmsle"]
    c = st.columns([3, 2])
    with c[0]:
        st.markdown("**Results on the test months**")
        st.dataframe(overall, hide_index=True, width="stretch",
                     column_config={"RMSLE (lower is better)": st.column_config.NumberColumn(format="%.3f"),
                                    "Average error (min)": st.column_config.NumberColumn(format="%.1f"),
                                    "Better than the simplest guess (%)": st.column_config.NumberColumn(format="%.0f")})
        st.markdown(f"""**What is RMSLE?** *Root Mean Squared Logarithmic Error*, the project's main score. For every
station stop: take log(1 + minutes) of the predicted and of the real delay, subtract, and square the difference.
Average over all stops and take the square root. 0 would be perfect; lower is better.

**Why RMSLE and not plain minutes?** Delays are very uneven: half of all arrivals are less than 16 minutes late,
but 1 in 100 is more than 5 hours late. With a plain error, those few huge delays would decide the score.
The log compares an error with the size of the delay:

| Real delay | Predicted | Off by | RMSLE error for this stop |
|---|---|---|---|
| 10 min | 20 min | 10 min | 0.65: big, the prediction is double the real delay |
| 300 min | 310 min | 10 min | 0.03: small, only 3% too high |

The model also learns log(1 + delay), so RMSLE scores exactly what it is trained on. We still show the average
error in minutes, because it is easier to picture.

**"Better than the simplest guess"** = how much lower the RMSLE is: (1 - {sel:.3f} / {pers:.3f}) x 100 =
{100 * (1 - sel / pers):.0f}% for TrackCast.""")
        st.caption("The one-step model scores slightly better on RMSLE, but TrackCast also gives the chance of being "
                   "late and is closer in minutes, so the app uses TrackCast.")
    with c[1]:
        st.markdown("**What the model relies on most**")
        imp = pd.DataFrame(t["importance"])
        imp = imp.assign(group=imp["group"].map(FEATURE_GROUP).fillna(imp["group"]))
        show(bars(imp, "group:N", "rmsle_increase:Q",
                  [alt.Tooltip("group:N", title="Information"),
                   alt.Tooltip("rmsle_increase:Q", title="Extra error without it", format=".2f")],
                  x_title=None, y_title="How much worse without it", sort="-x", horizontal=True, height=260))
        st.caption("By far the most important: how late the train was at the last stations.")


def skill_dots(df: pd.DataFrame, label: str, height: int):
    """How much better than persistence per group: dot + 95% range, sorted, one blue series, hover tooltip."""
    order = list(df.sort_values("skill_pct", ascending=False)[label])
    y = alt.Y(f"{label}:N", sort=order, title=None)
    tip = [alt.Tooltip(f"{label}:N", title="Group"),
           alt.Tooltip("skill_pct:Q", title="Better than the simplest guess (%)", format=".0f"),
           alt.Tooltip("skill_lo:Q", title="Range: from (%)", format=".0f"),
           alt.Tooltip("skill_hi:Q", title="Range: to (%)", format=".0f"),
           alt.Tooltip("rows:Q", title="Station stops tested", format=",")]
    base = alt.Chart(df).encode(y=y)
    rule = base.mark_rule(color=ORDINAL[1], strokeWidth=3).encode(
        x=alt.X("skill_lo:Q", title="Better than the simplest guess (%)"), x2="skill_hi:Q", tooltip=tip)
    dots = base.mark_point(filled=True, size=90, color=BLUE, opacity=1).encode(x="skill_pct:Q", tooltip=tip)
    return (rule + dots).properties(height=height)


def _results_frames(res_json: dict | None, res: dict | None):
    """(class, corridor, bottlenecks, full_year). The committed full-year results (reports/results.json) when
    present, so the hosted demo shows the same numbers as the report; else the served source's own tables."""
    if res_json:
        return (pd.DataFrame(res_json["class"]), pd.DataFrame(res_json["corridor"]),
                pd.DataFrame(res_json["bottlenecks"]), True)
    if res is None:
        return None
    return res["scores_class"], res["scores_corridor"], res["through"], False


def tab_results(res_json: dict | None, res: dict | None, full: bool):
    frames = _results_frames(res_json, res)
    if frames is None:
        st.info(f"No results for this data yet. Run `python -m trackcast.results{' --full' if full else ''}`.")
        return
    cls, cor, thr, full_year = frames
    st.markdown("**Does the model work for every kind of train?**")
    by = st.radio("Group by", ["Type of train", "Route (railway zones)"], horizontal=True, key="res_by",
                  label_visibility="collapsed")
    if by == "Type of train":
        df = cls.copy()
        label = "label" if "label" in df.columns else "group"
    else:
        df, label = cor.assign(name=cor["group"].map(corridor_name)), "name"
    df = df.head(10).dropna(subset=["skill_pct"]).reset_index(drop=True)
    best, worst = df.loc[df["skill_pct"].idxmax()], df.loc[df["skill_pct"].idxmin()]
    c = st.columns([3, 2])
    with c[0]:
        show(skill_dots(df, label, height=max(200, 34 * len(df))))
    with c[1]:
        what = "type of train" if by == "Type of train" else "route (the railway zones where the train starts and ends)"
        st.markdown(f"""**How to read this chart**
- Each row is one {what}.
- The **dot** shows how much better TrackCast is than the simplest guess ("the train will be as late as it is
  now") for that group, on the test months (Dec 2025 to Feb 2026). 33% means its error (RMSLE) is a third lower.
- The **line** is the range we are 95% sure of. We re-scored the model 1,000 times on randomly re-picked test
  days. A short line means many test stops, so we are sure; a long line means fewer stops.
- A line that crossed 0 would mean the model might be no better than the simple guess for that group.""")
        crosses = df[df["skill_lo"] <= 0]
        verdict = ("every line is well above 0, so the model helps every group" if crosses.empty else
                   f"for {', '.join(crosses[label])} the line reaches 0, so there we cannot be sure it helps")
        st.markdown(f"**What it shows:** every group is between {df['skill_pct'].min():.0f}% and "
                    f"{df['skill_pct'].max():.0f}% better, and {verdict}. Best: **{best[label]}** "
                    f"({best['skill_pct']:.0f}%). Lowest: **{worst[label]}** ({worst['skill_pct']:.0f}%).")
        if full_year:
            st.caption("Why the difference? The model gains most where trains usually run on time and small delays "
                       "come and go. Where trains are already very late, the delay mostly just stays, so the "
                       "simple guess is hard to beat.")

    st.markdown("**Stations that add the most delay** (bottlenecks): total minutes of delay added each day by all "
                "trains passing through the station")
    if not full_year:
        thr = thr[thr["station_class"].isin(C.BOTTLENECK_CLASSES)
                  & (thr["n_through"] >= C.PROFILE_MIN_OBS["station"][1 if full else 0])]
    top = thr.nlargest(10, "minutes_per_day")
    top = top.assign(label=top["station_name"] + " · " + top["station_full_name"].fillna(""))
    show(bars(top, "label:N", "minutes_per_day:Q",
              [alt.Tooltip("label:N", title="Station"),
               alt.Tooltip("minutes_per_day:Q", title="Minutes added per day", format=",.0f"),
               alt.Tooltip("through_change:Q", title="Minutes added per train", format="+.0f"),
               alt.Tooltip("through_per_day:Q", title="Trains per day", format=".0f")],
              x_title=None, y_title="Minutes of delay added per day", height=340, sort="-x", horizontal=True))
    st.caption("Kanpur Central (CNB), on the busy Delhi line, and Kharagpur (KGP), in the east, add the most.")


# --------------------------------------------------------------------------------------------------
TABS = ["Overview", "Predict a train's delay", "How delays spread", "How the model works", "Results & bottlenecks"]


def main():
    st.title("🚆 TrackCast")
    st.caption("Forecasting Cascading Train Delays and Schedule Recovery: Abhay Nagesh Rao (26CSM1R01), "
               "DSF Project, NIT Warangal.")
    sources = {s.name: s for s in S.available_sources()}
    if not sources:
        st.error("No artifacts found. Run `python -m trackcast.train` and `python -m trackcast.profiles` (or build "
                 "the demo bundle: `python -m trackcast.serving demo`).")
        st.stop()
    # A picker only when there is a choice (local builds); the hosted demo has one source, so none is shown.
    name = st.sidebar.radio("Data", list(sources)) if len(sources) > 1 else next(iter(sources))
    src = sources[name]
    st.sidebar.markdown(f"A train counts as **late** if it arrives more than **{LATE} minutes** after its "
                        "scheduled time.")
    pr_fp = S.fingerprint(src.profiles)
    pr = load_profiles(str(src.profiles), pr_fp)
    pr = {**pr, "train_labels": train_labels(str(src.profiles), pr_fp, pr["trains"])}
    lm = load_model(str(src.model), S.fingerprint(src.model))
    if lm.model is None:   # only a real problem is shown; version notes are in `python -m trackcast.serving check`
        st.sidebar.error(f"Model not served: {lm.error}")
    tt_key = S.fingerprint(*src.timetable_files())
    tabs = st.tabs(TABS)
    with tabs[0]:
        tab_overview(pr, load_report(str(C.DATA_QUALITY_JSON), S.fingerprint(C.DATA_QUALITY_JSON)),
                     load_report(str(C.PROFILES_JSON), S.fingerprint(C.PROFILES_JSON)))
    with tabs[1]:
        tab_forecast(pr, lm, load_timetable(src, tt_key), tt_key)
    with tabs[2]:
        tab_cascade(pr)
    with tabs[3]:
        tab_model(model_report())
    with tabs[4]:
        tab_results(load_report(str(C.RESULTS_JSON), S.fingerprint(C.RESULTS_JSON)),
                    load_results(str(src.results), S.fingerprint(src.results)), src.full)


main()
