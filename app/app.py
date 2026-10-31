"""
Dead Internet Observatory — Streamlit Dashboard
Research-grade interface tracking the Internet Aliveness Index.

An experimental public research instrument for sampled web text.
"""

import sqlite3
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from huggingface_hub import hf_hub_download
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
CONFIG_PATH = str(ROOT / "config" / "config.yaml")

try:
    from analytics.anomaly_detector import label_anomalies
    ANALYTICS_OK = True
except ImportError:
    def label_anomalies(df, col):  # noqa: E302
        return df
    ANALYTICS_OK = False


CACHE_TTL_SECONDS = 300
DATASET_REPO = "jupiternull/dead-internet-observatory"
DATASET_API_URL = f"https://huggingface.co/api/datasets/{DATASET_REPO}"

REQUIRED_DATABASE_SCHEMA = {
    "composite_index": {
        "date", "aliveness_index", "smoothed_index", "n_docs",
        "anomaly_flag", "anomaly_reason",
    },
    "daily_index": {"date", "source", "mean_score", "aliveness_index", "n_docs"},
    "meta": {"key", "value"},
}

EXPERIMENTAL_SIGNAL_LABEL = "Experimental synthetic-content signal"


@st.cache_resource()
def _database_path_state() -> dict:
    return {"last_valid_path": None, "lock": threading.Lock()}


def _http_session() -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=8)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _dataset_revision() -> str:
    response = _http_session().get(DATASET_API_URL, timeout=(5, 20))
    response.raise_for_status()
    return response.json()["sha"]


@st.cache_resource(ttl=CACHE_TTL_SECONDS)
def _database_path(revision: str) -> str:
    path = hf_hub_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        filename="observatory.db",
        revision=revision,
    )
    _validate_database(path)
    state = _database_path_state()
    with state["lock"]:
        state["last_valid_path"] = path
    return path


def _validate_database(path: str) -> None:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        result = conn.execute("PRAGMA quick_check").fetchone()
        if result is None or result[0] != "ok":
            raise sqlite3.DatabaseError("database quick_check failed")
        for table, required_columns in REQUIRED_DATABASE_SCHEMA.items():
            columns = {
                row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')
            }
            if not required_columns.issubset(columns):
                raise sqlite3.DatabaseError(
                    f"database is missing required schema for {table}"
                )


def _resolve_database_path() -> str:
    try:
        return _database_path(_dataset_revision())
    except Exception:
        state = _database_path_state()
        with state["lock"]:
            fallback = state["last_valid_path"]
        if fallback is None:
            raise
        _validate_database(fallback)
        return fallback


def _query(path: str, sql: str, params: tuple = ()) -> list:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


P = {
    "bg": "#f7f5ef", "ink": "#202e3d", "muted": "#56616c",
    "blue": "#24649a", "rust": "#9d4a36", "rule": "#d8d8cf",
}

PLOTLY_BASE = dict(
    template="plotly_white",
    paper_bgcolor=P["bg"], plot_bgcolor=P["bg"],
    font=dict(family="Arial, sans-serif", size=13, color=P["ink"]),
    hoverlabel=dict(bgcolor="#ffffff", font_size=13),
)

CSS = """
<style>
html { color-scheme: light; }
.stApp { background: #f7f5ef; color: #202e3d; }
.block-container, [data-testid="stMainBlockContainer"] {
    max-width: 1180px; padding-top: 3rem; padding-bottom: 3rem;
}
.stApp h1, .stApp h2, .stApp h3 {
    font-family: Georgia, 'Times New Roman', serif; color: #202e3d;
    font-weight: 400; letter-spacing: -.025em;
}
.stApp h1 { font-size: clamp(2.5rem, 5vw, 4.5rem); line-height: 1.08; padding-top: .6rem; }
.stApp h2 { font-size: 2rem; }
.stApp p, .stApp li { line-height: 1.65; }
.stApp a { color: #24649a; text-underline-offset: .2em; }
.stApp a:focus-visible, .stApp button:focus-visible,
.stApp input:focus-visible, .stApp [tabindex]:focus-visible {
    outline: 3px solid #24649a !important; outline-offset: 3px;
}
.kicker { font-size: .75rem; font-weight: 600; letter-spacing: .14em;
    text-transform: uppercase; color: #56616c; border-top: 2px solid #202e3d; padding-top: 1rem; }
.deck { font-size: 1.2rem; max-width: 760px; color: #56616c; margin: .4rem 0 1.6rem; }
.measure { border-top: 1px solid #d8d8cf; padding: 1.3rem 0 1rem; }
.measure-label { font-size: .78rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; }
.measure-value { font-family: Georgia, 'Times New Roman', serif; font-size: 5rem;
    color: #24649a; line-height: 1.15; font-variant-numeric: lining-nums tabular-nums; }
.measure-value small { font-family: Arial, sans-serif; font-size: 1.1rem; color: #56616c; }
.measure-note { color: #56616c; font-size: .9rem; margin-top: .45rem; }
.section { border-top: 1px solid #d8d8cf; margin-top: 2rem; padding-top: 1rem; }
.scope { border-left: 3px solid #9d4a36; padding-left: 1rem; margin: 1rem 0; max-width: 780px; }
[data-testid="stMetricValue"] { font-size: 1.6rem; font-variant-numeric: tabular-nums; }
[data-testid="stMetricLabel"] { color: #56616c; }
[data-testid="stExpander"] { border-color: #d8d8cf; border-radius: 3px; }
@media (max-width: 640px) {
    .block-container, [data-testid="stMainBlockContainer"] { padding-top: 1.5rem; }
    .measure-value { font-size: 4rem; }
    .deck { font-size: 1.05rem; }
}
</style>
"""


PANEL = ('hackernews', 'mastodon', 'steam', 'youtube', 'bluesky', 'fourchan')
TIMELINE_DIR = ROOT / 'data' / 'timeline'


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_timeline_csv(kind: str, directory: str = str(TIMELINE_DIR)) -> pd.DataFrame:
    if kind not in ('cc_crawls', 'biweekly'):
        raise ValueError('Timeline kind must be cc_crawls or biweekly')
    column = 'median_date' if kind == 'cc_crawls' else 'period_start'
    return pd.read_csv(Path(directory) / f'{kind}.csv', parse_dates=[column])


def compute_composite(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """Equal mean of panel medians, only for complete adequately sampled weeks.

    Platform selection never changes the panel or its weights.
    """
    columns = ['period_start', 'median']
    if weekly_df.empty:
        return pd.DataFrame(columns=columns)
    panel = weekly_df.loc[weekly_df['source'].isin(PANEL)].copy()
    rows = []
    for week, group in panel.groupby('period_start'):
        if (len(group) == len(PANEL) and set(group['source']) == set(PANEL)
                and group['n'].ge(100).all() and group['median'].notna().all()):
            rows.append({'period_start': week, 'median': group['median'].mean()})
    return pd.DataFrame(rows, columns=columns).sort_values('period_start').reset_index(drop=True)


def platform_options(frame):
    return sorted(set(frame['source']) - {'common_crawl'})


def split_timeline_gaps(df: pd.DataFrame, date_column: str) -> list:
    ordered = df.sort_values(date_column).copy()
    dates = pd.to_datetime(ordered[date_column])
    gaps = dates.diff().dt.days.gt(120 if date_column == 'median_date' else 14)
    return [group for _, group in ordered.groupby(gaps.cumsum())]


def score_axis_range(*frames, columns=('median', 'p25', 'p75'), pad=3.0):
    """Fit the score axis to plotted values with padding, clamped to 0–100."""
    parts = [f[c] for f in frames if f is not None and not f.empty for c in columns if c in f.columns]
    values = pd.concat(parts, ignore_index=True).dropna() if parts else pd.Series(dtype=float)
    if values.empty:
        return [45.0, 70.0]
    return [max(0.0, float(values.min()) - pad), min(100.0, float(values.max()) + pad)]


def chart_source_medians(df, date_column, *, band=False, spread=False, composite=None):
    fig = go.Figure()
    colors = ['#24649a', '#9d4a36', '#38836a', '#81679c', '#b08025', '#527c89', '#a95177', '#646464', '#758d35', '#c07142']
    for color, (source, observations) in zip(colors, df.groupby('source', sort=True)):
        for i, run in enumerate(split_timeline_gaps(observations, date_column)):
            bands = [('ci_low', 'ci_high', .18)] if band else []
            if spread:
                bands.insert(0, ('p25', 'p75', .07))
            for lower, upper, opacity in bands:
                fig.add_trace(go.Scatter(x=run[date_column], y=run[lower], mode='lines',
                                         line=dict(width=0), showlegend=False, hoverinfo='skip', connectgaps=False))
                fig.add_trace(go.Scatter(x=run[date_column], y=run[upper], mode='lines',
                                         line=dict(width=0), fill='tonexty', fillcolor=f'rgba(36,100,154,{opacity})',
                                         showlegend=False, hoverinfo='skip', connectgaps=False))
            fig.add_trace(go.Scatter(
                x=run[date_column], y=run['median'], mode='lines+markers',
                name=source.replace('_', ' ').title(), legendgroup=source, showlegend=i == 0,
                line=dict(color=color, width=2), marker=dict(size=4), connectgaps=False,
                customdata=run[['n', 'p25', 'p75']],
                hovertemplate='%{x|%d %b %Y}<br>Median: %{y:.1f}<br>Documents: %{customdata[0]:,.0f}'
                              '<br>IQR: %{customdata[1]:.1f}–%{customdata[2]:.1f}<extra>%{fullData.name}</extra>'))
    if composite is not None and not composite.empty:
        for i, run in enumerate(split_timeline_gaps(composite, 'period_start')):
            fig.add_trace(go.Scatter(x=run['period_start'], y=run['median'], mode='lines+markers',
                                     name='Fixed panel composite', showlegend=i == 0, legendgroup='composite',
                                     line=dict(color='#202e3d', width=4, dash='dash'), connectgaps=False))
    if date_column == 'median_date':
        fig.add_shape(type='line', x0='2022-11-30', x1='2022-11-30', y0=0, y1=1,
                      yref='paper', line=dict(dash='dash', color=P['rust']))
        fig.add_annotation(x='2022-11-30', y=1, yref='paper', text='ChatGPT launch',
                           showarrow=False, yshift=12)
    fig.update_layout(**PLOTLY_BASE, height=440, margin=dict(l=20, r=20, t=70, b=30),
                      yaxis=dict(title='Observational score · 0–100', gridcolor=P['rule'],
                                 range=score_axis_range(df, composite, columns=('median',) + (('ci_low', 'ci_high') if band else ()) + (('p25', 'p75') if spread else ()))),
                      xaxis=dict(title='Median WARC date per crawl' if date_column == 'median_date' else 'Two-week period starting Monday', showgrid=False),
                      legend=dict(orientation='h', y=1.15), hovermode='closest')
    return fig


def common_crawl_caption(monthly):
    """Describe observed crawl medians, without inventing annual coverage."""
    years = monthly['median_date'].dt.year
    periods = [('2014', years.eq(2014)), ('2017–19', years.between(2017, 2019)),
               ('2020', years.eq(2020)), ('since 2023', years.ge(2023))]
    summaries = [monthly.loc[mask, 'median'].median() for _, mask in periods]
    values = [f"{label}: {value:.1f}" for (label, _), value in zip(periods, summaries) if pd.notna(value)]
    narrative = ('This one-source series shows a gradual decline that predates the ChatGPT launch. '
                 if all(pd.notna(value) for value in summaries)
                 and all(a > b for a, b in zip(summaries, summaries[1:]))
                 else 'This record describes one source; period coverage and scores may vary. ')
    return (narrative +
            'Median of observed crawl medians (' + '; '.join(values) + '). '
            'Each point represents one crawl at its median WARC date; lines and bands break across gaps over 120 days. '
            'Documents have at least 100 tokens and are deduplicated within source. '
            'Crawls require at least 50 documents. The main band is a bootstrap 95% confidence interval of the median (1000 resamples, fixed seed, at most 20000 sampled documents); optional faint p25–p75 bands show document spread.')


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_timeline(path: str, days: int = None) -> pd.DataFrame:
    cutoff = ((datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
              if days is not None else "0001-01-01")
    data = _query(
        path,
        """SELECT date, aliveness_index, smoothed_index, n_docs,
                  anomaly_flag, anomaly_reason
           FROM composite_index
           WHERE date >= ?
           ORDER BY date ASC""",
        (cutoff,),
    )
    df = pd.DataFrame(data)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        # Keep the existing minimum sample threshold for displayed observations.
        df = df[df["n_docs"] >= 20].reset_index(drop=True)
        return label_anomalies(df, "aliveness_index")
    return df


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_sources(path: str) -> pd.DataFrame:
    return pd.DataFrame(_query(
        path,
        """SELECT source, date,
                  CASE WHEN SUM(n_docs) > 0
                       THEN SUM(mean_score * n_docs) / SUM(n_docs)
                       ELSE AVG(mean_score)
                  END AS mean_score,
                  SUM(n_docs) AS n_docs
           FROM daily_index
           WHERE mean_score IS NOT NULL
             AND date = (
                 SELECT MAX(latest.date)
                 FROM daily_index latest
                 WHERE latest.source = daily_index.source
                   AND latest.mean_score IS NOT NULL
             )
           GROUP BY source, date
           ORDER BY source"""
    ))


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_score(path: str) -> float:
    data = _query(
        path,
        "SELECT smoothed_index FROM composite_index ORDER BY date DESC LIMIT 1"
    )
    if not data:
        raise RuntimeError("composite_index returned no current score")
    return float(data[0]["smoothed_index"])


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_total_docs(path: str) -> int:
    data = _query(path, "SELECT value FROM meta WHERE key = 'total_scored_count'")
    if not data:
        raise RuntimeError("meta.total_scored_count returned no value")
    return int(data[0]["value"])


def _table_columns(path: str, table: str) -> set:
    rows = _query(path, f'PRAGMA table_info("{table}")')
    return {row["name"] for row in rows}


def _dashboard_panel_settings(config_path: str = CONFIG_PATH):
    """Read panel weights from config without requiring PyYAML at dashboard runtime."""
    weights = {}
    panel = []
    minimum = 50
    section = None
    subsection = None
    for raw in Path(config_path).read_text().splitlines():
        code = raw.split("#", 1)[0].rstrip()
        if not code.strip():
            continue
        indent = len(code) - len(code.lstrip(" "))
        line = code.strip()
        if indent == 0 and line.endswith(":"):
            section = line[:-1].strip()
            subsection = None
            continue
        if section != "analytics":
            continue
        if indent == 2 and line.endswith(":"):
            subsection = line[:-1].strip()
            continue
        if subsection == "source_weights" and indent >= 4 and ":" in line:
            key, value = line.split(":", 1)
            weights[key.strip()] = float(value.strip())
        elif subsection == "core_source_panel" and line.startswith("- "):
            panel.append(line[2:].strip())
        elif line.startswith("minimum_source_samples:"):
            minimum = int(float(line.split(":", 1)[1].strip()))
            subsection = None
    if not panel:
        panel = list(weights)
    return weights, panel, minimum


def _headline_view(**overrides) -> dict:
    view = {
        "date": None,
        "score": None,
        "n_docs": None,
        "source_coverage": None,
        "coverage_sufficient": None,
        "sample_sufficient": None,
        "ci_low": None,
        "ci_high": None,
        "withheld": True,
        "reason": "No composite row is available.",
        "quality_available": False,
    }
    view.update(overrides)
    return view


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_headline(path: str) -> dict:
    """Latest smoothed score, hidden when stored coverage or sample gates fail.

    Snapshots that predate index_quality still return the stored smoothed value.
    """
    rows = _query(
        path,
        "SELECT date, smoothed_index, n_docs FROM composite_index ORDER BY date DESC LIMIT 1",
    )
    quality_cols = _table_columns(path, "index_quality")
    if not rows:
        return _headline_view(quality_available=bool(quality_cols))
    row = rows[0]
    score = row["smoothed_index"]
    score = float(score) if score is not None and pd.notna(score) else None
    view = _headline_view(
        date=row["date"],
        score=score,
        n_docs=row["n_docs"],
        withheld=score is None,
        reason=None if score is not None else "No smoothed score is available for the latest day.",
        quality_available=False,
    )
    needed = {"source_coverage", "coverage_sufficient", "sample_sufficient"}
    if not needed.issubset(quality_cols):
        return view
    select_ci = ", ci_low, ci_high" if {"ci_low", "ci_high"}.issubset(quality_cols) else ""
    quality = _query(
        path,
        "SELECT source_coverage, coverage_sufficient, sample_sufficient"
        f"{select_ci} FROM index_quality WHERE date = ?",
        (row["date"],),
    )
    view["quality_available"] = True
    if quality:
        item = quality[0]
        view["source_coverage"] = item["source_coverage"]
        if item["coverage_sufficient"] is not None:
            view["coverage_sufficient"] = bool(item["coverage_sufficient"])
        if item["sample_sufficient"] is not None:
            view["sample_sufficient"] = bool(item["sample_sufficient"])
        if select_ci:
            view["ci_low"] = item["ci_low"]
            view["ci_high"] = item["ci_high"]
    coverage_ok = view["coverage_sufficient"] is True
    sample_ok = view["sample_sufficient"] is True
    if coverage_ok and sample_ok and score is not None:
        view["withheld"] = False
        view["reason"] = None
        return view
    view["score"] = None
    view["withheld"] = True
    if score is None:
        view["reason"] = "No smoothed score is available for the latest day."
    elif not coverage_ok:
        view["reason"] = "Core-source coverage is below the configured minimum."
    else:
        view["reason"] = "The sample is below the configured minimum."
    return view


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_movement(path: str, config_path: str = CONFIG_PATH):
    """Latest fixed-panel change, split into within-source and composition parts."""
    from analytics.source_behavior import attribute_index_movement

    weights, panel, minimum = _dashboard_panel_settings(config_path)
    rows = _query(
        path,
        """SELECT date, source,
                  SUM(n_docs) AS n_docs,
                  CASE WHEN SUM(n_docs) > 0
                       THEN SUM(mean_score * n_docs) / SUM(n_docs)
                       ELSE AVG(mean_score)
                  END AS mean_score
           FROM daily_index
           WHERE mean_score IS NOT NULL
           GROUP BY date, source""",
    )
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    if panel:
        frame = frame[frame["source"].isin(set(panel))]
    frame = frame[pd.to_numeric(frame["n_docs"], errors="coerce").fillna(0) >= minimum]
    if frame.empty:
        return None
    if not weights:
        weights = {source: 1.0 for source in sorted(set(frame["source"]))}
    movement = attribute_index_movement(frame, weights)
    if movement.empty:
        return None
    last = movement.iloc[-1]
    score_effect = last["score_effect"]
    composition = last["composition_effect"]
    current = last["index"]
    previous = last["previous_index"]
    total = None
    if current is not None and previous is not None and pd.notna(current) and pd.notna(previous):
        total = float(current) - float(previous)
    return {
        "date": last["date"],
        "previous_date": last["previous_date"],
        "score_effect": None if score_effect is None or pd.isna(score_effect) else float(score_effect),
        "composition_effect": None if composition is None or pd.isna(composition) else float(composition),
        "total_delta": total,
        "entered": list(last["entered"]),
        "exited": list(last["exited"]),
    }


def movement_caption(movement: dict) -> str:
    if not movement or movement.get("score_effect") is None or movement.get("composition_effect") is None:
        return "Score movement cannot be split until two dated panel observations exist."
    total = movement.get("total_delta")
    if total is None:
        total = movement["score_effect"] + movement["composition_effect"]
    return (
        f"Since {movement['previous_date']}, fixed-panel source means moved {total:+.2f}. "
        f"Change within sources present on both days: {movement['score_effect']:+.2f}. "
        f"Source-composition change: {movement['composition_effect']:+.2f}."
    )


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_source_intervals(path: str) -> pd.DataFrame:
    needed = {"date", "source", "ci_low", "ci_high"}
    if not needed.issubset(_table_columns(path, "source_interval")):
        return pd.DataFrame(columns=["date", "source", "ci_low", "ci_high"])
    return pd.DataFrame(_query(
        path,
        "SELECT date, source, ci_low, ci_high FROM source_interval",
    ))


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_platform_trends(path: str) -> pd.DataFrame:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=180)).date().isoformat()
    data = _query(
        path,
        """SELECT date, source, aliveness_index
           FROM daily_index
           WHERE source IN ('reddit', 'hackernews', 'bluesky', 'youtube', 'fourchan', 'steam')
             AND date >= ?
           ORDER BY date ASC""",
        (cutoff,),
    )
    df = pd.DataFrame(data)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_coverage(path: str) -> dict:
    rows = _query(path, """SELECT MIN(date) AS first_date, MAX(date) AS last_date,
                                   COUNT(DISTINCT source) AS source_count
                            FROM daily_index""")
    latest = _query(path, """SELECT date, n_docs FROM composite_index
                              ORDER BY date DESC LIMIT 1""")
    return {**rows[0], "latest": latest[0] if latest else None}


def chart_timeline(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    # Separate runs so unobserved dates are never drawn as a continuous trend.
    runs = df.sort_values("date").copy()
    runs["run"] = runs["date"].diff().dt.days.gt(1).cumsum()
    for col, name, color, width in (
        ("aliveness_index", "Daily composite", P["muted"], 1),
        ("smoothed_index", "Published smoothed index", P["blue"], 2.5),
    ):
        for i, (_, group) in enumerate(runs.groupby("run")):
            fig.add_trace(go.Scatter(
                x=group["date"], y=group[col], mode="lines+markers",
                name=name, legendgroup=col, showlegend=i == 0,
                line=dict(color=color, width=width),
                marker=dict(size=3 if col == "aliveness_index" else 4),
                customdata=group["n_docs"], connectgaps=False,
                hovertemplate="%{x|%d %b %Y}<br>" + name +
                    ": %{y:.1f}<br>Documents: %{customdata:,.0f}<extra></extra>",
            ))
    if {"ci_low", "ci_high"}.issubset(df.columns):
        for i, (_, group) in enumerate(runs.groupby("run")):
            fig.add_trace(go.Scatter(
                x=group["date"], y=group["ci_low"], mode="lines",
                line=dict(width=0), showlegend=False, hoverinfo="skip",
                connectgaps=False, legendgroup="ci",
            ))
            fig.add_trace(go.Scatter(
                x=group["date"], y=group["ci_high"], mode="lines",
                line=dict(width=0), fill="tonexty",
                fillcolor="rgba(36,100,154,0.15)",
                name="Sampling interval", legendgroup="ci", showlegend=i == 0,
                hoverinfo="skip", connectgaps=False,
            ))
    flags = df[df["anomaly_flag"].fillna(0).astype(bool)]
    if not flags.empty:
        fig.add_trace(go.Scatter(
            x=flags["date"], y=flags["aliveness_index"], mode="markers",
            name="Published deviation flag", marker=dict(color=P["rust"], size=8, symbol="circle-open"),
            hovertemplate="%{x|%d %b %Y}<br>Flagged daily score: %{y:.1f}<extra></extra>",
        ))
    fig.update_layout(
        **PLOTLY_BASE, height=440, margin=dict(l=20, r=20, t=85, b=30),
        xaxis=dict(title="Observation date", showgrid=False, zeroline=False,
                   tickfont=dict(color=P["muted"])),
        yaxis=dict(title="Internet Aliveness Index · 0–100", range=[0, 100],
                   dtick=20, gridcolor=P["rule"], zeroline=False,
                   tickfont=dict(color=P["muted"])),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
        hovermode="closest",
    )
    return fig


def chart_sources(df: pd.DataFrame) -> go.Figure:
    ordered = df.sort_values("mean_score").copy()
    extra = {}
    if {"ci_low", "ci_high"}.issubset(ordered.columns):
        center = pd.to_numeric(ordered["mean_score"], errors="coerce")
        low = pd.to_numeric(ordered["ci_low"], errors="coerce")
        high = pd.to_numeric(ordered["ci_high"], errors="coerce")
        extra["error_x"] = dict(
            type="data",
            symmetric=False,
            array=(high - center).tolist(),
            arrayminus=(center - low).tolist(),
            color=P["muted"],
            thickness=1.2,
            width=0,
        )
    fig = go.Figure(go.Scatter(
        x=ordered["mean_score"], y=ordered["source"].str.replace("_", " ").str.title(),
        mode="markers", marker=dict(color=P["blue"], size=10),
        customdata=ordered[["date", "n_docs"]],
        hovertemplate="%{y}<br>Mean score: %{x:.1f}<br>Observation: %{customdata[0]}"
                      "<br>Documents: %{customdata[1]:,.0f}<extra></extra>",
        **extra,
    ))
    fig.update_layout(
        **PLOTLY_BASE, height=max(280, 32 * len(ordered) + 100),
        margin=dict(l=10, r=20, t=10, b=20),
        xaxis=dict(title="Mean document score · 0–100", range=[0, 100],
                   dtick=20, gridcolor=P["rule"], zeroline=False,
                   tickfont=dict(color=P["muted"])),
        yaxis=dict(showgrid=False, automargin=True, tickfont=dict(color=P["muted"])),
        showlegend=False,
    )
    return fig


def report_load_failure(label: str, exc: Exception, *, critical: bool = False):
    message = (
        f"{label} could not be loaded from the published dataset snapshot. "
        f"Data for that section is unavailable. {type(exc).__name__}: {exc}"
    )
    if critical:
        st.error(message)
    else:
        st.warning(message)


def section(title: str):
    st.markdown('<div class="section"></div>', unsafe_allow_html=True)
    st.subheader(title)


def main():
    st.set_page_config(page_title="Dead Internet Observatory", layout="wide",
                       initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="kicker">Public web research / Open data</div>', unsafe_allow_html=True)
    st.title("Dead Internet Observatory")
    st.markdown('<p class="deck">An experimental record of linguistic and behavioural '
                'signals in sampled public web text. Explore the index, its coverage, '
                'and the assumptions behind it.</p>', unsafe_allow_html=True)

    headline, total_docs, coverage = None, None, {}
    movement = None
    src_df = pd.DataFrame()
    try:
        database_path = _resolve_database_path()
    except Exception as exc:
        report_load_failure("Published dataset snapshot", exc, critical=True)
        database_path = None

    if database_path is not None:
        for label, loader, key, critical in (
            ("Current Internet Aliveness Index", load_headline, "headline", True),
            ("Source scores", load_sources, "sources", False),
            ("Document count", load_total_docs, "count", False),
            ("Coverage metadata", load_coverage, "coverage", False),
        ):
            try:
                value = loader(database_path)
                if key == "headline":
                    candidate = value.get("score")
                    if candidate is not None and (not pd.notna(candidate) or not 0 <= float(candidate) <= 100):
                        raise ValueError("current score is not a finite value within 0–100")
                    headline = value
                elif key == "sources":
                    src_df = value
                elif key == "count":
                    total_docs = value
                else:
                    coverage = value
            except Exception as exc:
                report_load_failure(label, exc, critical=critical)
        try:
            movement = load_movement(database_path)
        except Exception as exc:
            report_load_failure("Score movement", exc, critical=False)
        try:
            intervals = load_source_intervals(database_path)
            if not src_df.empty and not intervals.empty:
                src_df = src_df.merge(intervals, on=["source", "date"], how="left")
        except Exception as exc:
            report_load_failure("Source intervals", exc, critical=False)

    published = None if headline is None or headline.get("withheld") else headline.get("score")
    left, right = st.columns([1, 2], gap="large")
    with left:
        value = f'{published:.1f}<small> / 100</small>' if published is not None else '<small>Unavailable</small>'
        latest = coverage.get("latest") or {}
        date = (headline or {}).get("date") or latest.get("date")
        st.markdown(
            f'<div class="measure"><div class="measure-label">{EXPERIMENTAL_SIGNAL_LABEL}</div>'
            f'<div class="measure-value">{value}</div>'
            '<div class="measure-note">Smoothed value shown only when coverage and sample minimums are met</div></div>',
            unsafe_allow_html=True,
        )
        if date:
            st.caption(f"Observation date: {date}")
        if headline and headline.get("reason"):
            st.caption(headline["reason"])
        if headline and headline.get("quality_available"):
            coverage_value = headline.get("source_coverage")
            coverage_text = f"{float(coverage_value):.0%}" if coverage_value is not None else "unavailable"
            coverage_gate = "met" if headline.get("coverage_sufficient") else "not met"
            sample_gate = "met" if headline.get("sample_sufficient") else "not met"
            st.caption(
                f"Core-source coverage {coverage_text}. Coverage minimum {coverage_gate}. "
                f"Sample minimum {sample_gate}."
            )
        if published is not None and headline.get("ci_low") is not None and headline.get("ci_high") is not None:
            st.caption(
                f"Sampling interval {float(headline['ci_low']):.1f}–{float(headline['ci_high']):.1f}. "
                "The interval is cluster-bootstrap uncertainty in this sample."
            )
        if movement:
            st.caption(movement_caption(movement))
        st.caption("Higher scores indicate stronger human-like signals under the scoring rules.")
    with right:
        st.markdown('<div class="measure"></div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        a.metric("Documents scored · cumulative", f"{total_docs:,}" if total_docs is not None else "Unavailable")
        b.metric("Sources represented · all dates", coverage.get("source_count", "Unavailable"))
        if coverage.get("first_date") and coverage.get("last_date"):
            st.caption(f"Corpus dates: {coverage['first_date']} to {coverage['last_date']}")
        if latest.get("n_docs") is not None:
            st.caption(f"Latest composite sample: {latest['n_docs']:,} documents")
        st.markdown('<p class="scope">This score is an experimental synthetic-content signal, not a calibrated '
                    'probability of AI authorship. It does not estimate how much of '
                    'the public web was written by a model. A missing headline means the coverage '
                    'or sample minimum was not met.</p>', unsafe_allow_html=True)
    st.caption("Provenance: published Hugging Face SQLite snapshot. Refresh checks are cached for five minutes; "
               "a previously validated snapshot may be used when the upstream service is unavailable. "
               "Observation dates describe the corpus, not the download time.")

    section("Long-term record: Common Crawl")
    try:
        common_crawl = load_timeline_csv("cc_crawls")
        if common_crawl.empty:
            st.info("No Common Crawl crawls meet the timeline sample threshold.")
        else:
            spread = st.toggle("Show spread (25th–75th percentile)", value=False)
            st.plotly_chart(chart_source_medians(common_crawl.assign(source='common_crawl'), "median_date", band=True, spread=spread),
                            width="stretch", config={"displaylogo": False})
            st.caption(common_crawl_caption(common_crawl))
    except Exception as exc:
        st.warning(f"Repository-local crawl timeline unavailable: {exc}")

    section("Dead Internet Index (two-week bins, 2026)")
    try:
        biweekly = load_timeline_csv("biweekly")
        platforms = platform_options(biweekly)
        selected = st.multiselect("Platforms", platforms, default=platforms)
        composite = compute_composite(biweekly)
        shown = biweekly.loc[biweekly["source"].isin(selected)]
        if composite.empty:
            st.info("The fixed panel composite is unavailable: no two-week bin has at least 100 eligible "
                    "documents in every panel source (Hacker News, Mastodon, Steam, YouTube, "
                    "Bluesky and 4chan). Missing platforms are never reweighted.")
        if not shown.empty or not composite.empty:
            st.plotly_chart(chart_source_medians(shown, "period_start", band=len(selected) == 1,
                                                composite=composite),
                            width="stretch", config={"displaylogo": False})
        st.caption("Per-platform two-week medians for 2026; at least 50 tokens per document and 100 documents per bin. "
                   "Common Crawl is excluded. Bins are aligned to Monday 2026-01-05. A bootstrap 95% CI "
                   "appears for one selected platform. The composite is the equal-weight mean of the six "
                   "fixed panel medians, shown only for complete bins. Selecting platforms does not change "
                   "the composite. Lines break across missing bins. Timelines use repository-local CSVs.")
    except Exception as exc:
        st.warning(f"Repository-local biweekly timeline unavailable: {exc}")

    section("Across sources")
    st.write("Latest available daily mean document score for each source, weighted by document count "
             "across categories. Observation dates differ; these are not simultaneous platform rankings.")
    if src_df.empty:
        st.info("Source observations are unavailable in this snapshot.")
    else:
        st.plotly_chart(chart_sources(src_df), width="stretch", config={"displaylogo": False})
        with st.expander("Inspect source observations"):
            table = src_df.rename(columns={
                "source": "Source", "date": "Latest observation",
                "mean_score": "Mean document score", "n_docs": "Documents",
                "ci_low": "Interval low", "ci_high": "Interval high",
            })
            columns = ["Source", "Latest observation", "Mean document score", "Documents"]
            if "Interval low" in table.columns:
                columns.extend(["Interval low", "Interval high"])
            st.dataframe(table[columns], hide_index=True, width="stretch")
        st.caption("Source: daily_index in the same SQLite snapshot. Horizontal bars, when present, "
                   "are source sampling intervals and require at least two clusters. "
                   "Sample size and genre affect comparability. "
                   "A source represented here may have historical data without current collection.")

    section("Methodology & interpretation")
    method, limits = st.columns(2, gap="large")
    with method:
        st.markdown("### From text to an index")
        st.write("The timelines use per-source medians with bootstrap 95% confidence intervals (1000 resamples, fixed seed, sample capped at 20000). Optional p25–p75 bands show spread. "
                 "Documents are deduplicated by (source, content_hash); missing hashes and degenerate "
                 "4chan hashes fall back to doc_id. Two-week bins exclude Common Crawl and use documents with at least 50 tokens "
                 "and require n ≥ 100; per-crawl Common Crawl points use at least 100 tokens and require n ≥ 50. "
                 "Common Crawl uses created_dt (WARC-Date), falling back to its ISO crawl week when absent. "
                 "Social sources use created_dt. Null, pre-2000 and future dates are excluded, as are "
                 "Wikipedia, Wayback and GitHub. The two-week timeline covers 2026; the long-term chart "
                 "shows Common Crawl alone. There is no pooled cross-source historical timeline because "
                 "the source mix changes. The two-week composite uses a fixed six-source equal-weight panel "
                 "and requires every panel source to meet the sample threshold.")
        st.write("CC harvests use a separate 50000-document pipeline quota and a 10000-row per-crawl cap. Completion requires published gold confirmation at 80% of target; attempts wait 12 hours for confirmation. The backfill requeues exactly 65 empty crawls. Progress never resets automatically by default.")
        st.write("The current score and source summaries above retain the published SQLite daily means "
                 "and smoothed composite; they use different aggregation rules from these local timelines. "
                 "The headline number is hidden when core-source coverage or the sample minimum is missed.")
        st.write("Six configured signals form the provisional headline. These are heuristic features rather than "
                 "independent tests of authorship. Hapax ratio, Yule's K, windowed lexical diversity and "
                 "type-token ratio are still stored for calibration. They are omitted from the headline so "
                 "vocabulary richness is not counted more than once alongside MTLD. Burstiness is a source-level "
                 "timing summary and is not part of the document score. Perplexity is not part of the headline composite.")
        st.dataframe(pd.DataFrame({
            "Signal": [
                "MTLD lexical diversity", "Sentence length variance", "Bigram repetition",
                "Zipf alignment", "Compression ratio", "Character entropy",
            ],
            "Provisional weight": ["25%", "18%", "17%", "12%", "14%", "14%"],
        }), hide_index=True, width="stretch")
        st.write("Weights and the maps from raw features to sub-scores are provisional. "
                 "A labeled calibration file can replace the scaling constants. "
                 "Confidence intervals are omitted when a source has fewer than two clusters. "
                 "An interval describes resampling uncertainty in the observed sample. "
                 "The movement line separates a change within sources present on both days "
                 "from a source-composition change.")
    with limits:
        st.markdown("### Scope & limitations")
        st.markdown("- **A collected sample, not a census.** Coverage depends on collector access, "
                    "API limits, search terms, archives, and processing caps.\n"
                    "- **Language is not authorship.** Genre, length, editing, and repeated phrasing can "
                    "change these signals in human and generated text alike. Short texts score lower, so length "
                    "confounding remains even after token thresholds. Scores are observational, not calibrated "
                    "AI probabilities. No calibrated error rates "
                    "for AI authorship are supplied by this dashboard. Bootstrap intervals describe sampling uncertainty, "
                    "in medians for the local timelines and in cluster resamples for the headline.\n"
                    "- **Historical samples differ.** Archive and crawl dates mix with recent platform "
                    "observations. Changing source composition and scoring modes limit comparisons over time. "
                    "The movement line separates those composition changes from movement within a fixed source set.\n"
                    "- **Access gaps matter.** Twitter/X and Substack collectors are blocked; their absence "
                    "does not imply an absence of activity. Bluesky collection uses topic searches, not a full firehose.")
        st.write("Consult the collection and scoring code before interpreting a shift as a change "
                 "in who wrote the sampled text. Statistical flags identify unusual scores and do not explain their cause.")

    section("Data & code")
    st.markdown(
        "[Published dataset & SQLite snapshot](https://huggingface.co/datasets/jupiternull/dead-internet-observatory)"
        " · [Source repository](https://github.com/jupiternull/dead-internet-observatory)"
        " · [Scoring implementation](https://github.com/jupiternull/dead-internet-observatory/tree/master/detection)"
        " · [Index aggregation](https://github.com/jupiternull/dead-internet-observatory/blob/master/analytics/aliveness_index.py)"
    )
    st.caption("Source code is MIT licensed. Collected web content remains subject to its original terms; "
               "public availability does not make it public domain. Published text may include personal information. "
               "See the repository README for dataset removal and security reporting procedures.")


if __name__ == "__main__":
    main()
