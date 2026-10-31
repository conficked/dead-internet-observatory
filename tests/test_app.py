import sqlite3
import tempfile
import unittest
from pathlib import Path

import pytest
from types import ModuleType
from unittest.mock import Mock, patch

import pandas as pd
import plotly.graph_objects  # Load before the Streamlit-only module patch.


streamlit = ModuleType("streamlit")
_resource_cache = {}


def cache_resource(**_kwargs):
    def decorate(function):
        key = (function.__module__, function.__qualname__)

        def cached(*args, **kwargs):
            cache_key = (key, args, tuple(sorted(kwargs.items())))
            if cache_key not in _resource_cache:
                _resource_cache[cache_key] = function(*args, **kwargs)
            return _resource_cache[cache_key]

        def clear():
            for cache_key in list(_resource_cache):
                if cache_key[0] == key:
                    del _resource_cache[cache_key]

        cached.clear = Mock(side_effect=clear)
        return cached

    return decorate


streamlit.cache_resource = cache_resource
streamlit.cache_data = cache_resource

with patch.dict("sys.modules", {"streamlit": streamlit}):
    from app import app


class DatabasePathTests(unittest.TestCase):
    def setUp(self):
        state = app._database_path_state()
        with state["lock"]:
            state["last_valid_path"] = None

    def _database(self, directory):
        path = Path(directory) / "observatory.db"
        with sqlite3.connect(path) as conn:
            conn.execute(
                "CREATE TABLE composite_index "
                "(date TEXT, aliveness_index REAL, smoothed_index REAL, n_docs INTEGER, "
                "anomaly_flag INTEGER, anomaly_reason TEXT)"
            )
            conn.execute(
                "CREATE TABLE daily_index "
                "(date TEXT, source TEXT, mean_score REAL, aliveness_index REAL, n_docs INTEGER)"
            )
            conn.execute("CREATE TABLE meta (key TEXT, value TEXT)")
        return str(path)

    def test_database_path_downloads_revision_with_huggingface_hub(self):
        app._database_path.clear()
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            with patch.object(
                app, "hf_hub_download", return_value=database
            ) as download:
                path = app._database_path("dataset-revision")

            download.assert_called_once_with(
                repo_id=app.DATASET_REPO,
                repo_type="dataset",
                filename="observatory.db",
                revision="dataset-revision",
            )
            self.assertEqual(path, database)

    def test_api_failure_uses_last_valid_database(self):
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            state = app._database_path_state()
            state["last_valid_path"] = database
            with patch.object(app, "_dataset_revision", side_effect=RuntimeError("API down")):
                self.assertEqual(app._resolve_database_path(), database)

    def test_download_failure_uses_last_valid_database(self):
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            state = app._database_path_state()
            state["last_valid_path"] = database
            with (
                patch.object(app, "_dataset_revision", return_value="new-revision"),
                patch.object(app, "_database_path", side_effect=RuntimeError("CAS down")),
            ):
                self.assertEqual(app._resolve_database_path(), database)

    def test_invalid_fallback_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "observatory.db"
            invalid.write_bytes(b"not sqlite")
            state = app._database_path_state()
            state["last_valid_path"] = str(invalid)
            with patch.object(app, "_dataset_revision", side_effect=RuntimeError("API down")):
                with self.assertRaises(sqlite3.DatabaseError):
                    app._resolve_database_path()

    def test_wrong_schema_fallback_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "observatory.db"
            with sqlite3.connect(invalid) as conn:
                conn.execute("CREATE TABLE unrelated (value TEXT)")
            state = app._database_path_state()
            state["last_valid_path"] = str(invalid)
            with patch.object(app, "_dataset_revision", side_effect=RuntimeError("API down")):
                with self.assertRaises(sqlite3.DatabaseError):
                    app._resolve_database_path()

    def test_last_valid_database_survives_streamlit_script_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            with patch.object(app, "hf_hub_download", return_value=database):
                app._database_path("rerun-regression-revision")

            def rerun_database_path_state():
                return {"last_valid_path": None, "lock": object()}

            rerun_database_path_state.__module__ = app.__name__
            rerun_database_path_state.__qualname__ = "_database_path_state"
            app._database_path_state = cache_resource()(rerun_database_path_state)

            with patch.object(app, "_dataset_revision", side_effect=RuntimeError("API down")):
                self.assertEqual(app._resolve_database_path(), database)

    def test_query_loaders_keep_the_resolved_snapshot(self):
        snapshot = "/cache/snapshots/revision/observatory.db"
        responses = [
            [{"smoothed_index": 58.1}],
            [{"value": "3440000"}],
            [],
            [],
        ]
        with (
            patch.object(app, "_query", side_effect=responses) as query,
            patch.object(app.pd, "DataFrame", return_value=Mock(empty=True)),
        ):
            app.load_score(snapshot)
            app.load_total_docs(snapshot)
            app.load_timeline(snapshot)
            app.load_sources(snapshot)

        self.assertEqual([call.args[0] for call in query.call_args_list], [snapshot] * 4)


class ResearchDisplayTests(unittest.TestCase):
    def test_timeline_breaks_at_missing_dates_and_preserves_stored_flags(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-02-01"]),
            "aliveness_index": [52.0, 53.0, 61.0],
            "smoothed_index": [52.5, 53.5, 60.0],
            "n_docs": [20, 30, 40], "anomaly_flag": [0, 0, 1],
        })
        chart = app.chart_timeline(df)
        self.assertEqual(len(chart.data), 5)
        for trace in chart.data[:4]:
            self.assertLessEqual(len(trace.x), 2)
            self.assertFalse(trace.connectgaps)
        self.assertEqual(list(chart.data[-1].y), [61.0])
        self.assertEqual(list(chart.layout.yaxis.range), [0, 100])

    def test_latest_source_means_are_document_weighted_and_individually_dated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sources.db"
            with sqlite3.connect(path) as conn:
                conn.execute("CREATE TABLE daily_index "
                             "(date TEXT, source TEXT, mean_score REAL, n_docs INTEGER)")
                conn.executemany("INSERT INTO daily_index VALUES (?, ?, ?, ?)", [
                    ("2026-01-01", "reddit", 99, 100),
                    ("2026-01-02", "reddit", 20, 10),
                    ("2026-01-02", "reddit", 80, 30),
                    ("2025-12-20", "news", 0, 5),
                ])
            sources = app.load_sources(str(path)).set_index("source")
            self.assertEqual(sources.loc["reddit", "mean_score"], 65)
            self.assertEqual(sources.loc["reddit", "n_docs"], 40)
            self.assertEqual(sources.loc["news", "date"], "2025-12-20")
            self.assertEqual(sources.loc["news", "mean_score"], 0)
            chart = app.chart_sources(sources.reset_index())
            self.assertEqual(list(chart.data[0].x), [0, 65])

    def test_full_timeline_retains_archive_dates_and_filters_sparse_days(self):
        with tempfile.TemporaryDirectory() as directory:
            path = DatabasePathTests()._database(directory)
            with sqlite3.connect(path) as conn:
                conn.executemany("INSERT INTO composite_index VALUES (?, ?, ?, ?, ?, ?)", [
                    ("2008-08-07", 60, 60, 20, 0, ""),
                    ("2013-01-01", 61, 60, 5, 0, ""),
                    ("2026-01-01", 55, 56, 30, 0, ""),
                ])
            timeline = app.load_timeline(path)
            self.assertEqual(timeline["date"].dt.year.tolist(), [2008, 2026])

    def test_coverage_handles_an_empty_valid_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = DatabasePathTests()._database(directory)
            coverage = app.load_coverage(path)
            self.assertIsNone(coverage["first_date"])
            self.assertEqual(coverage["source_count"], 0)
            self.assertIsNone(coverage["latest"])


if __name__ == "__main__":
    unittest.main()


def test_fixed_panel_composite_requires_every_source_and_minimum_n():
    rows = [dict(source=source, period_start='2026-06-08', n=100, median=i * 10)
            for i, source in enumerate(app.PANEL)]
    complete = pd.DataFrame(rows)
    assert app.compute_composite(complete).iloc[0]['median'] == 10 * (len(app.PANEL) - 1) / 2
    assert app.compute_composite(complete.iloc[:-1]).empty
    complete.loc[0, 'n'] = 99
    assert app.compute_composite(complete).empty


def test_crawl_gap_split_at_120_days():
    frame = pd.DataFrame({'median_date': pd.to_datetime(['2024-01-01', '2024-04-30', '2024-08-30'])})
    assert [len(run) for run in app.split_timeline_gaps(frame, 'median_date')] == [2, 1]


def test_timeline_csv_loader_uses_local_files(tmp_path):
    pd.DataFrame({'source': ['common_crawl'], 'median_date': ['2014-04-01'],
                  'n': [100], 'median': [62], 'p25': [50], 'p75': [70]}).to_csv(tmp_path / 'cc_crawls.csv', index=False)
    result = app.load_timeline_csv('cc_crawls', str(tmp_path))
    assert result.iloc[0]['median'] == 62
    assert result['median_date'].dt.year.tolist() == [2014]


def test_common_crawl_caption_does_not_invent_an_unobserved_decline():
    increasing = pd.DataFrame({'median_date': pd.to_datetime(['2014-01-01', '2018-01-01', '2020-01-01', '2024-01-01']),
                               'median': [40, 50, 60, 70]})
    assert 'gradual decline' not in app.common_crawl_caption(increasing)
    assert 'predates' not in app.common_crawl_caption(increasing.iloc[-1:])


def test_panel_excludes_reddit_and_has_six_sources():
    assert 'reddit' not in app.PANEL
    assert len(app.PANEL) == 6


def test_score_axis_fits_data_with_padding():
    df = pd.DataFrame({'source': ['a', 'a'], 'week_start': pd.to_datetime(['2026-06-01', '2026-06-08']),
                       'n': [40, 40], 'median': [55.0, 60.0], 'p25': [50.0, 52.0], 'p75': [62.0, 66.0]})
    assert app.score_axis_range(df, columns=('median',)) == [52.0, 63.0]
    assert app.score_axis_range(df) == [47.0, 69.0]
    chart = app.chart_source_medians(df, 'week_start')
    assert list(chart.layout.yaxis.range) == [52.0, 63.0]
    assert chart.layout.yaxis.title.text == 'Observational score · 0–100'


def test_cc_excluded_from_platform_options():
    assert app.platform_options(pd.DataFrame({'source': ['common_crawl', 'reddit']})) == ['reddit']


def test_headline_is_hidden_when_quality_thresholds_fail(tmp_path):
    path = tmp_path / "observatory.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE composite_index (date TEXT, aliveness_index REAL, smoothed_index REAL, "
            "n_docs INTEGER, anomaly_flag INTEGER, anomaly_reason TEXT)"
        )
        conn.execute(
            "CREATE TABLE index_quality (date TEXT, source_coverage REAL, coverage_sufficient INTEGER, "
            "sample_sufficient INTEGER, eligible_sources INTEGER, required_sources INTEGER, "
            "ci_low REAL, ci_high REAL)"
        )
        conn.execute("INSERT INTO composite_index VALUES ('2026-01-02', 40, 41, 10, 0, '')")
        conn.execute("INSERT INTO index_quality VALUES ('2026-01-02', 0.4, 0, 1, 2, 10, 30, 50)")
    view = app.load_headline(str(path))
    assert view["score"] is None
    assert view["withheld"] is True
    assert view["source_coverage"] == 0.4


def test_headline_shows_score_and_interval_when_sufficient(tmp_path):
    path = tmp_path / "observatory.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE composite_index (date TEXT, aliveness_index REAL, smoothed_index REAL, "
            "n_docs INTEGER, anomaly_flag INTEGER, anomaly_reason TEXT)"
        )
        conn.execute(
            "CREATE TABLE index_quality (date TEXT, source_coverage REAL, coverage_sufficient INTEGER, "
            "sample_sufficient INTEGER, eligible_sources INTEGER, required_sources INTEGER, "
            "ci_low REAL, ci_high REAL)"
        )
        conn.execute("INSERT INTO composite_index VALUES ('2026-01-02', 40, 41, 800, 0, '')")
        conn.execute("INSERT INTO index_quality VALUES ('2026-01-02', 0.9, 1, 1, 8, 10, 30, 50)")
    view = app.load_headline(str(path))
    assert view["score"] == 41
    assert view["ci_low"] == 30
    assert view["ci_high"] == 50
    assert view["withheld"] is False
    assert "Experimental synthetic-content signal" in app.EXPERIMENTAL_SIGNAL_LABEL


def test_legacy_snapshot_without_quality_table_still_returns_score(tmp_path):
    path = DatabasePathTests()._database(str(tmp_path))
    with sqlite3.connect(path) as conn:
        conn.execute("INSERT INTO composite_index VALUES ('2026-01-01', 50, 52, 20, 0, '')")
    view = app.load_headline(path)
    assert view["score"] == 52
    assert view["withheld"] is False
    assert view["quality_available"] is False


def test_movement_separates_score_change_from_source_composition(tmp_path):
    path = tmp_path / "movement.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE daily_index (date TEXT, source TEXT, mean_score REAL, n_docs INTEGER)")
        conn.executemany("INSERT INTO daily_index VALUES (?,?,?,?)", [
            ("2026-01-01", "reddit", 40, 100),
            ("2026-01-01", "news", 80, 100),
            ("2026-01-02", "reddit", 60, 100),
            ("2026-01-02", "news", 80, 100),
        ])
    movement = app.load_movement(str(path))
    assert movement["composition_effect"] == pytest.approx(0, abs=1e-6)
    assert movement["score_effect"] > 0
    caption = app.movement_caption(movement)
    assert "Source-composition" in caption
    assert "within sources" in caption


def test_dashboard_copy_calls_the_score_an_experimental_signal():
    source = Path(__file__).resolve().parents[1].joinpath("app", "app.py").read_text()
    assert "Experimental synthetic-content signal" in source
    assert "Temporal burstiness receives" not in source
    assert "synthetic content %" not in source.lower()


def test_crawl_chart_ci_spread_and_gaps():
    frame = pd.DataFrame({'source':['common_crawl']*3,
        'median_date':pd.to_datetime(['2020-01-01','2020-04-30','2020-08-30']),
        'n':[100]*3, 'median':[50]*3, 'ci_low':[49]*3, 'ci_high':[51]*3,
        'p25':[20]*3, 'p75':[80]*3})
    chart = app.chart_source_medians(frame, 'median_date', band=True)
    assert len(chart.data) == 6
    assert list(chart.layout.yaxis.range) == [46,54]
    assert str(chart.layout.shapes[0].x0) == '2022-11-30'
    chart = app.chart_source_medians(frame, 'median_date', band=True, spread=True)
    assert len(chart.data) == 10
    assert list(chart.layout.yaxis.range) == [17,83]
