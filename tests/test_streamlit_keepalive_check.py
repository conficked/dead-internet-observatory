import pytest

from scripts.streamlit_keepalive_check import evaluate_dashboard_text


def test_rejects_login_gate_content():
    with pytest.raises(RuntimeError, match="authentication"):
        evaluate_dashboard_text(
            "Sign in\nLog in to continue\n",
            final_url="https://deadinternetobservatory.streamlit.app/-/login",
        )


def test_rejects_missing_dashboard_signal():
    with pytest.raises(RuntimeError, match="did not render expected dashboard"):
        evaluate_dashboard_text(
            "Some generic Streamlit shell text",
            final_url="https://deadinternetobservatory.streamlit.app/",
        )


def test_accepts_dashboard_content():
    summary = evaluate_dashboard_text(
        "Dead Internet Observatory\nDocuments scored · cumulative\n",
        final_url="https://deadinternetobservatory.streamlit.app/~/+/",
    )
    assert summary["ok"] is True
