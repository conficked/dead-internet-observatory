from pathlib import Path


def test_keepalive_checks_out_repo_before_running_script():
    workflow = Path(".github/workflows/streamlit_keepalive.yml").read_text()
    checkout = "- uses: actions/checkout@v6"
    invoke = "python scripts/streamlit_keepalive_check.py"
    assert checkout in workflow, "workflow must checkout repository files"
    assert invoke in workflow, "workflow must execute keepalive checker script"
    assert workflow.index(checkout) < workflow.index(
        invoke
    ), "checkout must happen before script invocation"
