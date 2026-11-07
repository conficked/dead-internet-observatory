from pathlib import Path
import re


def test_keepalive_checks_out_repo_before_running_script():
    workflow = Path(".github/workflows/streamlit_keepalive.yml").read_text()
    checkout_match = re.search(r"- uses:\s*actions/checkout@[\w\.\-]+", workflow)
    invoke = "python scripts/streamlit_keepalive_check.py"
    assert checkout_match, "workflow must checkout repository files"
    assert invoke in workflow, "workflow must execute keepalive checker script"
    assert checkout_match.start() < workflow.index(
        invoke
    ), "checkout must happen before script invocation"
