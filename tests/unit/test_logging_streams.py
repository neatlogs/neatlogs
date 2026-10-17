"""SDK log lines must stay off stdout and out of the host app's root logger."""

import subprocess
import sys
import textwrap


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_sdk_logs_go_to_stderr_not_stdout():
    result = _run("""
        import neatlogs
        neatlogs.init(api_key=None, workflow_name="t", disable_export=True)
        neatlogs.shutdown()
        print("DATA")
        """)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "DATA"
    assert "neatlogs - INFO" in result.stderr


def test_debug_init_leaves_the_root_logger_alone():
    result = _run("""
        import logging
        import neatlogs
        neatlogs.init(api_key=None, workflow_name="t", disable_export=True, debug=True)
        root = logging.getLogger()
        print(logging.getLevelName(root.level), len(root.handlers))
        print(logging.getLevelName(logging.getLogger("neatlogs").level))
        neatlogs.shutdown()
        """)
    assert result.returncode == 0, result.stderr
    root_line, neatlogs_line = result.stdout.splitlines()[:2]
    assert root_line == "WARNING 0"
    assert neatlogs_line == "DEBUG"
