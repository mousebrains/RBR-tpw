import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest

# Known false positive: the checker only accepts the literal "CF-1.11".
_CF_FALSE_POSITIVES = ('Conventions global attribute does not contain "CF-1.11"',)


@pytest.fixture
def cf_problems():
    """Returns f(path) -> list of IOOS compliance-checker cf:1.11 problems (the files declare CF-1.13,
    for which no checker exists yet)."""
    cc = pytest.importorskip("compliance_checker.runner")
    cc.CheckSuite.load_all_available_checkers()

    def check(path: Path) -> list[str]:
        buf = io.StringIO()
        with redirect_stdout(buf):
            cc.ComplianceChecker.run_checker(str(path), checker_names=["cf:1.11"], verbose=0,
                                             criteria="strict", output_filename="-", output_format="text")
        problems = []
        for line in buf.getvalue().splitlines():
            s = line.strip()
            if not s.startswith("*"):
                continue
            msg = s.lstrip("*").strip()
            if not msg or "potential issues" in msg.lower() or any(fp in msg for fp in _CF_FALSE_POSITIVES):
                continue
            problems.append(msg)
        return problems

    return check
