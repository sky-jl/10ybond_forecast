"""Smoke test of the Streamlit app (synthetic data when no cached dataset exists)."""

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _run(at):
    at.run(timeout=120)
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    return at


def test_app_runs_and_reacts_to_inputs():
    at = _run(st_testing.AppTest.from_file(APP))
    assert len(at.metric) == 4
    before = at.metric[1].value
    neutral = next(s for s in at.sidebar.slider if s.label.startswith("Fed neutral"))
    neutral.set_value(neutral.value + 1.0)
    _run(at)
    assert at.metric[1].value != before   # higher neutral → different US 10Y path
