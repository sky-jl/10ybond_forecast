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


def _us_end(at):
    return next(m.value for m in at.metric if "US 10Y" in m.label and "weighted" in m.label)


def test_app_runs_and_reacts_to_inputs():
    at = _run(st_testing.AppTest.from_file(APP))
    assert any(m.label == "Fiscal fair value" for m in at.metric)       # macro tab rendered
    before = _us_end(at)
    neutral = next(s for s in at.sidebar.number_input if s.label.startswith("Fed neutral"))
    neutral.set_value(neutral.value + 1.0)
    _run(at)
    assert _us_end(at) != before   # higher neutral → different US 10Y path


def test_fiscal_mode_and_addons_move_forecast():
    at = _run(st_testing.AppTest.from_file(APP))
    before = _us_end(at)
    next(r for r in at.sidebar.radio if r.label == "Target").set_value("Fiscal fair value (regression)")
    _run(at)
    debt = next(n for n in at.sidebar.number_input if n.label.startswith("Debt held by public"))
    debt.set_value(debt.value + 20)
    _run(at)
    after_fiscal = _us_end(at)
    assert after_fiscal != before
    next(n for n in at.sidebar.number_input if n.label == "Fed independence").set_value(25)
    _run(at)
    assert _us_end(at) != after_fiscal
