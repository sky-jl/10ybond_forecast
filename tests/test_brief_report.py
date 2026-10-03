import io
import json
import sys
import types

import pytest

pypdf = pytest.importorskip("pypdf")

from brief_report import BRIEF_SCHEMA, build_context, generate_brief  # noqa: E402
from forecasting import run_forecast  # noqa: E402


@pytest.fixture(scope="module")
def result(monthly, config):
    return run_forecast(monthly, config)


def test_template_pdf_is_at_most_two_pages(result, config):
    pdf, com = generate_brief(result, config, use_ai=False)
    assert pdf[:5] == b"%PDF-"
    assert len(pypdf.PdfReader(io.BytesIO(pdf)).pages) <= 2
    assert set(BRIEF_SCHEMA["required"]) <= set(com)


def test_claude_path_parses_structured_output(result, config, monkeypatch):
    payload = {k: (["point one", "point two"] if BRIEF_SCHEMA["properties"][k]["type"] == "array"
                   else "Some text.") for k in BRIEF_SCHEMA["required"]}
    captured = {}

    class FakeMessages:
        def create(self, **kw):
            captured.update(kw)
            block = types.SimpleNamespace(type="text", text=json.dumps(payload))
            return types.SimpleNamespace(content=[block], stop_reason="end_turn", model=kw["model"])

    class FakeClient:
        def __init__(self, **_):
            self.beta = types.SimpleNamespace(messages=FakeMessages())

    fake = types.ModuleType("anthropic")
    fake.Anthropic = FakeClient
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    pdf, com = generate_brief(result, config, use_ai=True)
    assert captured["model"] == "claude-opus-5-5"
    assert captured["output_config"]["format"]["type"] == "json_schema"
    assert com["headline"] == "Some text." and "Claude" in com["_source"]
    assert len(pypdf.PdfReader(io.BytesIO(pdf)).pages) <= 2


def test_context_is_json_serialisable(result, config):
    json.dumps(build_context(result, config), default=float)


def test_long_commentary_still_fits_two_pages(result, config):
    from brief_report import build_pdf, template_commentary
    ctx = build_context(result, config)
    s = "A long sentence about the policy path, the term premium and fiscal supply pressures. " * 3
    com = {k: ([s] * 4 if isinstance(v, list) else s) for k, v in template_commentary(ctx).items()}
    pdf = build_pdf(result, com, config, ctx)
    assert len(pypdf.PdfReader(io.BytesIO(pdf)).pages) <= 2


def test_context_has_macro_attribution(result, config):
    ctx = build_context(result, config)
    att = ctx["last_quarter_attribution"]
    assert {"Expectations vs term premium", "Real vs inflation", "Front end vs curve"} <= set(att["splits"])
    assert ctx["term_premium_fiscal"]["drivers"]
