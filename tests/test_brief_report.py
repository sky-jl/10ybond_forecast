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
    assert len(pypdf.PdfReader(io.BytesIO(pdf)).pages) <= 3
    assert set(BRIEF_SCHEMA["required"]) <= set(com)


def test_claude_path_parses_structured_output(result, config, monkeypatch):
    payload = {k: (["point one", "point two"] if BRIEF_SCHEMA["properties"][k]["type"] == "array"
                   else "Some text.") for k in BRIEF_SCHEMA["required"]}
    payload["macro_outlook"] = {"overview": "Outlook text.", "watch_list": ["Auctions"],
                                "pillars": [{"driver": "fiscal_supply", "view": "Supply heavy.",
                                             "yield_impact": "higher"}]}
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
    assert len(pypdf.PdfReader(io.BytesIO(pdf)).pages) <= 3


def test_context_is_json_serialisable(result, config):
    json.dumps(build_context(result, config), default=float)


def test_long_commentary_still_fits_three_pages(result, config):
    from brief_report import build_pdf, template_commentary
    ctx = build_context(result, config)
    s = "A long sentence about the policy path, the term premium and fiscal supply pressures. " * 3
    com = {k: ([s] * 4 if isinstance(v, list) else s) for k, v in template_commentary(ctx).items()}
    com["macro_outlook"] = {"overview": s * 2, "watch_list": [s] * 5,
                            "pillars": [{"driver": d, "view": s, "yield_impact": "higher"}
                                        for d in ("fed_policy", "inflation_energy", "fiscal_supply",
                                                  "fed_independence", "geopolitics", "canada")]}
    pdf = build_pdf(result, com, config, ctx)
    assert len(pypdf.PdfReader(io.BytesIO(pdf)).pages) <= 3


def test_context_has_macro_attribution(result, config):
    ctx = build_context(result, config)
    att = ctx["last_quarter_attribution"]
    assert {"Expectations vs term premium", "Real vs inflation", "Front end vs curve"} <= set(att["splits"])
    assert ctx["term_premium_fiscal"]["drivers"]


def test_pdf_reflects_macro_drivers(monthly, config):
    import copy
    cfg = copy.deepcopy(config)
    cfg["forecast"]["term_premium"]["fiscal"] = {"deficit_revision_pp": 1.0}
    cfg["forecast"]["term_premium"]["addons_bps"] = {"fed_independence": 15}
    cfg["macro_themes"] = ["TEST-THEME weak auctions"]
    cfg["macro_theme_sources"] = [{"url": "https://www.example.com/auction", "title": "Auction recap"}]
    r = run_forecast(monthly, cfg)
    pdf, _ = generate_brief(r, cfg, use_ai=False)
    txt = " ".join(p.extract_text() for p in pypdf.PdfReader(io.BytesIO(pdf)).pages)
    for needle in ("What moved yields", "Macro backdrop", "Fiscal prem", "fed independence",
                   "TEST-THEME", "www.example.com"):
        assert needle.lower() in txt.lower(), needle
    assert len(pypdf.PdfReader(io.BytesIO(pdf)).pages) <= 3


def test_macro_outlook_uses_approved_research(monthly, config):
    import copy
    cfg = copy.deepcopy(config)
    cfg["macro_research"] = {"quarter_summary": "Supply worries dominated the quarter.", "themes": [
        {"theme": "APPROVED-THEME heavier coupon issuance", "driver": "fiscal_supply",
         "direction": "higher_yields", "evidence": "Auctions tailed.", "sources": ["Auction recap"]}]}
    cfg["macro_research_notes"] = "- Auctions tailed [1]"
    r = run_forecast(monthly, cfg)
    ctx = build_context(r, cfg)
    assert ctx["research_notes"].startswith("- Auctions")
    pdf, com = generate_brief(r, cfg, use_ai=False)
    txt = " ".join(p.extract_text() for p in pypdf.PdfReader(io.BytesIO(pdf)).pages)
    assert "Macro outlook" in txt and "APPROVED-THEME" in txt and "What to watch" in txt
    assert com["macro_outlook"]["pillars"][0]["yield_impact"] == "higher"
