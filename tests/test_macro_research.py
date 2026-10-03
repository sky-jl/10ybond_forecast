import json
import sys
import types

import pytest

import macro_research as mr
from forecasting.attribution import attribute


def _cite(url, title):
    return types.SimpleNamespace(type="web_search_result_location", url=url, title=title,
                                 cited_text="quoted passage")


def _text(text, cites=()):
    return types.SimpleNamespace(type="text", text=text, citations=list(cites))


class _Stream:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.msg


@pytest.fixture
def fake_anthropic(monkeypatch):
    calls = {"stream": [], "create": []}
    paused = types.SimpleNamespace(stop_reason="pause_turn", content=[
        _text("Treasury auction sizes rose and 10Y auctions tailed.",
              [_cite("https://example.com/auction", "Auction recap")])])
    done = types.SimpleNamespace(stop_reason="end_turn", content=[
        _text("Oil rose on supply disruptions.", [_cite("https://example.com/oil", "Oil news")]),
        _text(" No other verified drivers.")])
    structured = {
        "quarter_summary": "Term premium and supply drove the move.",
        "themes": [
            {"theme": "Heavier coupon issuance lifted the term premium", "driver": "fiscal_supply",
             "direction": "higher_yields", "evidence": "Auctions tailed.", "source_ids": [1]},
            {"theme": "Oil rally lifted breakevens", "driver": "inflation_energy",
             "direction": "higher_yields", "evidence": "Oil rose.", "source_ids": [2]},
            {"theme": "Unsourced claim", "driver": "geopolitics", "direction": "mixed",
             "evidence": "none", "source_ids": []},
            {"theme": "Bogus source", "driver": "other", "direction": "mixed",
             "evidence": "none", "source_ids": [99]},
        ]}

    class Messages:
        def stream(self, **kw):
            calls["stream"].append(kw)
            return _Stream(paused if len(calls["stream"]) == 1 else done)

        def create(self, **kw):
            calls["create"].append(kw)
            return types.SimpleNamespace(stop_reason="end_turn",
                                         content=[types.SimpleNamespace(type="text", text=json.dumps(structured))])

    class Client:
        def __init__(self, **_):
            self.beta = types.SimpleNamespace(messages=Messages())

    mod = types.ModuleType("anthropic")
    mod.Anthropic = Client
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    return calls


def test_research_keeps_only_sourced_themes(monthly, config, fake_anthropic):
    out = mr.research_macro_themes(attribute(monthly), config, api_key="k")
    assert [t["driver"] for t in out["themes"]] == ["fiscal_supply", "inflation_energy"]
    assert out["dropped_unsourced"] == 2
    assert out["themes"][0]["sources"][0]["url"] == "https://example.com/auction"
    assert len(out["sources"]) == 2 and "[1]" in out["notes"] and "[2]" in out["notes"]


def test_pause_turn_is_resumed_and_tools_sent(monthly, config, fake_anthropic):
    mr.research_macro_themes(attribute(monthly), config, api_key="k")
    first, second = fake_anthropic["stream"]
    assert first["tools"][0]["type"] == "web_search_20260209"
    assert first["model"] == "claude-opus-5-5"
    assert second["messages"][-1]["role"] == "assistant"      # resumed, no extra user turn
    fmt = fake_anthropic["create"][0]["output_config"]["format"]
    assert fmt["type"] == "json_schema"


def test_no_key_raises(monthly, config, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        mr.research_macro_themes(attribute(monthly), {**config, "anthropic_api_key": ""})


def test_save_and_load(tmp_path):
    r = {"generated": "2026-10-03", "themes": [], "sources": []}
    mr.save_research(r, tmp_path)
    assert mr.load_latest_research(tmp_path)["generated"] == "2026-10-03"


def _search_result(url, title):
    return types.SimpleNamespace(type="web_search_result", url=url, title=title)


def test_sources_from_search_results_without_citations(monkeypatch, monthly, config, tmp_path):
    """Dynamic filtering may return text without citations: use the retrieved result pages."""
    content = [
        types.SimpleNamespace(type="server_tool_use", name="web_search", input={"query": "10y treasury Q3"}),
        types.SimpleNamespace(type="web_search_tool_result",
                              content=[_search_result("https://example.com/a", "A"),
                                       _search_result("https://example.com/b", "B")]),
        _text("- Auctions tailed (https://example.com/a)\n- Made-up claim (https://fake.example/x)"),
    ]
    notes, sources = mr._notes_and_sources(content)
    assert [s["url"] for s in sources] == ["https://example.com/a", "https://example.com/b"]
    assert "[1]" in notes and "fake.example" not in [s["url"] for s in sources]
    d = mr.diagnose(content, ["end_turn"])
    assert d["searches"] == ["web_search: 10y treasury Q3"] and d["results_returned"] == 2


def test_failure_writes_log_and_diagnostics(monkeypatch, monthly, config, tmp_path):
    err = types.SimpleNamespace(type="web_search_tool_result",
                                content=types.SimpleNamespace(type="web_search_tool_result_error",
                                                              error_code="unavailable"))
    msg = types.SimpleNamespace(stop_reason="end_turn", content=[
        types.SimpleNamespace(type="server_tool_use", name="web_search", input={"query": "q"}), err,
        _text("Could not verify anything.")])

    class Messages:
        def stream(self, **kw):
            return _Stream(msg)

    class Client:
        def __init__(self, **_):
            self.beta = types.SimpleNamespace(messages=Messages())

    mod = types.ModuleType("anthropic")
    mod.Anthropic = Client
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    with pytest.raises(mr.ResearchError) as ei:
        mr.research_macro_themes(attribute(monthly), config, api_key="k", log_dir=tmp_path)
    assert "unavailable" in str(ei.value)
    assert ei.value.diagnostics["tool_errors"] == ["web_search_tool_result: unavailable"]
    log = json.loads(open(ei.value.log_path).read())
    assert log["error"] and log["diagnostics"]["searches"] == ["web_search: q"]
