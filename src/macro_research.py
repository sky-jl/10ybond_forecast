"""
Research the last quarter's macro themes for the 10Y outlook with Claude + web search.

Anti-fabrication design:
  1. Research call — Claude searches the web (server-side web_search tool). Every statement in its
     notes carries citations to pages it actually retrieved; we collect those into a numbered
     source list.
  2. Structuring call — Claude turns the cited notes into 4–6 themes as JSON, each listing the
     source numbers it relies on.
  3. Validation — any theme without at least one valid source number is dropped.
The user then edits / approves the themes in the app before they reach a report.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL = "claude-opus-5-5"
WEB_SEARCH = {"type": "web_search_20260209", "name": "web_search", "max_uses": 10}
MAX_CONTINUATIONS = 5

DRIVERS = ["fed_policy", "inflation_energy", "fiscal_supply", "fed_independence", "geopolitics",
           "growth_labor", "global_rates", "other"]

RESEARCH_SYSTEM = """You are a fixed-income research analyst. Use web search to find what drove the
US 10-year Treasury yield (and Canada 10-year) during the stated period. Cover: Fed policy and
communication, inflation data and energy prices, fiscal deficits / debt / Treasury issuance and
auctions, Fed independence and political pressure, geopolitics, growth and labour data, and the Bank
of Canada. Only state facts you found in search results, and say so when you could not verify
something. Do not speculate or fill gaps from memory. Write concise research notes."""

STRUCTURE_SYSTEM = """You turn cited research notes into macro themes for a bond-yield report. Use ONLY
information present in the notes. Every theme must list the source numbers [n] from the notes that
support it; if a point has no source number in the notes, leave it out. Do not add facts, numbers or
events that are not in the notes."""

THEMES_SCHEMA = {
    "type": "object",
    "properties": {
        "quarter_summary": {"type": "string",
                            "description": "2-3 sentences: what the bond market traded in the period"},
        "themes": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "theme": {"type": "string", "description": "One-line theme for the report (under 25 words)"},
                "driver": {"type": "string", "enum": DRIVERS},
                "direction": {"type": "string", "enum": ["higher_yields", "lower_yields", "mixed"]},
                "evidence": {"type": "string", "description": "1-2 sentences of supporting facts from the notes"},
                "source_ids": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["theme", "driver", "direction", "evidence", "source_ids"],
            "additionalProperties": False,
        }},
    },
    "required": ["quarter_summary", "themes"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------- helpers
def _client(api_key: str | None):
    import anthropic
    return anthropic.Anthropic(api_key=api_key or None)


def _research_prompt(att: dict, config: dict) -> str:
    facts = {k: att.get(k) for k in ("start", "end", "us_10y_change_bps", "splits", "reading")}
    return (f"Period: {att['start']} to {att['end']} (report quarter: {config.get('quarter')}).\n"
            f"Today is {date.today():%d %B %Y}.\n\n"
            "Our data shows this decomposition of the US 10Y move over the period (monthly averages):\n"
            f"{json.dumps(facts, indent=1, default=float)}\n\n"
            "Research the news and data releases that explain this move: what was the market trading? "
            "Explain in particular the components that moved most (expected short rates vs term premium, "
            "real yields vs breakevens). Then note the main macro themes likely to matter for 10Y yields "
            "over the next two years.")


def _notes_and_sources(content) -> tuple[str, list[dict]]:
    """Concatenate text blocks, tagging each cited passage with [n] and collecting the sources."""
    sources: list[dict] = []
    index: dict[str, int] = {}
    parts: list[str] = []
    for block in content:
        if getattr(block, "type", None) != "text":
            continue
        tags = []
        for c in getattr(block, "citations", None) or []:
            url = getattr(c, "url", None)
            if not url:
                continue
            if url not in index:
                sources.append({"id": len(sources) + 1, "url": url, "title": getattr(c, "title", "") or url,
                                "quote": (getattr(c, "cited_text", "") or "")[:300]})
                index[url] = len(sources)
            tags.append(index[url])
        tag = (" " + "".join(f"[{n}]" for n in sorted(set(tags)))) if tags else ""
        parts.append(block.text + tag)
    return "".join(parts).strip(), sources


def _run_research(client, prompt: str):
    """Web-search turn with pause_turn continuation. Returns the final assistant content."""
    messages = [{"role": "user", "content": prompt}]
    acc: list = []
    for _ in range(MAX_CONTINUATIONS + 1):
        with client.beta.messages.stream(
            model=MODEL, max_tokens=32000,
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            output_config={"effort": "high"},
            system=RESEARCH_SYSTEM, tools=[WEB_SEARCH], messages=messages,
        ) as stream:
            resp = stream.get_final_message()
        if resp.stop_reason == "refusal":
            raise RuntimeError("Claude declined the research request")
        acc += list(resp.content)
        if resp.stop_reason != "pause_turn":
            return acc
        # Resume the paused server-tool turn: resend the assistant content, no extra user message
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": acc}]
    return acc


def validate_themes(raw: dict, sources: list[dict]) -> dict:
    """Keep only themes backed by at least one real source; attach the source records."""
    by_id = {s["id"]: s for s in sources}
    kept, dropped = [], 0
    for t in raw.get("themes", []):
        ids = [i for i in t.get("source_ids", []) if i in by_id]
        if not ids or t.get("driver") not in DRIVERS:
            dropped += 1
            continue
        kept.append({**t, "source_ids": ids, "sources": [by_id[i] for i in ids]})
    return {"quarter_summary": raw.get("quarter_summary", ""), "themes": kept, "dropped_unsourced": dropped}


# --------------------------------------------------------------------------- main entry
def research_macro_themes(att: dict, config: dict, api_key: str | None = None) -> dict:
    """Run the two-step research. Raises on API errors (the app shows them)."""
    key = api_key or config.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set — add it to .env")
    client = _client(key)

    content = _run_research(client, _research_prompt(att, config))
    notes, sources = _notes_and_sources(content)
    if not sources:
        raise RuntimeError("Web search returned no citable sources — nothing to base themes on")

    src_list = "\n".join(f"[{s['id']}] {s['title']} — {s['url']}" for s in sources)
    resp = client.beta.messages.create(
        model=MODEL, max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": THEMES_SCHEMA}},
        system=STRUCTURE_SYSTEM,
        messages=[{"role": "user", "content":
                   f"Research notes (with source numbers):\n\n{notes}\n\nSources:\n{src_list}\n\n"
                   "Produce 4-6 themes for the period, most important first."}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("Claude declined to structure the themes")
    raw = json.loads(next(b.text for b in resp.content if b.type == "text"))
    out = validate_themes(raw, sources)
    out.update({"period": f"{att['start']} – {att['end']}", "generated": date.today().isoformat(),
                "model": MODEL, "sources": sources, "notes": notes})
    return out


def save_research(result: dict, out_dir) -> Path:
    path = Path(out_dir) / f"macro_research_{result['generated']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return path


def load_latest_research(out_dir) -> dict | None:
    files = sorted(Path(out_dir).glob("macro_research_*.json"))
    if not files:
        return None
    try:
        return json.loads(files[-1].read_text())
    except (OSError, json.JSONDecodeError):
        return None
