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
import re
from datetime import datetime
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
something. Do not speculate or fill gaps from memory. Write concise research notes as bullet
points, and end every bullet with the URL(s) of the page(s) it comes from in parentheses."""

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


def _get(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


class _Sources:
    """Registry of pages actually retrieved by the server tools (the only citable sources)."""

    def __init__(self):
        self.items: list[dict] = []
        self.index: dict[str, int] = {}

    def add(self, url, title="", quote="") -> int | None:
        if not url:
            return None
        url = str(url).rstrip(").,;")
        if url not in self.index:
            self.items.append({"id": len(self.items) + 1, "url": url, "title": title or url,
                               "quote": (quote or "")[:300]})
            self.index[url] = len(self.items)
        elif quote and not self.items[self.index[url] - 1]["quote"]:
            self.items[self.index[url] - 1]["quote"] = quote[:300]
        return self.index[url]


URL_RE = re.compile(r"https?://[^\s)\]>\"']+")


def _notes_and_sources(content) -> tuple[str, list[dict]]:
    """
    Collect sources from (a) text-block citations, (b) web_search_tool_result result lists,
    (c) web_fetch_tool_result documents — i.e. only pages the tools really returned — and tag the
    notes with [n]: from citations, and from URLs written in the text that match a retrieved page.
    """
    reg = _Sources()
    for block in content:
        btype = _get(block, "type")
        if btype == "web_search_tool_result":
            res = _get(block, "content")
            if isinstance(res, list):
                for r in res:
                    reg.add(_get(r, "url"), _get(r, "title", ""))
        elif btype == "web_fetch_tool_result":
            res = _get(block, "content")
            reg.add(_get(res, "url"), _get(_get(res, "content"), "title", "") if res is not None else "")
    parts: list[str] = []
    for block in content:
        if _get(block, "type") != "text":
            continue
        tags = set()
        for c in _get(block, "citations") or []:
            n = reg.add(_get(c, "url"), _get(c, "title", ""), _get(c, "cited_text", ""))
            if n:
                tags.add(n)
        text = _get(block, "text", "")
        for url in URL_RE.findall(text):
            n = reg.index.get(url.rstrip(").,;"))
            if n:
                tags.add(n)
        parts.append(text + ((" " + "".join(f"[{n}]" for n in sorted(tags))) if tags else ""))
    return "".join(parts).strip(), reg.items


def diagnose(content, stop_reasons: list[str]) -> dict:
    """What happened during the research turn — saved to the log and shown on failure."""
    counts: dict[str, int] = {}
    queries, errors, n_results, n_citations = [], [], 0, 0
    for block in content:
        t = _get(block, "type", "?")
        counts[t] = counts.get(t, 0) + 1
        if t == "server_tool_use":
            inp = _get(block, "input") or {}
            q = _get(inp, "query") or _get(inp, "url")
            if q:
                queries.append(f"{_get(block, 'name')}: {q}")
        elif t in ("web_search_tool_result", "web_fetch_tool_result"):
            res = _get(block, "content")
            if isinstance(res, list):
                n_results += len(res)
            elif _get(res, "error_code"):
                errors.append(f"{t}: {_get(res, 'error_code')}")
            elif res is not None:
                n_results += 1
        elif t == "text":
            n_citations += len(_get(block, "citations") or [])
    return {"stop_reasons": stop_reasons, "block_counts": counts, "searches": queries,
            "tool_errors": errors, "results_returned": n_results, "text_citations": n_citations}


def _dump(block):
    """JSON-safe dump of a content block without the large encrypted fields."""
    d = block.model_dump() if hasattr(block, "model_dump") else (
        dict(block) if isinstance(block, dict) else dict(vars(block)))

    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if "encrypted" not in k}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x if isinstance(x, (str, int, float, bool, type(None))) else str(x)
    return strip(d)


class ResearchError(RuntimeError):
    def __init__(self, msg: str, diagnostics: dict | None = None, log_path: str | None = None):
        super().__init__(msg)
        self.diagnostics = diagnostics or {}
        self.log_path = log_path


def _run_research(client, prompt: str):
    """Web-search turn with pause_turn continuation. Returns the final assistant content."""
    messages = [{"role": "user", "content": prompt}]
    acc: list = []
    stops: list[str] = []
    for _ in range(MAX_CONTINUATIONS + 1):
        with client.beta.messages.stream(
            model=MODEL, max_tokens=32000,
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            output_config={"effort": "high"},
            system=RESEARCH_SYSTEM, tools=[WEB_SEARCH], messages=messages,
        ) as stream:
            resp = stream.get_final_message()
        stops.append(str(resp.stop_reason))
        logger.info("Research turn: stop_reason=%s, %d blocks", resp.stop_reason, len(resp.content))
        if resp.stop_reason == "refusal":
            raise ResearchError("Claude declined the research request", {"stop_reasons": stops})
        acc += list(resp.content)
        if resp.stop_reason != "pause_turn":
            return acc, stops
        # Resume the paused server-tool turn: resend the assistant content, no extra user message
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": acc}]
    return acc, stops


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
def research_macro_themes(att: dict, config: dict, api_key: str | None = None,
                          log_dir=None) -> dict:
    """
    Run the two-step research. Every run writes a log (search queries, tool errors, sources,
    notes, raw blocks) to <log_dir>/macro_research_<timestamp>.json. Raises ResearchError with
    diagnostics on failure.
    """
    key = api_key or config.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ResearchError("ANTHROPIC_API_KEY not set — add it to .env")
    client = _client(key)
    log: dict = {"started": datetime.now().isoformat(timespec="seconds"), "model": MODEL,
                 "tool": WEB_SEARCH, "period": f"{att.get('start')} – {att.get('end')}"}
    log_path = None
    if log_dir is not None:
        log_path = Path(log_dir) / f"macro_research_{datetime.now():%Y%m%d-%H%M%S}.json"

    def write_log():
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False, default=str))
            logger.info("Research log: %s", log_path)

    try:
        content, stops = _run_research(client, _research_prompt(att, config))
        diag = diagnose(content, stops)
        notes, sources = _notes_and_sources(content)
        log.update({"diagnostics": diag, "notes": notes, "sources": sources,
                    "raw_blocks": [_dump(b) for b in content]})
        logger.info("Research: %d searches, %d results, %d citations, %d sources, errors=%s",
                    len(diag["searches"]), diag["results_returned"], diag["text_citations"],
                    len(sources), diag["tool_errors"])
        if not sources:
            hint = ("web search returned errors: " + ", ".join(diag["tool_errors"]) if diag["tool_errors"]
                    else "Claude did not run any web search" if not diag["searches"]
                    else "searches ran but no result pages could be identified")
            raise ResearchError(f"No citable sources ({hint})", diag, str(log_path) if log_path else None)

        src_list = "\n".join(f"[{s_['id']}] {s_['title']} — {s_['url']}" for s_ in sources)
        resp = client.beta.messages.create(
            model=MODEL, max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": THEMES_SCHEMA}},
            system=STRUCTURE_SYSTEM,
            messages=[{"role": "user", "content":
                       f"Research notes (with source numbers where known):\n\n{notes}\n\n"
                       f"Retrieved sources (cite by number):\n{src_list}\n\n"
                       "Produce 4-6 themes for the period, most important first. Cite the source "
                       "numbers each theme relies on."}],
        )
        if resp.stop_reason == "refusal":
            raise ResearchError("Claude declined to structure the themes", diag,
                                str(log_path) if log_path else None)
        raw = json.loads(next(b.text for b in resp.content if b.type == "text"))
        log["structured_raw"] = raw
        out = validate_themes(raw, sources)
        out.update({"period": f"{att['start']} – {att['end']}", "generated": date.today().isoformat(),
                    "model": MODEL, "sources": sources, "notes": notes, "diagnostics": diag,
                    "log_path": str(log_path) if log_path else None})
        log["result"] = {k: v for k, v in out.items() if k not in ("notes", "sources")}
        return out
    except ResearchError as exc:
        log["error"] = str(exc)
        exc.log_path = exc.log_path or (str(log_path) if log_path else None)
        raise
    except Exception as exc:
        log["error"] = f"{type(exc).__name__}: {exc}"
        raise ResearchError(f"{type(exc).__name__}: {exc}", log.get("diagnostics"),
                            str(log_path) if log_path else None) from exc
    finally:
        write_log()


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
