#!/usr/bin/env python3
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8152/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "local-gguf")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))
OUT_JSON = Path(os.environ.get("OUT_JSON", "llama-cpp-vulkan/out/gemma4-compaction/compaction_report.json"))
OUT_MD = Path(os.environ.get("OUT_MD", "llama-cpp-vulkan/out/gemma4-compaction/compaction_report.md"))


def chat(messages: List[Dict[str, str]], max_tokens: int = MAX_TOKENS, temperature: float = 0.2) -> Dict:
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.8,
        "top_k": 20,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        f"{API_BASE}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    obj = json.loads(body)
    if isinstance(obj, dict) and obj.get("error"):
        raise RuntimeError(f"Model error: {obj['error']}")
    return obj


def get_text(resp_obj: Dict) -> str:
    choices = resp_obj.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return (message.get("content") or "").strip()


def approx_tokens_from_chars(s: str) -> int:
    # Simple local heuristic used when usage.prompt_tokens is unavailable.
    return max(1, len(s) // 4)


@dataclass
class Scenario:
    name: str
    brief: str
    transcript_turns: List[str]
    final_question: str
    expected_keywords: List[str]


def build_scenarios() -> List[Scenario]:
    coding_turns = [
        "User: Build a C# genetic optimizer for double[]; deterministic seed is mandatory.",
        "Assistant: Acknowledged; seed-driven RNG and reproducible runs.",
        "User: Add cancellation token and max generations cap.",
        "Assistant: Will implement cancellation checks each generation.",
        "User: Keep population size small for local CPU fallback.",
        "Assistant: Proposed defaults: pop=48, elite=6.",
        "User: Fitness evaluations should support parallel workers.",
        "Assistant: Use Parallel.ForEach with bounded degree of parallelism.",
        "User: Must keep thread-safety for shared best score updates.",
        "Assistant: Use lock-free compare-exchange for best score state.",
        "User: Add benchmark target: minimize Rosenbrock in 20 dimensions.",
        "Assistant: Add benchmark fixture with strict timeout.",
        "User: Need checkpoint save/load between runs.",
        "Assistant: Add JSON checkpoint serializer with schema version.",
        "User: We need clear logs for each generation.",
        "Assistant: Add structured generation metrics output.",
        "User: Document complexity and memory behavior.",
        "Assistant: Include Big-O estimates and memory notes in docs.",
        "User: Add unit tests for deterministic behavior and checkpoint roundtrip.",
        "Assistant: Plan tests: same seed reproducibility + checkpoint parity.",
    ]
    coding_question = (
        "/no_think Based on the whole transcript, produce a concise implementation plan with exact acceptance criteria. "
        "Must explicitly mention deterministic seed, thread-safety, max generations, checkpoint save/load, and unit tests."
    )

    reasoning_turns = [
        "User: Plan migration from cloud notebooks to local AI workstation.",
        "Assistant: We should inventory workloads and risk.",
        "User: Constraint: sensitive data cannot leave local network.",
        "Assistant: Then prioritize local storage, redaction, and offline paths.",
        "User: Team has mixed GPU driver reliability issues.",
        "Assistant: Add fallback paths and a watchdog harness.",
        "User: We need runbooks for crashes and deadlocks.",
        "Assistant: Add triage flow with logs and restart scripts.",
        "User: Keep weekly summary of failures and fixes.",
        "Assistant: Define template with root cause, mitigation, owner.",
        "User: Training budget is limited; optimize for practical wins.",
        "Assistant: Focus on inference quality and reproducibility first.",
        "User: Need auditability for model/version changes.",
        "Assistant: Track hashes, Docker tags, and config snapshots.",
        "User: We must keep onboarding simple for new developers.",
        "Assistant: Provide one-command harness and examples.",
        "User: Add success metrics over 30 days.",
        "Assistant: Measure crash rate, rerun success, and response latency.",
    ]
    reasoning_question = (
        "/no_think Give a prioritized 30-day action plan from this transcript. "
        "Must include data-safety, fallback strategy, crash runbook, auditability, onboarding, and measurable KPIs."
    )

    return [
        Scenario(
            name="coding_long_history",
            brief="Long coding coordination history with many constraints.",
            transcript_turns=coding_turns,
            final_question=coding_question,
            expected_keywords=["seed", "thread", "generation", "checkpoint", "test"],
        ),
        Scenario(
            name="reasoning_long_history",
            brief="Long non-coding planning/problem-solving history.",
            transcript_turns=reasoning_turns,
            final_question=reasoning_question,
            expected_keywords=["data", "fallback", "runbook", "audit", "onboarding", "kpi"],
        ),
    ]


def chunk_list(items: List[str], size: int) -> List[List[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def summarize_chunk(chunk: List[str], scenario_name: str) -> str:
    chunk_text = "\n".join(chunk)
    messages = [
        {"role": "system", "content": "You summarize technical conversation history for long-context compression."},
        {
            "role": "user",
            "content": (
                f"/no_think Summarize this {scenario_name} transcript into at most 5 concise bullets. "
                f"Keep only hard constraints, key decisions, and pending tasks. "
                f"Output must be <= 600 characters.\n\n"
                f"{chunk_text}"
            ),
        },
    ]
    resp = chat(messages, max_tokens=220, temperature=0.1)
    return get_text(resp)


def build_compacted_memory(turns: List[str], scenario_name: str, keep_recent: int = 6, chunk_size: int = 4) -> Tuple[str, List[str]]:
    old = turns[:-keep_recent] if len(turns) > keep_recent else []
    recent = turns[-keep_recent:] if len(turns) > keep_recent else turns
    memory_text = summarize_chunk(old, scenario_name) if old else ""
    return memory_text, recent


def evaluate_keywords(answer: str, keywords: List[str]) -> Dict[str, object]:
    lowered = answer.lower()
    found = [k for k in keywords if k.lower() in lowered]
    missing = [k for k in keywords if k.lower() not in lowered]
    return {
        "score": len(found) / max(1, len(keywords)),
        "found": found,
        "missing": missing,
    }


def run_scenario(s: Scenario) -> Dict[str, object]:
    full_transcript = "\n".join(s.transcript_turns)
    baseline_prompt = (
        f"Scenario: {s.name}\n{s.brief}\n\nFull transcript:\n{full_transcript}\n\n"
        f"Final request:\n{s.final_question}"
    )

    baseline_messages = [
        {"role": "system", "content": "You are a precise planning assistant."},
        {"role": "user", "content": baseline_prompt},
    ]

    t0 = time.time()
    baseline_resp = chat(baseline_messages, max_tokens=MAX_TOKENS, temperature=0.2)
    t1 = time.time()
    baseline_text = get_text(baseline_resp)

    memory_text, recent_turns = build_compacted_memory(s.transcript_turns, s.name, keep_recent=4, chunk_size=4)
    compact_prompt = (
        f"Scenario: {s.name}\n{s.brief}\n\nCompacted memory:\n{memory_text}\n\nRecent turns:\n"
        + "\n".join(recent_turns)
        + f"\n\nFinal request:\n{s.final_question}"
    )

    compact_messages = [
        {"role": "system", "content": "You are a precise planning assistant."},
        {"role": "user", "content": compact_prompt},
    ]

    t2 = time.time()
    compact_resp = chat(compact_messages, max_tokens=MAX_TOKENS, temperature=0.2)
    t3 = time.time()
    compact_text = get_text(compact_resp)

    baseline_usage = baseline_resp.get("usage") if isinstance(baseline_resp, dict) else None
    compact_usage = compact_resp.get("usage") if isinstance(compact_resp, dict) else None

    baseline_prompt_tokens = (baseline_usage or {}).get("prompt_tokens") or approx_tokens_from_chars(baseline_prompt)
    compact_prompt_tokens = (compact_usage or {}).get("prompt_tokens") or approx_tokens_from_chars(compact_prompt)

    return {
        "scenario": s.name,
        "brief": s.brief,
        "baseline": {
            "latency_s": round(t1 - t0, 3),
            "prompt_chars": len(baseline_prompt),
            "prompt_tokens_est": baseline_prompt_tokens,
            "response": baseline_text,
            "keyword_eval": evaluate_keywords(baseline_text, s.expected_keywords),
        },
        "compact": {
            "latency_s": round(t3 - t2, 3),
            "prompt_chars": len(compact_prompt),
            "prompt_tokens_est": compact_prompt_tokens,
            "response": compact_text,
            "keyword_eval": evaluate_keywords(compact_text, s.expected_keywords),
            "memory_blocks_chars": len(memory_text),
            "recent_turn_count": len(recent_turns),
        },
        "reduction": {
            "prompt_chars_reduction_pct": round((1 - (len(compact_prompt) / max(1, len(baseline_prompt)))) * 100.0, 2),
            "prompt_tokens_reduction_pct": round((1 - (compact_prompt_tokens / max(1, baseline_prompt_tokens))) * 100.0, 2),
        },
    }


def to_markdown(report: Dict[str, object]) -> str:
    lines = []
    lines.append("# Gemma4 Context Compaction Demo")
    lines.append("")
    lines.append(f"- api_base: `{report['api_base']}`")
    lines.append(f"- model: `{report['model']}`")
    lines.append("")
    for row in report["scenarios"]:
        lines.append(f"## {row['scenario']}")
        lines.append(f"- brief: {row['brief']}")
        lines.append(f"- baseline prompt chars: `{row['baseline']['prompt_chars']}`")
        lines.append(f"- compact prompt chars: `{row['compact']['prompt_chars']}`")
        lines.append(f"- chars reduction: `{row['reduction']['prompt_chars_reduction_pct']}%`")
        lines.append(f"- token reduction: `{row['reduction']['prompt_tokens_reduction_pct']}%`")
        lines.append(f"- baseline keyword score: `{row['baseline']['keyword_eval']['score']}`")
        lines.append(f"- compact keyword score: `{row['compact']['keyword_eval']['score']}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    scenarios = build_scenarios()
    rows = [run_scenario(s) for s in scenarios]
    report = {
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "api_base": API_BASE,
        "model": MODEL_NAME,
        "scenarios": rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    OUT_MD.write_text(to_markdown(report), encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
