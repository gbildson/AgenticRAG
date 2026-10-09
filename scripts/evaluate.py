from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agent import ResearchAgent  # noqa: E402
from app.config import Settings  # noqa: E402
from app.env import load_env_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate routing and answer terms")
    parser.add_argument("--questions-file", type=Path, default=ROOT / "examples/helpdesk/questions.json")
    parser.add_argument("--output", type=Path, default=ROOT / "eval_results.json")
    parser.add_argument("--base-url", help="Use a running API instead of a local agent")
    args = parser.parse_args()
    if not args.base_url:
        load_env_file(ROOT / ".env")

    questions = json.loads(args.questions_file.read_text())
    if not isinstance(questions, list) or not questions or any(
        not isinstance(item, dict) or not item.get("id") or not item.get("question")
        for item in questions
    ):
        raise SystemExit("Questions must be a nonempty JSON array with id and question")
    agent = None if args.base_url else ResearchAgent(Settings())
    details = []
    for item in questions:
        started = time.perf_counter()
        if args.base_url:
            response = requests.post(
                f"{args.base_url.rstrip('/')}/api/chat",
                json={"question": item["question"]}, timeout=180,
            )
            response.raise_for_status()
            result = response.json()
        else:
            result = agent.answer(item["question"]).to_dict()
        used = {
            step["tool"] for step in result.get("trace", [])
            if step.get("status") == "ok"
        }
        required = set(item.get("required_tools", []))
        expected = item.get("expected_terms", [])
        answer = result["answer"]
        details.append({
            "id": item["id"],
            "question": item["question"],
            "answer": answer,
            "used_tools": sorted(used),
            "trace": result.get("trace", []),
            "routing_ok": required.issubset(used) if required else None,
            "terms_ok": all(term.lower() in answer.lower() for term in expected) if expected else None,
            "evidence_count": len(result.get("evidence", [])),
            "latency_seconds": round(time.perf_counter() - started, 3),
        })
        print(f"{item['id']}: routes={sorted(used)}, evidence={details[-1]['evidence_count']}")
    scored_routes = [item["routing_ok"] for item in details if item["routing_ok"] is not None]
    scored_terms = [item["terms_ok"] for item in details if item["terms_ok"] is not None]
    summary = {
        "question_count": len(details),
        "routing_accuracy": sum(scored_routes) / len(scored_routes) if scored_routes else None,
        "term_accuracy": sum(scored_terms) / len(scored_terms) if scored_terms else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "questions": details}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
