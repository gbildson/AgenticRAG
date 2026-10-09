from __future__ import annotations

import argparse
import json

import requests


def ask(base_url: str, question: str, timeout: int = 180) -> dict:
    response = requests.post(
        f"{base_url.rstrip('/')}/api/chat",
        json={"question": question},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def print_result(result: dict, show_evidence: bool) -> None:
    print(result["answer"])
    if not show_evidence:
        return
    print("\n--- Evidence ---")
    for item in result.get("evidence", []):
        if item.get("source_type") == "document":
            location = item["source"]
            if item.get("page"):
                location += f", p. {item['page']}"
            if item.get("section"):
                location += f", {item['section']}"
            print(f"\n[{location}]")
            print(item["excerpt"])
        elif item.get("source_type") == "sql":
            print(f"\nSQL: {item['sql']}")
            print(json.dumps(item.get("rows", []), indent=2))
    tools = [step["tool"] for step in result.get("trace", [])]
    if tools:
        print(f"\nTools: {' -> '.join(tools)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask the running local research assistant"
    )
    parser.add_argument("question", nargs="*", help="One-shot question")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Assistant server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--show-evidence",
        action="store_true",
        help="Print retrieved evidence and the tool trace",
    )
    args = parser.parse_args()

    one_shot = " ".join(args.question).strip()
    if one_shot:
        try:
            print_result(ask(args.base_url, one_shot), args.show_evidence)
        except requests.RequestException as error:
            raise SystemExit(f"Request failed: {error}") from error
        return

    print("Agentic RAG CLI")
    print("Enter a question, or type 'quit' to exit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"quit", "exit", ":q"}:
            break
        if not question:
            continue
        try:
            print_result(ask(args.base_url, question), args.show_evidence)
        except requests.RequestException as error:
            print(f"Request failed: {error}")


if __name__ == "__main__":
    main()
