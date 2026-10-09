from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from app.config import Settings
from app.models import ChatModel, OpenAICompatibleClient
from app.retrieval import DocumentRetriever
from app.sql_tool import ReadOnlySQLTool


SYSTEM_PROMPT = """
You are a research assistant. Answer using only evidence gathered from the tools.
Use search_documents for policy, instructions, explanations, and other document
claims. Use query_database for exact records, counts, and calculations when a
database is available. Use both when a question needs both kinds of evidence.
Make focused follow-up calls when needed.

{database_schema}

Never invent a value or document claim. State any evidence gap. Cite document
claims with the source filename and page or section when available. For SQL
values, identify the source table and calculation. Keep answers concise.
""".strip()


DOCUMENT_TOOL = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": "Search the supplied documents for grounded passages.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Focused search query."},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

SQL_TOOL = {
    "type": "function",
    "function": {
        "name": "query_database",
        "description": "Run one read-only SQLite SELECT query on allowed tables and columns.",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "One SELECT or WITH query."}
            },
            "required": ["sql"],
            "additionalProperties": False,
        },
    },
}


@dataclass
class AgentResult:
    answer: str
    evidence: list[dict]
    trace: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


class ResearchAgent:
    def __init__(
        self,
        settings: Settings,
        client: ChatModel | None = None,
        retriever: DocumentRetriever | None = None,
        sql_tool: ReadOnlySQLTool | None = None,
    ):
        self.settings = settings
        if client is None:
            settings.validate_assets(require_model=True)
        self.client = client or OpenAICompatibleClient(settings)
        self.retriever = retriever or DocumentRetriever(settings)
        self.sql_tool = sql_tool or (
            ReadOnlySQLTool(settings.sql_database, allowed_schema=settings.sql_allowed_schema)
            if settings.sql_database is not None else None
        )
        self.tools = [DOCUMENT_TOOL] + ([SQL_TOOL] if self.sql_tool else [])
        database_schema = (
            "Database schema:\n" + self.sql_tool.schema_description
            if self.sql_tool else "No database is configured."
        )
        self.system_prompt = SYSTEM_PROMPT.format(database_schema=database_schema)

    def _execute_tool(self, name: str, arguments: dict[str, Any]) -> dict:
        if name == "query_database" and self.sql_tool:
            return self.sql_tool.execute(arguments["sql"])
        if name == "search_documents":
            return {
                "source_type": "document_search",
                "query": arguments["query"],
                "results": self.retriever.search(
                    query=arguments["query"],
                    top_k=arguments.get("top_k"),
                ),
            }
        raise ValueError(f"Unknown tool: {name}")

    def answer(self, question: str) -> AgentResult:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty")
        messages: list[Any] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        evidence: list[dict] = []
        trace: list[dict] = []
        for step in range(self.settings.max_agent_steps):
            response = self.client.chat(
                messages=messages,
                tools=self.tools,
                tool_choice="required" if step == 0 else "auto",
                temperature=0.1,
            )
            message = response.choices[0].message
            messages.append(message.model_dump(exclude_none=True) if hasattr(message, "model_dump") else message)
            calls = message.tool_calls or []
            if not calls:
                answer = (message.content or "").strip()
                if answer:
                    return AgentResult(answer, evidence, trace)
                break
            for call in calls:
                name = call.function.name
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                    result = self._execute_tool(name, arguments)
                    status = "ok"
                    if result.get("source_type") == "sql":
                        evidence.append(result)
                    else:
                        evidence.extend(result.get("results", []))
                except Exception as error:
                    arguments = {}
                    result = {"error": str(error)}
                    status = "error"
                trace.append({"step": step + 1, "tool": name, "status": status, "arguments": arguments})
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, default=str),
                })
        messages.append({
            "role": "user",
            "content": "Tool-call limit reached. Answer from the evidence already gathered; state any gap.",
        })
        response = self.client.chat(
            messages=messages, tools=self.tools, tool_choice="none", temperature=0.1
        )
        answer = (response.choices[0].message.content or "").strip()
        return AgentResult(
            answer or "I could not produce a grounded answer from the available evidence.",
            evidence,
            trace,
        )
