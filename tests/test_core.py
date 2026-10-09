from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import pymupdf
from fastapi.testclient import TestClient
from openai import APIStatusError

from app.agent import AgentResult, ResearchAgent
from app.api import app, get_agent
from app.config import Settings
from app.env import load_env_file
from app.ingest import build_index, extract_chunks, index_is_current
from app.retrieval import DocumentRetriever
from app.sql_tool import ReadOnlySQLTool, SQLValidationError

ROOT = Path(__file__).resolve().parents[1]


def test_env_file_keeps_exported_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# Demo settings\n"
        "CHAT_MODEL=from-file\n"
        "LLM_API_KEY='demo key'\n"
        "SQL_ALLOWED_SCHEMA='{\"tickets\":[\"id\"]}'\n"
    )
    monkeypatch.setenv("CHAT_MODEL", "from-shell")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("SQL_ALLOWED_SCHEMA", raising=False)

    load_env_file(env_file)

    assert os.environ["CHAT_MODEL"] == "from-shell"
    assert os.environ["LLM_API_KEY"] == "demo key"
    assert json.loads(os.environ["SQL_ALLOWED_SCHEMA"]) == {"tickets": ["id"]}


def test_helpdesk_documents_are_indexed_and_cited(tmp_path):
    settings = replace(Settings(), index_dir=tmp_path / "index")
    chunks = extract_chunks(settings.documents_dir)
    assert {chunk.source for chunk in chunks} == {
        "device-replacement.md", "password-reset.md", "vpn.md"
    }
    assert any(chunk.section == "Eligibility" for chunk in chunks)
    manifest = build_index(settings)
    assert manifest["embedding_model"] is None
    assert index_is_current(settings)
    assert not settings.vectors_file.exists()
    hits = DocumentRetriever(settings, build_if_missing=False).search("replacement repair")
    assert hits[0]["source"] == "device-replacement.md"
    assert hits[0]["section"] == "Eligibility"
    assert "two business days" in hits[0]["excerpt"]
    assert any(
        hit["section"] == "Eligibility"
        for hit in DocumentRetriever(settings, build_if_missing=False).search("Eligibility")
    )


def test_fts_query_drops_filler_but_keeps_negation():
    query = DocumentRetriever._fts_query(
        "What should I do if I cannot use my registered second factor?"
    )
    assert query == '"cannot" OR "use" OR "registered" OR "second" OR "factor"'


def test_text_and_pdf_documents_keep_source_locations(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.txt").write_text("Escalate network outages to the service desk.")
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Printer support instructions")
    pdf.save(docs / "printer.pdf")
    pdf.close()
    chunks = extract_chunks(docs)
    assert any(chunk.source == "notes.txt" and chunk.page is None for chunk in chunks)
    assert any(chunk.source == "printer.pdf" and chunk.page == 1 for chunk in chunks)


class FakeEmbedder:
    def embed(self, texts):
        return [
            [float(text.lower().count("replacement")), float(text.lower().count("password")), 1.0]
            for text in texts
        ]


def test_optional_embeddings_add_vector_search(tmp_path):
    settings = replace(Settings(), index_dir=tmp_path / "index", embedding_model="fake")
    embedder = FakeEmbedder()
    manifest = build_index(settings, embedder=embedder)
    assert manifest["embedding_dimensions"] == 3
    assert np.load(settings.vectors_file).shape[1] == 3
    hits = DocumentRetriever(settings, embedder=embedder, build_if_missing=False).search("replacement")
    assert hits and hits[0]["source"] == "device-replacement.md"


def test_sql_allowlist_protects_custom_database(tmp_path):
    database = tmp_path / "support.sqlite"
    connection = sqlite3.connect(database)
    connection.executescript(
        "CREATE TABLE tickets(id INTEGER, status TEXT, private_note TEXT);"
        "CREATE TABLE staff(id INTEGER, salary INTEGER);"
        "INSERT INTO tickets VALUES (1, 'open', 'internal');"
        "INSERT INTO staff VALUES (1, 100000);"
    )
    connection.close()
    tool = ReadOnlySQLTool(database, allowed_schema={"tickets": ["id", "status"]})
    assert tool.schema_description == "tickets(id, status)"
    assert tool.execute("SELECT status FROM tickets")["rows"] == [{"status": "open"}]
    for sql in ("SELECT private_note FROM tickets", "SELECT salary FROM staff", "DELETE FROM tickets"):
        with pytest.raises(SQLValidationError):
            tool.execute(sql)


class FakeChat:
    def __init__(self):
        self.calls = 0

    def chat(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            assert kwargs["tool_choice"] == "required"
            calls = [
                SimpleNamespace(
                    id="sql-1",
                    function=SimpleNamespace(
                        name="query_database",
                        arguments=json.dumps({"sql": "SELECT model FROM assets WHERE status='available' AND device_type='laptop'"}),
                    ),
                ),
                SimpleNamespace(
                    id="docs-1",
                    function=SimpleNamespace(
                        name="search_documents",
                        arguments=json.dumps({"query": "replacement eligibility repair"}),
                    ),
                ),
            ]
            message = SimpleNamespace(content=None, tool_calls=calls)
        else:
            assert len([m for m in kwargs["messages"] if isinstance(m, dict) and m.get("role") == "tool"]) == 2
            message = SimpleNamespace(
                content="CedarBook 14 and 16 are available. A replacement depends on repair taking over two business days [device-replacement.md, Eligibility].",
                tool_calls=[],
            )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_agent_combines_sql_and_document_evidence(tmp_path):
    settings = replace(Settings(), index_dir=tmp_path / "index")
    build_index(settings)
    agent = ResearchAgent(settings, client=FakeChat())
    result = agent.answer("Which laptops are available and when can I get a replacement?")
    assert [step["tool"] for step in result.trace] == ["query_database", "search_documents"]
    assert {item["source_type"] for item in result.evidence} == {"sql", "document"}
    assert "assets(" in agent.system_prompt


def test_documents_only_configuration(tmp_path):
    settings = replace(Settings(), index_dir=tmp_path / "index", sql_database=None)
    build_index(settings)
    agent = ResearchAgent(settings, client=FakeChat())
    assert [tool["function"]["name"] for tool in agent.tools] == ["search_documents"]


def test_agent_search_ignores_model_supplied_source(tmp_path):
    settings = replace(Settings(), index_dir=tmp_path / "index", sql_database=None)
    build_index(settings)
    agent = ResearchAgent(settings, client=FakeChat())
    assert "source" not in agent.tools[0]["function"]["parameters"]["properties"]

    result = agent._execute_tool(
        "search_documents",
        {"query": "second factor unavailable", "source": "User_Authentication_Guide.pdf"},
    )
    assert any(hit["source"] == "password-reset.md" for hit in result["results"])
    assert agent.retriever.search("second factor unavailable", source="password-reset.md")


def test_api_contract_with_overridden_agent():
    class FakeAgent:
        def answer(self, question: str):
            return AgentResult(f"Grounded answer for: {question}", [], [])

    app.dependency_overrides[get_agent] = lambda: FakeAgent()
    try:
        client = TestClient(app)
        response = client.post("/api/chat", json={"question": "test"})
        assert response.status_code == 200
        assert response.json()["answer"] == "Grounded answer for: test"
        assert client.post("/api/chat", json={"question": ""}).status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_api_preserves_model_provider_status():
    class ProviderError(APIStatusError):
        def __init__(self):
            self.status_code = 429
            self.message = "No credits remaining"

    class FailingAgent:
        def answer(self, _question: str):
            raise ProviderError()

    app.dependency_overrides[get_agent] = lambda: FailingAgent()
    try:
        response = TestClient(app).post("/api/chat", json={"question": "test"})
        assert response.status_code == 429
        assert "No credits remaining" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_evaluator_accepts_question_only_dataset(tmp_path, monkeypatch):
    from scripts import evaluate

    monkeypatch.setattr(os, "environ", os.environ.copy())
    class FakeAgent:
        def __init__(self, settings):
            assert settings.chat_model == "demo-model"

        def answer(self, question: str):
            return AgentResult(f"Answer for {question}", [], [])

    questions = tmp_path / "questions.json"
    output = tmp_path / "results.json"
    (tmp_path / ".env").write_text("CHAT_MODEL=demo-model\n")
    questions.write_text(json.dumps([{"id": "q1", "question": "Test?"}]))
    monkeypatch.setattr(evaluate, "ROOT", tmp_path)
    monkeypatch.delenv("CHAT_MODEL", raising=False)
    monkeypatch.setattr(evaluate, "ResearchAgent", FakeAgent)
    monkeypatch.setattr("sys.argv", ["evaluate.py", "--questions-file", str(questions), "--output", str(output)])
    evaluate.main()
    assert json.loads(output.read_text())["summary"]["question_count"] == 1
