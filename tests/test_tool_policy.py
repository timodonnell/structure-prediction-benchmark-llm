"""Tool-free controls must be checked before any inference reaches the network."""

import json
from types import SimpleNamespace

import pytest

from pdbthink.evaluation.runner import ModelConfig, ProviderError, _post
from pdbthink.evaluation.tool_policy import require_tool_free, response_tool_events


@pytest.mark.parametrize("override", [
    {"tools": "none"}, {"tools": [{"type": "function"}]}, {"tool_choice": "auto"},
    {"functions": [{"name": "python"}]}, {"function_call": "auto"},
    {"plugins": [{"id": "web"}]}, {"model": "model:online:batch"},
    {"messages": [{"role": "tool", "content": "external result"}]},
])
def test_unsafe_requests_never_reach_network(monkeypatch, override):
    called = []
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: called.append(a))
    with pytest.raises(ProviderError):
        _post("http://localhost/v1/chat/completions",
              {"tools": [], "tool_choice": "none", **override}, {}, ModelConfig(model_id="test"))
    assert not called


def test_openrouter_requires_plugin_disabled_even_without_tools():
    body = {"tools": [], "tool_choice": "none"}
    with pytest.raises(ValueError, match="web plugin"):
        require_tool_free(body, "openai_chat", "https://openrouter.ai/api/v1")
    require_tool_free({**body, "plugins": [{"id": "web", "enabled": False}]},
                      "openai_chat", "https://openrouter.ai/api/v1")


def test_native_provider_controls():
    require_tool_free({"tools": None, "tool_choice": "none"}, "openai_chat", "http://localhost:8000/v1")
    with pytest.raises(ValueError, match="explicit tools"):
        require_tool_free({"tool_choice": "none"}, "openai_chat", "http://localhost:8000/v1")
    require_tool_free({"tools": [], "tool_choice": {"type": "none"}}, "anthropic_messages", "https://api.anthropic.com")
    require_tool_free({"tools": []}, "ollama_chat", "http://localhost:11434")


def test_batch_rejects_one_unsafe_request_before_submission(monkeypatch):
    from pdbthink.evaluation.openrouter_batch import OpenRouterClient

    called = []
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: called.append(a))
    safe = {"tools": [], "tool_choice": "none", "plugins": [{"id": "web", "enabled": False}]}
    payload = {"model": "test:batch", "requests": [{"body": safe}, {"body": {}}]}
    with pytest.raises(ValueError, match="tools"):
        OpenRouterClient("https://openrouter.ai/api/v1", "dummy")._request(
            "POST", "/api/v1/batches", json.dumps(payload).encode())
    assert not called


def test_together_rejects_tools_before_upload(tmp_path):
    from pdbthink.evaluation.batch import TogetherBatchClient

    called = []
    client = TogetherBatchClient.__new__(TogetherBatchClient)
    client._client = SimpleNamespace(files=SimpleNamespace(upload=lambda **k: called.append(k)))
    path = tmp_path / "request.jsonl"
    path.write_text(json.dumps({"body": {"tools": [{"type": "function"}]}}))
    with pytest.raises(ValueError, match="tools"):
        client.upload(path)
    assert not called


def test_tool_audit_detects_even_empty_tool_finish():
    assert response_tool_events({"choices": [{"message": {}, "finish_reason": "tool_calls"}]})
    assert response_tool_events({"choices": [{"message": {"function_call": {"name": "x"}}}]})
    assert response_tool_events({"choices": [{"message": {"content": "FINAL: A"}}]}) == []
