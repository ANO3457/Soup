"""soup serve: /v1/tools/python needs OS-level isolation, as /v1/tools/bash does.

Where the code sandbox has only its in-process guards to offer (the
``best-effort`` isolation strategy: Windows, or a Linux interpreter without
``os.unshare``), the python route answers 501 and does not start the
sandbox, the same answer the bash route gives there.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

PYTHON_DETAIL = "python sandbox requires OS-level isolation"
BASH_DETAIL = "bash sandbox requires OS-level isolation"
TOKEN = "s3cret-token"


def _client(**kwargs) -> TestClient:
    from soup_cli.commands.serve import _create_app

    base = dict(
        model_obj=MagicMock(),
        tokenizer=MagicMock(),
        device="cpu",
        model_name="test-model",
        max_tokens_default=128,
    )
    base.update(kwargs)
    return TestClient(_create_app(**base), base_url="http://127.0.0.1")


@pytest.fixture
def sandbox_calls(monkeypatch) -> list[str]:
    """Replace the code sandbox with a recorder that returns fixed stdout."""
    calls: list[str] = []

    def _fake_sandbox(code: str) -> str:
        calls.append(code)
        return "ran"

    monkeypatch.setattr("soup_cli.trainer.rewards._run_code_sandbox", _fake_sandbox)
    return calls


def _set_strategy(monkeypatch, strategy: str) -> None:
    monkeypatch.setattr("soup_cli.trainer.rewards._get_isolation_strategy", lambda: strategy)


def test_python_tool_refused_without_os_isolation(monkeypatch, sandbox_calls):
    _set_strategy(monkeypatch, "best-effort")
    resp = _client().post("/v1/tools/python", json={"code": "print(1)"})
    assert resp.status_code == 501, resp.text
    assert resp.json()["detail"] == PYTHON_DETAIL
    assert sandbox_calls == []


@pytest.mark.parametrize("strategy", ["namespaces", "sandbox-exec"])
def test_python_tool_runs_with_os_isolation(monkeypatch, sandbox_calls, strategy):
    _set_strategy(monkeypatch, strategy)
    resp = _client().post("/v1/tools/python", json={"code": "print(1)"})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"stdout": "ran", "stderr": "", "exit_code": 0, "timed_out": False}
    assert sandbox_calls == ["print(1)"]


@pytest.mark.parametrize(
    ("path", "body", "detail"),
    [
        ("/v1/tools/python", {"code": "print(1)"}, PYTHON_DETAIL),
        ("/v1/tools/bash", {"command": "echo 1"}, BASH_DETAIL),
    ],
)
def test_code_tools_give_the_same_answer_without_os_isolation(
    monkeypatch, sandbox_calls, path, body, detail
):
    _set_strategy(monkeypatch, "best-effort")
    resp = _client().post(path, json=body)
    assert resp.status_code == 501, (path, resp.text)
    assert resp.json()["detail"] == detail


def test_invalid_request_is_still_400_without_os_isolation(monkeypatch, sandbox_calls):
    _set_strategy(monkeypatch, "best-effort")
    resp = _client().post("/v1/tools/python", json={})
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == "Invalid request"


def test_token_is_checked_before_isolation(monkeypatch, sandbox_calls):
    _set_strategy(monkeypatch, "best-effort")
    client = _client(auth_token=TOKEN)
    resp = client.post(
        "/v1/tools/python",
        json={"code": "print(1)"},
        headers={"Authorization": "Bearer wrong"},
    )
    assert resp.status_code == 401, resp.text
    assert resp.json()["detail"] == "Invalid or missing bearer token"
    assert sandbox_calls == []


def test_correct_token_does_not_lift_the_isolation_requirement(monkeypatch, sandbox_calls):
    _set_strategy(monkeypatch, "best-effort")
    client = _client(auth_token=TOKEN)
    resp = client.post(
        "/v1/tools/python",
        json={"code": "print(1)"},
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert resp.status_code == 501, resp.text
    assert resp.json()["detail"] == PYTHON_DETAIL
    assert sandbox_calls == []
