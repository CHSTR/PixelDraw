"""Tests de pixeldraw.probe_opencode y pixeldraw.utils (no requieren OpenCode ni red)."""
import json
import os
from subprocess import CompletedProcess

import pytest

import pixeldraw.probe_opencode as po
from pixeldraw.utils import read_jsonl


class Args:
    opencode = "opencode"
    model = "opencode-go/deepseek-v4.1-flash"
    workdir = "workdir"
    timeout = 30


def test_parse_events_extrae_texto_finish_reason_y_tokens():
    stdout = "\n".join(json.dumps(e) for e in [
        {"type": "step_start", "part": {"type": "step-start"}},
        {"type": "text", "part": {"type": "text", "text": "hola"}},
        {"type": "step_finish", "part": {"type": "step-finish", "reason": "stop",
                                        "tokens": {"input": 10, "output": 5}}},
    ])

    content, reasoning, tools, meta = po.parse_events(stdout)

    assert content == "hola"
    assert reasoning == ""
    assert tools == 0
    assert meta["finish_reason"] == "stop"
    assert meta["tokens"]["output"] == 5
    assert meta["error"] is None


def test_run_one_pasa_pwd_al_subproceso(monkeypatch, tmp_path):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured.update(kwargs)
        return CompletedProcess(cmd, 0, stdout=json.dumps(
            {"type": "text", "part": {"type": "text", "text": "ok"}}), stderr="")

    monkeypatch.setattr(po.subprocess, "run", fake_run)
    args = Args()
    args.workdir = str(tmp_path)

    po.run_one("hola", args)

    assert captured["env"]["PWD"] == os.path.abspath(str(tmp_path))
    assert captured["cwd"] == str(tmp_path)


def test_run_one_expone_el_error_json_de_opencode(monkeypatch, tmp_path):
    error = {"type": "error", "error": {
        "type": "provider.free-limit", "message": "Free usage limit reached.",
        "status": 429, "retryAfterMs": 7143000}}

    def fake_run(cmd, **kwargs):
        return CompletedProcess(cmd, 1, stdout=json.dumps(error), stderr="")

    monkeypatch.setattr(po.subprocess, "run", fake_run)

    with pytest.raises(po.OpencodeError) as exc:
        po.run_one("hola", Args())

    assert exc.value.status == 429
    assert exc.value.retry_after_ms == 7143000
    assert "Free usage limit" in str(exc.value)


def test_write_agent_usa_esquema_v2_y_temperature_en_el_body(tmp_path):
    po.write_agent(str(tmp_path), "opencode-go/deepseek-v4.1-flash", 0.7)

    cfg = json.loads((tmp_path / "opencode.json").read_text())
    agent = cfg["agents"]["pixeldraw"]

    assert "agent" not in cfg
    assert agent["system"] == po.SYSTEM_PROMPT
    assert agent["permissions"] == [{"action": "*", "resource": "*", "effect": "deny"}]
    assert cfg["providers"]["opencode-go"]["models"]["deepseek-v4.1-flash"]["body"]["temperature"] == 0.7


def test_read_jsonl_salta_lineas_corruptas(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text('{"task_id": "a", "sample": 0}\n{"task_id": "b"\n')

    rows = read_jsonl(str(p), warn=False)

    assert rows == [{"task_id": "a", "sample": 0}]


def test_prompt_sha256_es_estable_y_distingue():
    assert po.prompt_sha256("hola") == po.prompt_sha256("hola")
    assert po.prompt_sha256("hola") != po.prompt_sha256("otra")


def test_done_keys_filtra_por_modelo_temperatura_y_prompt():
    rows = [
        {"task_id": "a", "sample": 0, "model": "m", "temperature": 0.7,
         "prompt_sha256": "abc", "finish_reason": "stop"},
        {"task_id": "b", "sample": 0, "model": "m", "temperature": 0.7,
         "prompt_sha256": "otro", "finish_reason": "stop"},
        {"task_id": "c", "sample": 0, "model": "otro", "temperature": 0.7,
         "prompt_sha256": "abc", "finish_reason": "stop"},
        {"task_id": "d", "sample": 0, "model": "m", "temperature": 0.7,
         "finish_reason": "stop"},
        {"task_id": "e", "sample": 0, "model": "m", "temperature": 0.7,
         "prompt_sha256": "abc", "finish_reason": "error:timeout"},
    ]

    assert po.done_keys(rows, "m", 0.7, "abc") == {("a", 0)}
    assert po.done_keys(rows, "m", 0.7, "abc", allow_legacy=True) == {("a", 0), ("d", 0)}


def test_workdir_extras_ignora_opencode_json(tmp_path):
    (tmp_path / "opencode.json").write_text("{}")
    (tmp_path / "AGENTS.md").write_text("x")

    assert po.workdir_extras(str(tmp_path)) == ["AGENTS.md"]
    assert po.workdir_extras(str(tmp_path / "no-existe")) == []
