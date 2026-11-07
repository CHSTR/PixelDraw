"""Mide un modelo al que solo se accede a través de OpenCode (por ejemplo, DeepSeek V4.1-Flash en opencode-go).

Llama a `opencode run` una vez por muestra, sin interacción, con un agente propio:
  - un prompt de sistema que pide un plan breve (<plan>) antes de <answer>, para trazas cortas y verificables;
  - temperatura fija;
  - todas las herramientas denegadas: el modelo solo puede responder texto.
El agente se escribe en un directorio de trabajo vacío (--workdir), así OpenCode no ve los archivos del repo.

Los resultados tienen el mismo formato que los de pixeldraw.probe, así que sirven igual para
`--summarize`, `--rescore`, pixeldraw.derive, pixeldraw.report y pixeldraw.render.

Ejemplo:
  python -m pixeldraw.probe_opencode --tasks data/eval.jsonl --model opencode-go/deepseek-v4.1-flash \\
      --k 2 --per-cell 10 --workers 2 --out results/eval/deepseek-v4.1-flash.jsonl

Si el proveedor responde con un límite de uso largo (por ejemplo, la cuota gratuita agotada), la
corrida se detiene en vez de reintentar en vano; basta relanzar el mismo comando más tarde.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .probe import result_row, summarize
from .provenance import write_manifest
from .utils import is_server_error, read_jsonl, select_tasks

SYSTEM_PROMPT = ("Eres un asistente que resuelve la tarea del usuario. Usa exactamente este formato:\n"
                 "<plan>\n"
                 "1-3 líneas: calcula qué celdas ocupa cada parte del dibujo (filas y columnas) y cómo se "
                 "cumplen las restricciones.\n"
                 "</plan>\n"
                 "<answer>\n"
                 "la respuesta en el formato exacto que pide la tarea\n"
                 "</answer>\n"
                 "No escribas nada más. No repitas el enunciado. No uses herramientas ni leas archivos.")


class OpencodeError(RuntimeError):
    """Error devuelto por OpenCode en el evento JSON (no en stderr)."""

    def __init__(self, message, *, status=None, retry_after_ms=None, rc=None):
        super().__init__(message)
        self.status = status
        self.retry_after_ms = float(retry_after_ms) if retry_after_ms is not None else None
        self.rc = rc


def prompt_sha256(prompt=SYSTEM_PROMPT):
    """Huella corta del prompt de sistema, para no mezclar corridas distintas."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


def done_keys(rows, model, temperature, sha, allow_legacy=False):
    """Claves (task_id, sample) reanudables solo si coinciden modelo, temperatura y prompt."""
    done = set()
    for r in rows:
        if "task_id" not in r or "sample" not in r:
            continue
        if is_server_error(r):
            continue
        if r.get("model") != model or r.get("temperature") != temperature:
            continue
        if r.get("prompt_sha256") is None:
            if not allow_legacy:
                continue
        elif r.get("prompt_sha256") != sha:
            continue
        done.add((r["task_id"], r["sample"]))
    return done


def workdir_extras(workdir):
    """Archivos ajenos a opencode.json que OpenCode podría leer como contexto."""
    if not os.path.isdir(workdir):
        return []
    return sorted(name for name in os.listdir(workdir) if name != "opencode.json")


def write_agent(workdir, model, temperature):
    os.makedirs(workdir, exist_ok=True)
    provider, _, model_id = model.partition("/")
    cfg = {
        "$schema": "https://opencode.ai/config.json",
        "agents": {
            "pixeldraw": {
                "description": "Resuelve tareas de PixelDraw sin herramientas",
                "mode": "primary",
                "model": model,
                "system": SYSTEM_PROMPT,
                "permissions": [{"action": "*", "resource": "*", "effect": "deny"}],
            }
        },
    }
    if temperature is not None and provider and model_id:
        # En OpenCode V2 `temperature` a nivel de agente no se envía; va en el body del modelo.
        cfg["providers"] = {
            provider: {"models": {model_id.split("#", 1)[0]: {"body": {"temperature": temperature}}}}
        }
    with open(os.path.join(workdir, "opencode.json"), "w") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def parse_events(stdout: str):
    """Une el texto de la salida --format json y devuelve también metadatos del step_finish."""
    text, reasoning, tools, parsed = [], [], 0, False
    meta = {"finish_reason": None, "tokens": None, "error": None}
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        parsed = True
        typ = str(ev.get("type", ""))
        part = ev.get("part") or {}
        ptype = str(part.get("type", typ))
        if typ == "error" or ev.get("error"):
            meta["error"] = ev.get("error") or {}
            continue
        if typ == "step_finish" or ptype == "step-finish":
            meta["finish_reason"] = part.get("reason") or meta["finish_reason"]
            meta["tokens"] = part.get("tokens") or meta["tokens"]
        if "tool" in typ or "tool" in ptype:
            tools += 1
        t = part.get("text") if isinstance(part, dict) else None
        if t is None:
            t = ev.get("text")
        if not t:
            continue
        if "reason" in typ or "reason" in ptype:
            reasoning.append(t)
        elif typ == "text" or ptype == "text":
            text.append(t)
    if not parsed:                       # versión sin --format json: usar la salida cruda
        return stdout, "", 0, meta
    return "".join(text), "\n".join(reasoning), tools, meta


def run_one(prompt, a):
    cmd = [a.opencode, "run", "--agent", "pixeldraw", "-m", a.model, "--format", "json", prompt]
    # OpenCode resuelve el proyecto/config con PWD, no con el cwd real del proceso.
    env = {**os.environ, "PWD": os.path.abspath(a.workdir)}
    p = subprocess.run(cmd, cwd=a.workdir, env=env, capture_output=True, text=True, timeout=a.timeout)
    content, reasoning, tools, meta = parse_events(p.stdout)
    if p.returncode != 0 or meta["error"] is not None:
        err = meta["error"] or {}
        msg = (err.get("message") or p.stderr.strip()
               or (p.stdout.strip()[:300] if p.stdout.strip() else f"rc={p.returncode}"))
        if err.get("type"):
            msg = f"{err['type']}: {msg}"
        raise OpencodeError(msg, status=err.get("status"),
                            retry_after_ms=err.get("retryAfterMs"), rc=p.returncode)
    return content, reasoning, tools, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", required=True, help="JSONL de tareas (data/*.jsonl)")
    ap.add_argument("--model", default="opencode-go/deepseek-v4.1-flash")
    ap.add_argument("--opencode", default="opencode", help="ruta al ejecutable de OpenCode")
    ap.add_argument("--workdir", default="opencode_workdir", help="directorio vacío donde corre OpenCode")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--k", type=int, default=4, help="muestras por tarea")
    ap.add_argument("--per-cell", type=int, default=0, help="tareas por celda nivel × tamaño (0 = todas)")
    ap.add_argument("--levels", nargs="*", help="solo estos niveles, p. ej. L2 L3")
    ap.add_argument("--sizes", type=int, nargs="*", help="solo estos tamaños de grilla")
    ap.add_argument("--workers", type=int, default=2, help="llamadas en paralelo")
    ap.add_argument("--timeout", type=int, default=600, help="segundos por llamada")
    ap.add_argument("--retries", type=int, default=5, help="intentos por muestra")
    ap.add_argument("--seed", type=int, default=0, help="semilla de la selección de tareas")
    ap.add_argument("--out", default="results_opencode.jsonl")
    ap.add_argument("--resume-legacy", action="store_true",
                    help="retomar filas antiguas sin prompt_sha256 (asume que el prompt no cambió)")
    a = ap.parse_args()

    sha = prompt_sha256()
    extras = workdir_extras(a.workdir)
    if extras:
        print(f"ADVERTENCIA: {a.workdir} no está vacío; OpenCode podría leer {extras[:5]}", flush=True)
    write_agent(a.workdir, a.model, a.temperature)
    tasks = select_tasks(read_jsonl(a.tasks), a.levels, a.sizes, a.per_cell, a.seed)

    previous = read_jsonl(a.out) if os.path.exists(a.out) else []
    legacy = sum(1 for r in previous if r.get("prompt_sha256") is None)
    done = done_keys(previous, a.model, a.temperature, sha, a.resume_legacy)
    if legacy and not a.resume_legacy:
        print(f"AVISO: {legacy} filas antiguas sin prompt_sha256 no se retoman "
              f"(usa --resume-legacy si el prompt no cambió)", flush=True)
    jobs = [(t, i) for t in tasks for i in range(a.k) if (t["id"], i) not in done]
    print(f"{len(tasks)} tareas × {a.k} muestras; pendientes {len(jobs)} (ya hechas {len(done)})", flush=True)
    write_manifest(a.out, a, inputs=[a.tasks],
                   extra={"tareas_elegidas": len(tasks), "prompt_sistema": SYSTEM_PROMPT})

    stop = threading.Event()

    def run(job):
        task, i = job
        content, reasoning, tools, finish, tokens = "", "", 0, "stop", None
        for attempt in range(a.retries):
            if stop.is_set():
                finish = "error:abortado"
                break
            try:
                content, reasoning, tools, meta = run_one(task["prompt"], a)
                finish = meta.get("finish_reason") or "stop"
                tokens = (meta.get("tokens") or {}).get("output")
                break
            except OpencodeError as e:
                finish = f"error:{e}"
                if e.retry_after_ms is not None and e.retry_after_ms > 300_000:
                    stop.set()
                    print(f"ABORTANDO: {e} (reintentar en ~{e.retry_after_ms / 60000:.0f} min)", flush=True)
                    break
                wait = e.retry_after_ms / 1000 if e.retry_after_ms is not None else min(60, 2 ** attempt)
                time.sleep(wait + random.random())
            except Exception as e:  # noqa: BLE001
                finish = f"error:{e}"
                time.sleep(min(60, 2 ** attempt) + random.random())
        response = {"content": content, "reasoning": reasoning, "finish_reason": finish,
                    "completion_tokens": tokens}
        return result_row(task, i, a.model, False, response,
                          extra_fields={"via": "opencode", "temperature": a.temperature,
                                        "prompt_sha256": sha, "tool_events": tools})

    lock, t0, n_done = threading.Lock(), time.time(), 0
    with open(a.out, "a", encoding="utf-8") as fout, ThreadPoolExecutor(a.workers) as pool:
        for fut in as_completed([pool.submit(run, j) for j in jobs]):
            row = fut.result()
            with lock:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                n_done += 1
                if n_done % 10 == 0 or n_done == len(jobs):
                    elapsed = time.time() - t0
                    remaining = elapsed / n_done * (len(jobs) - n_done)
                    print(f"  {n_done}/{len(jobs)}  {elapsed / 60:.1f} min  (~{remaining / 60:.1f} min restantes)",
                          flush=True)
    if stop.is_set():
        print("Corrida detenida por un límite de uso del proveedor; relanza el mismo comando más tarde.", flush=True)

    chosen = {t["id"] for t in tasks}
    rows = [r for r in read_jsonl(a.out) if r.get("task_id") in chosen]
    n_tools = sum(r.get("tool_events", 0) > 0 for r in rows)
    if n_tools:
        print(f"ATENCIÓN: {n_tools} muestras registraron eventos de herramientas; revisa el agente.")
    summarize(rows)


if __name__ == "__main__":
    main()
