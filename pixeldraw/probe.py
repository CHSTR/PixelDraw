"""Mide pass@1 y pass@k de un modelo servido con una API compatible con OpenAI (por ejemplo, llama.cpp).

Ejemplos:
  # Qwen3-1.7B con razonamiento, 10 tareas por nivel y tamaño
  python -m pixeldraw.probe --tasks data/eval.jsonl --base-url http://localhost:8080/v1 \\
      --model qwen3-1.7b --k 4 --per-cell 10 --max-tokens 6000 --temperature 1.0 \\
      --out results/eval/qwen3-1.7b_think.jsonl

  # sin razonamiento (agrega /no_think al prompt)
  python -m pixeldraw.probe --tasks data/eval.jsonl --model qwen3-1.7b --k 8 --per-cell 10 \\
      --no-think --temperature 1.0 --out results/eval/qwen3-1.7b_nothink.jsonl

  # resumir o recalificar una corrida existente
  python -m pixeldraw.probe --summarize results/eval/qwen3-1.7b_think.jsonl
  python -m pixeldraw.probe --tasks data/eval.jsonl --rescore results/eval/qwen3-1.7b_think.jsonl

Cada respuesta se escribe apenas llega, así que la corrida se puede interrumpir: al relanzar el
mismo comando sigue con lo que falta y reintenta las muestras que terminaron en error del servidor.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
import urllib.request
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from .provenance import write_manifest
from .utils import is_server_error, latest_samples, pass_at_k, read_jsonl, select_tasks, write_jsonl
from .verifier import verify


# ------------------------------------------------------------------ cliente

def chat(base_url, model, prompt, *, temperature, top_p, max_tokens, api_key, timeout, seed, extra):
    """Una llamada a /chat/completions. Devuelve el texto, el razonamiento (si viene aparte) y metadatos."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens, "seed": seed, **extra}
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    choice = data["choices"][0]
    msg = choice.get("message", {})
    return {"content": msg.get("content") or "",
            "reasoning": msg.get("reasoning_content") or "",
            "finish_reason": choice.get("finish_reason"),
            "completion_tokens": (data.get("usage") or {}).get("completion_tokens")}


def result_row(task, sample, model, no_think, response, extra_fields=None):
    """Califica una respuesta y arma la fila de resultados (formato común a probe y probe_opencode)."""
    v = verify(task["spec"], response["content"])
    row = {"task_id": task["id"], "level": task["level"], "size": task["size"], "family": task["family"],
           "sample": sample, "model": model, "no_think": no_think,
           "solved": v["solved"], "score": v["score"], "score_adj": v["score_adj"], "error": v["error"],
           "failed_checks": [name for name, ok in v["checks"] if not ok],
           "finish_reason": response["finish_reason"], "completion_tokens": response["completion_tokens"],
           "content": response["content"], "reasoning": response.get("reasoning", "")}
    if extra_fields:
        row.update(extra_fields)
    return row


# ------------------------------------------------------------------ resumen

def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(rows, k=None):
    """Imprime pass@1, pass@k, crédito parcial, errores de formato, truncamiento y calibración."""
    n_err = sum(is_server_error(r) for r in latest_samples(rows, drop_errors=False))
    rows = latest_samples(rows)
    if n_err:
        print(f"(se excluyen {n_err} muestras con error del servidor; relanza el mismo comando para reintentarlas)")
    by_task, meta = defaultdict(list), {}
    for r in rows:
        by_task[r["task_id"]].append(r)
        meta[r["task_id"]] = (r["level"], r["size"], r["family"])
    if not by_task:
        print("sin resultados")
        return
    k = k or max(len(v) for v in by_task.values())

    def aggregate(key_fn):
        groups = defaultdict(lambda: defaultdict(list))
        for tid, rs in by_task.items():
            g = groups[key_fn(meta[tid])]
            n, c = len(rs), sum(r["solved"] for r in rs)
            g["p1"].append(c / n)
            g["pk"].append(pass_at_k(n, c, min(k, n)))
            g["partial"].append(sum(r["score_adj"] for r in rs) / n)
            g["format"].append(sum(r["error"] is not None for r in rs) / n)
            g["trunc"].append(sum(r["finish_reason"] == "length" for r in rs) / n)
            tokens = [r["completion_tokens"] for r in rs if r.get("completion_tokens")]
            if tokens:
                g["tokens"].append(sum(tokens) / len(tokens))
        return groups

    def print_table(groups, title, key_name):
        print(f"\n== {title} ==")
        print(f"{key_name:<22}{'tareas':>7}{'pass@1':>9}{f'pass@{k}':>9}{'parcial':>9}"
              f"{'formato✗':>10}{'trunc':>8}{'tokens':>8}")
        sort_key = lambda key: tuple(str(v).zfill(4) for v in (key if isinstance(key, tuple) else (key,)))
        for key in sorted(groups, key=sort_key):
            d = groups[key]
            label = " ".join(map(str, key)) if isinstance(key, tuple) else str(key)
            print(f"{label:<22}{len(d['p1']):>7}{_mean(d['p1']):>9.3f}{_mean(d['pk']):>9.3f}"
                  f"{_mean(d['partial']):>9.3f}{_mean(d['format']):>10.3f}{_mean(d['trunc']):>8.3f}"
                  f"{_mean(d['tokens']):>8.0f}")

    print_table(aggregate(lambda m: m[0]), "por nivel", "nivel")
    print_table(aggregate(lambda m: (m[0], m[1])), "por nivel × tamaño", "nivel tamaño")
    print_table(aggregate(lambda m: (m[0], m[2])), "por nivel × familia", "nivel familia")

    # calibración: tareas que el modelo resuelve nunca, a veces (zona aprendible) o siempre
    zones = defaultdict(lambda: [0, 0, 0])
    for tid, rs in by_task.items():
        c = sum(r["solved"] for r in rs)
        zones[meta[tid][0]][0 if c == 0 else 2 if c == len(rs) else 1] += 1
    print("\n== calibración (por tarea: nunca / a veces / siempre resuelta) ==")
    for level in sorted(zones):
        never, some, always = zones[level]
        total = never + some + always
        print(f"{level:<6} nunca {never / total:6.1%}   a veces {some / total:6.1%}   siempre {always / total:6.1%}")

    failures = defaultdict(int)
    for r in rows:
        for name in r.get("failed_checks") or []:
            failures[(r["level"], "region" if name.startswith("region") else name)] += 1
        if r["error"]:
            failures[(r["level"], "PARSE:" + r["error"].split("_")[0])] += 1
    print("\n== restricciones más falladas (nivel, restricción): conteo ==")
    for (level, name), count in sorted(failures.items(), key=lambda x: -x[1])[:25]:
        print(f"{level:<4} {name:<34} {count}")


def rescore(tasks_path, results_path):
    """Recalifica una corrida con el verificador actual y la guarda como <archivo>.rescored.jsonl."""
    tasks = {t["id"]: t for t in read_jsonl(tasks_path)}
    rows = read_jsonl(results_path)
    changed = 0
    for r in rows:
        v = verify(tasks[r["task_id"]]["spec"], r["content"])
        changed += v["solved"] != r["solved"]
        r.update(solved=v["solved"], score=v["score"], score_adj=v["score_adj"], error=v["error"],
                 failed_checks=[name for name, ok in v["checks"] if not ok])
    out = results_path.removesuffix(".jsonl") + ".rescored.jsonl"
    write_jsonl(out, rows)
    print(f"{changed} de {len(rows)} muestras cambiaron de veredicto -> {out}")
    summarize(rows)


# ------------------------------------------------------------------ CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", help="JSONL de tareas (data/*.jsonl)")
    ap.add_argument("--base-url", default="http://localhost:8080/v1")
    ap.add_argument("--model", default="local")
    ap.add_argument("--api-key", default=os.environ.get("PD_API_KEY", "none"),
                    help="clave de la API (o variable de entorno PD_API_KEY)")
    ap.add_argument("--retries", type=int, default=6, help="reintentos con espera exponencial (429, timeouts)")
    ap.add_argument("--k", type=int, default=8, help="muestras por tarea")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--no-think", action="store_true", help="agrega /no_think al prompt (Qwen3)")
    ap.add_argument("--extra", default="{}", help='JSON extra para el body, p. ej. \'{"top_k":20}\'')
    ap.add_argument("--workers", type=int, default=8, help="peticiones en paralelo (= slots del servidor)")
    ap.add_argument("--timeout", type=int, default=1200, help="segundos por petición")
    ap.add_argument("--levels", nargs="*", help="solo estos niveles, p. ej. L2 L3")
    ap.add_argument("--sizes", type=int, nargs="*", help="solo estos tamaños de grilla")
    ap.add_argument("--per-cell", type=int, default=0, help="tareas por celda nivel × tamaño (0 = todas)")
    ap.add_argument("--seed", type=int, default=0, help="semilla de la selección de tareas")
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--no-resume", action="store_true", help="no retomar: rehace todas las muestras")
    ap.add_argument("--summarize", metavar="RESULTADOS", help="solo resume un archivo de resultados")
    ap.add_argument("--rescore", metavar="RESULTADOS", help="recalifica con el verificador actual (requiere --tasks)")
    a = ap.parse_args()

    if a.rescore:
        return rescore(a.tasks, a.rescore)
    if a.summarize:
        return summarize(read_jsonl(a.summarize))
    if not a.tasks:
        ap.error("falta --tasks")

    tasks = select_tasks(read_jsonl(a.tasks), a.levels, a.sizes, a.per_cell, a.seed)
    done = set()
    if os.path.exists(a.out) and not a.no_resume:
        done = {(r["task_id"], r["sample"]) for r in read_jsonl(a.out) if not is_server_error(r)}
    jobs = [(t, i) for t in tasks for i in range(a.k) if (t["id"], i) not in done]
    print(f"{len(tasks)} tareas × {a.k} muestras; pendientes {len(jobs)} (ya hechas {len(done)})", flush=True)

    write_manifest(a.out, a, inputs=[a.tasks], extra={"tareas_elegidas": len(tasks)})
    extra = json.loads(a.extra)

    def run(job):
        task, i = job
        prompt = task["prompt"] + (" /no_think" if a.no_think else "")
        seed = zlib.crc32(f"{task['id']}|{i}|{a.seed}".encode())   # estable entre ejecuciones
        for attempt in range(a.retries):
            try:
                response = chat(a.base_url, a.model, prompt, temperature=a.temperature, top_p=a.top_p,
                                max_tokens=a.max_tokens, api_key=a.api_key, timeout=a.timeout,
                                seed=seed, extra=extra)
                break
            except Exception as e:  # noqa: BLE001  (429, timeouts, cortes de red...)
                response = {"content": "", "reasoning": "", "finish_reason": f"error:{e}",
                            "completion_tokens": None}
                time.sleep(min(60, 2 ** attempt) + random.random())
        return result_row(task, i, a.model, a.no_think, response)

    lock, n_done, t0 = threading.Lock(), 0, time.time()
    with open(a.out, "a", encoding="utf-8") as fout, ThreadPoolExecutor(a.workers) as pool:
        for fut in as_completed([pool.submit(run, j) for j in jobs]):
            row = fut.result()
            with lock:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                n_done += 1
                if n_done % 25 == 0 or n_done == len(jobs):
                    elapsed = time.time() - t0
                    remaining = elapsed / n_done * (len(jobs) - n_done)
                    print(f"  {n_done}/{len(jobs)}  {elapsed / 60:.1f} min  (~{remaining / 60:.1f} min restantes)",
                          flush=True)

    chosen = {t["id"] for t in tasks}
    summarize([r for r in read_jsonl(a.out) if r["task_id"] in chosen], a.k)


if __name__ == "__main__":
    main()
