"""Construye los datos derivados (trazas SFT y pares DPO) a partir de corridas sobre el split train.

Flujo:
  1. Correr el modelo grande (profesor) sobre train con pixeldraw.probe o pixeldraw.probe_opencode.
  2. Correr el modelo objetivo sobre las mismas tareas: sus fallos son los rechazos on-policy de DPO.
  3. Filtrar:
       python -m pixeldraw.derive --tasks data/train.jsonl \\
           --results results/train/*.jsonl \\
           --sft-models opencode-go/deepseek-v4.1-flash --out-dir derived/

Filtros:
  - cada respuesta se vuelve a calificar con el verificador actual (no se confía en el veredicto guardado);
  - se descartan las respuestas truncadas y las que terminaron en error del servidor;
  - solo se usan tareas de train, para no contaminar eval ni test;
  - las respuestas idénticas de una misma tarea cuentan una vez.

Salidas:
  derived/sft.jsonl  respuestas correctas, las más cortas primero (hasta --sft-per-task por tarea), en
                     formato de mensajes: {"messages": [usuario, asistente], ...}
  derived/dpo.jsonl  pares {"prompt", "chosen", "rejected", ...}: preferida correcta y rechazada incorrecta,
                     priorizando las rechazadas casi correctas (legibles y con más crédito parcial)
  derived/stats.json resumen por nivel
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict

from .provenance import write_manifest
from .utils import is_server_error, read_jsonl, write_jsonl
from .verifier import verify


def completion_text(row, include_reasoning):
    if include_reasoning and row.get("reasoning"):
        return f"<think>\n{row['reasoning'].strip()}\n</think>\n\n{row['content'].strip()}"
    return row["content"]


def load_answers(tasks, result_paths, include_reasoning, allow_non_train):
    """Agrupa por tarea las respuestas válidas, ya recalificadas. Devuelve (por_tarea, descartes)."""
    dropped = Counter()
    latest = {}
    for path in result_paths:                 # si una muestra se reintentó, vale la última
        for r in read_jsonl(path):
            latest[(path, r["task_id"], r["sample"])] = r

    by_task = defaultdict(list)
    for r in latest.values():
        task = tasks.get(r["task_id"])
        if task is None:
            dropped["tarea_desconocida"] += 1
        elif task["split"] != "train" and not allow_non_train:
            dropped["split_no_train"] += 1
        elif r.get("finish_reason") == "length":
            dropped["truncada"] += 1
        elif is_server_error(r):
            dropped["error_servidor"] += 1
        else:
            v = verify(task["spec"], r["content"])
            by_task[r["task_id"]].append({
                "text": completion_text(r, include_reasoning), "content": r["content"],
                "model": r.get("model", "?"), "solved": v["solved"], "score_adj": v["score_adj"],
                "error": v["error"], "failed": [name for name, ok in v["checks"] if not ok]})
    return by_task, dropped


def unique_by_content(answers):
    seen, out = set(), []
    for x in answers:
        key = x["content"].strip()
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", required=True, help="JSONL de tareas")
    ap.add_argument("--results", nargs="+", required=True, help="corridas de pixeldraw.probe / probe_opencode")
    ap.add_argument("--out-dir", default="derived")
    ap.add_argument("--sft-per-task", type=int, default=2, help="máximo de trazas SFT por tarea")
    ap.add_argument("--dpo-per-task", type=int, default=2, help="máximo de pares DPO por tarea")
    ap.add_argument("--sft-models", nargs="*", help="modelos que pueden aportar trazas SFT (por defecto, todos)")
    ap.add_argument("--include-reasoning", action="store_true",
                    help="antepone el razonamiento (reasoning_content) como <think>…</think>")
    ap.add_argument("--allow-non-train", action="store_true", help="permite tareas fuera de train")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    tasks = {t["id"]: t for path in a.tasks for t in read_jsonl(path)}
    by_task, dropped = load_answers(tasks, a.results, a.include_reasoning, a.allow_non_train)

    sft, dpo = [], []
    stats = defaultdict(Counter)
    for tid, answers in by_task.items():
        task = tasks[tid]
        level = task["level"]
        meta = {"task_id": tid, "level": level, "size": task["size"], "family": task["family"]}
        unique = unique_by_content(answers)
        correct = [x for x in unique if x["solved"]]
        wrong = [x for x in unique if not x["solved"]]
        teacher = [x for x in correct if not a.sft_models or x["model"] in a.sft_models]
        stats[level]["tareas"] += 1
        stats[level]["respuestas"] += len(answers)
        stats[level]["tareas_con_acierto"] += bool(teacher)

        # SFT: las respuestas correctas más cortas (menos ruido)
        for i, x in enumerate(sorted(teacher, key=lambda x: len(x["text"]))[:a.sft_per_task]):
            sft.append({"id": f"{tid}-sft{i}", **meta, "source_model": x["model"],
                        "messages": [{"role": "user", "content": task["prompt"]},
                                     {"role": "assistant", "content": x["text"]}]})
            stats[level]["sft"] += 1

        # DPO: rechazos casi correctos primero (grilla legible y mayor crédito parcial)
        if correct and wrong:
            ranked = sorted(wrong, key=lambda x: (x["error"] is None, x["score_adj"]), reverse=True)
            for i, rejected in enumerate(ranked[:a.dpo_per_task]):
                chosen = rng.choice(correct)
                dpo.append({"id": f"{tid}-dpo{i}", **meta, "prompt": task["prompt"],
                            "chosen": chosen["text"], "rejected": rejected["text"],
                            "chosen_model": chosen["model"], "rejected_model": rejected["model"],
                            "rejected_score_adj": rejected["score_adj"],
                            "rejected_failed_checks": rejected["failed"] or [f"PARSE:{rejected['error']}"]})
                stats[level]["dpo"] += 1

    os.makedirs(a.out_dir, exist_ok=True)
    write_jsonl(os.path.join(a.out_dir, "sft.jsonl"), sft)
    write_jsonl(os.path.join(a.out_dir, "dpo.jsonl"), dpo)
    summary = {"descartadas": dict(dropped), "por_nivel": {k: dict(v) for k, v in sorted(stats.items())},
               "total_sft": len(sft), "total_dpo": len(dpo)}
    with open(os.path.join(a.out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_manifest(os.path.join(a.out_dir, "derived.jsonl"), a, inputs=list(a.tasks) + list(a.results),
                   extra={"total_sft": len(sft), "total_dpo": len(dpo)})

    print(f"descartadas: {dict(dropped)}")
    print(f"{'nivel':<6}{'tareas':>8}{'con acierto':>13}{'SFT':>7}{'DPO':>7}")
    for level, c in sorted(stats.items()):
        print(f"{level:<6}{c['tareas']:>8}{c['tareas_con_acierto']:>13}{c['sft']:>7}{c['dpo']:>7}")
    print(f"total: {len(sft)} trazas SFT, {len(dpo)} pares DPO -> {os.path.normpath(a.out_dir)}/")


if __name__ == "__main__":
    main()
