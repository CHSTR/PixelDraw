"""Utilidades compartidas: lectura y escritura de JSONL, selección de tareas y pass@k."""
from __future__ import annotations

import json
import math
import os
import random
from collections import defaultdict
from typing import Iterable


def read_jsonl(path: str, warn: bool = True) -> list[dict]:
    """Lee un JSONL ignorando líneas vacías o incompletas (por ejemplo, una escritura cortada con Ctrl+C)."""
    rows, bad = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                bad += 1
    if bad and warn:
        print(f"AVISO: {bad} línea(s) incompleta(s) ignorada(s) en {path}")
    return rows


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def latest_samples(rows: Iterable[dict], drop_errors: bool = True) -> list[dict]:
    """Deja la última versión de cada (tarea, muestra) y, si se pide, descarta los errores del servidor.

    Una corrida retomada vuelve a escribir las muestras que fallaron por error del servidor, así que
    la última línea de cada muestra es la que vale.
    """
    latest = {}
    for r in rows:
        latest[(r["task_id"], r["sample"])] = r
    out = list(latest.values())
    if drop_errors:
        out = [r for r in out if not is_server_error(r)]
    return out


def is_server_error(row: dict) -> bool:
    return str(row.get("finish_reason", "")).startswith("error")


def select_tasks(tasks: list[dict], levels=None, sizes=None, per_cell: int = 0, seed: int = 0) -> list[dict]:
    """Filtra por nivel y tamaño y, si per_cell > 0, toma hasta per_cell tareas por celda (nivel, tamaño).

    Con la misma semilla y el mismo archivo, la selección es idéntica en todos los scripts.
    """
    if levels:
        tasks = [t for t in tasks if t["level"] in levels]
    if sizes:
        tasks = [t for t in tasks if t["size"] in sizes]
    if per_cell:
        rng = random.Random(seed)
        cells = defaultdict(list)
        for t in tasks:
            cells[(t["level"], t["size"])].append(t)
        tasks = [t for v in cells.values() for t in rng.sample(v, min(per_cell, len(v)))]
    return tasks


def pass_at_k(n: int, c: int, k: int) -> float:
    """Estimador insesgado de pass@k (Chen et al., 2021): 1 - C(n-c, k) / C(n, k).

    n = muestras de la tarea, c = muestras correctas. Devuelve NaN si k > n.
    """
    if k > n:
        return float("nan")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)
