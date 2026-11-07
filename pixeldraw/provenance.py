"""Procedencia: escribe un manifiesto JSON junto a cada archivo de resultados.

El manifiesto registra el comando, los parámetros, el hash del archivo de tareas y el hash del
código del verificador, para que cualquier número del informe se pueda rastrear y reproducir.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VERIFIER_FILES = ("core.py", "verifier.py")


def sha256_file(path, n=16):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def verifier_hash():
    h = hashlib.sha256()
    for name in VERIFIER_FILES:
        with open(os.path.join(HERE, name), "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


def write_manifest(out_path, args, inputs=(), extra=None):
    """Escribe <out_path>.manifest.json. `inputs` = archivos de entrada a registrar por hash."""
    man = {
        "creado": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "comando": " ".join([os.path.basename(sys.argv[0])] + sys.argv[1:]),
        "parametros": {k: v for k, v in vars(args).items() if k not in ("api_key",)},
        "verificador_sha256": verifier_hash(),
        "entradas": {p: sha256_file(p) for p in inputs if p and os.path.exists(p)},
        "python": platform.python_version(),
    }
    if extra:
        man.update(extra)
    base = out_path[:-6] if out_path.endswith(".jsonl") else out_path
    path = base + ".manifest.json"
    runs = []
    if os.path.exists(path):          # una corrida retomada agrega una entrada, no borra la anterior
        try:
            prev = json.load(open(path))
            runs = prev.get("corridas", [prev])
        except ValueError:
            runs = []
    runs.append(man)
    with open(path, "w") as f:
        json.dump({"corridas": runs}, f, ensure_ascii=False, indent=2)
    return path
