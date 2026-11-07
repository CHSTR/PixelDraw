"""Corre la batería del verificador sobre una muestra de eval (cada 10.ª tarea, ~74 tareas).

La batería completa sobre un split se corre con:  python -m tests.verifier_suite data/eval.jsonl
"""
from pathlib import Path

from pixeldraw.verifier import load_tasks
from tests.verifier_suite import FAILS, run_suite

EVAL = Path(__file__).resolve().parent.parent / "data" / "eval.jsonl"


def test_bateria_del_verificador_sin_fallas():
    tasks = load_tasks(str(EVAL))[::10]
    ok, total = run_suite(tasks, verbose=False)
    assert total > 5000
    assert ok == total, {cat: lst[:3] for cat, lst in FAILS.items()}
