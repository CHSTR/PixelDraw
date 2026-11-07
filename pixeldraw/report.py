"""Tablas de línea base y calibración, en Markdown y LaTeX (booktabs), a partir de varias corridas.

Escribe en --out-dir:
  1. línea base por modelo y nivel: N tareas, n muestras por tarea, pass@1, pass@k, crédito parcial,
     respuestas ilegibles y truncadas;
  2. calibración por modelo y nivel: porcentaje de tareas resueltas nunca, a veces o siempre;
  3. con --families, pass@1 y truncamiento por familia.

Cada corrida se pasa como ETIQUETA=ARCHIVO. Se usa la última versión de cada muestra y se excluyen los
errores del servidor. Con --tasks, todo se recalifica con el verificador actual.

Ejemplo:
  python -m pixeldraw.report --tasks data/eval.jsonl \\
      --run "Qwen3-1.7B sin razonamiento=results/eval/qwen3-1.7b_nothink.jsonl" \\
            "Qwen3-1.7B con razonamiento=results/eval/qwen3-1.7b_think.jsonl" \\
            "DeepSeek V4.1-Flash=results/eval/deepseek-v4.1-flash.jsonl" \\
      --k 2 4 8 --families --out-dir informe/
"""
from __future__ import annotations

import argparse
import math
import os
from collections import defaultdict

from .utils import latest_samples, pass_at_k, read_jsonl
from .verifier import verify


def load_run(path, tasks=None):
    """Última versión de cada muestra, sin errores del servidor; recalificada si se pasan las tareas."""
    rows = latest_samples(read_jsonl(path))
    if tasks is not None:
        kept = []
        for r in rows:
            t = tasks.get(r["task_id"])
            if t is None:
                continue
            v = verify(t["spec"], r.get("content", ""))
            r = dict(r, solved=v["solved"], score_adj=v["score_adj"], error=v["error"])
            kept.append(r)
        rows = kept
    return rows


def aggregate(rows, key, ks):
    by_task = defaultdict(list)
    meta = {}
    for r in rows:
        by_task[r["task_id"]].append(r)
        meta[r["task_id"]] = r
    groups = defaultdict(list)
    for tid, rs in by_task.items():
        groups[key(meta[tid])].append(rs)
    out = {}
    for g, tasks in groups.items():
        ns = [len(rs) for rs in tasks]
        d = {"N": len(tasks), "n": min(ns), "n_max": max(ns)}
        for k in ks:
            vals = [pass_at_k(len(rs), sum(r["solved"] for r in rs), k) for rs in tasks]
            vals = [v for v in vals if not math.isnan(v)]
            d[f"p{k}"] = sum(vals) / len(vals) if vals else float("nan")
        d["p1"] = sum(sum(r["solved"] for r in rs) / len(rs) for rs in tasks) / len(tasks)
        d["partial"] = sum(sum(r.get("score_adj", 0) for r in rs) / len(rs) for rs in tasks) / len(tasks)
        d["fmt"] = sum(sum(r.get("error") is not None for r in rs) / len(rs) for rs in tasks) / len(tasks)
        d["trunc"] = sum(sum(r.get("finish_reason") == "length" for r in rs) / len(rs) for rs in tasks) / len(tasks)
        cs = [(sum(r["solved"] for r in rs), len(rs)) for rs in tasks]
        d["never"] = sum(c == 0 for c, _ in cs) / len(cs)
        d["some"] = sum(0 < c < n for c, n in cs) / len(cs)
        d["always"] = sum(c == n for c, n in cs) / len(cs)
        out[g] = d
    return out


def pct(x):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}"


def tex_escape(s):
    return (s.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%")
             .replace("_", "\\_").replace("#", "\\#"))


def table(header, rows, caption, label):
    """Devuelve la misma tabla en Markdown y en LaTeX (booktabs), con una regla entre modelos."""
    md = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    cols = "l" * 2 + "r" * (len(header) - 2)
    tex = ["\\begin{table}[htbp]", "  \\centering\\small", f"  \\caption{{{caption}}}", f"  \\label{{{label}}}",
           f"  \\begin{{tabular}}{{@{{}}{cols}@{{}}}}", "    \\toprule",
           "    " + " & ".join(tex_escape(h) for h in header) + " \\\\", "    \\midrule"]
    prev = None
    for r in rows:
        if prev is not None and r[0] != prev:
            tex.append("    \\midrule")
        tex.append("    " + " & ".join(tex_escape(c) for c in r) + " \\\\")
        prev = r[0]
    tex += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    return "\n".join(md), "\n".join(tex)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs="+", required=True, help='"ETIQUETA=archivo.jsonl"')
    ap.add_argument("--tasks", nargs="*", help="si se indica, recalifica con el verificador vigente")
    ap.add_argument("--k", type=int, nargs="+", default=[2, 4, 8], help="valores de k a reportar")
    ap.add_argument("--families", action="store_true", help="agrega el desglose por familia")
    ap.add_argument("--out-dir", default="informe")
    a = ap.parse_args()

    tasks = None
    if a.tasks:
        tasks = {}
        for p in a.tasks:
            for t in read_jsonl(p):
                tasks[t["id"]] = t
    runs = []
    for spec in a.run:
        label, _, path = spec.partition("=")
        label, path = label.strip(), path.strip()
        if not os.path.exists(path):
            print(f"AVISO: no existe {path}; se omite «{label}»")
            continue
        rows = load_run(path, tasks)
        if not rows:
            print(f"AVISO: {path} no tiene respuestas válidas para estas tareas; se omite «{label}»")
            continue
        runs.append((label, rows))
    if not runs:
        raise SystemExit("Ninguna corrida disponible.")
    os.makedirs(a.out_dir, exist_ok=True)
    md_all, tex_all = [], []

    # 1) línea base
    hdr = ["Modelo", "Nivel", "N", "n", "pass@1"] + [f"pass@{k}" for k in a.k] + ["Parcial", "Ilegible", "Truncada"]
    body = []
    for label, rows in runs:
        agg = aggregate(rows, lambda r: r["level"], a.k)
        for lv in sorted(agg):
            d = agg[lv]
            n = str(d["n"]) if d["n"] == d["n_max"] else f'{d["n"]}–{d["n_max"]}'
            body.append([label, lv, str(d["N"]), n, pct(d["p1"])] + [pct(d[f"p{k}"]) for k in a.k]
                        + [pct(d["partial"]), pct(d["fmt"]), pct(d["trunc"])])
    md, tex = table(hdr, body,
                    "Línea base por modelo y nivel (porcentajes). N = tareas, n = muestras por tarea; "
                    "pass@k con el estimador insesgado; — = k mayor que n.", "tab:lineabase")
    md_all.append("## Línea base por modelo y nivel (%)\n\n" + md)
    tex_all.append(tex)

    # 2) calibración
    hdr = ["Modelo", "Nivel", "N", "Nunca", "A veces", "Siempre"]
    body = []
    for label, rows in runs:
        agg = aggregate(rows, lambda r: r["level"], a.k)
        for lv in sorted(agg):
            d = agg[lv]
            body.append([label, lv, str(d["N"]), pct(d["never"]), pct(d["some"]), pct(d["always"])])
    md, tex = table(hdr, body,
                    "Calibración: porcentaje de tareas que el modelo nunca, a veces o siempre resuelve. "
                    "Las tareas resueltas a veces son las que dan señal a RL.", "tab:calibracion")
    md_all.append("## Calibración (% de tareas)\n\n" + md)
    tex_all.append(tex)

    # 3) por familia
    if a.families:
        hdr = ["Modelo", "Nivel / familia", "N", "pass@1", "Truncada"]
        body = []
        for label, rows in runs:
            agg = aggregate(rows, lambda r: (r["level"], r["family"]), a.k)
            for key in sorted(agg):
                d = agg[key]
                body.append([label, f"{key[0]} {key[1]}", str(d["N"]), pct(d["p1"]), pct(d["trunc"])])
        md, tex = table(hdr, body, "pass@1 por familia de tareas (porcentajes).", "tab:familias")
        md_all.append("## Por familia (%)\n\n" + md)
        tex_all.append(tex)

    with open(os.path.join(a.out_dir, "tablas.md"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(md_all) + "\n")
    with open(os.path.join(a.out_dir, "tablas.tex"), "w", encoding="utf-8") as f:
        f.write("% Requiere \\usepackage{booktabs}\n\n" + "\n\n".join(tex_all) + "\n")
    print("\n\n".join(md_all))
    print(f"\n-> {a.out_dir}/tablas.md y {a.out_dir}/tablas.tex")


if __name__ == "__main__":
    main()
