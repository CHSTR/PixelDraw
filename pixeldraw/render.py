"""Galería HTML de lo que dibujan los modelos.

Lee las tareas y uno o más archivos de resultados (de pixeldraw.probe o pixeldraw.probe_opencode) y genera:
  - una página HTML autocontenida con una tabla de pass@1 por modelo y nivel y, por cada tarea, el
    enunciado, el dibujo de referencia y las respuestas de cada modelo dibujadas, con su veredicto,
    las restricciones que fallaron y el texto de la respuesta;
  - si se indica --png-dir, una figura PNG por tarea (requiere matplotlib).

Ejemplos:
  # comparar los modelos de la línea base
  python -m pixeldraw.render --tasks data/eval.jsonl --results "results/eval/*.jsonl" --out galeria.html
  # solo L4-L5 y solo tareas donde un modelo acierta y otro falla, con PNG para el informe
  python -m pixeldraw.render --tasks data/train.jsonl --results "results/train/*.jsonl" \\
      --levels L4 L5 --pick discrepan --png-dir figuras/ --out galeria_L4L5.html
  # revisar las trazas SFT
  python -m pixeldraw.render --tasks data/train.jsonl --sft derived/sft.jsonl --out galeria_sft.html
"""
from __future__ import annotations

import argparse
import glob
import html
import os
import random
from collections import defaultdict

from .core import PALETTE
from .utils import latest_samples, read_jsonl
from .verifier import ParseError, parse_grid

HEX = {
    "blanco": "#FFFFFF", "negro": "#1F1F1F", "rojo": "#D62828", "verde": "#2A9D3F", "azul": "#1F5FD1",
    "amarillo": "#F2C200", "café": "#8B5A2B", "naranja": "#F28C28", "morado": "#7B3FA0", "gris": "#8C8C8C",
}
COLORS = [HEX[p] for p in PALETTE]


def task_text(t):
    """El enunciado de la tarea, sin el encabezado de la paleta ni las instrucciones de formato."""
    p = t["prompt"]
    if "Tarea: " in p:
        p = p.split("Tarea: ", 1)[1]
    for cut in ("\n\nFormato de respuesta", "\n\nResponde"):
        p = p.split(cut, 1)[0]
    return p


def grid_html(g, px):
    n = len(g)
    cells = "".join(f'<i style="background:{COLORS[v]}"></i>' for row in g for v in row)
    return (f'<div class="grid" style="grid-template-columns:repeat({n},{px}px);'
            f'grid-auto-rows:{px}px">{cells}</div>')


def load_results(paths):
    """Filas válidas de varios archivos: la última versión de cada muestra, sin errores del servidor."""
    rows = []
    for path in paths:
        for r in latest_samples(read_jsonl(path)):
            rows.append(dict(r, model=r.get("model") or os.path.basename(path)))
    return rows


def pass_table(rows):
    agg = defaultdict(lambda: [0, 0])
    for r in rows:
        a = agg[(r["model"], r["level"])]
        a[0] += bool(r["solved"])
        a[1] += 1
    models = sorted({m for m, _ in agg})
    levels = sorted({lv for _, lv in agg})
    head = "".join(f"<th>{lv}</th>" for lv in levels)
    body = ""
    for m in models:
        tds = ""
        for lv in levels:
            s, n = agg.get((m, lv), (0, 0))
            if n:
                p = s / n
                tds += (f'<td style="background:rgba(42,157,63,{0.12 + 0.6 * p:.2f})">'
                        f'{100 * p:.0f}%<small> ({n})</small></td>')
            else:
                tds += "<td>—</td>"
        body += f"<tr><th>{html.escape(m)}</th>{tds}</tr>"
    return f'<table class="pass"><tr><th>pass@1</th>{head}</tr>{body}</table>'


def pick_tasks(tasks, by_task, a, rng):
    ids = [tid for tid in by_task if tid in tasks]
    if a.levels:
        ids = [i for i in ids if tasks[i]["level"] in a.levels]
    if a.families:
        ids = [i for i in ids if tasks[i]["family"] in a.families]
    if a.task_ids:
        ids = [i for i in ids if i in set(a.task_ids)]

    def has(tid, solved):
        return any(r["solved"] == solved for r in by_task[tid])

    if a.pick == "fallidas":
        ids = [i for i in ids if has(i, False)]
    elif a.pick == "resueltas":
        ids = [i for i in ids if has(i, True)]
    elif a.pick == "discrepan":
        # algún modelo acierta y otro (o el mismo en otra muestra) falla
        ids = [i for i in ids if has(i, True) and has(i, False)]
    groups = defaultdict(list)
    for i in ids:
        groups[(tasks[i]["level"], tasks[i]["family"], tasks[i]["size"])].append(i)
    chosen = []
    for key in sorted(groups):
        g = groups[key]
        rng.shuffle(g)
        chosen += g[:a.per_group]
    return chosen[:a.max_tasks]


def card(r, t, px):
    n = t["size"]
    try:
        g = parse_grid(r["content"], n)
        pic = grid_html(g, px)
    except ParseError as e:
        g = None
        pic = f'<div class="noparse" style="width:{n * px}px;height:{n * px}px">no se pudo leer<br><code>{html.escape(str(e))}</code></div>'
    ok = r["solved"]
    badge = ('<span class="ok">✓ resuelta</span>' if ok
             else f'<span class="bad">✗ parcial {r.get("score_adj", 0):.2f}</span>')
    fails = "" if ok else "<div class='fails'>" + html.escape(", ".join(r.get("failed_checks") or [r.get("error") or ""])) + "</div>"
    raw = html.escape((r.get("content") or "")[:2500])
    reas = r.get("reasoning") or ""
    reas_html = (f"<details><summary>razonamiento ({len(reas)} car.)</summary><pre>{html.escape(reas[:4000])}</pre></details>"
                 if reas else "")
    toks = f" · {r['completion_tokens']} tok" if r.get("completion_tokens") else ""
    return (f'<div class="card {"c-ok" if ok else "c-bad"}"><div class="who">{html.escape(r["model"])}'
            f' <small>#{r["sample"]}{toks}</small></div>{pic}{badge}{fails}'
            f'<details><summary>respuesta</summary><pre>{raw}</pre></details>{reas_html}</div>'), g


def save_png(t, ref, entries, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(COLORS)
    panels = [("referencia", ref, None)] + entries
    fig, axes = plt.subplots(1, len(panels), figsize=(2.1 * len(panels), 2.5))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, g, ok) in zip(axes, panels):
        ax.set_xticks([])
        ax.set_yticks([])
        if g is None:
            ax.text(0.5, 0.5, "ilegible", ha="center", va="center", fontsize=9)
            ax.set_facecolor("#f3f3f3")
        else:
            n = len(g)
            ax.imshow(g, cmap=cmap, vmin=0, vmax=9, interpolation="nearest")
            ax.set_xticks([x - 0.5 for x in range(1, n)], minor=True)
            ax.set_yticks([y - 0.5 for y in range(1, n)], minor=True)
            ax.grid(which="minor", color="#cccccc", linewidth=0.4)
            ax.tick_params(which="minor", length=0)
        mark = "" if ok is None else (" ✓" if ok else " ✗")
        ax.set_title(title + mark, fontsize=8, color="black" if ok in (None, True) else "#b00020")
    fig.suptitle(f'{t["level"]} · {t["family"]} · {t["size"]}×{t["size"]}', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


CSS = """
:root{--fg:#1d1d1f;--bg:#fafafa;--card:#fff;--muted:#666;--line:#ddd}
@media (prefers-color-scheme: dark){:root{--fg:#eee;--bg:#161616;--card:#222;--muted:#aaa;--line:#444}}
body{font:14px/1.45 system-ui,sans-serif;color:var(--fg);background:var(--bg);margin:0;padding:16px}
h1{font-size:20px;margin:0 0 8px} h2{font-size:15px;margin:0 0 6px}
.task{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin:14px 0}
.prompt{white-space:pre-wrap;color:var(--muted);font-size:13px;margin-bottom:10px;max-width:900px}
.row{display:flex;flex-wrap:wrap;gap:12px;align-items:flex-start}
.card{border:1px solid var(--line);border-radius:8px;padding:8px;max-width:340px}
.c-ok{border-color:#2A9D3F}.c-bad{border-color:#d9534f}
.who{font-weight:600;font-size:12px;margin-bottom:6px;word-break:break-all}
.grid{display:grid;gap:1px;background:#bbb;border:1px solid #bbb;width:max-content}
.grid i{display:block}
.ok{color:#2A9D3F;font-weight:600}.bad{color:#d9534f;font-weight:600}
.fails{font-size:11px;color:var(--muted);margin-top:2px}
.noparse{display:flex;align-items:center;justify-content:center;text-align:center;font-size:11px;
  background:repeating-linear-gradient(45deg,#eee,#eee 6px,#f8f8f8 6px,#f8f8f8 12px);color:#555}
pre{white-space:pre-wrap;font-size:11px;max-height:260px;overflow:auto;background:rgba(127,127,127,.1);padding:6px}
table.pass{border-collapse:collapse;margin:8px 0 16px}
table.pass th,table.pass td{border:1px solid var(--line);padding:4px 8px;text-align:center}
small{color:var(--muted)} .legend span{display:inline-block;margin-right:10px;font-size:12px}
.legend b{display:inline-block;width:12px;height:12px;border:1px solid #999;vertical-align:-2px;margin-right:4px}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--results", nargs="*", default=[], help="archivos de resultados (se aceptan comodines)")
    ap.add_argument("--sft", help="derived/sft.jsonl: muestra las trazas SFT como un 'modelo' más")
    ap.add_argument("--levels", nargs="*")
    ap.add_argument("--families", nargs="*")
    ap.add_argument("--task-ids", nargs="*")
    ap.add_argument("--pick", choices=["todas", "fallidas", "resueltas", "discrepan"], default="todas")
    ap.add_argument("--per-group", type=int, default=1, help="tareas por (nivel, familia, tamaño)")
    ap.add_argument("--max-tasks", type=int, default=60)
    ap.add_argument("--samples", type=int, default=2, help="muestras por modelo y tarea")
    ap.add_argument("--px", type=int, default=16, help="tamaño de celda en el HTML")
    ap.add_argument("--png-dir", help="si se indica, guarda una figura PNG por tarea (requiere matplotlib)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="galeria.html")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    tasks = {t["id"]: t for path in a.tasks for t in read_jsonl(path)}
    paths = sorted({f for pat in a.results for f in (glob.glob(pat) or [pat])})
    rows = load_results(paths)
    if a.sft:
        for x in read_jsonl(a.sft):
            rows.append({"task_id": x["task_id"], "sample": x["id"], "model": f"SFT ({x['source_model']})",
                         "level": x["level"], "family": x["family"], "size": x["size"], "solved": True,
                         "score_adj": 1.0, "failed_checks": [], "content": x["messages"][-1]["content"]})
    rows = [r for r in rows if r["task_id"] in tasks]
    if not rows:
        raise SystemExit("No hay respuestas que coincidan con las tareas indicadas (¿--tasks correcto?).")
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append(r)

    chosen = pick_tasks(tasks, by_task, a, rng)
    if a.png_dir:
        os.makedirs(a.png_dir, exist_ok=True)
    legend = "".join(f'<span><b style="background:{COLORS[i]}"></b>{i} {p}</span>' for i, p in enumerate(PALETTE))
    parts = [f"<!doctype html><html lang='es'><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width,initial-scale=1'><title>PixelDraw: galería</title>"
             f"<style>{CSS}</style></head><body><h1>PixelDraw — lo que dibujan los modelos</h1>"
             f"<div class='legend'>{legend}</div>"
             f"<p><small>Archivos: {html.escape(', '.join(os.path.basename(p) for p in paths) or '—')}"
             f"{' · ' + html.escape(a.sft) if a.sft else ''} · tareas mostradas: {len(chosen)} · "
             f"selección: {a.pick}</small></p>{pass_table(rows)}"]
    for tid in chosen:
        t = tasks[tid]
        rs = by_task[tid]
        per_model = defaultdict(list)
        for r in sorted(rs, key=lambda r: (r["model"], str(r["sample"]))):
            per_model[r["model"]].append(r)
        cards, png_entries = [], []
        for m, lst in per_model.items():
            for r in lst[:a.samples]:
                c, g = card(r, t, a.px)
                cards.append(c)
                png_entries.append((f'{m.split("/")[-1][:18]} #{r["sample"]}', g, r["solved"]))
        ref = (f'<div class="card"><div class="who">referencia del generador</div>'
               f'{grid_html(t["reference"], a.px)}<small>una solución válida, no la única</small></div>')
        parts.append(f'<div class="task"><h2>{t["level"]} · {t["family"]} · {t["size"]}×{t["size"]} '
                     f'<small>{tid}</small></h2><div class="prompt">{html.escape(task_text(t))}</div>'
                     f'<div class="row">{ref}{"".join(cards)}</div></div>')
        if a.png_dir:
            try:
                save_png(t, t["reference"], png_entries, os.path.join(a.png_dir, f"{tid}.png"))
            except ImportError:
                print("matplotlib no está instalado: se omiten los PNG (pip install matplotlib)")
                a.png_dir = None
    parts.append("</body></html>")
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    print(f"{len(chosen)} tareas -> {a.out}" + (f" y PNG en {a.png_dir}/" if a.png_dir else ""))


if __name__ == "__main__":
    main()
