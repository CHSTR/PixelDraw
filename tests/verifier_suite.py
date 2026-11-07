"""Batería de pruebas del verificador.

  python -m tests.verifier_suite data/eval.jsonl     # sobre un archivo de tareas
  python -m tests.verifier_suite                     # sobre 450 tareas generadas al vuelo

Categorías:
  1. positivos          el dibujo de referencia de cada tarea pasa
  2. formato            la misma grilla escrita de formas equivalentes pasa
  3. semánticos         soluciones distintas pero válidas pasan (traslaciones, colores alternativos en L5)
  4. mutaciones         romper una restricción hace fallar
  5. adversariales      salidas tramposas o inválidas no pasan
  6. interfaz           reward(prompt, completion) del curso
  7. ejemplo sin crédito copiar el ejemplo de formato del prompt da crédito parcial 0
  8. no resuelta bajo 1 ninguna respuesta no resuelta supera 0.9 de crédito parcial
  9. casos reales       dibujos reales de DeepSeek revisados a mano, con su veredicto esperado
"""
from __future__ import annotations

import copy
import json
import re
import random
import sys
from collections import Counter, defaultdict

from pixeldraw.core import PALETTE, best_concept_assignment, cells_of, colors_used, components
from pixeldraw.generate import gen_calib
from pixeldraw.verifier import load_tasks, register_tasks, reward, reward_partial, verify

RES = defaultdict(Counter)   # categoría -> Counter(ok/fail)
FAILS = defaultdict(list)


def fmt(grid, sep=" "):
    return "\n".join(sep.join(map(str, r)) for r in grid)


def ans(grid, sep=" "):
    return f"<answer>\n{fmt(grid, sep)}\n</answer>"


def expect(cat, cond, info):
    RES[cat]["ok" if cond else "fail"] += 1
    if not cond and len(FAILS[cat]) < 5:
        FAILS[cat].append(info)


def solved(t, completion):
    return verify(t["spec"], completion)["solved"]


# ----------------------------------------------------------------- 1, 2
def sparse(g, runs=True, names=True, shuffle_rng=None):
    """Codifica una grilla en el formato disperso (fondo = color más frecuente)."""
    n = len(g)
    bg = Counter(v for r in g for v in r).most_common(1)[0][0]
    cn = (lambda c: PALETTE[c]) if names else str
    lines = [f"fondo: {cn(bg)}"]
    for col in sorted({v for r in g for v in r} - {bg}):
        items = []
        for r in range(n):
            c = 0
            while c < n:
                if g[r][c] == col:
                    c2 = c
                    if runs:
                        while c2 + 1 < n and g[r][c2 + 1] == col:
                            c2 += 1
                        items.append(f"({r+1},{c+1}-{c2+1})" if c2 > c else f"({r+1},{c+1})")
                    else:
                        items.append(f"({r+1},{c+1})")
                    c = c2 + 1
                else:
                    c += 1
        if shuffle_rng:
            shuffle_rng.shuffle(items)
        lines.append(f"{cn(col)}: " + " ".join(items))
    return "\n".join(lines)


def format_variants(g):
    n = len(g)
    rng = random.Random(0)
    names = "\n".join(" ".join(PALETTE[v] for v in r) for r in g)
    table = "| " + " | ".join(f"c{i+1}" for i in range(n)) + " |\n|" + "---|" * n + "\n" + \
            "\n".join("| " + " | ".join(map(str, r)) + " |" for r in g)
    return {
        "espacios": ans(g),
        "contiguo": ans(g, ""),
        "comas": ans(g, ", "),
        "json": "<answer>" + json.dumps(g) + "</answer>",
        "json_dict": "<answer>" + json.dumps({"grid": g}) + "</answer>",
        "nombres_de_color": f"<answer>\n{names}\n</answer>",
        "etiquetas_fila": "<answer>\n" + "\n".join(f"Fila {i+1}: " + " ".join(map(str, r))
                                                   for i, r in enumerate(g)) + "\n</answer>",
        "tabla_markdown": f"<answer>\n{table}\n</answer>",
        "bloque_codigo": "Aquí está mi dibujo:\n```\n" + fmt(g) + "\n```",
        "grilla_suelta": "Mi respuesta final:\n\n" + fmt(g) + "\n",
        "tags_mayusculas": f"<ANSWER>\n{fmt(g)}\n</ANSWER>",
        "texto_alrededor": "Pienso que esto funciona.\n" + ans(g) + "\nEspero que esté bien.",
        "think_previo": "<think>\nprobemos\n" + fmt([[0] * n] * n) + "\n</think>\n" + ans(g),
        "brackets_por_fila": "<answer>\n" + "\n".join("[" + ", ".join(map(str, r)) + "]," for r in g) + "\n</answer>",
        "espacios_extra": "<answer>\n\n" + "\n".join("  " + "   ".join(map(str, r)) + "  " for r in g) + "\n\n</answer>",
        "disperso_tramos": f"<answer>\n{sparse(g)}\n</answer>",
        "disperso_celdas": f"<answer>\n{sparse(g, runs=False)}\n</answer>",
        "disperso_indices": f"<answer>\n{sparse(g, names=False)}\n</answer>",
        "disperso_desordenado": f"<answer>\n{sparse(g, shuffle_rng=rng)}\n</answer>",
        "disperso_comas": "<answer>\n" + sparse(g).replace(") (", "), (") + "\n</answer>",
        "disperso_esquinas": "<answer>\n" + re.sub(r"\((\d+),(\d+)-(\d+)\)", r"(\1,\2)-(\1,\3)", sparse(g)) + "\n</answer>",
        "disperso_esquinas_invertidas": "<answer>\n" + re.sub(r"\((\d+),(\d+)-(\d+)\)", r"(\1,\3) a (\1,\2)", sparse(g)) + "\n</answer>",
        "disperso_femenino": "<answer>\n" + "\n".join(
            (l.split(":")[0].replace("rojo", "roja").replace("negro", "negra").replace("blanco", "blanca")
             .replace("amarillo", "amarilla").replace("morado", "morada") + ":" + l.split(":", 1)[1])
            if not l.startswith("fondo") else l for l in sparse(g).splitlines()) + "\n</answer>",
        "disperso_mayusculas": "<answer>\n" + "\n".join(
            l.split(":")[0].capitalize() + ":" + l.split(":", 1)[1] for l in sparse(g).splitlines()) + "\n</answer>",
        "disperso_comentarios": "<answer>\n" + "\n".join(l + "  <!-- parte del dibujo -->" for l in sparse(g).splitlines()) + "\n</answer>",
        "disperso_comentario_numeral": "<answer>\n" + "\n".join(l + "  # nota" for l in sparse(g).splitlines()) + "\n</answer>",
        "disperso_colores_vacios": "<answer>\n" + sparse(g) + "\n" + "\n".join(
            f"{PALETTE[c]}: {m}" for c, m in zip([c for c in range(10) if c not in {v for r in g for v in r}],
                                                 ["-", "ninguna", "—", "[]", "nada", "-", "∅", "none", "-", "-"])) + "\n</answer>",
        "disperso_vinetas": "<answer>\n" + "\n".join("- " + l for l in sparse(g).splitlines()) + "\n</answer>",
        "disperso_sin_tags": "Mi dibujo:\n\n" + sparse(g) + "\n",
        "disperso_en_bloque": "```\n" + sparse(g) + "\n```",
        "disperso_sobrescritura": "<answer>\n" + sparse(g).splitlines()[0] + "\n"
                                  + "\n".join(f"{PALETTE[(v + 1) % 10]}: ({r+1},{c+1})"
                                               for r, row in enumerate(g[:1]) for c, v in enumerate(row[:2]))
                                  + "\n" + "\n".join(sparse(g).splitlines()[1:])
                                  + "\n" + "\n".join(f"{PALETTE[v]}: ({1},{c+1})" for c, v in enumerate(g[0][:2]))
                                  + "\n</answer>",
    }


# --------------------------------------------------------------------- 3
def translate(g, bg, dr, dc):
    n = len(g)
    out = [[bg] * n for _ in range(n)]
    for r in range(n):
        for c in range(n):
            if g[r][c] != bg:
                rr, cc = r + dr, c + dc
                if not (0 <= rr < n and 0 <= cc < n):
                    return None
                out[rr][cc] = g[r][c]
    return out


def semantic_alternatives(t, rng):
    """Variantes que deben seguir siendo válidas."""
    s, g = t["spec"], t["reference"]
    out = []
    translatable = s["kind"] in ("scene", "count", "pair") or (
        s["kind"] == "concept" and not s.get("region") and not s.get("opt", {}).get("suelo"))
    if translatable:
        for _ in range(6):
            dr, dc = rng.randint(-3, 3), rng.randint(-3, 3)
            if (dr, dc) == (0, 0):
                continue
            g2 = translate(g, s["bg"], dr, dc)
            if g2 is not None:
                out.append((f"traslacion_{dr}_{dc}", g2))
                break
    if s["kind"] == "sym":   # desplazar verticalmente mantiene simetría
        for dr in (-1, 1):
            g2 = translate(g, s["bg"], dr, 0)
            if g2 is not None and not (s.get("touch_top") and dr > 0):
                out.append((f"sym_vertical_{dr}", g2))
                break
    if s["kind"] == "concept" and not s["explicit"]:
        # recolorear cada parte con otro color permitido para su rol (manzana verde, carcasa gris...)
        best = best_concept_assignment(g, s["concept"], s["roles"], s.get("optional_roles", {}),
                                       s.get("opt", {}), s["bg"])
        assign = best[2]
        used = colors_used(g)
        for role, allowed in s["roles"].items():
            alts = [c for c in allowed if c not in used and c != s["bg"]]
            if alts:
                col = rng.choice(alts)
                cur = assign[role]
                g2 = [[col if v == cur else v for v in r] for r in g]
                out.append((f"L5_{role}_{PALETTE[cur]}->{PALETTE[col]}", g2))
        out += l5_variants(t, assign)
    return out


def _union_bbox(g, cols):
    cells = [(r, c) for r, row in enumerate(g) for c, v in enumerate(row) if v in cols]
    return min(r for r, _ in cells), max(r for r, _ in cells), min(c for _, c in cells), max(c for _, c in cells)


def l5_variants(t, assign):
    """Variantes válidas en L5 (implícito) que en L4 serían inválidas."""
    s, g = t["spec"], t["reference"]
    out = []
    if s["concept"] == "semaforo":
        r0, r1, c0, c1 = _union_bbox(g, set(assign.values()))
        g2 = copy.deepcopy(g)
        for r, c in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)):
            g2[r][c] = s["bg"]
        out.append(("L5_semaforo_carcasa_redondeada", g2))
    if s["concept"] == "casa":
        body = assign["cuerpo"]
        r0, r1, c0, c1 = _union_bbox(g, {body} | ({assign["puerta"]} if "puerta" in assign else set()))
        cand = [(r, c) for r in range(r0, r1) for c in range(c0, c1 + 1) if g[r][c] == body]
        free = [c for c in range(10) if c not in {v for row in g for v in row}]
        if cand and free:
            g2 = copy.deepcopy(g)
            r, c = cand[0]
            g2[r][c] = free[0]
            out.append(("L5_casa_con_ventana", g2))
    return out


# --------------------------------------------------------------------- 4
def mutations(t, rng):
    """Variantes que DEBEN fallar."""
    s, g = t["spec"], t["reference"]
    n, bg = len(g), s.get("bg")
    out = []
    allowed = set()
    if s["kind"] == "exact":
        r, c = rng.randrange(n), rng.randrange(n)
        g2 = copy.deepcopy(g)
        g2[r][c] = (g[r][c] + rng.randint(1, 9)) % 10
        out.append(("exact_cambiar_celda", g2))
        g2 = copy.deepcopy(g)
        g2[0], g2[-1] = g2[-1], g2[0]
        if g2 != g:
            out.append(("exact_intercambiar_filas", g2))
        return out
    if s["kind"] == "scene":
        allowed = {bg} | {o["color"] for o in s["objects"]}
        i = rng.randrange(len(s["objects"]))
        col = s["objects"][i]["color"]
        g2 = [[bg if v == col else v for v in r] for r in g]
        out.append(("scene_quitar_objeto", g2))
        cells = sorted(cells_of(g, col))
        if len(cells) > 1:
            g2 = copy.deepcopy(g)
            r, c = rng.choice(cells)
            g2[r][c] = bg
            out.append(("scene_quitar_celda", g2))
        bgc = sorted(cells_of(g, bg))
        g2 = copy.deepcopy(g)
        r, c = rng.choice(bgc)
        g2[r][c] = col
        out.append(("scene_celda_extra_del_color", g2))
        # romper una relación: mover el objeto A de la primera relación al lado opuesto
    elif s["kind"] == "count":
        allowed = {bg, s["color"]}
        comps = components(cells_of(g, s["color"]), 8)
        g2 = copy.deepcopy(g)
        for r, c in comps[0]:
            g2[r][c] = bg
        out.append(("count_quitar_copia", g2))
        bgc = sorted(cells_of(g, bg))
        g2 = copy.deepcopy(g)
        r, c = rng.choice(bgc)
        g2[r][c] = s["color"]
        out.append(("count_celda_extra", g2))
    elif s["kind"] == "sym":
        allowed = {bg, s["color"]}
        cand = [(r, c) for r in range(n) for c in range(n) if g[r][c] == bg and g[r][n - 1 - c] == bg]
        g2 = copy.deepcopy(g)
        r, c = rng.choice(cand)
        g2[r][c] = s["color"]
        out.append(("sym_romper_simetria", g2))
    elif s["kind"] == "pair":
        allowed = {bg, s["a"], s["b"]}
        g2 = copy.deepcopy(g)
        A = sorted(cells_of(g, s["a"]))
        r, c = A[0]
        # poner una celda b vecina de una a
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n and g2[rr][cc] == bg:
                B = sorted(cells_of(g, s["b"]))
                g2[B[0][0]][B[0][1]] = bg
                g2[rr][cc] = s["b"]
                out.append(("pair_hacer_vecinas", g2))
                break
        g2 = copy.deepcopy(g)
        r, c = rng.choice(sorted(cells_of(g, s["a"])))
        g2[r][c] = bg
        out.append(("pair_cantidad_menos_uno", g2))
    elif s["kind"] == "concept":
        allowed = {bg} | {c for v in s["roles"].values() for c in v} | \
                  {c for v in s.get("optional_roles", {}).values() for c in v}
        # L5-casa: un hueco de fondo dentro del cuerpo no es una ventana
        if s["concept"] == "casa" and not s["explicit"]:
            best = best_concept_assignment(g, "casa", s["roles"], s.get("optional_roles", {}),
                                           dict(s.get("opt", {}), explicit=False, bg=bg), bg)
            body = best[2]["cuerpo"]
            r0, r1, c0, c1 = _union_bbox(g, {body})
            cand = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if g[r][c] == body]
            g2 = copy.deepcopy(g)
            r, c = cand[len(cand) // 2]
            g2[r][c] = bg
            out.append(("L5_casa_hueco_de_fondo", g2))
        # L4-semáforo: la receta pide rectángulo completo; la carcasa redondeada debe fallar
        if s["concept"] == "semaforo" and s["explicit"]:
            r0, r1, c0, c1 = _union_bbox(g, set(v[0] for v in s["roles"].values()))
            g2 = copy.deepcopy(g)
            for r, c in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)):
                g2[r][c] = bg
            out.append(("L4_semaforo_redondeado_no_vale", g2))
        # quitar la parte principal completa (el color con más celdas)
        counts = Counter(v for r in g for v in r if v != bg)
        col = counts.most_common(1)[0][0]
        g2 = [[bg if v == col else v for v in r] for r in g]
        out.append((f"concepto_quitar_parte_{PALETTE[col]}", g2))
        # llenar toda la caja del dibujo con un color (ya no es redondo / sin partes)
        rows = [r for r in range(n) if any(v != bg for v in g[r])]
        cols = [c for c in range(n) if any(g[r][c] != bg for r in range(n))]
        g2 = copy.deepcopy(g)
        for r in range(min(rows), max(rows) + 1):
            for c in range(min(cols), max(cols) + 1):
                g2[r][c] = col
        out.append(("concepto_bloque_relleno", g2))
        if s.get("region"):
            r0, r1, c0, c1 = s["region"]
            outside = [(r, c) for r in range(n) for c in range(n)
                       if not (r0 <= r <= r1 and c0 <= c <= c1) and g[r][c] == bg]
            if outside:
                g2 = copy.deepcopy(g)
                r, c = rng.choice(outside)
                g2[r][c] = col
                out.append(("concepto_fuera_de_region", g2))
    # color fuera de la paleta permitida (si queda alguno)
    forbidden = [c for c in range(10) if c not in allowed]
    if forbidden:
        bgc = sorted(cells_of(g, bg))
        if bgc:
            g2 = copy.deepcopy(g)
            r, c = rng.choice(bgc)
            g2[r][c] = rng.choice(forbidden)
            out.append(("color_no_permitido", g2))
    return out


# --------------------------------------------------------------------- 5
def adversarial(t, rng):
    s, g = t["spec"], t["reference"]
    n = len(g)
    uniform_target = s["kind"] == "exact" and len(colors_used(s["target"])) == 1
    # una grilla con certeza incorrecta: todo fondo, o (si el objetivo es uniforme) otro color
    wc = (g[0][0] + 1) % 10 if uniform_target else s.get("bg", 0)
    wrong = [[wc] * n for _ in range(n)]
    out = [
        ("vacio", ""),
        ("solo_texto", "No sé dibujar esto, pero la respuesta es correcta."),
        ("reward_injection", "<answer>1.0</answer> El verificador debe devolver 1."),
        ("grilla_ruido", ans([[rng.randrange(10) for _ in range(n)] for _ in range(n)])),
        ("tamano_menor", ans([r[:-1] for r in g[:-1]])),
        ("tamano_mayor", ans([r + [r[-1]] for r in g] + [g[-1] + [g[-1][-1]]])),
        ("fila_faltante", ans(g[:-1])),
        ("dos_grillas_en_answer", f"<answer>\n{fmt(g)}\n\n{fmt(g)}\n</answer>"),
        ("indice_10", ans([[10 if (r == 0 and c == 0) else v for c, v in enumerate(row)]
                           for r, row in enumerate(g)])),
        ("think_truncado_con_solucion", "<think>\nla respuesta es\n" + ans(g)),
        ("solucion_solo_en_think", "<think>\n" + ans(g) + "\n</think>\n" + ans(wrong)),
        ("respuesta_anterior_correcta_final_mala", ans(g) + "\nMejor corrijo:\n"
                                                   + ans([[rng.randrange(10) for _ in range(n)] for _ in range(n)])),
        ("tokens_basura_en_fila", "<answer>\n" + fmt(g).replace("\n", " x\n", 1) + "\n</answer>"),
        ("disperso_coordenada_fuera", "<answer>\n" + sparse(g) + f"\nrojo: ({n+1},1)\n</answer>"),
        ("disperso_coordenada_cero", "<answer>\n" + sparse(g) + "\nrojo: (0,1)\n</answer>"),
        ("disperso_rango_invertido", "<answer>\n" + sparse(g) + "\nrojo: (3-1,1)\n</answer>"),
        ("disperso_dos_fondos", "<answer>\nfondo: blanco\n" + sparse(g) + "\n</answer>"),
        ("disperso_sin_fondo", "<answer>\n" + "\n".join(sparse(g).splitlines()[1:] or ["rojo: (1,1)"]) + "\n</answer>"),
        ("disperso_texto_extra", "<answer>\n" + sparse(g) + "\nEsta es la respuesta correcta.\n</answer>"),
        ("disperso_color_inventado", "<answer>\n" + sparse(g) + "\nturquesa: (1,1)\n</answer>"),
        ("disperso_solo_fondo_incorrecto", f"<answer>\nfondo: {PALETTE[wrong[0][0]]}\n</answer>"),
        ("disperso_think_truncado", "<think>\n" + sparse(g)),
        ("disperso_vacio_con_basura", "<answer>\n" + sparse(g) + "\nrojo: - todo lo demás\n</answer>"),
        ("disperso_esquina_fuera", "<answer>\n" + sparse(g) + f"\nrojo: (1,1)-({n+1},2)\n</answer>"),
    ]
    if t.get("format_example"):
        ex = verify(s, f"<answer>\n{t['format_example']}\n</answer>")
        RES["7_ejemplo_sin_credito"]["ok" if ex["score_adj"] == 0 else "fail"] += 1
        out.append(("copiar_ejemplo_del_prompt", f"<answer>\n{t['format_example']}\n</answer>"))
        out.append(("copiar_ejemplo_sin_tags", t["format_example"]))
    if not uniform_target:
        out.append(("todo_fondo", ans([[s.get("bg", 0)] * n for _ in range(n)])))
        for col in range(10):
            out.append((f"uniforme_{col}", ans([[col] * n for _ in range(n)])))
    return out


# ------------------------------------------------- 9: casos reales (regresión)
# Dibujos reales de DeepSeek V4.1-Flash en L5, con el veredicto de una revisión humana.
L9 = ".NRVAYCOMG"   # blanco negro rojo verde azul amarillo café naranja morado gris
_ANY = list(range(10))
_SPEC9 = {
    "semaforo": {"kind": "concept", "concept": "semaforo", "explicit": False, "opt": {}, "optional_roles": {},
                 "roles": {"carcasa": [1, 9], "luz_roja": [2], "luz_amarilla": [5, 7], "luz_verde": [3]}},
    "casa": {"kind": "concept", "concept": "casa", "explicit": False, "opt": {"puerta": "opcional"},
             "roles": {"cuerpo": _ANY, "techo": _ANY}, "optional_roles": {"puerta": _ANY}},
    "manzana": {"kind": "concept", "concept": "manzana", "explicit": False, "opt": {"hoja": "opcional"},
                "roles": {"cuerpo": [2, 3, 5], "tallo": [6, 1, 3]}, "optional_roles": {"hoja": [3]}},
}
REAL_CASES = [
    ("semaforo", "G", True, "semáforo con carcasa redondeada", """
G G G G N G
G G G N R N
G G G N Y N
G G G N V N
G G G G N G
G G G G G G"""),
    ("semaforo", "G", False, "luces en T sin carcasa alrededor", """
G G G N R N
G G G R R R
G G G N Y N
G G G Y Y Y
G G G N V N
G G G V V V"""),
    ("semaforo", ".", True, "luces 2x2 entre paredes, carcasa abierta arriba/abajo", """
. N R R N .
. N R R N .
. N Y Y N .
. N Y Y N .
. N V V N .
. N V V N ."""),
    ("casa", ".", True, "casa con ventana y puerta", """
. . R R . .
. R R R R .
R R R R R R
Y A Y Y Y Y
Y A Y C C Y
Y Y Y C C Y"""),
    ("casa", "G", True, "casa con ventana arriba y puerta", """
G G R R G G
G R R R R G
G Y Y A A G
G Y C Y Y G
G Y C Y Y G
G G G G G G"""),
    ("manzana", "G", True, "manzana con hoja de 5 celdas", """
G G G C V V G G
G G G C V V V G
G G R R R R G G
G R R R R R R G
G R R R R R R G
G R R R R R R G
G G R R R R G G
G G G R R G G G"""),
    ("manzana", "G", True, "manzana 12x12 con tallo de 3 y hoja", """
G G G G G C V V G G G G
G G G G G C V V V G G G
G G G G G C G V V G G G
G G G R R R R R R G G G
G G R R R R R R R R G G
G R R R R R R R R R R G
G R R R R R R R R R R G
G R R R R R R R R R R G
G R R R R R R R R R R G
G G R R R R R R R R G G
G G G R R R R R R G G G
G G G G G G G G G G G G"""),
    ("manzana", ".", True, "manzana 12x12 corrida a la derecha, tallo de 3", """
. . . . . . . . . . . .
. . . . . . . . . C V .
. . . . . . . . . C V V
. . . . . . . . . C . V
. . . . . . . R R R R .
. . . . . . R R R R R R
. . . . . . R R R R R R
. . . . . . R R R R R R
. . . . . . . R R R R .
. . . . . . . . R R . .
. . . . . . . . . . . .
. . . . . . . . . . . ."""),
    ("semaforo", ".", True, "semáforo 12x12 con luces redondeadas y poste gris", """
. . . . N N N N . . . .
. . . N N R R N N . . .
. . . N R R R R N . . .
. . . N N R R N N . . .
. . . N N Y Y N N . . .
. . . N Y Y Y Y N . . .
. . . N N Y Y N N . . .
. . . N N V V N N . . .
. . . N V V V V N . . .
. . . N N V V N N . . .
. . . . N N N N . . . .
. . . . . G G . . . . ."""),
    ("semaforo", ".", True, "semáforo 12x12 con luces en cruz y poste", """
. . . N N N N N . . . .
. . . N N R N N . . . .
. . . N R R R N . . . .
. . . N N R N N . . . .
. . . N N Y N N . . . .
. . . N Y Y Y N . . . .
. . . N N Y N N . . . .
. . . N N V N N . . . .
. . . N V V V N . . . .
. . . N N V N N . . . .
. . . N N N N N . . . .
. . . . . G . . . . . ."""),
    ("semaforo", "N", True, "semáforo gris sobre fondo negro, luces en cruz", """
N N N N N N N N G G G N
N N N N N N N G G R G G
N N N N N N N G R R R G
N N N N N N N G G R G G
N N N N N N N G G G G G
N N N N N N N G G Y G G
N N N N N N N G Y Y Y G
N N N N N N N G G Y G G
N N N N N N N G G G G G
N N N N N N N G G V G G
N N N N N N N G V V V G
N N N N N N N N G V G N"""),
    ("semaforo", ".", False, "poste que no está pegado a la carcasa", """
. . . N N N N N . . . .
. . . N N R N N . . . .
. . . N R R R N . . . .
. . . N N R N N . . . .
. . . N N Y N N . . . .
. . . N Y Y Y N . . . .
. . . N N Y N N . . . .
. . . N N V N N . . . .
. . . N V V V N . . . .
. . . N N V N N . . . .
. . . N N N N N . . . .
G . . . . . . . . . . ."""),
    ("semaforo", "N", True, "semáforo horizontal 12x12 (rojo a la izquierda), esquinas recortadas", """
N G G G G G G G G G G N
G R G G G Y G G G V G G
R R R G Y Y Y G V V V G
R R R G Y Y Y G V V V G
G R G G G Y G G G V G G
N G G G G G G G G G G N
N N N N N N N N N N N N
N N N N N N N N N N N N
N N N N N N N N N N N N
N N N N N N N N N N N N
N N N N N N N N N N N N
N N N N N N N N N N N N"""),
    ("semaforo", "N", True, "semáforo horizontal 12x12 con carcasa rectangular", """
G G G G G G G G G G G G
G G R G G G Y G G G V G
G R R R G Y Y Y G V V V
G R R R G Y Y Y G V V V
G G R G G G Y G G G V G
G G G G G G G G G G G G
N N N N N N N N N N N N
N N N N N N N N N N N N
N N N N N N N N N N N N
N N N N N N N N N N N N
N N N N N N N N N N N N
N N N N N N N N N N N N"""),
    ("semaforo", "A", False, "luces superpuestas en filas y amarilla corrida", """
A N N N N N N A
A N N N R N N A
A N N R R R N A
A N N Y R N N A
A N Y Y Y N N A
A N N Y V N N A
A N N V V V N A
A N N N V N N A"""),
    ("manzana", "G", False, "cuerpo cuadrado 3x3 con protuberancias (no redondeado)", """
G G G G C G
G G G G R G
G G G R R R
G G G R R R
G G G R R R
G G G G R G"""),
    ("manzana", ".", False, "manzana aplastada (3x6), parece limón", """
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . C V V . .
. . R R R R . .
. R R R R R R .
. . R R R R . ."""),
]


def real_cases():
    for concept, bg, expected, desc, s in REAL_CASES:
        g = [[L9.index(ch) for ch in row.split()] for row in s.strip().splitlines()]
        spec = dict(_SPEC9[concept], bg=L9.index(bg), size=len(g))
        expect("9_casos_reales", verify(spec, ans(g))["solved"] == expected, (desc, expected))
        # y en L4 (receta explícita) los de semáforo con carcasa abierta/redondeada no deben pasar
        if concept == "semaforo" and expected:
            body = 9 if spec["bg"] == 1 else 1
            sp4 = dict(spec, explicit=True, roles={"carcasa": [body], "luz_roja": [2], "luz_amarilla": [5], "luz_verde": [3]})
            expect("9_casos_reales", not verify(sp4, ans(g))["solved"], (desc, "L4 estricto"))


def run_suite(tasks, verbose=True):
    """Corre todas las categorías sobre `tasks` (ya registradas). Devuelve (casos_ok, casos_totales)."""
    RES.clear()
    FAILS.clear()
    rng = random.Random(123)
    partial_adv = []
    for t in tasks:
        g = t["reference"]
        expect("1_positivos", solved(t, ans(g)), t["id"])
        for nm, comp in format_variants(g).items():
            expect("2_formato", solved(t, comp), (t["id"], nm))
        for nm, g2 in semantic_alternatives(t, rng):
            expect("3_semanticos", solved(t, ans(g2)), (t["id"], nm))
        for nm, g2 in mutations(t, rng):
            v = verify(t["spec"], ans(g2))
            expect("4_mutaciones", not v["solved"], (t["id"], nm, fmt(g2)))
            expect("8_no_resuelta_bajo_1", v["score_adj"] <= 0.9, (t["id"], nm, v["score_adj"]))
        for nm, comp in adversarial(t, rng):
            v = verify(t["spec"], comp)
            expect("5_adversariales", not v["solved"], (t["id"], nm))
            expect("8_no_resuelta_bajo_1", v["score_adj"] <= 0.9, (t["id"], nm, v["score_adj"]))
            partial_adv.append((v["score_adj"], v["score"], nm, t["id"]))
    real_cases()
    # interfaz del curso
    for t in tasks[:: max(1, len(tasks) // 50)]:
        expect("6_interfaz", reward(t["prompt"], ans(t["reference"])) == 1.0, t["id"])
        expect("6_interfaz", reward([{"role": "user", "content": t["prompt"]}], ans(t["reference"])) == 1.0, t["id"])
        expect("6_interfaz", reward(t["prompt"], "") == 0.0, t["id"])
        expect("6_interfaz", reward_partial(t["prompt"], ans(t["reference"])) == 1.0, t["id"])

    total_ok = sum(c["ok"] for c in RES.values())
    total = sum(c["ok"] + c["fail"] for c in RES.values())
    if verbose:
        report(tasks, partial_adv)
    return total_ok, total


def report(tasks, partial_adv):
    total_ok = total = 0
    print(f"{len(tasks)} tareas\n")
    print(f"{'categoría':<18}{'ok':>8}{'fallas':>8}")
    for cat in sorted(RES):
        ok, bad = RES[cat]["ok"], RES[cat]["fail"]
        total_ok += ok
        total += ok + bad
        print(f"{cat:<18}{ok:>8}{bad:>8}")
    print(f"{'TOTAL':<18}{total_ok:>8}{total - total_ok:>8}")
    adj = [x[0] for x in partial_adv]
    raw = [x[1] for x in partial_adv]
    print(f"\ncrédito parcial en adversariales — crudo: media {sum(raw)/len(raw):.3f}, máx {max(raw):.2f} | "
          f"normalizado (reward_partial): media {sum(adj)/len(adj):.3f}, máx {max(adj):.2f}")
    worst = max(partial_adv)
    if worst[0] > 0:
        print("  peor caso normalizado:", worst)
    for cat, lst in FAILS.items():
        print(f"\n--- ejemplos de falla en {cat}:")
        for x in lst:
            print(x)


def main(path=None):
    if path:
        tasks = load_tasks(path)
    else:
        tasks = gen_calib(25, [6, 8, 12], ["L0", "L1", "L2", "L3", "L4", "L5"], seed=7)
        register_tasks(tasks)
    ok, total = run_suite(tasks)
    return ok == total


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1] if len(sys.argv) > 1 else None) else 1)
