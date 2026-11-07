"""Generador de tareas de PixelDraw.

Cada tarea es un diccionario con:
  id, core_id   identificador y huella del enunciado sin el ejemplo de formato (para deduplicar)
  level, size, family, split
  prompt        el texto que recibe el modelo
  spec          las restricciones que revisa el verificador
  reference     un dibujo construido por el generador que cumple la especificación; prueba que la
                tarea tiene solución, pero no se usa para calificar las tareas abiertas
  format_example el ejemplo de formato incluido en el prompt (copiarlo nunca da recompensa)

Uso:
  python -m pixeldraw.generate --mode full --out-dir data/          # splits train/eval/test/test_ood
  python -m pixeldraw.generate --mode calib --per-cell 30 --out calib.jsonl

El generador es determinista dada la semilla. Los archivos de data/ son la versión canónica: además
de generarse con este script, se les quitaron las tareas usadas para ajustar el verificador
(ver data/limpieza.json).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random

from .core import C, PALETTE, REL_TEXT, describe_shape, rel_holds, shape_cells, shape_dims
from .utils import write_jsonl

COLS = list(range(10))


def empty(n, bg):
    return [[bg] * n for _ in range(n)]


def paint(g, cells, color, off=(0, 0)):
    for r, c in cells:
        g[r + off[0]][c + off[1]] = color


def shifted(cells, off):
    return {(r + off[0], c + off[1]) for r, c in cells}


def name(c):
    return PALETTE[c]


FEM = {"blanco": "blanca", "negro": "negra", "rojo": "roja", "amarillo": "amarilla", "morado": "morada"}


def fname(c):
    """Color concordado con un sustantivo femenino ('la figura roja')."""
    return FEM.get(PALETTE[c], PALETTE[c])


def header(n):
    pal = ", ".join(f"{i} {p}" for i, p in enumerate(PALETTE))
    return (f"Vas a dibujar en una grilla de {n}×{n} usando esta paleta (índice: color): {pal}.\n"
            f"Las filas se numeran de 1 a {n} de arriba hacia abajo y las columnas de 1 a {n} "
            f"de izquierda a derecha.\n\n")


def sparse_example(n, rng):
    """Ejemplo concreto y aleatorio del formato disperso (otro dibujo, no la tarea).
    Devuelve (texto_de_respuesta, frase_explicativa)."""
    bg, a, b = rng.sample(range(10), 3)
    cells = sorted({(rng.randint(1, n), rng.randint(1, n)) for _ in range(rng.randint(1, 3))})
    kind = rng.choice(["fila", "columna", "bloque"])
    if kind == "fila":
        r, c1 = rng.randint(1, n), rng.randint(1, n - 2)
        c2 = rng.randint(c1 + 1, n)
        item, expl = f"({r},{c1}-{c2})", f"la fila {r} desde la columna {c1} hasta la {c2}"
    elif kind == "columna":
        c, r1 = rng.randint(1, n), rng.randint(1, n - 2)
        r2 = rng.randint(r1 + 1, n)
        item, expl = f"({r1}-{r2},{c})", f"la columna {c} desde la fila {r1} hasta la {r2}"
    else:
        r1, c1 = rng.randint(1, n - 1), rng.randint(1, n - 1)
        r2, c2 = rng.randint(r1 + 1, min(n, r1 + 2)), rng.randint(c1 + 1, min(n, c1 + 2))
        item, expl = f"({r1}-{r2},{c1}-{c2})", f"el bloque de las filas {r1} a {r2} y columnas {c1} a {c2}"
    ans = (f"fondo: {PALETTE[bg]}\n{PALETTE[a]}: " + " ".join(f"({r},{c})" for r, c in cells)
           + f"\n{PALETTE[b]}: {item}")
    desc = (f"este ejemplo pinta de {PALETTE[a]} " + ", ".join(f"({r},{c})" for r, c in cells)
            + f" y de {PALETTE[b]} {expl}, sobre fondo {PALETTE[bg]}")
    return ans, desc


def footer(n, fmt="grid", example=None):
    if fmt == "sparse":
        ex_ans, ex_desc = example
        return ("\n\nFormato de respuesta: dentro de <answer></answer>, una línea «fondo: color» y luego una "
                "línea por cada color pintado con sus celdas como (fila,columna), numeradas desde 1. Un tramo "
                "se escribe con guion: (f,c1-c2) es un tramo de fila, (f1-f2,c) un tramo de columna y "
                "(f1-f2,c1-c2) un bloque. Las celdas no listadas quedan del color de fondo. Escribe el nombre "
                "real de cada color.\n"
                "Ejemplo del formato (corresponde a OTRO dibujo, no a esta tarea; "
                f"{ex_desc}):\n<answer>\n{ex_ans}\n</answer>")
    return (f"\n\nResponde con la grilla final dentro de <answer></answer>: exactamente {n} líneas, "
            f"cada una con {n} índices separados por espacios. No incluyas nada más dentro de <answer>.")


def pick_bg(rng, p_white=0.6):
    return C["blanco"] if rng.random() < p_white else rng.choice([C["negro"], C["gris"], C["azul"], C["amarillo"]])


def other_colors(rng, k, exclude):
    pool = [c for c in COLS if c not in exclude]
    return rng.sample(pool, k)


# ---------------------------------------------------------------- formas
def sample_shape(rng, max_dim, kinds=None):
    kinds = kinds or ["punto", "rect", "lineah", "lineav", "marco", "cruz"]
    for _ in range(100):
        k = rng.choice(kinds)
        if k == "punto":
            p = {}
        elif k == "rect":
            p = {"h": rng.randint(2, max(2, max_dim)), "w": rng.randint(2, max(2, max_dim))}
        elif k in ("lineah", "lineav"):
            p = {"L": rng.randint(2, max(2, max_dim))}
        elif k == "marco":
            if max_dim < 3:
                continue
            p = {"h": rng.randint(3, max_dim), "w": rng.randint(3, max_dim)}
        else:
            if max_dim < 3:
                continue
            p = {"a": rng.randint(1, (max_dim - 1) // 2)}
        h, w = shape_dims(k, p)
        if h <= max_dim and w <= max_dim:
            return k, p
    return "punto", {}


# ------------------------------------------------------------------- L0
def gen_L0(n, rng):
    v = rng.choice(["fill", "halves", "bands", "border"])
    if v == "fill":
        c = rng.choice(COLS)
        g = empty(n, c)
        text = f"Tarea: pinta toda la grilla de {name(c)}."
    elif v == "halves":
        a, b = rng.sample(COLS, 2)
        g = empty(n, a)
        h = n // 2
        if rng.random() < 0.5:
            for r in range(n):
                for c in range(h, n):
                    g[r][c] = b
            text = f"Tarea: pinta la mitad izquierda de la grilla de {name(a)} y la mitad derecha de {name(b)}."
        else:
            for r in range(h, n):
                g[r] = [b] * n
            text = f"Tarea: pinta la mitad superior de la grilla de {name(a)} y la mitad inferior de {name(b)}."
    elif v == "bands":
        ks = [k for k in (2, 3, 4) if n % k == 0 and k < n]
        k = rng.choice(ks)
        cols = rng.sample(COLS, k)
        g = empty(n, cols[0])
        horiz = rng.random() < 0.5
        bw = n // k
        for r in range(n):
            for c in range(n):
                g[r][c] = cols[(r if horiz else c) // bw]
        lst = ", ".join(name(c) for c in cols)
        if horiz:
            text = (f"Tarea: divide la grilla en {k} franjas horizontales de igual alto y píntalas, "
                    f"de arriba hacia abajo, de: {lst}.")
        else:
            text = (f"Tarea: divide la grilla en {k} franjas verticales de igual ancho y píntalas, "
                    f"de izquierda a derecha, de: {lst}.")
    else:
        bg, c = rng.sample(COLS, 2)
        w = rng.choice([1, 2]) if n >= 8 else 1
        g = empty(n, bg)
        for r in range(n):
            for cc in range(n):
                if r < w or r >= n - w or cc < w or cc >= n - w:
                    g[r][cc] = c
        text = (f"Tarea: pinta toda la grilla de {name(bg)}, excepto un borde de {w} celda"
                f"{'s' if w > 1 else ''} de grosor alrededor de toda la grilla, que debe ser de color {name(c)}.")
    return text, {"kind": "exact", "target": g, "variant": v}, g


# ------------------------------------------------------------------- L1
def gen_L1(n, rng):
    bg = pick_bg(rng)
    col = other_colors(rng, 1, {bg})[0]
    k, p = sample_shape(rng, max(2, n // 2 + 1))
    h, w = shape_dims(k, p)
    r0, c0 = rng.randint(0, n - h), rng.randint(0, n - w)
    g = empty(n, bg)
    paint(g, shape_cells(k, p), col, (r0, c0))
    desc = describe_shape(k, p)
    if k == "punto":
        where = f"en la fila {r0 + 1}, columna {c0 + 1}"
    elif k == "cruz":
        where = f"con su celda central en la fila {r0 + p['a'] + 1}, columna {c0 + p['a'] + 1}"
    else:
        where = f"con su esquina superior izquierda en la fila {r0 + 1}, columna {c0 + 1}"
    text = (f"Tarea: sobre un fondo {name(bg)}, dibuja {desc} de color {name(col)}, {where}. "
            f"Todas las demás celdas deben quedar de color {name(bg)}.")
    return text, {"kind": "exact", "target": g, "variant": k}, g


# ------------------------------------------------------------------- L2
def _place(n, rng, tmpl, occupied, region=None, tries=200):
    h = max(r for r, _ in tmpl) + 1
    w = max(c for _, c in tmpl) + 1
    rr0, rr1, cc0, cc1 = region or (0, n - 1, 0, n - 1)
    for _ in range(tries):
        if rr1 - h + 1 < rr0 or cc1 - w + 1 < cc0:
            return None
        off = (rng.randint(rr0, rr1 - h + 1), rng.randint(cc0, cc1 - w + 1))
        cells = shifted(tmpl, off)
        if not (cells & occupied):
            return cells
    return None


def gen_L2(n, rng):
    for _ in range(200):
        bg = pick_bg(rng)
        n_obj = 2 if n < 8 else rng.choice([2, 3, 3])
        colors = other_colors(rng, n_obj, {bg})
        objs, cells_list, occupied = [], [], set()
        use_inside = rng.random() < 0.3 and n >= 7
        ok = True
        for i in range(n_obj):
            if i == 0 and use_inside:
                k, p = "marco", {"h": rng.randint(4, min(n, 6)), "w": rng.randint(4, min(n, 6))}
            elif i == 1 and use_inside:
                inner = (objs[0]["params"]["h"] - 2, objs[0]["params"]["w"] - 2)
                k, p = sample_shape(rng, min(inner), ["punto", "rect", "lineah", "lineav"])
                if max(shape_dims(k, p)) > min(inner):
                    ok = False
                    break
            else:
                k, p = sample_shape(rng, max(2, n // 3))
            tmpl = shape_cells(k, p)
            region = None
            if i == 1 and use_inside:
                r0, r1, c0, c1 = (min(r for r, _ in cells_list[0]), max(r for r, _ in cells_list[0]),
                                  min(c for _, c in cells_list[0]), max(c for _, c in cells_list[0]))
                region = (r0 + 1, r1 - 1, c0 + 1, c1 - 1)
            cells = _place(n, rng, tmpl, occupied, region)
            if cells is None:
                ok = False
                break
            objs.append({"color": colors[i], "shape": k, "params": p})
            cells_list.append(cells)
            occupied |= cells
        if not ok:
            continue
        # una figura del mismo color no puede quedar partida en 2 componentes: ok por construcción
        cand = []
        for i in range(n_obj):
            for j in range(n_obj):
                if i == j:
                    continue
                for rel in ("izquierda", "arriba", "toca", "no_toca", "dentro"):
                    if rel == "dentro" and objs[j]["shape"] != "marco":
                        continue
                    if rel in ("toca", "no_toca") and i > j:
                        continue
                    if rel_holds(rel, cells_list[i], cells_list[j]):
                        cand.append({"rel": rel, "a": i, "b": j})
        if use_inside:
            must = [c for c in cand if c["rel"] == "dentro"]
            if not must:
                continue
        else:
            must = []
        rng.shuffle(cand)
        m = rng.randint(n_obj - 1, n_obj)
        chosen = must[:1]
        for c in cand:
            if len(chosen) >= m:
                break
            if c in chosen:
                continue
            pair_n = sum(1 for x in chosen if {x["a"], x["b"]} == {c["a"], c["b"]})
            if pair_n >= 2:
                continue
            # usar a veces 'derecha'/'abajo' (misma relación con los roles invertidos)
            if c["rel"] in ("izquierda", "arriba") and rng.random() < 0.5:
                c = {"rel": "derecha" if c["rel"] == "izquierda" else "abajo", "a": c["b"], "b": c["a"]}
            chosen.append(c)
        if len(chosen) < m:
            continue
        g = empty(n, bg)
        for o, cells in zip(objs, cells_list):
            paint(g, cells, o["color"])
        ref_name = {i: f"la figura {fname(o['color'])}" for i, o in enumerate(objs)}
        lines = [f"- Figura {fname(o['color'])}: {describe_shape(o['shape'], o['params'])}."
                 for o in objs]
        conds = []
        for c in chosen:
            t = REL_TEXT[c["rel"]].format(A=ref_name[c["a"]], B=ref_name[c["b"]])
            conds.append("- " + t[0].upper() + t[1:] + ".")
        text = (f"Tarea: sobre un fondo {name(bg)}, dibuja exactamente estas figuras "
                f"(todas las celdas de un color forman su figura; no uses otros colores):\n"
                + "\n".join(lines) + "\n\nCondiciones:\n" + "\n".join(conds)
                + "\n\nLa posición exacta es libre mientras se cumplan todas las condiciones. "
                  "Las figuras no pueden superponerse.")
        spec = {"kind": "scene", "bg": bg, "objects": objs, "relations": chosen,
                "variant": f"{n_obj}obj" + ("_dentro" if use_inside else "")}
        return text, spec, g
    raise RuntimeError("no se pudo generar L2")


# ------------------------------------------------------------------- L3
def gen_L3(n, rng):
    v = rng.choice(["count", "sym", "pair"])
    bg = pick_bg(rng)
    if v == "count":
        for _ in range(200):
            kinds = ["punto", "rect", "lineah", "lineav"] + (["cruz"] if n >= 8 else [])
            k = rng.choice(kinds)
            p = ({} if k == "punto" else {"h": 2, "w": 2} if k == "rect"
                 else {"L": rng.randint(2, 3)} if k.startswith("linea") else {"a": 1})
            K = rng.randint(2, 4 if n < 10 else 5)
            col = other_colors(rng, 1, {bg})[0]
            tmpl = shape_cells(k, p)
            placed, occupied, halo = [], set(), set()
            for _ in range(K):
                cells = _place(n, rng, tmpl, halo)
                if cells is None:
                    break
                placed.append(cells)
                occupied |= cells
                halo |= {(r + dr, c + dc) for r, c in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
            if len(placed) < K:
                continue
            g = empty(n, bg)
            paint(g, occupied, col)
            text = (f"Tarea: sobre un fondo {name(bg)}, dibuja exactamente {K} copias de "
                    f"{describe_shape(k, p)}, todas de color {name(col)}. Ninguna copia puede tocar a otra, "
                    f"ni siquiera en diagonal. No uses otros colores.")
            return text, {"kind": "count", "bg": bg, "color": col, "shape": k, "params": p, "k": K, "variant": v}, g
    if v == "sym":
        col = other_colors(rng, 1, {bg})[0]
        h = n // 2
        m = rng.randint(2, max(2, min(h * n // 3, 10)))
        start = (rng.randrange(n), h - 1)
        cells = {start}
        while len(cells) < m:
            r, c = rng.choice(sorted(cells))
            dr, dc = rng.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
            q = (r + dr, c + dc)
            if 0 <= q[0] < n and 0 <= q[1] < h:
                cells.add(q)
        full = cells | {(r, n - 1 - c) for r, c in cells}
        touch_top = rng.random() < 0.3
        if touch_top and not any(r == 0 for r, _ in full):
            # desplazar hacia arriba
            dr = min(r for r, _ in full)
            full = {(r - dr, c) for r, c in full}
        g = empty(n, bg)
        paint(g, full, col)
        text = (f"Tarea: sobre un fondo {name(bg)}, dibuja una única figura conectada de color {name(col)} "
                f"(sus celdas unidas por arriba/abajo/izquierda/derecha) con exactamente {len(full)} celdas, "
                f"de modo que la imagen completa sea simétrica respecto del eje vertical central "
                f"(cada fila se lee igual de izquierda a derecha que de derecha a izquierda)."
                + (" La figura debe tocar la primera fila." if touch_top else "")
                + " No uses otros colores.")
        return text, {"kind": "sym", "bg": bg, "color": col, "cells": len(full),
                      "touch_top": touch_top, "variant": v}, g
    # pair
    a, b = other_colors(rng, 2, {bg})
    for _ in range(200):
        na, nb = rng.randint(2, min(8, n)), rng.randint(2, min(8, n))
        allc = [(r, c) for r in range(n) for c in range(n)]
        rng.shuffle(allc)
        A = set(allc[:na])
        forb = A | {(r + dr, c + dc) for r, c in A for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))}
        free = [x for x in allc if x not in forb]
        if len(free) < nb:
            continue
        B = set(free[:nb])
        g = empty(n, bg)
        paint(g, A, a)
        paint(g, B, b)
        text = (f"Tarea: sobre un fondo {name(bg)}, pinta exactamente {na} celdas de {name(a)} y exactamente "
                f"{nb} celdas de {name(b)}, en cualquier posición, de modo que ninguna celda {fname(a)} sea "
                f"vecina (arriba/abajo/izquierda/derecha) de una celda {fname(b)}. No uses otros colores.")
        return text, {"kind": "pair", "bg": bg, "a": a, "b": b, "na": na, "nb": nb, "variant": v}, g
    raise RuntimeError("no se pudo generar L3")


# --------------------------------------------------------------- conceptos
def round_blob(h, w):
    if (h, w) == (3, 3):
        return {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    return {(r, c) for r in range(h) for c in range(w)} - corners


BLOBS = [(3, 3), (3, 4), (4, 3), (4, 4), (4, 5), (5, 4), (5, 5), (5, 6), (6, 6)]


def fit_blobs(maxh, maxw):
    return [b for b in BLOBS if b[0] <= maxh and b[1] <= maxw]


ROUND_DEF = ("«Redondeado» significa: relleno, sin huecos, con alto y ancho de al menos 3 celdas y "
             "proporción entre 2/3 y 3/2, y con las esquinas recortadas (no un rectángulo completo).")


def concept_ref(concept, n, rng, a, opt, bg):
    """Construye un dibujo de referencia para el concepto con la asignación a."""
    g = empty(n, bg)
    if concept == "manzana":
        s = rng.choice([1, 2])
        opts = fit_blobs(n - s, n - 1)
        if not opts:
            return None
        hb, wb = rng.choice(opts)
        r0, c0 = rng.randint(s, n - hb), rng.randint(0, n - wb)
        sc = c0 + wb // 2
        paint(g, round_blob(hb, wb), a["cuerpo"], (r0, c0))
        for i in range(1, s + 1):
            g[r0 - i][sc] = a["tallo"]
        if "hoja" in a:
            lc = sc + 1 if sc + 1 < n else sc - 1
            g[r0 - s][lc] = a["hoja"]
        return g
    if concept == "casa":
        ws = [w for w in (3, 5, 7) if w <= n]
        for _ in range(50):
            w = rng.choice(ws)
            hb = rng.choice([2, 3, 4])
            base = w + 2 if (w + 2 <= n and rng.random() < 0.5) else w
            rows = (base + 1) // 2
            if rows + hb > n:
                continue
            r0 = rng.randint(rows, n - hb)          # fila superior del cuerpo
            cb = rng.randint(0, n - w)               # columna izquierda del cuerpo
            cr = cb - (base - w) // 2                # columna izquierda de la base del techo
            if cr < 0 or cr + base > n:
                continue
            for r in range(r0, r0 + hb):
                for c in range(cb, cb + w):
                    g[r][c] = a["cuerpo"]
            for i in range(rows):
                wi = base - 2 * i
                for c in range(cr + i, cr + i + wi):
                    g[r0 - 1 - i][c] = a["techo"]
            if "puerta" in a:
                dh = rng.randint(1, hb - 1)
                dc = cb + w // 2
                for r in range(r0 + hb - dh, r0 + hb):
                    g[r][dc] = a["puerta"]
            return g
        return None
    if concept == "arbol":
        for _ in range(50):
            ht = rng.randint(2, 3)
            opts = fit_blobs(n - ht, n)
            if not opts:
                return None
            hc, wc = rng.choice(opts)
            if opt.get("suelo"):
                rt1 = n - 1
            else:
                rt1 = rng.randint(hc + ht - 1, n - 1)
            rc0 = rt1 - ht - hc + 1
            if rc0 < 0:
                continue
            c0 = rng.randint(0, n - wc)
            paint(g, round_blob(hc, wc), a["copa"], (rc0, c0))
            tw = 2 if (wc >= 5 and wc % 2 == 0 and rng.random() < 0.5) else 1
            tc = c0 + wc // 2 - (1 if tw == 2 else 0)
            for r in range(rc0 + hc, rt1 + 1):
                for c in range(tc, tc + tw):
                    g[r][c] = a["tronco"]
            return g
        return None
    if concept == "semaforo":
        H = 7 if (n >= 7 and rng.random() < 0.5) else 5
        if H > n or 3 > n:
            return None
        r0, c0 = rng.randint(0, n - H), rng.randint(0, n - 3)
        for r in range(r0, r0 + H):
            for c in range(c0, c0 + 3):
                g[r][c] = a["carcasa"]
        rows = (1, 2, 3) if H == 5 else (1, 3, 5)
        for k, rr in zip(("luz_roja", "luz_amarilla", "luz_verde"), rows):
            g[r0 + rr][c0 + 1] = a[k]
        return g
    if concept == "sol":
        k = opt["k"]
        big = n >= 7 and (k > 4 or rng.random() < 0.5)
        if k > 4 and not big:
            return None
        hc = 5 if big else 3
        span = hc + 2
        if span > n:
            return None
        r0, c0 = rng.randint(1, n - hc - 1), rng.randint(1, n - hc - 1)
        core = round_blob(hc, hc)
        paint(g, core, a["nucleo"], (r0, c0))
        m = hc // 2
        ortho = [((r0 - 1, c0 + m), (-1, 0)), ((r0 + hc, c0 + m), (1, 0)),
                 ((r0 + m, c0 - 1), (0, -1)), ((r0 + m, c0 + hc), (0, 1))]
        diag = []
        if big:
            diag = [((r0, c0), (-1, -1)), ((r0, c0 + hc - 1), (-1, 1)),
                    ((r0 + hc - 1, c0), (1, -1)), ((r0 + hc - 1, c0 + hc - 1), (1, 1))]
        cand = ortho + diag
        if k > len(cand):
            return None
        for (r, c), (dr, dc) in rng.sample(cand, k):
            g[r][c] = a["rayos"]
            r2, c2 = r + dr, c + dc
            if rng.random() < 0.5 and 0 <= r2 < n and 0 <= c2 < n and g[r2][c2] == bg:
                g[r2][c2] = a["rayos"]
        return g
    raise ValueError(concept)


L4_CONCEPTS = ["manzana", "casa", "arbol", "semaforo", "sol"]
L5_CONCEPTS = ["manzana", "casa", "arbol", "semaforo"]
ARTICLE = {"manzana": "una manzana", "casa": "una casa", "arbol": "un árbol", "semaforo": "un semáforo",
           "sol": "un sol"}


def gen_L4(n, rng, concepts=L4_CONCEPTS):
    for _ in range(200):
        concept = rng.choice(concepts)
        bg = C["blanco"] if rng.random() < 0.7 else rng.choice([C["azul"], C["gris"], C["negro"]])
        canon = rng.random() < 0.5
        opt = {}
        if concept == "manzana":
            leaf = rng.random() < 0.5
            cols = ([C["rojo"], C["café"]] + ([C["verde"]] if leaf else [])) if canon \
                else other_colors(rng, 3 if leaf else 2, {bg})
            if bg in cols:
                continue
            a = {"cuerpo": cols[0], "tallo": cols[1]}
            if leaf:
                a["hoja"] = cols[2]
                opt["hoja"] = "requerida"
            text = (f"Tarea: dibuja una manzana sobre fondo {name(bg)}: un cuerpo redondeado de color "
                    f"{name(a['cuerpo'])}, y un tallo vertical de color {name(a['tallo'])} de 1 o 2 celdas apoyado "
                    f"justo encima del cuerpo (su celda inferior toca una celda de la fila superior del cuerpo)"
                    + (f", y una hoja de color {name(a['hoja'])} de 1 a 3 celdas que toque el tallo (puede ser en diagonal)"
                       if leaf else "") + f". {ROUND_DEF} No uses otros colores.")
        elif concept == "casa":
            cols = [C["amarillo"], C["rojo"], C["café"]] if canon else other_colors(rng, 3, {bg})
            if bg in cols:
                continue
            a = {"cuerpo": cols[0], "techo": cols[1], "puerta": cols[2]}
            opt["puerta"] = "requerida"
            text = (f"Tarea: dibuja una casa sobre fondo {name(bg)}: un cuerpo rectangular relleno de color "
                    f"{name(a['cuerpo'])} de al menos 2 filas y 3 columnas; un techo triangular de color "
                    f"{name(a['techo'])} justo encima del cuerpo (cada fila del techo es un tramo continuo "
                    f"contenido en la fila de abajo, la fila superior del techo es más angosta que la inferior, "
                    f"el techo tiene al menos 2 filas y su fila inferior cubre todo el ancho del cuerpo); y una "
                    f"puerta rectangular de color {name(a['puerta'])} dentro del cuerpo, apoyada en su fila inferior, "
                    f"sin tocar los bordes izquierdo, derecho ni superior del cuerpo. Puerta y cuerpo juntos forman "
                    f"un rectángulo completo. No uses otros colores.")
        elif concept == "arbol":
            cols = [C["verde"], C["café"]] if canon else other_colors(rng, 2, {bg})
            if bg in cols:
                continue
            a = {"copa": cols[0], "tronco": cols[1]}
            opt["suelo"] = rng.random() < 0.5
            text = (f"Tarea: dibuja un árbol sobre fondo {name(bg)}: una copa redondeada de color {name(a['copa'])} "
                    f"y, justo debajo, un tronco vertical de color {name(a['tronco'])} (un rectángulo de 1 o 2 columnas "
                    f"de ancho y al menos 2 filas de alto). Cada columna del tronco debe tener una celda de copa "
                    f"inmediatamente encima, y la copa debe ser más ancha que el tronco"
                    + (". El tronco debe llegar hasta la última fila de la grilla" if opt["suelo"] else "")
                    + f". {ROUND_DEF} No uses otros colores.")
        elif concept == "semaforo":
            body = rng.choice([C["negro"], C["gris"]]) if canon else \
                other_colors(rng, 1, {bg, C["rojo"], C["amarillo"], C["verde"]})[0]
            if bg in (body, C["rojo"], C["amarillo"], C["verde"]):
                continue
            a = {"carcasa": body, "luz_roja": C["rojo"], "luz_amarilla": C["amarillo"], "luz_verde": C["verde"]}
            text = (f"Tarea: dibuja un semáforo sobre fondo {name(bg)}: una carcasa rectangular de color "
                    f"{name(body)}, más alta que ancha, y dentro de ella, sin tocar su borde, tres luces: roja "
                    f"arriba, amarilla al medio y verde abajo, alineadas en la misma columna. Cada luz es un "
                    f"cuadrado de 1×1 o 2×2. Carcasa y luces juntas forman un rectángulo completo. "
                    f"No uses otros colores.")
        else:  # sol
            k = rng.choice([3, 4, 4, 5, 6, 8]) if n >= 7 else rng.choice([2, 3, 4])
            cols = [C["amarillo"], C["naranja"]] if canon else other_colors(rng, 2, {bg})
            if bg in cols:
                continue
            a = {"nucleo": cols[0], "rayos": cols[1]}
            opt["k"] = k
            text = (f"Tarea: dibuja un sol sobre fondo {name(bg)}: un núcleo redondeado de color {name(a['nucleo'])} "
                    f"y exactamente {k} rayos de color {name(a['rayos'])}. Cada rayo es un segmento recto (horizontal, "
                    f"vertical o diagonal) de 1 o 2 celdas que toca el núcleo (puede ser en diagonal); los rayos no "
                    f"se tocan entre sí, ni siquiera en diagonal. {ROUND_DEF} No uses otros colores.")
        ref, reg, box, rtxt = concept_ref_region(concept, n, rng, a, opt, bg, p_region=0.3)
        if ref is None:
            continue
        if reg:
            text = text[:-len("No uses otros colores.")] + rtxt[0].upper() + rtxt[1:] + ". No uses otros colores."
        roles = {k: [v] for k, v in a.items()}
        spec = {"kind": "concept", "concept": concept, "bg": bg, "roles": roles, "optional_roles": {},
                "opt": opt, "explicit": True, "variant": concept, "region": box}
        return text, spec, ref
    raise RuntimeError("no se pudo generar L4")


def regions(n):
    h = n // 2
    return {
        "izquierda": ((0, n - 1, 0, h - 1), f"todo el dibujo debe quedar en la mitad izquierda de la grilla (columnas 1 a {h})"),
        "derecha": ((0, n - 1, n - h, n - 1), f"todo el dibujo debe quedar en la mitad derecha de la grilla (columnas {n - h + 1} a {n})"),
        "arriba": ((0, h - 1, 0, n - 1), f"todo el dibujo debe quedar en la mitad superior de la grilla (filas 1 a {h})"),
        "abajo": ((n - h, n - 1, 0, n - 1), f"todo el dibujo debe quedar en la mitad inferior de la grilla (filas {n - h + 1} a {n})"),
    }


def in_region(g, bg, box):
    r0, r1, c0, c1 = box
    return all(r0 <= r <= r1 and c0 <= c <= c1
               for r, row in enumerate(g) for c, v in enumerate(row) if v != bg)


def concept_ref_region(concept, n, rng, a, opt, bg, p_region=0.4):
    """Referencia + (opcional) restricción de región. Devuelve (ref, region_name, box, texto)."""
    want = rng.random() < p_region
    if want:
        name_, (box, txt) = rng.choice(sorted(regions(n).items()))
        for _ in range(300):
            ref = concept_ref(concept, n, rng, a, opt, bg)
            if ref is not None and in_region(ref, bg, box):
                return ref, name_, box, txt
    ref = concept_ref(concept, n, rng, a, opt, bg)
    return ref, None, None, None


def gen_L5(n, rng, concepts=L5_CONCEPTS):
    anyc = [c for c in COLS]
    for _ in range(200):
        concept = rng.choice(concepts)
        bg = C["blanco"] if rng.random() < 0.5 else rng.choice([C["azul"], C["negro"], C["gris"]])
        if concept == "manzana":
            roles = {"cuerpo": [C["rojo"], C["verde"], C["amarillo"]], "tallo": [C["café"], C["negro"], C["verde"]]}
            opt_roles = {"hoja": [C["verde"]]}
            opt = {"hoja": "opcional"}
            ref_a = {"cuerpo": C["rojo"], "tallo": C["café"] if bg != C["café"] else C["negro"], "hoja": C["verde"]}
        elif concept == "casa":
            roles = {"cuerpo": anyc, "techo": anyc}
            opt_roles = {"puerta": anyc}
            opt = {"puerta": "opcional"}
            ref_a = {"cuerpo": C["amarillo"], "techo": C["rojo"], "puerta": C["café"]}
        elif concept == "arbol":
            roles = {"copa": [C["verde"], C["amarillo"], C["naranja"]], "tronco": [C["café"], C["negro"], C["gris"]]}
            opt_roles, opt = {}, {}
            ref_a = {"copa": C["verde"], "tronco": C["café"]}
        else:
            roles = {"carcasa": [C["negro"], C["gris"]], "luz_roja": [C["rojo"]],
                     "luz_amarilla": [C["amarillo"], C["naranja"]], "luz_verde": [C["verde"]]}
            opt_roles, opt = {}, {}
            ref_a = {"carcasa": C["gris"] if bg == C["negro"] else C["negro"], "luz_roja": C["rojo"],
                     "luz_amarilla": C["amarillo"], "luz_verde": C["verde"]}
        if bg in ref_a.values():
            continue
        ref, reg, box, rtxt = concept_ref_region(concept, n, rng, ref_a, opt, bg)
        if ref is None:
            continue
        text = (f"Tarea: dibuja {ARTICLE[concept]} reconocible sobre fondo {name(bg)}. "
                + (rtxt[0].upper() + rtxt[1:] + ". " if reg else "")
                + "Elige tú los colores apropiados de la paleta. Dibuja solo las partes "
                "principales del objeto, sin elementos extra (pasto, nubes, decoraciones). Las partes redondas "
                "deben verse redondeadas (rellenas y con las esquinas recortadas) y cada parte debe ser de un "
                "solo color, distinto del de las otras partes y del fondo.")
        spec = {"kind": "concept", "concept": concept, "bg": bg, "roles": roles,
                "optional_roles": opt_roles, "opt": opt, "explicit": False, "variant": concept,
                "region": box}
        return text, spec, ref
    raise RuntimeError("no se pudo generar L5")


GENS = {"L0": gen_L0, "L1": gen_L1, "L2": gen_L2, "L3": gen_L3, "L4": gen_L4, "L5": gen_L5}


ANSWER_FORMAT = "grid"   # se fija desde la CLI con --format


def make_task(level, n, rng, idx, split="calib", **kw):
    text, spec, ref = GENS[level](n, rng, **kw)
    spec = dict(spec, size=n, level=level, answer_format=ANSWER_FORMAT)
    example = None
    if ANSWER_FORMAT == "sparse":
        # copiar el ejemplo no puede dar recompensa: ni binaria ni parcial
        from .verifier import verify
        for _ in range(200):
            example = sparse_example(n, rng)
            if verify(spec, f"<answer>\n{example[0]}\n</answer>")["score_adj"] == 0.0:
                break
        else:
            raise RuntimeError("no se encontró un ejemplo con recompensa 0 para la tarea")
    prompt = header(n) + text + footer(n, ANSWER_FORMAT, example)
    tid = f"{level}-{n}-" + hashlib.sha1(prompt.encode()).hexdigest()[:10]
    # identidad de la tarea sin el ejemplo de formato: se usa para deduplicar entre splits
    core = hashlib.sha1((header(n) + text).encode()).hexdigest()[:16]
    task = {"id": tid, "core_id": core, "level": level, "size": n, "split": split,
            "family": spec.get("variant"), "prompt": prompt, "spec": spec, "reference": ref}
    if example:
        task["format_example"] = example[0]
    return task


def gen_calib(per_cell, sizes, levels, seed):
    rng = random.Random(seed)
    tasks, seen = [], set()
    for lv in levels:
        for n in sizes:
            k = tries = 0
            while k < per_cell and tries < per_cell * 50:
                tries += 1
                t = make_task(lv, n, rng, k)
                if t["core_id"] in seen:
                    continue
                seen.add(t["core_id"])
                tasks.append(t)
                k += 1
    return tasks


def _unique_pool(level, n, rng, m, seen, split, **kw):
    """Hasta m tareas con enunciado distinto (core_id) que no estén en `seen`."""
    pool, tries = [], 0
    while len(pool) < m and tries < m * 50:
        tries += 1
        t = make_task(level, n, rng, len(pool), split=split, **kw)
        if t["core_id"] in seen:
            continue
        seen.add(t["core_id"])
        pool.append(t)
    return pool


def gen_full(seed, per_cell_train, per_cell_eval, per_cell_test, sizes, ood_sizes):
    """train/eval/test en tamaños in-distribution (L4 sin 'sol'); test_ood: tamaños más grandes
    + el concepto 'sol', nunca visto en train/eval.
    Sin fugas: ningún enunciado (core_id) aparece en dos splits. Cuando un nivel tiene menos
    enunciados distintos que los pedidos (L5, L0 en grillas chicas), el conjunto disponible se
    reparte en proporción train:eval:test en vez de agotarse en train."""
    rng = random.Random(seed)
    seen, out = set(), {"train": [], "eval": [], "test": [], "test_ood": []}
    want = {"train": per_cell_train, "eval": per_cell_eval, "test": per_cell_test}
    total = sum(want.values())
    for lv in GENS:
        kw = {"concepts": ["manzana", "casa", "arbol", "semaforo"]} if lv == "L4" else {}
        for n in sizes:
            pool = _unique_pool(lv, n, rng, total, seen, "pool", **kw)
            rng.shuffle(pool)
            if len(pool) >= total:
                cuts = {s: want[s] for s in want}
            else:   # reparto proporcional del conjunto disponible
                ev = max(1, round(len(pool) * want["eval"] / total))
                te = max(1, round(len(pool) * want["test"] / total))
                cuts = {"eval": ev, "test": te, "train": max(0, len(pool) - ev - te)}
            i = 0
            for s in ("train", "eval", "test"):
                for t in pool[i:i + cuts[s]]:
                    t["split"] = s
                    out[s].append(t)
                i += cuts[s]
    for lv in GENS:
        kw = {"concepts": L4_CONCEPTS} if lv == "L4" else {}
        for n in ood_sizes:
            out["test_ood"] += _unique_pool(lv, n, rng, per_cell_test, seen, "test_ood", **kw)
    for n in sizes:   # 'sol' en tamaños in-distribution también es OOD
        out["test_ood"] += _unique_pool("L4", n, rng, per_cell_test, seen, "test_ood", concepts=["sol"])
    return out


def main():
    global ANSWER_FORMAT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["calib", "full"], default="calib",
                    help="calib: un solo archivo; full: splits train/eval/test/test_ood")
    ap.add_argument("--per-cell", type=int, default=30, help="(calib) tareas por nivel × tamaño")
    ap.add_argument("--sizes", type=int, nargs="+", default=[6, 8, 12])
    ap.add_argument("--levels", nargs="+", default=list(GENS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="calib.jsonl", help="(calib) archivo de salida")
    ap.add_argument("--out-dir", default="data", help="(full) directorio de salida")
    ap.add_argument("--train", type=int, default=400, help="(full) tareas de train por nivel × tamaño")
    ap.add_argument("--eval", type=int, default=50, help="(full) tareas de eval por nivel × tamaño")
    ap.add_argument("--test", type=int, default=100, help="(full) tareas de test por nivel × tamaño")
    ap.add_argument("--ood-sizes", type=int, nargs="+", default=[10, 16], help="(full) tamaños de test_ood")
    ap.add_argument("--format", choices=["grid", "sparse"], default="sparse",
                    help="formato de respuesta pedido en el prompt (el verificador acepta ambos)")
    a = ap.parse_args()
    ANSWER_FORMAT = a.format
    if a.mode == "calib":
        tasks = gen_calib(a.per_cell, a.sizes, a.levels, a.seed)
        write_jsonl(a.out, tasks)
        print(f"{len(tasks)} tareas -> {a.out}")
    else:
        for split, tasks in gen_full(a.seed, a.train, a.eval, a.test, a.sizes, a.ood_sizes).items():
            write_jsonl(os.path.join(a.out_dir, f"{split}.jsonl"), tasks)
            print(f"{split}: {len(tasks)} tareas")


if __name__ == "__main__":
    main()
