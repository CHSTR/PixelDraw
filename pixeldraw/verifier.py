"""Parser y verificador de PixelDraw, con la interfaz común del curso.

    reward(prompt, completion) -> float          1.0 si la respuesta cumple todas las restricciones
    reward_partial(prompt, completion) -> float  crédito parcial en [0, 1] (señal densa para GRPO)
    verify(spec, completion) -> dict             veredicto detallado de una respuesta

reward y reward_partial buscan la especificación de la tarea a partir del prompt, en un registro que
se llena con load_tasks(ruta), register_tasks(tareas) o la variable de entorno PIXELDRAW_TASKS
(varias rutas separadas por ':' en Linux y macOS, o ';' en Windows).
"""
from __future__ import annotations

import hashlib
import json
import os
import re

from .core import (NAME2IDX, best_concept_assignment, cells_of, colors_used,
                   components, normalize, rel_holds, shape_cells)

# ------------------------------------------------------------------ parser
_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
_FENCE_RE = re.compile(r"```[a-zA-Z]*\n?(.*?)```", re.S)
_LABEL_RE = re.compile(r"^\s*(?:fila|row|f|r)?\s*\d{1,2}\s*[:)\-–]\s+", re.I)
_SEP_RE = re.compile(r"[\s,;|\[\]]+")


class ParseError(Exception):
    pass


def _strip_thinking(text: str) -> str:
    if re.search(r"<think>", text, re.I) and not re.search(r"</think>", text, re.I):
        return ""  # razonamiento truncado: no hay respuesta final
    text = _THINK_RE.sub("", text)
    # algunos servidores devuelven solo el cierre
    if re.search(r"</think>", text, re.I):
        text = re.split(r"</think>", text, flags=re.I)[-1]
    return text


def _parse_row(line: str):
    """Devuelve lista de índices o None si la línea no es una fila de grilla."""
    s = line.strip().strip("`").strip()
    if not s:
        return None
    s = _LABEL_RE.sub("", s)
    s = s.strip().strip("[](),;|").strip()
    if not s:
        return None
    if re.fullmatch(r"\d+", s):           # forma contigua: "00120"
        return [int(ch) for ch in s]
    toks = [t for t in _SEP_RE.split(s) if t and t not in {"-", "--", ":"}]
    out = []
    for t in toks:
        tl = t.lower().strip("'\".")
        if re.fullmatch(r"\d", tl):
            out.append(int(tl))
        elif tl in NAME2IDX:
            out.append(NAME2IDX[tl])
        else:
            return None                   # token extraño: no es una fila
    return out or None


def _grid_from_block(block: str, n: int):
    block = block.strip()
    if block.startswith(("[", "{")):
        try:
            data = json.loads(block)
            if isinstance(data, dict):
                data = data.get("grid") or data.get("grilla")
            if (isinstance(data, list) and all(isinstance(r, list) for r in data)):
                g = [[int(v) if not isinstance(v, str) else
                      (int(v) if v.isdigit() else NAME2IDX[v.lower()]) for v in r] for r in data]
                return g
        except (ValueError, KeyError, TypeError, AttributeError):
            pass
    rows = []
    for line in block.splitlines():
        if re.fullmatch(r"\s*\|?\s*:?-{2,}.*", line):   # separador de tabla markdown
            continue
        r = _parse_row(line)
        if r is not None:
            rows.append(r)
    return rows


# ---- formato disperso:  "fondo: blanco" + líneas "rojo: (2,3) (2,4-6) (5-6,1)"
_FONDO_RE = re.compile(r"^\s*[-*]?\s*(?:fondo|background|bg)\s*[:=]\s*(\S+?)\s*\.?\s*$", re.I | re.M)
_COLOR_LINE_RE = re.compile(r"^\s*[-*]?\s*([A-Za-záéíóúÁÉÍÓÚ]+|\d)\s*[:=]\s*(.*)$")
_ITEM_RE = re.compile(r"\(\s*(\d+)\s*(?:-\s*(\d+)\s*)?,\s*(\d+)\s*(?:-\s*(\d+)\s*)?\)")
# notación equivalente de esquina a esquina: (f1,c1)-(f2,c2)
_CORNER_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*(?:-|–|a|to|hasta)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", re.I)


# comentarios al final de línea (no cambian el dibujo) y marcas de lista vacía
_COMMENT_RE = re.compile(r"<!--.*?-->|\s(?:#|//).*$")
_EMPTY_RE = re.compile(r"\s*(?:-+|—|–|ningun[ao]s?|nada|none|vac[ií]o|\[\s*\]|\(\s*\)|∅)?\s*\.?\s*", re.I)


def _color_token(tok: str):
    t = tok.strip().lower().strip("'\".")
    if re.fullmatch(r"\d", t):
        return int(t)
    if t in NAME2IDX:
        return NAME2IDX[t]
    raise ParseError("color_desconocido")


def _sparse_from_block(block: str, n: int):
    """Renderiza el formato disperso. Estricto: cualquier línea o token que no
    sea fondo/color/coordenada invalida la respuesta (no se adivina)."""
    lines = [_COMMENT_RE.sub("", l).rstrip() for l in block.strip().splitlines()]
    lines = [l for l in lines if l.strip()]
    fondos = [l for l in lines if _FONDO_RE.match(l)]
    if len(fondos) != 1:
        raise ParseError("fondo_ausente_o_repetido")
    bg = _color_token(_FONDO_RE.match(fondos[0]).group(1))
    g = [[bg] * n for _ in range(n)]
    for line in lines:
        line = _COMMENT_RE.sub("", line).rstrip()
        if not line.strip() or _FONDO_RE.match(line):
            continue
        m = _COLOR_LINE_RE.match(line)
        if not m:
            raise ParseError("linea_extrana")
        col = _color_token(m.group(1))
        rest = m.group(2)
        if _EMPTY_RE.fullmatch(rest):          # "rojo: -" / "rojo: ninguna" = no pinta nada
            continue
        boxes = []
        for it in _CORNER_RE.finditer(rest):
            a, b, c, d = map(int, it.groups())
            boxes.append((min(a, c), max(a, c), min(b, d), max(b, d)))
        rest = _CORNER_RE.sub(" ", rest)
        if _ITEM_RE.sub("", rest).strip(" ,;.\t") != "":
            raise ParseError("token_extrano")
        for it in _ITEM_RE.finditer(rest):
            r1, r2, c1, c2 = it.groups()
            r1, c1 = int(r1), int(c1)
            r2 = int(r2) if r2 else r1
            c2 = int(c2) if c2 else c1
            boxes.append((r1, r2, c1, c2))
        for r1, r2, c1, c2 in boxes:
            if not (1 <= r1 <= r2 <= n and 1 <= c1 <= c2 <= n):
                raise ParseError("coordenada_fuera")
            for r in range(r1 - 1, r2):
                for c in range(c1 - 1, c2):
                    g[r][c] = col
    return g


def _is_sparse(block: str) -> bool:
    return any(_FONDO_RE.match(_COMMENT_RE.sub("", l).rstrip()) for l in (block or "").splitlines())


def parse_grid(completion: str, n: int):
    """Extrae una grilla n×n. Prioridad: último <answer>, último bloque ```,
    últimas n filas-de-grilla del texto. Lanza ParseError si no hay grilla válida."""
    text = _strip_thinking(completion or "")
    if not text.strip():
        raise ParseError("sin_respuesta")
    answers = _ANSWER_RE.findall(text)
    if answers:
        if _is_sparse(answers[-1]):
            return _sparse_from_block(answers[-1], n)
        rows = _grid_from_block(answers[-1], n)
    else:
        fences = _FENCE_RE.findall(text)
        if fences and _is_sparse(fences[-1]):
            return _sparse_from_block(fences[-1], n)
        rows = _grid_from_block(fences[-1], n) if fences else None
        if not rows and _is_sparse(text):
            # respaldo disperso sin etiquetas: desde la última línea "fondo:" mientras sigan líneas de color
            lines = text.splitlines()
            i = max(k for k, l in enumerate(lines) if _FONDO_RE.match(_COMMENT_RE.sub("", l).rstrip()))
            blk = [lines[i]]
            for l in lines[i + 1:]:
                if not l.strip():
                    if len(blk) > 1:
                        break
                    continue
                lc = _COMMENT_RE.sub("", l).rstrip()
                if _COLOR_LINE_RE.match(lc) and (_ITEM_RE.search(lc) or _EMPTY_RE.fullmatch(_COLOR_LINE_RE.match(lc).group(2))):
                    blk.append(l)
                else:
                    break
            return _sparse_from_block("\n".join(blk), n)
        if not rows:
            # respaldo: último tramo contiguo de filas válidas
            runs, cur = [], []
            for line in text.splitlines():
                r = _parse_row(line)
                if r is not None and len(r) == n:
                    cur.append(r)
                else:
                    if cur:
                        runs.append(cur)
                    cur = []
            if cur:
                runs.append(cur)
            rows = runs[-1] if runs else []
    if len(rows) != n or any(len(r) != n for r in rows):
        raise ParseError(f"dimensiones_{len(rows)}x{max((len(r) for r in rows), default=0)}")
    if any(v < 0 or v > 9 for r in rows for v in r):
        raise ParseError("indice_fuera_de_paleta")
    return rows


# -------------------------------------------------------------- verificación
def _check_exact(grid, spec):
    target = spec["target"]
    out = []
    for col in sorted(colors_used(target)):
        out.append((f"region_{col}", cells_of(grid, col) == cells_of(target, col)))
    return out


def _find_object(grid, color, kind, params):
    cells = cells_of(grid, color)
    comps = components(cells, 4)
    if len(comps) == 1 and normalize(comps[0]) == shape_cells(kind, params):
        return comps[0]
    return None


def _check_scene(grid, spec):
    bg = spec["bg"]
    objs = spec["objects"]
    allowed = {bg} | {o["color"] for o in objs}
    out = [("solo_colores_permitidos", colors_used(grid) <= allowed)]
    found = []
    for i, o in enumerate(objs):
        f = _find_object(grid, o["color"], o["shape"], o["params"])
        found.append(f)
        out.append((f"objeto_{i}_forma", f is not None))
    for rr in spec["relations"]:
        A, B = found[rr["a"]], found[rr["b"]]
        out.append((f"rel_{rr['rel']}_{rr['a']}_{rr['b']}", bool(A and B and rel_holds(rr["rel"], A, B))))
    return out


def _check_count(grid, spec):
    bg, col = spec["bg"], spec["color"]
    out = [("solo_colores_permitidos", colors_used(grid) <= {bg, col})]
    comps = components(cells_of(grid, col), 8)
    out.append(("cantidad_exacta", len(comps) == spec["k"]))
    tmpl = shape_cells(spec["shape"], spec["params"])
    out.append(("todas_con_la_forma", bool(comps) and all(normalize(c) == tmpl for c in comps)))
    return out


def _check_sym(grid, spec):
    bg, col = spec["bg"], spec["color"]
    cells = cells_of(grid, col)
    out = [("solo_colores_permitidos", colors_used(grid) <= {bg, col})]
    out.append(("cantidad_celdas", len(cells) == spec["cells"]))
    out.append(("conectada", len(components(cells, 4)) == 1))
    out.append(("simetrica", all(row == row[::-1] for row in grid)))
    if spec.get("touch_top"):
        out.append(("toca_borde_superior", any(r == 0 for r, _ in cells)))
    return out


def _check_pair(grid, spec):
    bg, a, b = spec["bg"], spec["a"], spec["b"]
    A, B = cells_of(grid, a), cells_of(grid, b)
    return [("solo_colores_permitidos", colors_used(grid) <= {bg, a, b}),
            ("cantidad_a", len(A) == spec["na"]),
            ("cantidad_b", len(B) == spec["nb"]),
            ("no_se_tocan", not (A and B and rel_holds("toca", A, B)))]


def _check_concept(grid, spec):
    roles = {k: v for k, v in spec["roles"].items()}
    opt_roles = spec.get("optional_roles", {})
    opt = dict(spec.get("opt", {}), explicit=spec.get("explicit", True), bg=spec["bg"])
    best = best_concept_assignment(grid, spec["concept"], roles, opt_roles, opt, spec["bg"])
    checks = [("asignacion_de_colores", False)] if best is None else list(best[1])
    box = spec.get("region")
    if box:
        r0, r1, c0, c1 = box
        bg = spec["bg"]
        inside = all(r0 <= r <= r1 and c0 <= c <= c1
                     for r, row in enumerate(grid) for c, v in enumerate(row) if v != bg)
        checks.append(("dentro_de_la_region", inside))
    return checks


CHECKERS = {
    "exact": _check_exact,
    "scene": _check_scene,
    "count": _check_count,
    "sym": _check_sym,
    "pair": _check_pair,
    "concept": _check_concept,
}


# Restricciones de "higiene": obligatorias para resolver la tarea, pero sin crédito
# parcial propio (casi cualquier salida las cumple y darían recompensa gratis).
HYGIENE = {"solo_colores_permitidos", "sin_colores_extra", "dentro_de_la_region"}


def _score_grid(spec, grid):
    checks = CHECKERS[spec["kind"]](grid, spec)
    ok = sum(v for _, v in checks)
    sub = [v for name, v in checks if name not in HYGIENE]
    sub_score = (sum(sub) / len(sub)) if sub else ok / len(checks)
    return checks, ok / len(checks), sub_score


def _hygiene_frac(checks):
    h = [v for name, v in checks if name in HYGIENE]
    return sum(h) / len(h) if h else 1.0


_BASELINE_CACHE: dict[str, float] = {}


def trivial_baseline(spec) -> float:
    """Mejor crédito sustantivo alcanzable con una salida trivial (grilla de un solo color).
    El crédito parcial se mide por encima de este piso para que no sea explotable."""
    key = json.dumps(spec, sort_keys=True)
    if key not in _BASELINE_CACHE:
        n = spec["size"]
        _BASELINE_CACHE[key] = max(_score_grid(spec, [[c] * n for _ in range(n)])[2] for c in range(10))
    return _BASELINE_CACHE[key]


def verify(spec: dict, completion: str) -> dict:
    """Devuelve {'solved', 'score', 'score_adj', 'checks', 'error'}.
    score     = fracción de restricciones cumplidas, todas incluidas (0 si la grilla no se pudo leer)
    score_adj = 1.0 si la tarea está resuelta. Si no:
                0.9 · max(0, (s - b) / (1 - b)) · (0.5 + 0.5 · h)
                con s = fracción de restricciones sustantivas cumplidas, b = piso trivial y
                h = fracción de restricciones de higiene cumplidas. Una respuesta no resuelta
                nunca alcanza a una resuelta, y violar la higiene descuenta."""
    try:
        grid = parse_grid(completion, spec["size"])
    except ParseError as e:
        return {"solved": False, "score": 0.0, "score_adj": 0.0, "checks": [], "error": str(e)}
    checks, score, sub = _score_grid(spec, grid)
    solved = all(v for _, v in checks)
    if solved:
        adj = 1.0
    else:
        b = trivial_baseline(spec)
        base = 0.0 if b >= 1 else max(0.0, (sub - b) / (1 - b))
        adj = 0.9 * base * (0.5 + 0.5 * _hygiene_frac(checks))
    return {"solved": solved, "score": score, "score_adj": adj, "checks": checks, "error": None}


# ------------------------------------------------------- interfaz del curso
_REGISTRY: dict[str, dict] = {}


def _key(prompt) -> str:
    if isinstance(prompt, list):  # formato chat: usar el último mensaje de usuario
        prompt = [m["content"] for m in prompt if m.get("role") == "user"][-1]
    return hashlib.sha1(prompt.strip().encode()).hexdigest()


def register_tasks(tasks):
    for t in tasks:
        _REGISTRY[_key(t["prompt"])] = t["spec"]


def load_tasks(path):
    with open(path, encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if line.strip()]
    register_tasks(tasks)
    return tasks


def _ensure_registry():
    if not _REGISTRY and os.environ.get("PIXELDRAW_TASKS"):
        for p in os.environ["PIXELDRAW_TASKS"].split(os.pathsep):
            load_tasks(p)


def reward(prompt, completion) -> float:
    """Recompensa binaria (1.0 si se cumplen todas las restricciones)."""
    _ensure_registry()
    spec = _REGISTRY.get(_key(prompt))
    if spec is None:
        raise KeyError("prompt no registrado: cargue las tareas con load_tasks()")
    return 1.0 if verify(spec, completion)["solved"] else 0.0


def reward_partial(prompt, completion) -> float:
    """Crédito parcial sobre el piso trivial (señal densa para GRPO; ver verify)."""
    _ensure_registry()
    spec = _REGISTRY.get(_key(prompt))
    if spec is None:
        raise KeyError("prompt no registrado: cargue las tareas con load_tasks()")
    return verify(spec, completion)["score_adj"]
