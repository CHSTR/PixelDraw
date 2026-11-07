"""Núcleo compartido por el generador y el verificador: paleta, formas, relaciones y conceptos.

Grilla = lista de listas de enteros 0..9 (índices de PALETTE).
Coordenadas internas 0-indexadas (r, c); en los prompts se muestran 1-indexadas.
"""
from __future__ import annotations

from itertools import product

PALETTE = ["blanco", "negro", "rojo", "verde", "azul", "amarillo",
           "café", "naranja", "morado", "gris"]
C = {name: i for i, name in enumerate(PALETTE)}

# Nombres aceptados al parsear (respuestas equivalentes escritas con palabras)
NAME2IDX = {
    "blanco": 0, "white": 0, "negro": 1, "black": 1, "rojo": 2, "red": 2,
    "verde": 3, "green": 3, "azul": 4, "blue": 4, "amarillo": 5, "yellow": 5,
    "cafe": 6, "café": 6, "marron": 6, "marrón": 6, "brown": 6,
    "naranja": 7, "orange": 7, "morado": 8, "purple": 8, "violeta": 8,
    "gris": 9, "gray": 9, "grey": 9,
    # formas femeninas y plurales (el prompt dice "la figura roja")
    "blanca": 0, "blancas": 0, "blancos": 0, "negra": 1, "negras": 1, "negros": 1,
    "roja": 2, "rojas": 2, "rojos": 2, "verdes": 3, "azules": 4,
    "amarilla": 5, "amarillas": 5, "amarillos": 5, "cafés": 6, "marrones": 6,
    "naranjas": 7, "morada": 8, "moradas": 8, "morados": 8, "grises": 9,
}

N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
N8 = N4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))


# ----------------------------------------------------------------- formas
def shape_cells(kind: str, p: dict) -> frozenset:
    """Plantilla de una forma con origen en (0,0)."""
    if kind == "punto":
        return frozenset({(0, 0)})
    if kind == "rect":
        return frozenset((r, c) for r in range(p["h"]) for c in range(p["w"]))
    if kind == "lineah":
        return frozenset((0, c) for c in range(p["L"]))
    if kind == "lineav":
        return frozenset((r, 0) for r in range(p["L"]))
    if kind == "marco":
        h, w = p["h"], p["w"]
        return frozenset((r, c) for r in range(h) for c in range(w)
                         if r in (0, h - 1) or c in (0, w - 1))
    if kind == "cruz":
        a = p["a"]
        cells = {(a, a)}
        for d in range(1, a + 1):
            cells |= {(a - d, a), (a + d, a), (a, a - d), (a, a + d)}
        return frozenset(cells)
    raise ValueError(kind)


def shape_dims(kind: str, p: dict) -> tuple[int, int]:
    cells = shape_cells(kind, p)
    return (max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1)


def describe_shape(kind: str, p: dict) -> str:
    if kind == "punto":
        return "un punto (una sola celda)"
    if kind == "rect":
        if p["h"] == p["w"]:
            return f"un cuadrado relleno de {p['h']}×{p['w']}"
        return f"un rectángulo relleno de {p['h']} filas × {p['w']} columnas"
    if kind == "lineah":
        return f"una línea horizontal de {p['L']} celdas"
    if kind == "lineav":
        return f"una línea vertical de {p['L']} celdas"
    if kind == "marco":
        return (f"un marco (rectángulo hueco: solo el borde, de 1 celda de grosor) "
                f"de {p['h']} filas × {p['w']} columnas")
    if kind == "cruz":
        s = 2 * p["a"] + 1
        return (f"una cruz (+) de {s}×{s}: una celda central y brazos de {p['a']} "
                f"celda{'s' if p['a'] > 1 else ''} hacia arriba, abajo, izquierda y derecha")
    raise ValueError(kind)


def normalize(cells) -> frozenset:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), max(rs), min(cs), max(cs)


# ----------------------------------------------------------- componentes
def cells_of(grid, color) -> set:
    return {(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == color}


def components(cells: set, conn=4) -> list[set]:
    nb = N4 if conn == 4 else N8
    left = set(cells)
    comps = []
    while left:
        start = left.pop()
        comp, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in nb:
                q = (r + dr, c + dc)
                if q in left:
                    left.remove(q)
                    comp.add(q)
                    stack.append(q)
        comps.append(comp)
    return comps


def colors_used(grid) -> set:
    return {v for row in grid for v in row}


# ------------------------------------------------------------- relaciones
def touches4(A, B) -> bool:
    return any((r + dr, c + dc) in B for r, c in A for dr, dc in N4)


def touches8(A, B) -> bool:
    return any((r + dr, c + dc) in B for r, c in A for dr, dc in N8)


def rel_holds(rel: str, A: set, B: set) -> bool:
    if not A or not B:
        return False
    ar0, ar1, ac0, ac1 = bbox(A)
    br0, br1, bc0, bc1 = bbox(B)
    if rel == "izquierda":
        return ac1 < bc0
    if rel == "derecha":
        return ac0 > bc1
    if rel == "arriba":
        return ar1 < br0
    if rel == "abajo":
        return ar0 > br1
    if rel == "toca":
        return touches4(A, B)
    if rel == "no_toca":
        return not touches8(A, B)
    if rel == "dentro":  # A dentro del interior de B (B es un marco)
        return all(br0 < r < br1 and bc0 < c < bc1 for r, c in A)
    raise ValueError(rel)


REL_TEXT = {
    "izquierda": "{A} debe estar completamente a la izquierda de {B} (todas las columnas de {A} menores que todas las de {B})",
    "derecha": "{A} debe estar completamente a la derecha de {B} (todas las columnas de {A} mayores que todas las de {B})",
    "arriba": "{A} debe estar completamente arriba de {B} (todas las filas de {A} menores que todas las de {B})",
    "abajo": "{A} debe estar completamente abajo de {B} (todas las filas de {A} mayores que todas las de {B})",
    "toca": "{A} debe tocar a {B} (al menos una celda de {A} vecina arriba/abajo/izquierda/derecha de una celda de {B})",
    "no_toca": "{A} no debe tocar a {B}, ni siquiera en diagonal",
    "dentro": "{A} debe quedar dentro de {B} (en su interior, sin tocar el borde de {B})",
}


# --------------------------------------------------------- formas difusas
def row_col_convex(cells) -> bool:
    """Cada fila y cada columna de la figura es un tramo contiguo (sin huecos)."""
    rows, cols = {}, {}
    for r, c in cells:
        rows.setdefault(r, []).append(c)
        cols.setdefault(c, []).append(r)
    return all(max(v) - min(v) + 1 == len(v) for d in (rows, cols) for v in d.values())


def is_round_blob(cells, min_side=3) -> bool:
    """'Redondeado': compacto, sin huecos, no rectangular, proporción ~1."""
    if not cells or len(components(cells)) != 1:
        return False
    r0, r1, c0, c1 = bbox(cells)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if min(h, w) < min_side or max(h, w) / min(h, w) > 1.5:
        return False
    fill = len(cells) / (h * w)
    return 0.55 <= fill <= 0.95 and row_col_convex(cells)


def is_filled_rect(cells) -> bool:
    if not cells:
        return False
    r0, r1, c0, c1 = bbox(cells)
    return len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1)


def is_vertical_bar(cells, max_w=2, min_h=1, max_h=99) -> bool:
    if not is_filled_rect(cells):
        return False
    r0, r1, c0, c1 = bbox(cells)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    return w <= max_w and min_h <= h <= max_h and h >= w


def is_straight_ray(cells, max_len=2) -> bool:
    """Segmento recto horizontal, vertical o diagonal de 1..max_len celdas."""
    n = len(cells)
    if n == 0 or n > max_len:
        return False
    if n == 1:
        return True
    pts = sorted(cells)
    dr, dc = pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]
    if max(abs(dr), abs(dc)) != 1:
        return False
    return all((pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) == (dr, dc)
               for i in range(n - 1))


# ---------------------------------------------------------- conceptos
# Cada checker recibe (grid, asignación rol->color, opciones) y devuelve
# una lista ordenada de (nombre_restricción, bool).

def _single(grid, color, conn=4):
    comps = components(cells_of(grid, color), conn)
    return comps[0] if len(comps) == 1 else None


def check_manzana(grid, a, opt):
    res = []
    body = _single(grid, a["cuerpo"])
    res.append(("cuerpo_redondeado", body is not None and is_round_blob(body)))
    stem = _single(grid, a["tallo"])
    # L4: 1 o 2 celdas (receta). L5: hasta la mitad de la altura del cuerpo (mínimo 2).
    max_stem = 2
    if not opt.get("explicit", True) and body:
        max_stem = max(2, (bbox(body)[1] - bbox(body)[0] + 1) // 2)
    res.append(("tallo_vertical", stem is not None and is_vertical_bar(stem, max_w=1, max_h=max_stem)))
    on_top = False
    if body and stem:
        br0 = bbox(body)[0]
        sr1, sc = bbox(stem)[1], bbox(stem)[2]
        on_top = sr1 == br0 - 1 and (br0, sc) in body
    res.append(("tallo_sobre_cuerpo", on_top))
    if opt.get("hoja") == "requerida" or (opt.get("hoja") == "opcional" and "hoja" in a
                                           and cells_of(grid, a["hoja"])):
        leaf = _single(grid, a["hoja"])
        # L4 (receta explícita): 1 a 3 celdas. L5 (implícito): hasta un tercio del cuerpo.
        max_leaf = 3 if opt.get("explicit", True) else max(3, len(body or ()) // 3)
        ok = leaf is not None and 1 <= len(leaf) <= max_leaf and stem is not None and touches8(leaf, stem)
        res.append(("hoja_junto_al_tallo", ok))
    return res


def _casa_openings(grid, body, roof_color, bg):
    """L5: aberturas (puerta/ventanas) = celdas dentro de la caja del cuerpo que no son cuerpo.
    Devuelve (ok, R, colores_de_aberturas)."""
    if not body:
        return False, None, set()
    R = bbox(body)
    if R[1] - R[0] < 1 or R[3] - R[2] < 2:
        return False, R, set()
    inside = {(r, c) for r in range(R[0], R[1] + 1) for c in range(R[2], R[3] + 1)}
    openings = inside - body
    cols = {grid[r][c] for r, c in openings}
    if bg in cols or roof_color in cols:
        return False, R, cols                 # huecos de fondo o techo dentro del cuerpo
    if len(openings) > len(inside) / 2:
        return False, R, cols
    for col in cols:
        for comp in components({p for p in openings if grid[p[0]][p[1]] == col}):
            if not is_filled_rect(comp):
                return False, R, cols
    return True, R, cols


def casa_allowed_extra(grid, a, opt, bg):
    """Colores de puerta/ventanas permitidos en L5 (todas sus celdas dentro del cuerpo)."""
    if opt.get("explicit", True) or "cuerpo" not in a:
        return set()
    ok, R, cols = _casa_openings(grid, cells_of(grid, a["cuerpo"]), a.get("techo"), bg)
    if not ok:
        return set()
    return {col for col in cols
            if all(R[0] <= r <= R[1] and R[2] <= c <= R[3] for r, c in cells_of(grid, col))}


def check_casa(grid, a, opt):
    if not opt.get("explicit", True):
        return _check_casa_l5(grid, a, opt)
    res = []
    body = cells_of(grid, a["cuerpo"])
    door_req = opt.get("puerta") == "requerida"
    door = cells_of(grid, a["puerta"]) if "puerta" in a else set()
    full = body | door
    rect_ok = (body and len(components(body)) == 1 and is_filled_rect(full)
               and (lambda b: b[1] - b[0] >= 1 and b[3] - b[2] >= 2)(bbox(full)))
    res.append(("cuerpo_rectangular", bool(rect_ok)))
    if door_req or door:
        ok = False
        if rect_ok and door and len(components(door)) == 1 and is_filled_rect(door):
            R = bbox(full)
            D = bbox(door)
            ok = (D[1] == R[1] and D[0] > R[0] and D[2] > R[2] and D[3] < R[3])
        res.append(("puerta_abajo_dentro", ok))
    roof = _single(grid, a["techo"])
    tri = False
    if roof:
        rows = {}
        for r, c in roof:
            rows.setdefault(r, []).append(c)
        rr = sorted(rows)
        runs = [(min(rows[r]), max(rows[r])) for r in rr]
        contiguous = all(max(rows[r]) - min(rows[r]) + 1 == len(rows[r]) for r in rr)
        nested = all(runs[i][0] >= runs[i + 1][0] and runs[i][1] <= runs[i + 1][1]
                     for i in range(len(runs) - 1))
        widths = [b - a_ + 1 for a_, b in runs]
        tri = (contiguous and nested and len(rr) >= 2 and widths[0] < widths[-1])
    res.append(("techo_triangular", tri))
    placed = False
    if roof and rect_ok:
        R = bbox(full)
        rb = max(r for r, _ in roof)
        base = [c for r, c in roof if r == rb]
        placed = rb == R[0] - 1 and min(base) <= R[2] and max(base) >= R[3]
    res.append(("techo_sobre_cuerpo", placed))
    return res


def _roof_ok(roof):
    rows = {}
    for r, c in roof:
        rows.setdefault(r, []).append(c)
    rr = sorted(rows)
    runs = [(min(rows[r]), max(rows[r])) for r in rr]
    contiguous = all(max(rows[r]) - min(rows[r]) + 1 == len(rows[r]) for r in rr)
    nested = all(runs[i][0] >= runs[i + 1][0] and runs[i][1] <= runs[i + 1][1] for i in range(len(runs) - 1))
    widths = [b - a_ + 1 for a_, b in runs]
    return contiguous and nested and len(rr) >= 2 and widths[0] < widths[-1]


def _check_casa_l5(grid, a, opt):
    """Casa implícita: cuerpo rectangular con aberturas opcionales (puerta, ventanas) de cualquier
    color, rectangulares, dentro del cuerpo y que no ocupen más de la mitad; techo triangular encima."""
    bg = opt.get("bg")
    body = cells_of(grid, a["cuerpo"])
    ok, R, _ = _casa_openings(grid, body, a.get("techo"), bg)
    res = [("cuerpo_rectangular", ok)]
    roof = _single(grid, a["techo"])
    res.append(("techo_triangular", bool(roof) and _roof_ok(roof)))
    placed = False
    if roof and ok:
        rb = max(r for r, _ in roof)
        base = [c for r, c in roof if r == rb]
        placed = rb == R[0] - 1 and min(base) <= R[2] and max(base) >= R[3]
    res.append(("techo_sobre_cuerpo", placed))
    return res


def check_arbol(grid, a, opt):
    res = []
    crown = _single(grid, a["copa"])
    res.append(("copa_redondeada", crown is not None and is_round_blob(crown)))
    trunk = _single(grid, a["tronco"])
    res.append(("tronco_vertical", trunk is not None and is_vertical_bar(trunk, max_w=2, min_h=2)))
    ok = False
    if crown and trunk:
        cr0, cr1, cc0, cc1 = bbox(crown)
        tr0, tr1, tc0, tc1 = bbox(trunk)
        ok = (tr0 == cr1 + 1 and all((cr1, c) in crown for c in range(tc0, tc1 + 1))
              and (cc1 - cc0) > (tc1 - tc0))
    res.append(("copa_sobre_tronco", ok))
    if opt.get("suelo"):
        n = len(grid)
        res.append(("tronco_hasta_el_suelo", bool(trunk) and bbox(trunk)[1] == n - 1))
    return res


def check_semaforo(grid, a, opt):
    if not opt.get("explicit", True):
        return _check_semaforo_l5(grid, a, opt)
    res = []
    body = cells_of(grid, a["carcasa"])
    lights = {k: cells_of(grid, a[k]) for k in ("luz_roja", "luz_amarilla", "luz_verde")}
    full = body.union(*lights.values())
    rect_ok = False
    if body and len(components(body)) == 1 and is_filled_rect(full):
        R = bbox(full)
        rect_ok = (R[1] - R[0]) > (R[3] - R[2])
    res.append(("carcasa_vertical", rect_ok))
    lights_ok = True
    for k, L in lights.items():
        ok = bool(L) and len(components(L)) == 1 and is_filled_rect(L) and len(L) <= 4
        if ok and rect_ok:
            R = bbox(full)
            ok = all(R[0] < r < R[1] and R[2] < c < R[3] for r, c in L)
        lights_ok &= ok
    res.append(("tres_luces_dentro", lights_ok))
    order = (lights_ok and rel_holds("arriba", lights["luz_roja"], lights["luz_amarilla"])
             and rel_holds("arriba", lights["luz_amarilla"], lights["luz_verde"]))
    res.append(("orden_rojo_amarillo_verde", order))
    aligned = False
    if lights_ok:
        spans = [bbox(L)[2:] for L in lights.values()]
        aligned = max(s[0] for s in spans) <= min(s[1] for s in spans)
    res.append(("luces_alineadas", aligned))
    return res


def _check_semaforo_l5(grid, a, opt):
    """Semáforo implícito: vertical u horizontal.
    - carcasa (con las luces) compacta y sin huecos, alargada en un eje (esquinas recortadas permitidas)
    - cada luz cuadrada o redondeada y encerrada por la carcasa en el eje corto
      (vertical: a izquierda y derecha; horizontal: arriba y abajo); puede tocar los extremos
    - vertical: rojo arriba, amarillo al medio, verde abajo
      horizontal: amarillo al medio y rojo/verde en los extremos (el lado del rojo varía según el país)"""
    res = []
    body = cells_of(grid, a["carcasa"])
    lights = {k: cells_of(grid, a[k]) for k in ("luz_roja", "luz_amarilla", "luz_verde")}
    full = body.union(*lights.values())
    orient, shape_ok = None, False
    if body and len(components(full)) == 1 and row_col_convex(full):
        R = bbox(full)
        h, w = R[1] - R[0] + 1, R[3] - R[2] + 1
        if len(full) >= 0.7 * h * w and min(h, w) >= 3 and h != w:
            orient, shape_ok = ("vertical" if h > w else "horizontal"), True
    res.append(("carcasa_alargada", shape_ok))
    n = len(grid)
    lights_ok = True
    for L in lights.values():
        ok = (bool(L) and len(components(L)) == 1 and len(L) <= 16
              and (is_filled_rect(L) or is_round_blob(L)))
        if ok and shape_ok:
            if orient == "vertical":
                ok = all(any((r, cc) in body for cc in range(c)) and
                         any((r, cc) in body for cc in range(c + 1, n)) for r, c in L)
            else:
                ok = all(any((rr, c) in body for rr in range(r)) and
                         any((rr, c) in body for rr in range(r + 1, n)) for r, c in L)
        lights_ok &= ok
    res.append(("tres_luces_dentro", lights_ok))
    Rl, Yl, Gl = lights["luz_roja"], lights["luz_amarilla"], lights["luz_verde"]
    order = aligned = False
    if lights_ok and shape_ok:
        if orient == "vertical":
            order = rel_holds("arriba", Rl, Yl) and rel_holds("arriba", Yl, Gl)
            spans = [bbox(L)[2:] for L in lights.values()]
        else:
            order = ((rel_holds("izquierda", Rl, Yl) and rel_holds("izquierda", Yl, Gl)) or
                     (rel_holds("derecha", Rl, Yl) and rel_holds("derecha", Yl, Gl)))
            spans = [bbox(L)[:2] for L in lights.values()]
        aligned = max(s[0] for s in spans) <= min(s[1] for s in spans)
    res.append(("orden_rojo_amarillo_verde", order))
    res.append(("luces_alineadas", aligned))
    return res


def check_sol(grid, a, opt):
    res = []
    core = _single(grid, a["nucleo"])
    res.append(("nucleo_redondeado", core is not None and is_round_blob(core)))
    rays = components(cells_of(grid, a["rayos"]), conn=8)
    ok_rays = [R for R in rays if is_straight_ray(R) and core is not None and touches8(R, core)]
    res.append(("rayos_rectos_tocando", len(rays) > 0 and len(ok_rays) == len(rays)))
    res.append((f"exactamente_{opt['k']}_rayos", len(rays) == opt["k"]))
    return res


def semaforo_allowed_extra(grid, a, opt, bg):
    """L5: un poste opcional = barra vertical (≤2 columnas) pegada bajo la carcasa y dentro de su ancho."""
    if opt.get("explicit", True) or "carcasa" not in a:
        return set()
    used = set(a.values())
    full = set().union(*(cells_of(grid, a[k]) for k in ("carcasa", "luz_roja", "luz_amarilla", "luz_verde")))
    if not full:
        return set()
    r0, r1, c0, c1 = bbox(full)
    allowed = set()
    for col in colors_used(grid) - used - {bg}:
        P = cells_of(grid, col)
        # rectángulo de hasta 2 columnas (puede ser un trozo corto si la grilla se acaba)
        if len(components(P)) == 1 and is_filled_rect(P) and bbox(P)[3] - bbox(P)[2] + 1 <= 2:
            pr0, pr1, pc0, pc1 = bbox(P)
            if pr0 == r1 + 1 and c0 <= pc0 and pc1 <= c1:
                allowed.add(col)
    return allowed


ALLOWED_EXTRA = {"casa": casa_allowed_extra, "semaforo": semaforo_allowed_extra}

CONCEPTS = {
    "manzana": check_manzana,
    "casa": check_casa,
    "arbol": check_arbol,
    "semaforo": check_semaforo,
    "sol": check_sol,
}


def best_concept_assignment(grid, concept, roles: dict, optional_roles: dict, opt, bg):
    """Prueba todas las asignaciones rol->color permitidas (colores distintos entre sí,
    distintos del fondo y presentes en la grilla) y devuelve el mejor resultado.
    roles: rol -> lista de colores permitidos (obligatorios)
    optional_roles: rol -> lista de colores permitidos (pueden no aparecer)."""
    present = colors_used(grid) - {bg}
    req = list(roles)
    opt_names = list(optional_roles)
    best = None
    req_choices = [[c for c in roles[k] if c in present] or [roles[k][0]] for k in req]
    for combo in product(*req_choices):
        if len(set(combo)) != len(combo) or bg in combo:
            continue
        a = dict(zip(req, combo))
        opt_choices = [[None] + [c for c in optional_roles[k] if c in present and c not in combo]
                       for k in opt_names]
        for ocombo in product(*opt_choices):
            if len([c for c in ocombo if c is not None]) != len({c for c in ocombo if c is not None}):
                continue
            aa = dict(a)
            for k, c in zip(opt_names, ocombo):
                if c is not None:
                    aa[k] = c
            checks = CONCEPTS[concept](grid, aa, opt)
            allowed = ALLOWED_EXTRA[concept](grid, aa, opt, bg) if concept in ALLOWED_EXTRA else set()
            extra = present - set(aa.values()) - allowed
            checks = checks + [("sin_colores_extra", not extra)]
            score = sum(ok for _, ok in checks) / len(checks)
            key = (all(ok for _, ok in checks), score)
            if best is None or key > best[0]:
                best = (key, checks, aa)
    return best
