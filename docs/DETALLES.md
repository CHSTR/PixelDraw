# PixelDraw — detalles técnicos

Este documento explica las decisiones de diseño del entorno y por qué se tomaron. Para instalar y usar el entorno, ver el [README](../README.md).

## Niveles

| Nivel | Qué pide | Respuesta | Qué verifica |
|---|---|---|---|
| L0 | Relleno: mitades, franjas, borde | Única | Igualdad por región de color |
| L1 | Una forma en una posición absoluta: punto, rectángulo, línea, marco, cruz | Única | Igualdad por región de color |
| L2 | 2–3 formas con relaciones: izquierda/derecha/arriba/abajo, toca, no toca, dentro de un marco | Abierta | Cada color forma exactamente su figura y cada relación se cumple |
| L3 | Restricciones globales: k copias que no se tocan, figura simétrica de n celdas, conteos de dos colores sin vecindad | Abierta | Conteos, conectividad, simetría, vecindad |
| L4 | Concepto con receta explícita (manzana, casa, árbol, semáforo; sol solo en test_ood), con colores dados y región opcional | Abierta | Gramática del concepto: formas y relaciones entre sus partes |
| L5 | Concepto implícito ("dibuja un árbol reconocible") | Abierta | La misma gramática, más permisiva, probando todas las asignaciones de colores posibles a las partes |

El tamaño de la grilla (6, 8, 12; 10 y 16 en test_ood) es un segundo eje de dificultad, independiente del nivel.

## Splits

| Split | Total | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 6.139 | 1.200 | 1.200 | 1.200 | 1.200 | 1.200 | 139 |
| eval | 738 | 141 | 150 | 150 | 145 | 144 | 8 |
| test | 1.471 | 273 | 297 | 300 | 291 | 295 | 15 |
| test_ood | 1.450 | 200 | 200 | 200 | 200 | 490 | 160 |

- **Distribución.** train, eval y test usan grillas de 6, 8 y 12. `test_ood` usa grillas de 10 y 16, y además el concepto *sol*, que no aparece en los otros splits.
- **Sin fugas.** La deduplicación usa el enunciado sin el ejemplo de formato (`core_id`), así que ninguna tarea aparece en dos splits ni siquiera con otro ejemplo. Antes de esta corrección, 61 tareas de test compartían el enunciado con train.
- **Limpieza.** Durante el desarrollo, el verificador se ajustó mirando las respuestas a dos conjuntos de calibración de 540 tareas, que compartían enunciados con los splits. Se quitaron esas tareas de eval (29), test (65) y test_ood (10); train no se tocó. `data/limpieza.json` lista cada tarea quitada.
- **L5 es chico por construcción.** La consigna implícita solo varía en concepto, fondo, región y tamaño, así que hay 162 tareas de L5 entre train, eval y test. Cuando un nivel tiene menos enunciados que los pedidos, el conjunto se reparte en proporción entre train, eval y test, en vez de agotarse en train.
- **Corrección posterior.** Una tarea de test_ood (`L5-16-ffa875243c`) tenía un ejemplo de formato que, después de flexibilizar el tallo de la manzana en L5, obtenía 0,15 de crédito parcial. Se le cambió el ejemplo y pasó a ser `L5-16-21b5689b56`.

## Formato de respuesta

El prompt pide el formato disperso, pero el verificador también acepta la grilla densa (`python -m pixeldraw.generate --format grid` genera prompts que la piden).

**Disperso.** Se declara el fondo y se listan solo las celdas pintadas, con tramos opcionales: `(f,c1-c2)` es un tramo de fila, `(f1-f2,c)` uno de columna y `(f1-f2,c1-c2)` un bloque. También se acepta la notación de esquina a esquina `(f1,c1)-(f2,c2)`.

```
<answer>
fondo: blanco
rojo: (2,3) (2,4) (3,3-5)
azul: (5-6,1)
</answer>
```

**Por qué no la grilla densa.** Con la grilla densa, Qwen3-1.7B sin razonamiento respondió la grilla en blanco en el 57% de las muestras, incluso en tareas con figuras, y quedó en ~0% en todos los niveles: copiaba la forma de la respuesta e ignoraba el contenido. En el formato disperso no dibujar nada queda explícito, y el modelo razona en coordenadas en vez de llenar una matriz.

**Ejemplo concreto en cada prompt.** Cada prompt trae un ejemplo del formato con otro dibujo, otros colores y otras coordenadas, en vez de una plantilla con marcadores. Con `<color>: (f,c)…`, el 1.7B copiaba los marcadores literalmente. El generador garantiza que copiar el ejemplo da recompensa 0, binaria y parcial, y la batería de tests lo comprueba en todas las tareas.

**Estricto donde importa.** Exactamente una línea `fondo:`, solo líneas `color: (f,c)…`, coordenadas dentro de 1..N y rangos crecientes; cualquier otra cosa invalida la respuesta. Si una celda aparece dos veces, vale la última línea.

## Verificador

- **Parser.** Si el bloque tiene una línea `fondo:`, usa el formato disperso. Si no, toma el último `<answer>`; si no hay, el último bloque de código; si tampoco, el último tramo de filas de grilla.
  - Acepta dígitos separados por espacios o comas, dígitos contiguos, JSON (lista o `{"grid": ...}`), nombres de colores, etiquetas "Fila k:", tablas Markdown y corchetes por fila.
  - Ignora el contenido de `<think>`; si el `<think>` quedó sin cerrar, la recompensa es 0.
  - Exige exactamente N×N.
- **Restricciones.** Todo es geométrico: componentes 4-conexas, igualdad exacta con la plantilla de la forma, cajas contenedoras para las relaciones, vecindad 4 u 8, y filas y columnas contiguas para "redondeado".
- **Crédito parcial.** `reward_partial` vale 1.0 si la tarea está resuelta. Si no:

  0,9 · max(0, (s − b)/(1 − b)) · (0,5 + 0,5·h)

  - s es la fracción de restricciones **sustantivas** cumplidas (formas, relaciones, conteos).
  - b es el mejor puntaje sustantivo de una grilla de un solo color.
  - h es la fracción de restricciones de higiene cumplidas (colores permitidos, región).

  Así, una salida trivial no recibe señal, la higiene descuenta pero no anula el crédito, y una respuesta no resuelta nunca alcanza a una resuelta (tope 0,9). Cada pieza responde a un problema encontrado:
  - una grilla uniforme sacaba 0,8 en simetría;
  - cualquier dibujo de dos colores sacaba 0,2 en la casa de L5;
  - una respuesta con colores extra podía igualar a una solución.

## Tests

`python -m tests.verifier_suite data/eval.jsonl` sobre las 738 tareas de eval:

| Categoría | Casos | Fallas |
|---|---:|---:|
| Positivos (referencias) | 738 | 0 |
| Formatos equivalentes (15 densos + 16 dispersos por tarea: tramos, esquinas, femenino, comentarios, colores vacíos, sin etiquetas…) | 22.878 | 0 |
| Soluciones alternativas válidas (traslaciones, colores alternativos en L5) | 236 | 0 |
| Mutaciones que deben fallar | 1.895 | 0 |
| Adversariales (vacío, ruido, uniforme, tamaño incorrecto, dos grillas, solución solo en `<think>`, inyección "1.0", coordenadas fuera de rango, dos fondos, texto extra, color inventado, copiar el ejemplo del prompt, etc.) | 27.306 | 0 |
| Interfaz `reward(prompt, completion)` | 212 | 0 |
| Casos reales: dibujos de DeepSeek revisados a mano, con su veredicto | 24 | 0 |
| El ejemplo del prompt da crédito parcial 0 | 738 | 0 |
| Ninguna respuesta no resuelta supera 0,9 de crédito parcial | 29.201 | 0 |
| **Total** | **83.228** | **0** |

Sobre `data/test_ood.jsonl` (1.450 tareas) pasan los 163.954 casos. `python -m pytest` corre la misma batería sobre una muestra de eval, más pruebas unitarias de `probe_opencode`.

**Falsos negativos encontrados con respuestas reales del 1.7B y corregidos:**
- colores en femenino ("roja"), que dejaban como error de formato el 44% de las respuestas de L2;
- líneas de color vacías (`negro: -`), que rechazaban respuestas correctas de "pinta todo de un color";
- comentarios al final de línea;
- la notación de esquina a esquina `(f1,c1)-(f2,c2)`.

Cada uno quedó cubierto por un test de formato equivalente.

## Medición

`pixeldraw.probe` funciona con cualquier API compatible con OpenAI:

- `--no-think` agrega `/no_think` al prompt (Qwen3). Si el servidor no lo respeta, usa `--extra '{"chat_template_kwargs":{"enable_thinking":false}}'`.
- `--workers` debería coincidir con los slots del servidor (`-np` en llama.cpp).
- Con una API en la nube, la clave va en `PD_API_KEY`. Los errores del servidor (429, timeouts) se reintentan con espera exponencial (`--retries`). Si alguno igual falla, queda fuera del resumen, y basta relanzar el mismo comando para reintentarlo.
- `--per-cell N` toma N tareas por celda nivel × tamaño. Con la misma `--seed`, `probe` y `probe_opencode` eligen exactamente las mismas tareas.

**Cómo leer la calibración.** La zona útil para GRPO es la de tareas resueltas "a veces":
- si un nivel queda casi siempre en "nunca", conviene bajar el tamaño de la grilla o la cantidad de objetos y relaciones;
- si queda en "siempre", conviene subirlos;
- si la mayoría de los fallos son de formato, el problema es la escritura de la grilla y no el dibujo.

**OpenCode.** Algunos modelos (como `opencode-go/deepseek-v4.1-flash`) solo están disponibles dentro de OpenCode. `pixeldraw.probe_opencode` llama a `opencode run --format json` una vez por muestra, con un agente propio:
- todas las herramientas denegadas;
- temperatura fija;
- un prompt de sistema que pide un **plan breve** (`<plan>`, 1–3 líneas sobre qué celdas ocupa cada parte) antes de `<answer>`.

Así las trazas son cortas (~130–230 tokens) y enseñan una estrategia verificable. Cada fila guarda una huella del prompt de sistema (`prompt_sha256`), para no mezclar corridas con prompts distintos al retomar. OpenCode puede añadir contexto propio al prompt, así que esta vía es algo menos controlada que una API directa.

## Galerías

`pixeldraw.render` arma una página HTML autocontenida:
- una tabla de pass@1 por modelo y nivel;
- por cada tarea: el enunciado, el dibujo de referencia y las respuestas de cada modelo dibujadas, con su veredicto, las restricciones falladas y el texto de la respuesta.

```bash
python -m pixeldraw.render --tasks data/eval.jsonl --results "results/eval/*.jsonl" --out galeria.html
python -m pixeldraw.render --tasks data/train.jsonl --results "results/train/*.jsonl" \
    --levels L4 L5 --pick discrepan --png-dir figuras/ --out galeria_L4L5.html
python -m pixeldraw.render --tasks data/train.jsonl --sft derived/sft.jsonl --out galeria_sft.html
```

Opciones de selección:
- `--pick todas|fallidas|resueltas|discrepan`;
- `--levels`, `--families` y `--task-ids`;
- `--per-group`: tareas por nivel, familia y tamaño;
- `--samples`: muestras por modelo;
- `--px`: tamaño de celda.

La opción `--pick resueltas` sirve para auditar a mano una muestra de aciertos.

## Datos derivados

- **SFT** (`derived/sft.jsonl`, formato `messages`):
  - solo respuestas resueltas del profesor (`--sft-models`), recalificadas con el verificador actual;
  - sin truncadas y sin duplicados;
  - hasta 2 por tarea, las más cortas.
- **DPO** (`derived/dpo.jsonl`, `prompt` / `chosen` / `rejected`):
  - un par para cada tarea con al menos una respuesta correcta y una incorrecta;
  - como rechazo se prefieren las "casi correctas" (grilla legible y mayor crédito parcial), que son las comparaciones más informativas;
  - los rechazos del modelo objetivo hacen que los pares sean *on-policy* respecto del modelo base.
- Solo se usan tareas de train (`--allow-non-train` lo desactiva).
- `derived/stats.json` resume por nivel cuántas tareas tuvieron algún acierto y cuántas trazas y pares salieron.

| Nivel | Tareas | Con acierto del profesor | SFT | DPO |
|---|---:|---:|---:|---:|
| L0 | 30 | 30 | 60 | 21 |
| L1 | 30 | 30 | 59 | 13 |
| L2 | 120 | 120 | 238 | 100 |
| L3 | 120 | 120 | 239 | 82 |
| L4 | 120 | 120 | 240 | 180 |
| L5 | 115 | 91 | 165 | 178 |
| **Total** | | | **1.001** | **574** |

**Advertencia para DPO.** Las respuestas preferidas (DeepSeek) traen `<plan>` y las rechazadas (Qwen3-1.7B) no, porque el razonamiento del 1.7B no se incluye. Un entrenamiento DPO podría aprender a escribir un plan en vez de a dibujar mejor. Si eso importa, conviene quitar el `<plan>` de `chosen` o regenerar los rechazos con un modelo ya ajustado con SFT, que también escribe planes.

## Convenciones y limitaciones del verificador

- **"Redondeado"** exige esquinas recortadas, alto y ancho de al menos 3 y un relleno entre 55% y 95%.
- **L4 vs. L5.** En L4 la regla va en el prompt y se aplica estricta. En L5 (implícito), las gramáticas son más permisivas. Se ajustaron a partir de dibujos reales de DeepSeek que el verificador rechazaba sin razón:
  - el semáforo puede ser vertical u horizontal; en horizontal, el amarillo va al medio y el rojo en cualquier extremo, porque varía según el país;
  - la carcasa del semáforo puede tener las esquinas recortadas o estar abierta en los extremos, siempre que cada luz quede encerrada en el eje corto;
  - las luces pueden ser cuadradas o redondeadas, de hasta 16 celdas;
  - se admite un poste opcional: un rectángulo de hasta 2 columnas pegado bajo la carcasa;
  - el tallo de la manzana puede medir hasta la mitad de la altura del cuerpo, y la hoja hasta un tercio del cuerpo;
  - la casa admite aberturas (puerta y ventanas) de cualquier color, rectangulares, dentro del cuerpo y que no ocupen más de la mitad.
- **Lo que sigue rechazándose** son dibujos que no se leen como el objeto: una manzana de proporción 2:1, un cuerpo cuadrado con protuberancias, luces que ocupan todo el ancho del semáforo o que se superponen.
- **L0 y L1 tienen respuesta única.** Sirven de ancla para medir el control de la grilla, más que como tareas abiertas.
