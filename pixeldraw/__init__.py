"""PixelDraw: un entorno para entrenar a un LLM a dibujar en grillas, con recompensa verificable.

Uso mínimo:

    from pixeldraw import load_tasks, reward
    tasks = load_tasks("data/train.jsonl")
    reward(tasks[0]["prompt"], completion)   # 1.0 o 0.0
"""
from .verifier import ParseError, load_tasks, parse_grid, register_tasks, reward, reward_partial, verify

__all__ = ["ParseError", "load_tasks", "parse_grid", "register_tasks", "reward", "reward_partial", "verify"]
__version__ = "1.0.0"
