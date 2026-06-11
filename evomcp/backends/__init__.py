"""Backend adapters for evomcp.

Today: mutation LM backends. Tomorrow: evaluator executors.
"""
from evomcp.backends.mutation import build_mutation_lm

__all__ = ["build_mutation_lm"]
