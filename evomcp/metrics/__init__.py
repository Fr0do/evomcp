"""Metric library for evomcp evaluators.

Metrics are pure functions: (gold, pred) -> (primary, secondary, failure_class).
They do not touch I/O and have no side effects, so they can be reused across
Python evaluator, Rust evaluator, and offline analysis.
"""
from evomcp.metrics.gore_captcha import score_gore_captcha

__all__ = ["score_gore_captcha"]
