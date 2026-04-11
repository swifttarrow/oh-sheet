"""Shared score fixtures for engrave quality tests.

Re-exports :func:`load_score_fixture` for convenience::

    from tests.fixtures import load_score_fixture

    score = load_score_fixture("c_major_scale")
"""
from tests.fixtures._builders import FIXTURE_NAMES, load_score_fixture

__all__ = ["FIXTURE_NAMES", "load_score_fixture"]
