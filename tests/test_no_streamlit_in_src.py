"""src/ must never import streamlit (docs/04-APP-AND-DEPLOY-SPEC.md section 1).

If it does, the model stops being callable from the notebook, the eval harness
and the tests -- quietly breaking the brief's "don't leave it stuck in a
notebook" requirement. A few lines to protect a graded property.

Matches real import statements rather than the substring, so a docstring
mentioning Streamlit does not trip it and an aliased import does not slip past.
"""
from __future__ import annotations

import re

from src import config

IMPORT_RE = re.compile(r"^\s*(?:import\s+streamlit|from\s+streamlit\b)", re.M)


def test_src_does_not_import_streamlit():
    offenders = [
        str(path.relative_to(config.PROJECT_ROOT))
        for path in (config.PROJECT_ROOT / "src").rglob("*.py")
        if IMPORT_RE.search(path.read_text())
    ]
    assert not offenders, f"src/ must not depend on streamlit: {offenders}"
