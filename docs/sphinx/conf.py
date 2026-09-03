"""Sphinx config for the kws-de firmware requirements-traceability site.

Two extensions do the work:
- sphinx-needs: requirements (`req`) and tests (`test`) as linkable, tabulable
  "needs" objects (requirements.rst, tests.rst, traceability.rst).
- breathe: renders the Doxygen XML for firmware/main/*.h into api.rst.

The Doxygen XML is generated separately (`cd firmware && doxygen Doxyfile`)
and is NOT committed, so it is usually absent on a fresh checkout / this
machine. If it's missing, api.rst is dropped from the build (with a
warning) instead of failing.
"""

import os

from sphinx.util import logging as sphinx_logging

logger = sphinx_logging.getLogger(__name__)

project = "kws-de"
copyright = "kws-de contributors"
author = "kws-de contributors"

extensions = ["breathe", "sphinx_needs", "sphinx.ext.graphviz"]

exclude_patterns = ["_build"]

html_theme = "alabaster"

# -- graphviz (model-architecture diagrams, docs/sphinx/_generated/*.dot) ---
# SVGs scale cleanly at any zoom and stay text-searchable/selectable, unlike
# the default PNG output. `dot` itself must be on PATH (installed alongside
# doxygen in CI; see .github/workflows/docs.yml).
graphviz_output_format = "svg"

# -- breathe (Doxygen C API reference) --------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
_doxygen_xml_dir = os.path.join(_here, "..", "..", "firmware", "docs", "doxygen", "xml")
_doxygen_xml_present = os.path.isdir(_doxygen_xml_dir)

breathe_projects = {"kws_de_fw": _doxygen_xml_dir}
breathe_default_project = "kws_de_fw"

if not _doxygen_xml_present:
    exclude_patterns.append("api.rst")
    # index.rst's toctree still lists "api" (it must, for CI where the XML
    # IS present) — expected/harmless here, so don't let it warn.
    suppress_warnings = ["toc.excluded"]


def setup(app):
    if not _doxygen_xml_present:
        app.connect(
            "builder-inited",
            lambda app: logger.warning(
                "doxygen XML absent — API page skipped (run `cd firmware && doxygen Doxyfile`)"
            ),
        )


# -- sphinx-needs (requirements traceability) -------------------------------
needs_types = [
    dict(directive="req", title="Requirement", prefix="REQ_", color="#BFD8D2", style="node"),
    dict(directive="test", title="Test", prefix="TEST_", color="#DF744A", style="node"),
]
# All ids in this project are hand-assigned, uppercase, e.g. REQ_FW_MFCC_PARITY.
needs_id_regex = "^[A-Z][A-Z0-9_]{4,60}$"
needs_id_required = True
needs_title_optional = True
# status is free text here (implemented / verified / manual / open); no
# closed enum needed for a project this size.
needs_statuses = []
