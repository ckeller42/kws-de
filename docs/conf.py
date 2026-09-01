"""Sphinx configuration for kws-de (mirrors the open-california docs convention)."""

project = "kws-de"
author = "kws-de"

extensions = [
    "myst_parser",             # markdown pages on the site
    "sphinxcontrib.mermaid",   # ```mermaid fences
    "sphinx_likec4",           # .. likec4-view:: <id> — architecture diagrams
]
likec4_source_dir = "likec4"

myst_fence_as_directive = ["mermaid"]
myst_heading_anchors = 3

html_theme = "furo"
html_title = "kws-de"

# Internal planning artifacts are not part of the doc site.
exclude_patterns = ["_build", "superpowers/**", "requirements.txt"]
