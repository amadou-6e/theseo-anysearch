"""Sphinx configuration for the AnySearch documentation site."""

project = "AnySearch"
copyright = "2026, AnySearch contributors"
author = "AnySearch contributors"

extensions = [
    "myst_parser",
]

source_suffix = {
    ".md": "markdown",
}

master_doc = "index"

myst_enable_extensions = [
    "colon_fence",
]

# Suppress warnings about external/excluded document references
# These occur when MyST tries to resolve relative links to files outside the docs/
# source tree (e.g., usage/experiments/...) or to excluded files.
# The files and links themselves are valid; they simply point to resources not
# part of the Sphinx documentation build.
suppress_warnings = [
    "myst.xref_missing",
]

html_theme = "furo"
html_static_path = ["_static"]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "superpowers", "testing/ray-suite-timings-2026-08-02.md"]
