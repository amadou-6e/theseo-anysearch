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

root_doc = "index"

myst_enable_extensions = [
    "colon_fence",
]

html_theme = "furo"
html_static_path = ["_static"]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "superpowers"]
