"""Sphinx configuration for RSN Game QA documentation."""

import os
import sys

# -- Path setup ---------------------------------------------------------------
# Add the project root to sys.path so autodoc can find src/
sys.path.insert(0, os.path.abspath(".."))

# -- Project information ------------------------------------------------------
project = "RSN Game QA"
copyright = "2026, Roboter Schlafen Nicht"
author = "Roboter Schlafen Nicht"
release = "0.1.0"

# -- General configuration ----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # Google/NumPy-style docstrings
    "sphinx.ext.viewcode",  # [source] links in API docs
    "sphinx.ext.intersphinx",  # cross-ref to Python/Gymnasium docs
    "sphinx_autodoc_typehints",  # inline type hints in signatures
    "myst_parser",  # parse .md files (specs)
]

# MyST (Markdown) settings
myst_enable_extensions = [
    "colon_fence",  # ::: directive syntax
    "deflist",  # definition lists
    "tasklist",  # - [x] checkboxes
    "fieldlist",  # :field: value
]
myst_heading_anchors = 3

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_mock_imports = [
    "cv2",
    "numpy",
    "torch",
    "gymnasium",
    "stable_baselines3",
    "ultralytics",
    "psutil",
    "win32gui",
    "win32ui",
    "win32con",
    "win32api",
    "ctypes",
    "PIL",
    "imagehash",
    "pandas",
    "polars",
    "yaml",
]

# Napoleon settings (for Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# Source file suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The master doc
master_doc = "index"

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Suppress warnings for cross-references in included spec files
# (spec .md files use relative links that don't resolve in the Sphinx tree)
suppress_warnings = ["myst.xref_missing"]

# -- Options for HTML output ---------------------------------------------------
html_theme = "furo"
html_title = "RSN Game QA"
html_static_path = []

# Furo theme options
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}
