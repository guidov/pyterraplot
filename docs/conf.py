"""Sphinx configuration for pyterraplot."""
import os
import sys
from datetime import date

# Import the package from the repo root so autodoc reads the working tree.
sys.path.insert(0, os.path.abspath(".."))

project = "pyterraplot"
author = "Guido Vettoretti"
copyright = f"{date.today().year}, {author}"

try:
    from pyterraplot import __version__ as release
except Exception:          # docs must still build if the import fails
    release = "0.0.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_copybutton",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
# xarray/numpy are installed, but keep the optional extras out of the build.
autodoc_mock_imports = ["cf_xarray", "rioxarray", "fastapi", "uvicorn"]

napoleon_google_docstring = False
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}

myst_enable_extensions = ["colon_fence", "deflist", "linkify"]
myst_heading_anchors = 3

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

html_theme = "furo"
html_title = f"pyterraplot {release}"
html_static_path = []
