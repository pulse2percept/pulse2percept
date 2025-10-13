# -- stdlib / setup -----------------------------------------------------------
import os
import sys
import importlib.metadata

# Make local extensions importable first (e.g., github_link.py)
sys.path.insert(0, os.path.abspath('_ext'))

# Use a non-interactive backend for any plots
import matplotlib
matplotlib.use("Agg")

ON_RTD = os.environ.get("READTHEDOCS") == "True"

# -- Sphinx config ------------------------------------------------------------
from github_link import make_linkcode_resolve
import sphinx_rtd_theme
from sphinx_gallery.sorting import ExplicitOrder

project = "pulse2percept"
copyright = "2016 - 2025, pulse2percept developers"

# Never import the package here; use package metadata if present
try:
    release = importlib.metadata.version("pulse2percept")
    version = release
except importlib.metadata.PackageNotFoundError:
    # During first phase of RTD build, the package may not yet be installed
    version = release = os.environ.get("READTHEDOCS_VERSION", "0.0.dev")

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.todo",
    "sphinx_gallery.gen_gallery",
    "versionwarning.extension",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "inherited-members": True,
}
todo_include_todos = True

source_suffix = ".rst"
master_doc = "index"
templates_path = ["_templates"]
exclude_patterns = ["_build", "**/.ipynb_checkpoints"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

# Sphinx-Gallery: keep execution on (your examples produce images),
# but never fail the build if an example hiccups.
from sphinx_gallery.sorting import ExplicitOrder
sphinx_gallery_conf = {
    "examples_dirs": ["../examples"],
    "gallery_dirs": ["examples"],
    "reference_url": {"pulse2percept": None},
    "thumbnail_size": (320, 224),
    "remove_config_comments": True,
    "subsection_order": ExplicitOrder([
        "../examples/implants",
        "../examples/stimuli",
        "../examples/models",
        "../examples/datasets",
        "../examples/developers",
    ]),
    "only_warn_on_example_error": True,
}

# linkcode_resolve must be a function (avoid partial warning)
_resolver = make_linkcode_resolve(
    "pulse2percept",
    "https://github.com/pulse2percept/pulse2percept/blob/{revision}/{package}/{path}#L{lineno}",
)
def linkcode_resolve(domain, info):
    return _resolver(domain, info)

extlinks = {
    "pull":   ("https://github.com/pulse2percept/pulse2percept/pull/%s", "PR #%s"),
    "issue":  ("https://github.com/pulse2percept/pulse2percept/issues/%s", "Issue #%s"),
    "commit": ("https://github.com/pulse2percept/pulse2percept/commit/%s", "Commit %s"),
}

# MathJax (older URL that loads reliably on RTD)
mathjax_path = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_SVG"
