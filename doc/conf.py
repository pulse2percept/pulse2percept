# -- stdlib / setup -----------------------------------------------------------
import os
import sys
import types

# Put _ext on path (for github_link.py)
sys.path.insert(0, os.path.abspath('_ext'))

# Use a non-interactive backend for any plotting during doc builds
import matplotlib
matplotlib.use('Agg')

# -- RTD-only: stub compiled extensions so imports don't fail -----------------
def _stub_module(qualname: str, **attrs):
    """Create a minimal module and register in sys.modules."""
    m = types.ModuleType(qualname)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[qualname] = m

def _stub_func(return_value=None):
    def _f(*args, **kwargs):
        return return_value
    return _f

if os.environ.get("READTHEDOCS") == "True":
    # Utils / stimuli Cython bits
    _stub_module("pulse2percept.utils._fast_array",
                 fast_is_strictly_increasing=_stub_func(True))
    _stub_module("pulse2percept.stimuli._base",
                 fast_compress_space=_stub_func(None),
                 fast_compress_time=_stub_func(None))

    # Models Cython bits seen in your logs
    _stub_module("pulse2percept.models._temporal", fading_fast=_stub_func(None))
    _stub_module("pulse2percept.models._beyeler2019",
                 fast_scoreboard=_stub_func(None),
                 fast_axon_map=_stub_func(None),
                 fast_jansonius=_stub_func(None),
                 fast_find_closest_axon=_stub_func(None))
    _stub_module("pulse2percept.models._horsager2009",
                 temporal_fast=_stub_func(None))
    _stub_module("pulse2percept.models._nanduri2012",
                 spatial_fast=_stub_func(None),
                 temporal_fast=_stub_func(None))
    _stub_module("pulse2percept.models._granley2021",
                 fast_biphasic_axon_map=_stub_func(None))

# -- Sphinx basics ------------------------------------------------------------
import sphinx_gallery
from sphinx_gallery.sorting import ExplicitOrder
import sphinx_rtd_theme
from github_link import make_linkcode_resolve

# Project metadata
project = 'pulse2percept'
copyright = '2016 - 2025, pulse2percept developers (BSD License)'

# Retrieve version from package (be tolerant on RTD)
try:
    from pulse2percept import __version__
    version = release = __version__
except Exception:
    # Fallback so Sphinx doesn't die if import chain still fails
    version = release = os.environ.get("P2P_DOC_VERSION", "0.0.dev")

# Sphinx extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.linkcode',
    'sphinx.ext.todo',
    'sphinx_gallery.gen_gallery',
    'versionwarning.extension',
    'IPython.sphinxext.ipython_directive',
    'IPython.sphinxext.ipython_console_highlighting',
    'sphinx.ext.extlinks',
    'sphinx.ext.mathjax',
]

# Autodoc
autosummary_generate = True
autodoc_default_options = {
    'members': None,
    'member-order': 'bysource',
    'inherited-members': None,
}

# Napoleon
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = False
todo_include_todos = True

# MathJax
mathjax_path = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_SVG'

# Source files and paths
templates_path = ['_templates']
exclude_patterns = ['_build', '**/.ipynb_checkpoints']
source_suffix = '.rst'
master_doc = 'index'

# HTML output
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['css/custom.css']
html_last_updated_fmt = '%b %d, %Y'

# InterSphinx
intersphinx_mapping = {'python': ('https://docs.python.org/3', None)}

# Gallery configuration
sphinx_gallery_conf = {
    'examples_dirs': ['../examples'],
    'gallery_dirs': ['examples'],
    'reference_url': {'pulse2percept': None},
    'thumbnail_size': (320, 224),
    'remove_config_comments': True,
    'subsection_order': ExplicitOrder([
        '../examples/implants',
        '../examples/stimuli',
        '../examples/models',
        '../examples/datasets',
        '../examples/cortex',
        '../examples/developers',
    ]),
}

# GitHub linkcode resolver
linkcode_resolve = make_linkcode_resolve(
    'pulse2percept',
    'https://github.com/pulse2percept/pulse2percept/blob/{revision}/{package}/{path}#L{lineno}'
)

# External links
extlinks = {
    'pull':   ('https://github.com/pulse2percept/pulse2percept/pull/%s', 'PR #%s'),
    'issue':  ('https://github.com/pulse2percept/pulse2percept/issues/%s', 'Issue #%s'),
    'commit': ('https://github.com/pulse2percept/pulse2percept/commit/%s', 'Commit %s'),
}
