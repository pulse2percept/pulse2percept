# -- stdlib / setup -----------------------------------------------------------
import os
import sys
import types

# Put _ext on path (for github_link.py)
sys.path.insert(0, os.path.abspath('_ext'))

# Use a non-interactive backend for any plotting during doc builds
import matplotlib
matplotlib.use('Agg')

# -- tiny helpers for stubbing modules on RTD --------------------------------
def _stub_module(qualname: str, **attrs):
    """Create a minimal module and register in sys.modules."""
    m = types.ModuleType(qualname)
    for k, v in attrs.items():
        setattr(m, v.__name__ if callable(v) else k, v)
    sys.modules[qualname] = m
    return m

def _stub_package(qualname: str):
    """Create a minimal *package* (has __path__) and register it."""
    m = types.ModuleType(qualname)
    m.__path__ = []  # mark as a package
    sys.modules[qualname] = m
    return m

def _stub_func(return_value=None):
    def _f(*args, **kwargs):
        return return_value
    return _f

ON_RTD = os.environ.get("READTHEDOCS") == "True"

if ON_RTD:
    # --- Stub JAX/JAXLIB to prevent importing heavy binary wheels on RTD ----
    import numpy as _np
    jax_pkg = _stub_package("jax")
    # common jax APIs that might be referenced (no-ops)
    jax_pkg.jit = lambda f=None, *a, **k: (f if f is not None else (lambda x: x))
    jax_pkg.grad = _stub_func(None)
    jax_pkg.device_put = _stub_func(None)
    jax_pkg.random = types.SimpleNamespace(PRNGKey=_stub_func(0), normal=_stub_func(_np.array([])))
    # jax.numpy as a separate importable module; alias to numpy for basic ops
    _stub_package("jax.numpy")
    sys.modules["jax.numpy"].__dict__.update(_np.__dict__)

    jaxlib_pkg = _stub_package("jaxlib")
    _stub_module("jaxlib.xla_client")
    _stub_module("jaxlib.xla_extension")

    # --- Stub your Cython extensions that showed up in RTD logs -------------
    _stub_module("pulse2percept.utils._fast_array",
                 fast_is_strictly_increasing=_stub_func(True))
    _stub_module("pulse2percept.stimuli._base",
                 fast_compress_space=_stub_func(None),
                 fast_compress_time=_stub_func(None))
    _stub_module("pulse2percept.models._temporal", fading_fast=_stub_func(None))
    _stub_module("pulse2percept.models._beyeler2019",
                 fast_scoreboard=_stub_func(None),
                 fast_axon_map=_stub_func(None),
                 fast_jansonius=_stub_func(None),
                 fast_find_closest_axon=_stub_func(None))
    _stub_module("pulse2percept.models._horsager2009", temporal_fast=_stub_func(None))
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

# Version: NEVER import the package on RTD (avoids JAX/NumPy ABI crashes)
if ON_RTD:
    # Prefer RTD-provided version strings if available; otherwise a safe fallback
    version = release = os.environ.get("READTHEDOCS_VERSION", "0.0.dev")
else:
    try:
        from pulse2percept import __version__
        version = release = __version__
    except Exception:
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
