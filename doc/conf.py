import os
import sys
import types
import sphinx_gallery
from sphinx_gallery.sorting import ExplicitOrder
import sphinx_rtd_theme

# Ensure _ext is importable (for github_link)
sys.path.insert(0, os.path.abspath('_ext'))
from github_link import make_linkcode_resolve

# ----------------------
# Project metadata
# ----------------------
project = 'pulse2percept'
copyright = '2016 - 2025, pulse2percept developers (BSD License)'

# Get version WITHOUT importing the package (avoids compiled imports on RTD)
try:
    from importlib.metadata import version as _dist_version
    version = release = _dist_version("pulse2percept")
except Exception:
    # Fallback for local builds / PR previews
    version = release = os.environ.get("P2P_DOC_VERSION", "0.0.dev")

# ----------------------
# Mock compiled submodules early so imports succeed in docs
# ----------------------
def _mock(modname, attrs=None):
    if modname in sys.modules:
        return
    m = types.ModuleType(modname)
    for k, v in (attrs or {}).items():
        setattr(m, k, v)
    sys.modules[modname] = m

# These are the ones tripping your build
_mock('pulse2percept.utils._fast_array', {
    'fast_is_strictly_increasing': lambda *a, **k: True,
})
_mock('pulse2percept.stimuli._base', {
    'fast_compress_space': lambda *a, **k: None,
    'fast_compress_time': lambda *a, **k: None,
})
_mock('pulse2percept.models._temporal', {
    'fading_fast': lambda *a, **k: 0.0,
})
_mock('pulse2percept.models._beyeler2019', {
    'fast_scoreboard':       lambda *a, **k: None,
    'fast_axon_map':         lambda *a, **k: None,
    'fast_jansonius':        lambda *a, **k: None,
    'fast_find_closest_axon':lambda *a, **k: (None, None),
})

# Tell autodoc to treat them as mocked as well
autodoc_mock_imports = [
    'pulse2percept.utils._fast_array',
    'pulse2percept.stimuli._base',
    'pulse2percept.models._temporal',
    'pulse2percept.models._beyeler2019',
]

# ----------------------
# Sphinx extensions
# ----------------------
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

# Autodoc / Napoleon
autosummary_generate = True
autodoc_default_options = {
    'members': None,
    'member-order': 'bysource',
    'inherited-members': None,
}
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
    # On RTD, avoid executing examples to reduce chances of runtime import/ABI issues
    'plot_gallery': False if os.environ.get('READTHEDOCS') == 'True' else True,
}

# IPython: make sure we use a non-interactive backend; pre-set any env you need
ipython_execlines = [
    "import os; os.environ.setdefault('MPLBACKEND', 'Agg')",
]

# GitHub link code resolver — wrap the partial so Sphinx sees a function
_linkcode_partial = make_linkcode_resolve(
    'pulse2percept',
    'https://github.com/pulse2percept/pulse2percept/blob/{revision}/{package}/{path}#L{lineno}',
)
def linkcode_resolve(domain, info):
    return _linkcode_partial(domain, info)

# Short external links
extlinks = {
    'pull':   ('https://github.com/pulse2percept/pulse2percept/pull/%s', 'PR #%s'),
    'issue':  ('https://github.com/pulse2percept/pulse2percept/issues/%s', 'Issue #%s'),
    'commit': ('https://github.com/pulse2percept/pulse2percept/commit/%s', 'Commit %s'),
}
