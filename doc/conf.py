import os
import sys
import sphinx_gallery
from sphinx_gallery.sorting import ExplicitOrder
import sphinx_rtd_theme

# If extensions (or modules to document with autodoc) are in another
# directory, add these directories to sys.path here. If the directory
# is relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
sys.path.insert(0, os.path.abspath('_ext'))

from github_link import make_linkcode_resolve

# Project metadata
project = 'pulse2percept'
copyright = '2016 - 2025, pulse2percept developers (BSD License)'

# Retrieve version from package
from pulse2percept import __version__
version = release = __version__

# Sphinx extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.linkcode',
    'sphinx.ext.todo',
    # 'sphinx_gallery.gen_gallery',
    'versionwarning.extension',
    'IPython.sphinxext.ipython_directive',
    'IPython.sphinxext.ipython_console_highlighting',
    'sphinx.ext.extlinks', 
]


# Autosummary & Napoleon settings
autosummary_generate = True
autodoc_default_options = {
    'members': None,
    'member-order': 'bysource',
    'inherited-members': None,
    "imported-members": False,  # Don't document functions that were imported
}
autodoc_preserve_defaults = True
napoleon_google_docstring = False  # force consistency
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

# Math rendering
# For maths, use mathjax by default and svg if NO_MATHJAX env variable is set
# (useful for viewing the doc offline)
if os.environ.get('NO_MATHJAX'):
    extensions.append('sphinx.ext.imgmath')
    imgmath_image_format = 'svg'
else:
    extensions.append('sphinx.ext.mathjax')
    mathjax_path = ('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/'
                    'MathJax.js?config=TeX-AMS_SVG')

# Source files and paths
templates_path = ['_templates']
exclude_patterns = ['_build', '**/.ipynb_checkpoints']
source_suffix = '.rst'
master_doc = 'index'

# HTML output
html_theme = 'sphinx_rtd_theme'  # Switch to 'furo' if preferred
html_static_path = ['_static']
html_css_files = ['css/custom.css']
html_last_updated_fmt = '%b %d, %Y'

# InterSphinx mapping
intersphinx_mapping = {'python': ('https://docs.python.org/3', None)}

# Gallery configuration
sphinx_gallery_conf = {
    'examples_dirs': ['../examples'],
    'gallery_dirs': ['examples'],
    'backreferences_dir': '_api',
    'doc_module': 'pulse2percept',
    'reference_url': {'pulse2percept': None},
    'remove_config_comments': True,
    'thumbnail_size': (320, 224),
    'subsection_order': ExplicitOrder([
        '../examples/implants',
        '../examples/stimuli',
        '../examples/models',
        '../examples/datasets',
        '../examples/cortex',
        '../examples/developers'
    ])
}

# GitHub link code resolver
linkcode_resolve = make_linkcode_resolve(
    'pulse2percept',
    'https://github.com/pulse2percept/pulse2percept/blob/{revision}/{package}/{path}#L{lineno}'
)

extlinks = {
    'pull': ('https://github.com/pulse2percept/pulse2percept/pull/%s', 'PR #%s'),
    'issue': ('https://github.com/pulse2percept/pulse2percept/issues/%s', 'Issue #%s'),
    'commit': ('https://github.com/pulse2percept/pulse2percept/commit/%s', 'Commit %s'),
}

