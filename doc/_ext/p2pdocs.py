import os
from docutils.parsers.rst.roles import set_classes
from docutils import nodes


def run_apidoc(_):
    ignore_paths = [
        os.path.join('..', '..', 'pulse2percept', '*', 'tests')
    ]

    argv = [
        "-f",
        "-M",
        "-e",
        "-E",
        "-T",
        "-o", "aaapi",
        os.path.join('..', 'pulse2percept')
    ] + ignore_paths

    try:
        # Sphinx 1.7+
        from sphinx.ext import apidoc
        apidoc.main(argv)
    except ImportError:
        # Sphinx 1.6 (and earlier)
        from sphinx import apidoc
        argv.insert(0, apidoc.__file__)
        apidoc.main(argv)


def pull_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    ref = 'https://github.com/pulse2percept/pulse2percept/pull/' + text
    set_classes(options)
    node = nodes.reference(rawtext, 'PR #' + text, refuri=ref, **options)
    return [node], []


def issue_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    ref = 'https://github.com/pulse2percept/pulse2percept/issues/' + text
    set_classes(options)
    node = nodes.reference(rawtext, 'issue #' + text, refuri=ref, **options)
    return [node], []


def commit_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    ref = 'https://github.com/pulse2percept/pulse2percept/commit/' + text
    set_classes(options)
    node = nodes.reference(rawtext, 'commit ' + text, refuri=ref, **options)
    return [node], []


def setup(app):
    app.add_role('pull', pull_role)
    app.add_role('issue', issue_role)
    app.add_role('commit', commit_role)
    # app.connect('builder-inited', run_apidoc)
