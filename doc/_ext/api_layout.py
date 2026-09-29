"""Assign every public API object one page, for the autosummary templates.

An object's page lives at the shallowest package that lists it in ``__all__``
(``pulse2percept.implants.retina.ArgusII``, not ``...retina.argus.ArgusII``).
Objects no ``__all__`` lists stay on their defining module's page. A plain
module gets a page only if some object lives there; subpackages always do.

``CURATED`` modules list their objects and subpackages in hand-written
autosummary tables in their docstring; those names are taken as given and get
no generated tables.
"""
import importlib
import inspect
import pkgutil
import re
import warnings

ROOT = 'pulse2percept'
CURATED = tuple(f'{ROOT}.{m}' for m in (
    'datasets', 'implants', 'implants.cortex', 'implants.retina', 'models',
    'models.cortex', 'models.retina', 'percepts', 'plotting', 'stimuli',
    'topography', 'topography.cortex', 'topography.retina', 'units', 'utils',
    'vision'))
SKIP = re.compile(r'\.(tests|conftest)(\.|$)|\._|\.utils\.testing$')
KINDS = (('class', 'Classes'), ('function', 'Functions'),
         ('exception', 'Exceptions'))


def _kind(obj):
    if inspect.isclass(obj):
        return 'exception' if issubclass(obj, BaseException) else 'class'
    if inspect.isroutine(obj) or callable(obj):
        return 'function'
    return 'data'


def _curated_names(mod):
    """Return the object names listed in a module docstring's autosummaries"""
    names = []
    for block in re.findall(r'\.\. autosummary::\n(?:[ \t]+:.*\n)*\n'
                            r'((?:[ \t]+\S.*\n)+)', mod.__doc__ or ''):
        names += [line.strip().lstrip('~') for line in block.splitlines()]
    return names


def build_layout():
    """Return ``{module: {'modules', 'tables', 'data'}}`` for the templates"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        root = importlib.import_module(ROOT)
        mods = {ROOT: root}
        for info in pkgutil.walk_packages(root.__path__, ROOT + '.'):
            if not SKIP.search(info.name):
                mods[info.name] = importlib.import_module(info.name)

    home = {}  # id(obj) -> (module, name, kind)

    def claim(obj, modname, name, curated=False):
        key = id(obj)
        src = getattr(obj, '__module__', '') or ''

        def rank(mod, alias):
            # Shallower first, then the real name over a deprecated alias
            # (``ProsthesisSystem`` is ``Implant``), then the source package
            return (mod.count('.'), alias != getattr(obj, '__name__', alias),
                    not src.startswith(mod))

        if key in home and not curated:
            if rank(modname, name) >= rank(*home[key][:2]):
                return
        home[key] = (modname, name, _kind(obj))

    for modname, mod in mods.items():
        for name in getattr(mod, '__all__', ()):
            obj = getattr(mod, name, None)
            if obj is not None and not inspect.ismodule(obj):
                claim(obj, modname, name)
    for modname, mod in mods.items():
        for name, obj in vars(mod).items():
            public = not name.startswith('_') and id(obj) not in home
            own = ((inspect.isclass(obj) or inspect.isfunction(obj)) and
                   obj.__module__ == modname)
            if public and own:
                claim(obj, modname, name)
    curated = set()
    for modname in CURATED:
        for name in _curated_names(mods[modname]):
            obj = mods[modname]
            for part in name.split('.'):
                obj = getattr(obj, part)
            if inspect.ismodule(obj):
                continue
            claim(obj, *f'{modname}.{name}'.rsplit('.', 1), curated=True)
            curated.add(id(obj))

    layout = {m: {'modules': [], 'tables': [], 'data': []} for m in mods}
    by_module = {}
    for key, (modname, name, kind) in home.items():
        if key not in curated:
            by_module.setdefault(modname, []).append((name, kind))
    for modname, members in by_module.items():
        if modname in CURATED:
            continue
        for kind, title in KINDS:
            names = sorted(n for n, k in members if k == kind)
            if names:
                layout[modname]['tables'].append((title, names))
        layout[modname]['data'] = [n for n, k in members if k == 'data']

    has_page = {m for m in mods if hasattr(mods[m], '__path__') or
                layout[m]['tables'] or layout[m]['data']}
    for modname in sorted(has_page - {ROOT}):
        parent = modname.rsplit('.', 1)[0]
        # The root and curated docstrings list their own subpackages
        if parent != ROOT and parent not in CURATED:
            layout[parent]['modules'].append(modname.rsplit('.', 1)[1])
    return {m: v for m, v in layout.items() if m in has_page}


def _inject(app):
    app.config.autosummary_context['api_layout'] = build_layout()


def setup(app):
    # Before autosummary generates stubs (also on builder-inited, priority 500)
    app.connect('builder-inited', _inject, priority=400)
    return {'parallel_read_safe': True}
