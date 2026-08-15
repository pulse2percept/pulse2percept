""":py:class:`~pulse2percept.utils.deprecated`,
   :py:class:`~pulse2percept.utils.deprecate_parameter`,
   :py:class:`~pulse2percept.utils.deprecated_alias`,
   :py:class:`~pulse2percept.utils.rename_parameter`,
   :py:class:`~pulse2percept.utils.is_deprecated`"""

import sys
import inspect
import warnings
import functools


def _version_clause(deprecated_version=None, removed_version=None):
    """Build the " since version X, and will be removed in version Y" clause

    Shared by :py:class:`deprecated` and :py:class:`deprecate_parameter` so
    that all deprecation messages are worded the same way.
    """
    dep_msg = ""
    if deprecated_version is not None:
        dep_msg = f" since version {deprecated_version}"
    rmv_msg = ""
    if removed_version is not None:
        rmv_msg = f", and will be removed in version {removed_version}"
    return dep_msg + rmv_msg


def _callable_name(func):
    """Names a callable the way a user would refer to it"""
    obj_name = getattr(func, '__qualname__', None) or func.__name__
    # A decorated constructor is really about the class, so report
    # `MyClass`, not `MyClass.__init__`:
    if obj_name.endswith('.__init__'):
        obj_name = obj_name[:-len('.__init__')]
    return obj_name


class deprecated:
    """Decorator to mark deprecated functions and classes with a warning.

    .. seealso::

        Adapted from
        https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/deprecation.py.

    Parameters
    ----------
    alt_func : str
        If given, tell user what function to use instead.
    deprecated_version : float or str
        The package version in which the function/class was first marked as
        deprecated.
    removed_version : float or str
        The package version in which the deprecated function/class will be
        removed.
    """

    def __init__(self, alt_func=None, deprecated_version=None,
                 removed_version=None):
        self.alt_func = alt_func
        self.deprecated_version = deprecated_version
        self.removed_version = removed_version

    def __call__(self, obj):
        if isinstance(obj, type):
            return self._decorate_class(obj)
        elif isinstance(obj, property):
            # Note that this is only triggered properly if the `property`
            # decorator comes before the `deprecated` decorator, like so:
            #
            # @deprecated(msg)
            # @property
            # def deprecated_attribute_(self):
            #     ...
            return self._decorate_property(obj)
        else:
            return self._decorate_fun(obj)

    def _get_message(self, obj_name):
        """Builds the message string"""
        msg = f"{obj_name} is deprecated"
        alt_msg = ""
        if self.alt_func is not None:
            alt_msg = f"Use ``{self.alt_func}`` instead."
        clause = _version_clause(self.deprecated_version, self.removed_version)
        return msg + clause + ". " + alt_msg

    def _update_doc(self, old_doc, msg=None):
        """Updates the docstring"""
        if msg is None:
            msg = self._get_message("This feature")
        # Insert a deprecated directive:
        doc = f".. deprecated:: {self.deprecated_version}\n\n    {msg}"
        if old_doc:
            doc = f"{doc}\n\n{old_doc}"
        return doc

    def _decorate_class(self, cls):
        """Mark a class as deprecated"""
        msg = self._get_message(f"Class {cls.__name__}")

        # FIXME: we should probably reset __new__ for full generality
        init = cls.__init__

        def wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return init(*args, **kwargs)
        cls.__init__ = wrapped

        wrapped.__name__ = '__init__'
        wrapped.__doc__ = self._update_doc(init.__doc__, msg)
        wrapped.deprecated_original = init

        return cls

    def _decorate_property(self, prop):
        """Mark a class property as deprecated

        Note that this is only triggered properly if the `property` decorator
        comes before the `deprecated` decorator, like so:

        .. code-block:: python

            @deprecated()
            @property
            def deprecated_attribute_(self):
                ...
        """
        # Use the getter's name, not `prop.__name__`: properties only grew a
        # `__name__` attribute in Python 3.13.
        msg = self._get_message(f"Property {prop.fget.__name__}")

        @property
        def wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return prop.fget(*args, **kwargs)

        wrapped.__doc__ = self._update_doc(prop.__doc__, msg)
        return wrapped

    def _decorate_fun(self, fun):
        """Mark a function as deprecated"""
        msg = self._get_message(f"Function {fun.__name__}")

        @functools.wraps(fun)
        def wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return fun(*args, **kwargs)

        wrapped.__doc__ = self._update_doc(wrapped.__doc__, msg)

        return wrapped

class deprecate_parameter:
    """Decorator to mark a single function or method parameter as deprecated

    The decorated callable keeps *accepting* the parameter, so that existing
    code does not break, but the value is ignored. A ``DeprecationWarning`` is
    raised whenever the parameter is passed explicitly, whether by keyword or
    by position.

    Use this when a parameter is going away but the callable itself stays. To
    deprecate an entire function, class, or property, use
    :py:class:`~pulse2percept.utils.deprecated` instead. To keep a parameter
    that is merely being *renamed* working under its old name, use
    :py:class:`~pulse2percept.utils.rename_parameter` instead.

    .. versionadded:: 0.9.1

    .. note::

        This decorator only produces the *warning*. Document the parameter
        itself by adding a ``.. deprecated::`` directive to its entry in the
        docstring's ``Parameters`` section, which is where numpydoc expects it.

    .. seealso::

        Modeled on matplotlib's ``matplotlib._api.delete_parameter``.

    Parameters
    ----------
    name : str
        Name of the deprecated parameter. Must appear in the signature of the
        decorated callable, otherwise a ``ValueError`` is raised at decoration
        time (which catches the parameter being renamed or dropped).
    deprecated_version : float or str
        The package version in which the parameter was first marked as
        deprecated.
    removed_version : float or str
        The package version in which the parameter will be removed.
    addendum : str, optional
        Text appended to the warning, e.g. to spell out what the parameter
        used to do or how the behavior differs now that it is ignored.

    Examples
    --------
    >>> from pulse2percept.utils import deprecate_parameter
    >>> @deprecate_parameter('engine', deprecated_version='0.9.1',
    ...                      removed_version='0.10.0')
    ... def predict(data, engine=None):
    ...     return data
    >>> predict([1, 2])  # no warning
    [1, 2]

    """

    def __init__(self, name, deprecated_version=None, removed_version=None,
                 addendum=None):
        self.name = name
        self.deprecated_version = deprecated_version
        self.removed_version = removed_version
        self.addendum = addendum

    def _get_message(self, obj_name):
        """Builds the message string"""
        msg = (f"The '{self.name}' parameter of {obj_name} is deprecated"
               f"{_version_clause(self.deprecated_version, self.removed_version)}"
               f". It is ignored.")
        if self.addendum is not None:
            msg = f"{msg} {self.addendum}"
        return msg

    def _get_obj_name(self, func):
        """Names the decorated callable the way a user would refer to it"""
        return _callable_name(func)

    def __call__(self, func):
        signature = inspect.signature(func)
        obj_name = self._get_obj_name(func)
        if self.name not in signature.parameters:
            raise ValueError(f"'{self.name}' is not a parameter of "
                             f"{obj_name}. Its signature is {signature}.")

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            try:
                passed = self.name in signature.bind(*args, **kwargs).arguments
            except TypeError:
                # The call does not match the signature. Don't preempt the
                # error the wrapped callable is about to raise itself:
                passed = False
            if passed:
                # Build the message here rather than capturing it in the
                # closure: `is_deprecated` looks for the word "deprecated" in
                # closure cells, and the callable itself is *not* deprecated.
                warnings.warn(self._get_message(obj_name),
                              category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapped


class rename_parameter:
    """Decorator to rename a single function or method parameter

    Calls that use the old name keep working: the value is forwarded to the
    new parameter, so the callable behaves exactly as it did before, but a
    ``DeprecationWarning`` is raised. Use this when a parameter is only being
    renamed; when its value is no longer read at all, use
    :py:class:`~pulse2percept.utils.deprecate_parameter` instead.

    Only *keyword* use of the old name is forwarded, which is the only way it
    can be recognized: a positional argument is bound by position and never
    mentions either name.

    .. versionadded:: 0.10.0

    .. note::

        Models take their parameters as ``**params``, validated against
        ``get_default_params`` rather than declared in a signature, so this
        decorator cannot see them. Rename those with a
        :py:class:`~pulse2percept.utils.deprecated_alias` instead.

    Parameters
    ----------
    old_name : str
        The name being retired. Must *not* appear in the signature of the
        decorated callable, otherwise a ``ValueError`` is raised at decoration
        time (which catches the rename never having been made).
    new_name : str
        The name that replaces it. Must appear in the signature, otherwise a
        ``ValueError`` is raised at decoration time.
    deprecated_version : float or str
        The package version in which the old name was first marked as
        deprecated.
    removed_version : float or str
        The package version in which the old name will stop working.

    Examples
    --------
    >>> from pulse2percept.utils import rename_parameter
    >>> @rename_parameter('axlambda', 'lam', deprecated_version='0.10.0',
    ...                   removed_version='0.11.0')
    ... def decay(lam=1):
    ...     return lam
    >>> decay(lam=3)  # no warning
    3

    """

    def __init__(self, old_name, new_name, deprecated_version=None,
                 removed_version=None):
        self.old_name = old_name
        self.new_name = new_name
        self.deprecated_version = deprecated_version
        self.removed_version = removed_version

    def _get_message(self, obj_name):
        """Builds the message string"""
        clause = _version_clause(self.deprecated_version, self.removed_version)
        return (f"The '{self.old_name}' parameter of {obj_name} is deprecated"
                f"{clause}. Use '{self.new_name}' instead.")

    def __call__(self, func):
        signature = inspect.signature(func)
        obj_name = _callable_name(func)
        if self.new_name not in signature.parameters:
            raise ValueError(f"'{self.new_name}' is not a parameter of "
                             f"{obj_name}. Its signature is {signature}.")
        if self.old_name in signature.parameters:
            raise ValueError(f"'{self.old_name}' is still a parameter of "
                             f"{obj_name}. Rename it to '{self.new_name}' "
                             f"first.")

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            if self.old_name in kwargs:
                if self.new_name in kwargs:
                    raise TypeError(f"{obj_name} got both '{self.old_name}' "
                                    f"and '{self.new_name}', which are the "
                                    f"same parameter. Pass only "
                                    f"'{self.new_name}'.")
                # Build the message here rather than capturing it in the
                # closure: `is_deprecated` looks for the word "deprecated" in
                # closure cells, and the callable itself is *not* deprecated.
                warnings.warn(self._get_message(obj_name),
                              category=DeprecationWarning, stacklevel=2)
                kwargs[self.new_name] = kwargs.pop(self.old_name)
            return func(*args, **kwargs)

        return wrapped


class deprecated_alias:
    """Class attribute that keeps a renamed parameter usable under its old name

    Assign one in the class body, under the *old* name, to a
    :py:class:`~pulse2percept.utils.Parametrized` subclass whose parameters
    live in ``get_default_params`` rather than in a signature:

    .. code-block:: python

        class MyModel(BaseModel):

            axlambda = deprecated_alias('lam', deprecated_version='0.10.0')

            def get_default_params(self):
                return {'lam': 500}

    Reading or writing ``model.axlambda`` then reads or writes ``model.lam``
    and raises a ``DeprecationWarning``. The alias also registers itself in
    the owner's ``_renamed_params``, which is what lets the constructor,
    ``set_params`` and ``build`` keep accepting the old name as a keyword
    argument (see
    :py:func:`~pulse2percept.utils.rename_deprecated_params`).

    To rename a parameter that *is* declared in a signature, use
    :py:class:`~pulse2percept.utils.rename_parameter` instead.

    .. versionadded:: 0.10.0

    .. note::

        Document the rename by adding a ``.. versionchanged::`` directive to
        the *new* parameter's entry in the docstring's ``Parameters`` section.
        The old name has no entry of its own, since nothing should be written
        against it any more.

    Parameters
    ----------
    new_name : str
        Name of the parameter that replaces the alias.
    deprecated_version : float or str
        The package version in which the old name was first marked as
        deprecated.
    removed_version : float or str
        The package version in which the old name will stop working.

    """

    def __init__(self, new_name, deprecated_version=None,
                 removed_version=None):
        self.new_name = new_name
        self.deprecated_version = deprecated_version
        self.removed_version = removed_version
        # Both filled in by ``__set_name__`` when the class body is executed:
        self.old_name = None
        self.owner_name = None

    def __set_name__(self, owner, name):
        self.old_name = name
        self.owner_name = owner.__name__
        # Register on `owner` itself rather than mutate the dict it inherited,
        # which every other class in the hierarchy is looking at too:
        owner._renamed_params = {**getattr(owner, '_renamed_params', {}),
                                 name: self}

    def _get_message(self, obj_name=None):
        """Builds the message string"""
        clause = _version_clause(self.deprecated_version, self.removed_version)
        return (f"The '{self.old_name}' parameter of "
                f"{obj_name or self.owner_name} is deprecated{clause}. "
                f"Use '{self.new_name}' instead.")

    def __get__(self, obj, objtype=None):
        if obj is None:
            # Looked up on the class rather than on an instance, which is how
            # the attribute machinery asks whether the name exists at all.
            # Nothing is being read, so nothing is deprecated yet:
            return self
        # stacklevel=2 blames the line that touched the attribute:
        warnings.warn(self._get_message(), category=DeprecationWarning,
                      stacklevel=2)
        return getattr(obj, self.new_name)

    def __set__(self, obj, value):
        warnings.warn(self._get_message(), category=DeprecationWarning,
                      stacklevel=2)
        setattr(obj, self.new_name, value)


def warn_deprecated_params(obj_name, supplied, specs, stacklevel=3):
    """Warn about deprecated *model* parameters that were supplied by name

    pulse2percept models take their parameters as ``**params``, validated
    against ``get_default_params`` rather than declared in a signature, so
    :py:class:`~pulse2percept.utils.deprecate_parameter` cannot see them. This
    is the equivalent for that path: hand it the names the caller actually
    supplied, and it warns for the deprecated ones.

    .. versionadded:: 0.9.1

    Parameters
    ----------
    obj_name : str
        Name of the model, as it should appear in the warning.
    supplied : iterable of str
        Parameter names the caller passed explicitly. Names that are not
        deprecated are skipped, so it is fine to pass all of them.
    specs : dict
        Maps a deprecated parameter name to the
        :py:class:`~pulse2percept.utils.deprecate_parameter` describing it, so
        that signature-level and model-level deprecations word alike.
    stacklevel : int, optional
        Passed to ``warnings.warn``. Exact attribution is not possible through
        a chain of ``super().__init__`` calls of varying depth, so the message
        names the parameter and the model rather than relying on it.
    """
    for name in supplied:
        spec = specs.get(name)
        if spec is not None:
            warnings.warn(spec._get_message(obj_name),
                          category=DeprecationWarning, stacklevel=stacklevel)


def rename_deprecated_params(obj_name, params, specs, stacklevel=3):
    """Rewrite renamed *model* parameters that were supplied by their old name

    The counterpart of :py:func:`~pulse2percept.utils.warn_deprecated_params`
    for parameters that were renamed rather than retired: the value is kept,
    but moves to the new name, and the warning names the replacement.

    Handing the caller a rewritten dict, rather than letting the assignment
    fall through to the
    :py:class:`~pulse2percept.utils.deprecated_alias` descriptor, keeps the
    warning to one per parameter and lets it name the model the user actually
    called.

    .. versionadded:: 0.10.0

    Parameters
    ----------
    obj_name : str
        Name of the model, as it should appear in the warning.
    params : dict
        Parameters the caller supplied. Names that were not renamed are left
        alone, so it is fine to pass all of them.
    specs : dict
        Maps a renamed parameter's old name to the
        :py:class:`~pulse2percept.utils.deprecated_alias` describing it.
    stacklevel : int, optional
        Passed to ``warnings.warn``. Exact attribution is not possible through
        a chain of ``super().__init__`` calls of varying depth, so the message
        names the parameter and the model rather than relying on it.

    Returns
    -------
    params : dict
        ``params`` with every renamed key replaced by its new name. The
        original dict is returned untouched if none of the keys were renamed.

    """
    if not any(name in specs for name in params):
        return params
    renamed = {}
    for name, val in params.items():
        spec = specs.get(name)
        if spec is not None:
            warnings.warn(spec._get_message(obj_name),
                          category=DeprecationWarning, stacklevel=stacklevel)
            name = spec.new_name
        renamed[name] = val
    return renamed


def is_deprecated(func):
    """Helper to check if ``func`` is wrapped by the deprecated decorator"""
    closures = getattr(func, '__closure__', [])
    if closures is None:
        closures = []
    is_deprecated = ('deprecated' in ''.join([
        c.cell_contents for c in closures if isinstance(c.cell_contents, str)
    ]))
    return is_deprecated
