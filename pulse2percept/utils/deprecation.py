"""deprecated, is_deprecated"""

import sys
import warnings
import functools


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
        msg = "%s is deprecated" % obj_name
        alt_msg = ""
        if self.alt_func is not None:
            alt_msg = "Use ``%s`` instead." % self.alt_func
        dep_msg = ""
        if self.deprecated_version is not None:
            dep_msg = " since version %s" % self.deprecated_version
        rmv_msg = ""
        if self.removed_version is not None:
            rmv_msg = (", and will be removed in version %s" %
                       self.removed_version)
        return msg + dep_msg + rmv_msg + ". " + alt_msg

    def _update_doc(self, olddoc):
        newdoc = self._get_message("Deprecated")
        if olddoc:
            newdoc = "%s\n\n%s" % (newdoc, olddoc)
        return newdoc

    def _decorate_class(self, cls):
        """Mark a class as deprecated"""
        msg = self._get_message("Class %s" % cls.__name__)

        # FIXME: we should probably reset __new__ for full generality
        init = cls.__init__

        def wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return init(*args, **kwargs)
        cls.__init__ = wrapped

        wrapped.__name__ = '__init__'
        wrapped.__doc__ = self._update_doc(init.__doc__)
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
        msg = self._get_message("Property %s" % prop.__name__)

        @property
        def wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return prop.fget(*args, **kwargs)

        return wrapped

    def _decorate_fun(self, fun):
        """Mark a function as deprecated"""
        msg = self._get_message("Function %s" % fun.__name__)

        @functools.wraps(fun)
        def wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return fun(*args, **kwargs)

        wrapped.__doc__ = self._update_doc(wrapped.__doc__)

        return wrapped


def is_deprecated(func):
    """Helper to check if ``func`` is wrapped by the deprecated decorator"""
    if sys.version_info < (3, 5):
        raise NotImplementedError("This is only available for Python 3.5 "
                                  "or above")
    closures = getattr(func, '__closure__', [])
    if closures is None:
        closures = []
    is_deprecated = ('deprecated' in ''.join([
        c.cell_contents for c in closures if isinstance(c.cell_contents, str)
    ]))
    return is_deprecated
