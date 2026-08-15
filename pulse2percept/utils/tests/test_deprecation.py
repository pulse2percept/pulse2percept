import warnings
import numpy.testing as npt
import pytest
from pulse2percept.utils.deprecation import (deprecated, deprecate_parameter,
                                             deprecated_alias,
                                             rename_parameter,
                                             rename_deprecated_params,
                                             is_deprecated)
from pulse2percept.utils.testing import assert_warns_msg


@deprecated(alt_func='qwerty')
class MockClass1:
    pass


class MockClass2(object):

    @deprecated(deprecated_version=0.1, removed_version=0.2)
    def mymethod(self):
        pass


class MockClass3:

    @deprecated()
    def __init__(self):
        pass


class MockClass4:
    pass


@deprecated(deprecated_version=0.4)
def mock_function():
    return 10


def test_deprecated():
    assert_warns_msg(DeprecationWarning, MockClass1, 'Use ``qwerty`` instead')
    assert_warns_msg(DeprecationWarning, MockClass2().mymethod,
                     'since version 0.1, and will be removed in version 0.2')
    assert_warns_msg(DeprecationWarning, MockClass3, 'deprecated')
    assert_warns_msg(DeprecationWarning, mock_function, 'since version 0.4')


def test_is_deprecated():
    # Test if is_deprecated helper identifies wrapping via deprecated:
    # NOTE it works only for class methods and functions
    npt.assert_equal(is_deprecated(MockClass1.__init__), True)
    npt.assert_equal(is_deprecated(MockClass2().mymethod), True)
    npt.assert_equal(is_deprecated(MockClass3.__init__), True)
    npt.assert_equal(is_deprecated(MockClass4.__init__), False)
    npt.assert_equal(is_deprecated(mock_function), True)


class MockClassProperty:

    @deprecated(deprecated_version=0.5, alt_func='new_attribute')
    @property
    def deprecated_attribute(self):
        """Original docstring."""
        return 42


def test_deprecated_property():
    # A deprecated property warns on access, but still returns its value:
    obj = MockClassProperty()
    assert_warns_msg(DeprecationWarning, lambda: obj.deprecated_attribute,
                     'since version 0.5')
    npt.assert_equal(obj.deprecated_attribute, 42)

    # The warning names the property. `property` objects only have a
    # `__name__` on Python 3.13+, so the name has to come from the getter:
    assert_warns_msg(DeprecationWarning, lambda: obj.deprecated_attribute,
                     'Property deprecated_attribute is deprecated')

    # The deprecation directive is prepended to the original docstring:
    doc = MockClassProperty.deprecated_attribute.__doc__
    npt.assert_equal('.. deprecated:: 0.5' in doc, True)
    npt.assert_equal('Use ``new_attribute`` instead' in doc, True)
    npt.assert_equal('Original docstring.' in doc, True)


def test_deprecated_update_doc():
    # Without an explicit message, a generic one is generated:
    doc = deprecated(deprecated_version=0.6)._update_doc('Original.')
    npt.assert_equal('.. deprecated:: 0.6' in doc, True)
    npt.assert_equal('This feature is deprecated' in doc, True)
    npt.assert_equal('Original.' in doc, True)
    # An empty original docstring is fine:
    npt.assert_equal('Original.' in deprecated()._update_doc(''), False)


def test_is_deprecated_without_closure():
    # A function with no closure cells at all (`__closure__` is None):
    npt.assert_equal(is_deprecated(lambda: None), False)


@deprecate_parameter('old', deprecated_version=0.1, removed_version=0.2)
def mock_func_old_param(a, old=None, b=3):
    return a + b


class MockClassOldParam:

    @deprecate_parameter('old', deprecated_version=0.1, removed_version=0.2,
                         addendum='It used to do nothing.')
    def __init__(self, a, old=None):
        self.a = a


def test_deprecate_parameter():
    # Passing the parameter warns, by keyword or by position:
    assert_warns_msg(DeprecationWarning, mock_func_old_param,
                     "The 'old' parameter of mock_func_old_param is "
                     "deprecated since version 0.1, and will be removed in "
                     "version 0.2. It is ignored.", 1, old='x')
    assert_warns_msg(DeprecationWarning, mock_func_old_param,
                     "'old' parameter", 1, 'x')
    # But the parameter is ignored: the return value is unaffected:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        npt.assert_equal(mock_func_old_param(1, old='x'), 4)
        npt.assert_equal(mock_func_old_param(1, 'x', 10), 11)
    # Omitting the parameter does not warn:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        npt.assert_equal(mock_func_old_param(1), 4)
        npt.assert_equal(mock_func_old_param(1, b=10), 11)


def test_deprecate_parameter_method():
    # On a constructor, the warning names the class, not `__init__`:
    assert_warns_msg(DeprecationWarning, MockClassOldParam,
                     "parameter of MockClassOldParam is deprecated",
                     1, old='x')
    # The addendum is appended to the message:
    assert_warns_msg(DeprecationWarning, MockClassOldParam,
                     'It used to do nothing.', 1, old='x')
    # The wrapped callable still works:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        npt.assert_equal(MockClassOldParam(1, old='x').a, 1)
    # Docstring and name survive the wrapping:
    npt.assert_equal(mock_func_old_param.__name__, 'mock_func_old_param')


def test_deprecate_parameter_is_not_is_deprecated():
    # Deprecating a parameter does not deprecate the callable itself, so
    # `is_deprecated` must keep reporting False for it:
    npt.assert_equal(is_deprecated(mock_func_old_param), False)
    npt.assert_equal(is_deprecated(MockClassOldParam.__init__), False)


def test_deprecate_parameter_unknown_param():
    # A typo'd or already-removed parameter fails loudly at decoration time,
    # rather than silently never warning:
    with pytest.raises(ValueError):
        @deprecate_parameter('nonexistent')
        def func(a, b=2):
            return a

    # An invalid call is left to raise its own error, not preempted by the
    # decorator's signature binding:
    with pytest.raises(TypeError):
        mock_func_old_param()


@rename_parameter('old', 'new', deprecated_version=0.1, removed_version=0.2)
def mock_func_renamed_param(a, new=3):
    return a + new


class MockClassRenamedParam:

    @rename_parameter('old', 'new', deprecated_version=0.1,
                      removed_version=0.2)
    def __init__(self, new):
        self.new = new


def test_rename_parameter():
    # The old name warns, and names the replacement:
    assert_warns_msg(DeprecationWarning, mock_func_renamed_param,
                     "The 'old' parameter of mock_func_renamed_param is "
                     "deprecated since version 0.1, and will be removed in "
                     "version 0.2. Use 'new' instead.", 1, old=10)
    # But unlike a deprecated parameter, its value is *kept*:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        npt.assert_equal(mock_func_renamed_param(1, old=10), 11)
    # The new name behaves identically, and does not warn:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        npt.assert_equal(mock_func_renamed_param(1, new=10), 11)
        npt.assert_equal(mock_func_renamed_param(1), 4)
    # Docstring and name survive the wrapping, and the callable itself is not
    # deprecated:
    npt.assert_equal(mock_func_renamed_param.__name__,
                     'mock_func_renamed_param')
    npt.assert_equal(is_deprecated(mock_func_renamed_param), False)


def test_rename_parameter_method():
    # On a constructor, the warning names the class, not `__init__`:
    assert_warns_msg(DeprecationWarning, MockClassRenamedParam,
                     "parameter of MockClassRenamedParam is deprecated",
                     old=10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        npt.assert_equal(MockClassRenamedParam(old=10).new, 10)


def test_rename_parameter_both_names():
    # The two names are the same parameter, so passing both is an error
    # rather than a silent choice between them:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(TypeError):
            mock_func_renamed_param(1, old=10, new=20)


def test_rename_parameter_unknown_param():
    # The new name has to exist...
    with pytest.raises(ValueError):
        @rename_parameter('old', 'nonexistent')
        def func(a, b=2):
            return a

    # ...and the old one must be gone from the signature, which catches the
    # rename never having been made:
    with pytest.raises(ValueError):
        @rename_parameter('b', 'a')
        def func(a, b=2):
            return a


class MockClassAlias:

    old = deprecated_alias('new', deprecated_version=0.1, removed_version=0.2)

    def __init__(self):
        self.new = 42


def test_deprecated_alias():
    obj = MockClassAlias()
    # Reading through the alias warns, but returns the current value:
    assert_warns_msg(DeprecationWarning, lambda: obj.old,
                     "The 'old' parameter of MockClassAlias is deprecated "
                     "since version 0.1, and will be removed in version 0.2. "
                     "Use 'new' instead.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        npt.assert_equal(obj.old, 42)
        # Writing through it warns too, and writes the new name:
        obj.old = 7
    npt.assert_equal(obj.new, 7)
    assert_warns_msg(DeprecationWarning,
                     lambda: setattr(obj, 'old', 9), "Use 'new' instead")
    npt.assert_equal(obj.new, 9)
    # The new name stays silent:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        obj.new = 3
        npt.assert_equal(obj.new, 3)


class MockSubclassAlias(MockClassAlias):
    pass


def test_deprecated_alias_names_runtime_class():
    # A subclass inherits the alias, but the warning has to name the class the
    # user is actually holding, not the one the alias was declared on:
    assert_warns_msg(DeprecationWarning, lambda: MockSubclassAlias().old,
                     "The 'old' parameter of MockSubclassAlias is deprecated")
    assert_warns_msg(DeprecationWarning,
                     lambda: setattr(MockSubclassAlias(), 'old', 1),
                     "The 'old' parameter of MockSubclassAlias is deprecated")


def test_deprecated_alias_blames_caller():
    # A deprecation warning is only actionable if it points at the line that
    # used the old name, so check where it lands and not just what it says:
    obj = MockClassAlias()
    with pytest.warns(DeprecationWarning) as record:
        obj.old
    npt.assert_equal(record[0].filename, __file__)
    with pytest.warns(DeprecationWarning) as record:
        obj.old = 1
    npt.assert_equal(record[0].filename, __file__)


def test_deprecated_alias_on_class():
    # Looked up on the class, the alias returns itself rather than warning:
    # that is how the attribute machinery asks whether a name exists at all.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        npt.assert_equal(isinstance(MockClassAlias.old, deprecated_alias),
                         True)
        npt.assert_equal(hasattr(MockClassAlias, 'old'), True)
    # And it registered itself, so that `**params` constructors can find it:
    npt.assert_equal(MockClassAlias._renamed_params['old'].new_name, 'new')


def test_rename_deprecated_params():
    specs = MockClassAlias._renamed_params
    # A renamed name is warned about, under the name the caller used, and
    # rewritten:
    assert_warns_msg(DeprecationWarning, rename_deprecated_params,
                     "The 'old' parameter of MyModel is deprecated",
                     'MyModel', {'old': 1}, specs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        npt.assert_equal(rename_deprecated_params('MyModel', {'old': 1},
                                                  specs), {'new': 1})
    # Everything else is passed straight through, untouched and unwarned:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        params = {'other': 1}
        npt.assert_equal(rename_deprecated_params('MyModel', params, specs) is
                         params, True)
        npt.assert_equal(rename_deprecated_params('MyModel', params, {}) is
                         params, True)


def test_rename_deprecated_params_both_names():
    specs = MockClassAlias._renamed_params
    # The two names are the same parameter, so supplying both must raise
    # rather than let the order they were passed in decide which one wins.
    # `**kwargs` preserves insertion order, so check it really is symmetric:
    for params in ({'old': 1, 'new': 2}, {'new': 2, 'old': 1}):
        with pytest.raises(TypeError, match="same parameter"):
            rename_deprecated_params('MyModel', params, specs)
    # And it raises *instead of* warning, not after it, exactly as the
    # signature-level `rename_parameter` does:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(TypeError):
            rename_deprecated_params('MyModel', {'old': 1, 'new': 2}, specs)
