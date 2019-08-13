How to run a model
==================

Running an existing model
-------------------------

TODO

A model follows the following steps:

* instantiate
* build
* predict

Creating your own model
-----------------------

TODO

The `BaseModel` class defines which methods and attributes a model must
have.
These include the following methods for which you must provide an implementation:

* :py:meth:`~pulse2percept.models.BaseModel.__init__`
* :py:meth:`~pulse2percept.models.BaseModel._predict_pixel_percept`
* :py:meth:`~pulse2percept.models.BaseModel.get_tissue_coords`

Writing the constructor
~~~~~~~~~~~~~~~~~~~~~~~

The `BaseModel` class defines which methods and attributes a model must
have. You can create your own model by adding a class that derives from
`BaseModel`:

.. code-block:: python

    class MyModel(BaseModel):

The constructor is the only place where you can add new variables
(i.e., class attributes). The signature of your own constructor should
look like this:

.. code-block:: python

    def __init__(self, **kwargs):

meaning that all arguments are passed as keyword arguments. Also, make
sure to call :py:meth:`~pulse2percept.models.BaseModel.__init__` first thing.
So a complete example of a constructor could look like this:

.. code-block:: python

    class MyModel(BaseModel):

        def __init__(self, **kwargs):
            # Call the BaseModel constructor. This will load the default values
            # for all model parameters:
            super().__init__(self, **kwargs)
            # You can add a new variable here:
            self.newvar = 0

.. note::

   If `self.newvar` already exists in the BaseModel class, the last line of the
   above code snippet would overwrite it.

.. warning::

    The constructor is the only place where you can add new variables
    (i.e., class attributes).
    Trying to set `self.someothervar` outside the constructor will raise a
    `FreezeError`. Of course, you can always set `self.newvar = None` in
    the constructor to make sure the variable exists, and then assign a new
    value in other class methods.


Writing the get_tissue_coords method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO

Writing the _predict_pixel_percept method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO

:py:meth:`~pulse2percept.models.BaseModel.predict_percept` calls out to
:py:meth:`~pulse2percept.models.BaseModel._predict_pixel_percept`, which is
an abstract method that must be provided by the user.

It should return the stimulus at time `t` for pixel `p`.

Adding to the build method
~~~~~~~~~~~~~~~~~~~~~~~~~~

Every model must have a ```build`` method, which is meant to perform
all expensive one-time calculations. You must call ``build`` before
calling ``predict_percept``.

You can override ``build`` in your own model (for a good example, see
the AxonMapModel). You will want to make sure that:

- all `build_params` take effect,
- the flag `_is_built` is set,
- the method returns `self`.

One way to do this is to call the BaseModel's ``build`` method from
within your own model:

.. code-block:: python

    class MyModel(BaseModel):

        def build(self, **build_params):
            super(MyModel, self).build(self, **build_params)
            # Add your own code here...

Example models
--------------

Minimal example
~~~~~~~~~~~~~~~

Here's a small working example example:

.. ipython:: python

    from pulse2percept.models import BaseModel
    from pulse2percept.implants import ArgusI

    class MyModel(BaseModel):
        def get_tissue_coords(self, xdva, ydva):
            # Assume 1dva corresponds to 289um:
            factor = 289
            return xdva * factor, ydva * factor
        def _predict_pixel_percept(self, xygrid, implant, t=None):
            return 42

    # Instantiate the model:
    my_model = MyModel()

    # Build the model:
    my_model.build()

    # Build the implant + stimulus:
    implant = ArgusI(stim=np.ones(16))

    my_model.predict_percept(implant)

As mentioned above, trying to create a new variable outside the constructor
will result in a `FreezeError`:

.. ipython:: python

    my_model.my_new_var = 0

Trying to predict a percept before building a model will result in a
`NotBuiltError`:

.. ipython:: python

    not_built = MyModel()

    not_built.predict_percept(implant)

Using Watson (2014) to convert between dva and um
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the `get_tissue_coords` method from `Watson214ConversionMixin`:

.. code-block:: python

    from pulse2percept.models import BaseModel, Watson2014ConversionMixin

    class MyWatsonModel(Watson2014ConversionMixin, BaseModel):

        def _predict_pixel_percept(self, xygrid, implant, t=None):
            return 42

More complicated
~~~~~~~~~~~~~~~~

Make changes to everything:

.. code-block:: python

    import numpy as np
    from pulse2percept.models import BaseModel, Watson2014ConversionMixin

    class MyComplicatedModel(Watson2014ConversionMixin, BaseModel):

        def __init__(self, **kwargs):
            super().__init__(self, **kwargs)
            self.important_var = 100
            self.output = None

        def _get_default_params(self):
            params = super()._get_default_params()
            params.update({'alpha': 0, 'beta': 1})
            return params

        def build(self, **build_params):
            # Set additional parameters (they must be mentioned in the
            # constructor; you can't add new class attributes outside of 
            # that):
            for key, val in build_params.items():
                setattr(self, key, val)

            # Perform some expensive one-time computation:
            output = 0
            for x in range(100):
                # This is silly - I know:
                output += np.random.rand()
            self.output = output

            # Indicate that we're all done:
            self._is_built = True

        def _predict_pixel_percept(self, xygrid, implant, t=None):
            return self.output
