{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. The package page already documents this class, so this stub page is not
   indexed; cross-references resolve to the package page.

.. autoclass:: {{ objname }}
   :no-index:

   {% block methods %}
   .. automethod:: __init__
      :no-index:

   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
