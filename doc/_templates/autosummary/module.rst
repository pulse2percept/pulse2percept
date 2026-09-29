{{ fullname.split('.')[-1] | escape | underline}}

.. rst-class:: api-path

``{{ fullname }}``

.. Objects get their own pages (see doc/_ext/api_layout.py); this page only
   lists them.

.. automodule:: {{ fullname }}
   :no-members:
   :no-inherited-members:

{%- set page = api_layout.get(fullname, {}) %}
{%- if page.modules %}

.. rubric:: Modules

.. autosummary::
   :toctree:
{% for item in page.modules %}
   {{ item }}
{%- endfor %}
{%- endif %}
{%- for title, names in page.tables %}

.. rubric:: {{ title }}

.. autosummary::
   :toctree:
{% for item in names %}
   {{ item }}
{%- endfor %}
{%- endfor %}
{%- if page.data %}

.. rubric:: Module Attributes
{% for item in page.data %}

.. autodata:: {{ item }}
{%- endfor %}
{%- endif %}
