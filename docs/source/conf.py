import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../../src"))

project = "PySATL Core"
copyright = f"{datetime.now().year}, Leonid Elkin, Mikhail Mikhailov"
author = "Leonid Elkin, Mikhail Mikhailov"
release = "0.0.1a0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx_rtd_theme",
    "myst_nb",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx.ext.autosummary",
]
autosummary_generate = True
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- Napoleon (NumPy style docstrings) --
napoleon_google_docstring = False
napoleon_use_keyword = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = None

# -- Autodocumentation settings --
autodoc_default_options = {
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}

autodoc_typehints = "description"
autodoc_typehints_format = "short"

# -- Intersphinx --
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- MyST Parser --
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "tasklist",
]
myst_heading_anchors = 3
nb_execution_mode = "off"

# -- HTML --
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]

# ReadTheDocs
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
    "logo_only": False,
}

# Sidebar: show navigation tree, avoid listing all members of the current page
html_sidebars = {
    "**": [
        "about.html",
        "navigation.html",
        "relations.html",
        "searchbox.html",
    ]
}

html_logo = "_static/PySATL-logo.jpg"
html_favicon = "_static/PySATL-icon.jpg"

# -- Compile --
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_line_continuation_character = "\\"

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
    ".ipynb": "myst-nb",
}

suppress_warnings = [
    # 'autodoc.duplicate_object',
    "ref.misc",
]

# forward references (actually don't matter too much but better be here)
autodoc_type_aliases = {
    "In": "typing.Any",
    "Out": "typing.Any",
    "DistributionType": "pysatl_core.types.DistributionType",
    "Kind": "pysatl_core.types.Kind",
    "SamplingStrategy": "pysatl_core.distributions.strategies.SamplingStrategy",
    "ComputationStrategy": "pysatl_core.distributions.strategies.ComputationStrategy",
    "Parametrization": "pysatl_core.families.parametrizations.Parametrization",
    "Support": "pysatl_core.distributions.support.Support",
    "Sample": "pysatl_core.distributions.sampling.Sample",
    "Distribution": "pysatl_core.distributions.distribution.Distribution",
    "ParametricFamily": "pysatl_core.families.parametric_family.ParametricFamily",
}

# Some checks
nitpicky = False
