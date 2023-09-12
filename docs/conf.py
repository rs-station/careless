# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import inspect

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

try:
    from careless import __version__ as release
except ImportError:
    release = "unknown"


# -- Project information -----------------------------------------------------

project = "careless"
copyright = "2023, Kevin M. Dalton"
author = "Kevin M. Dalton"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # "jupyter_sphinx",
    "myst_parser",
    "sphinxcontrib.autoprogram",
    "sphinx_rtd_theme",
    # "numpydoc",
    # "sphinx.ext.autodoc",
    # "sphinx.ext.intersphinx",
    # "sphinx.ext.linkcode",
    # "sphinx.ext.mathjax",
    # "sphinx.ext.napoleon",
    # "sphinx_copybutton",
    # "sphinx_panels",
    # "sphinx_thebe",
    # "sphinx_togglebutton",
]

add_module_names = False
napoleon_google_docstring = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_numpy_docstring = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = False
napoleon_use_rtype = False
numpydoc_show_class_members = False

# Cross-referencing configuration
default_role = "py:obj"
primary_domain = "py"
nitpicky = True  # warn if cross-references are missing

# Intersphinx settings
intersphinx_mapping = {
    "ipywidgets": ("https://ipywidgets.readthedocs.io/en/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "python": ("https://docs.python.org/3", None),
}

# remove panels css to get wider main content
panels_add_bootstrap_css = False

# Settings for copybutton
copybutton_prompt_is_regexp = True
copybutton_prompt_text = r">>> |\.\.\. "  # doctest

# Settings for linkcheck
linkcheck_anchors = False
linkcheck_ignore = []  # type: ignore

execution_timeout = -1
jupyter_execute_notebooks = "off"
if "EXECUTE_NB" in os.environ:
    print("\033[93;1mWill run Jupyter notebooks!\033[0m")
    jupyter_execute_notebooks = "force"

# Settings for myst-parser
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "dollarmath",
    "smartquotes",
    "substitution",
]
suppress_warnings = [
    "myst.header",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "**ipynb_checkpoints",
    ".DS_Store",
    "Thumbs.db",
    "_build",
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_copy_source = True  # needed for download notebook button
# html_css_files = [
#     "custom.css",
# ]
html_sourcelink_suffix = ""
html_static_path = ["_static"]
html_theme = "sphinx_rtd_theme"

html_favicon = "images/rs-favicon_32x32.png"

# html_theme_options = {
#     "launch_buttons": {
#         "binderhub_url": "https://mybinder.org",
#         "colab_url": "https://colab.research.google.com",
#         "notebook_interface": "jupyterlab",
#         "thebe": True,
#         "thebelab": True,
#     },
#     "path_to_docs": "docs",
#     "repository_branch": "main",
#     "repository_url": "https://github.com/dennisbrookner/matchmaps",
#     "use_download_button": True,
#     "use_edit_page_button": True,
#     "use_issues_button": True,
#     "use_repository_button": True,
# }
html_title = "careless"

master_doc = "index"
# thebe_config = {
#     "repository_url": html_theme_options["repository_url"],
#     "repository_branch": html_theme_options["repository_branch"],
# }


# # based on pandas/doc/source/conf.py
# def linkcode_resolve(domain, info):
#     """
#     Determine the URL corresponding to Python object
#     """
#     if domain != "py":
#         return None

#     modname = info["module"]
#     fullname = info["fullname"]

#     submod = sys.modules.get(modname)
#     if submod is None:
#         return None

#     obj = submod
#     for part in fullname.split("."):
#         try:
#             obj = getattr(obj, part)
#         except AttributeError:
#             return None

#     try:
#         fn = inspect.getsourcefile(inspect.unwrap(obj))
#     except TypeError:
#         fn = None
#     if not fn:
#         return None

#     try:
#         source, lineno = inspect.getsourcelines(obj)
#     except OSError:
#         lineno = None

#     if lineno:
#         linespec = f"#L{lineno}-L{lineno + len(source) - 1}"
#     else:
#         linespec = ""

#     fn = os.path.relpath(fn, start=os.path.dirname("../matchmaps"))

#     return f"https://github.com/dennisbrookner/matchmaps/blob/main/matchmaps/{fn}{linespec}"  # noqa