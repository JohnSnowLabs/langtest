# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'nlptest'
copyright = '2023, John Snow Labs'
author = 'John Snow Labs'

# The full version, including alpha/beta/rc tags
release = '1.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "autoapi.extension",
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = [
    "css/custom.css",
]

# -- Options for autodoc --------------------------------------------------

# Look at the first line of the docstring for function and method signatures.
autodoc_docstring_signature = False
autosummary_generate = False
numpydoc_show_class_members = False  # Or add Method section in doc strings? https://stackoverflow.com/questions/65198998/sphinx-warning-autosummary-stub-file-not-found-for-the-methods-of-the-class-c
# autoclass_content = "both"  # use __init__ as doc as well

autoapi_options = [
    "members",
    "show-module-summary",
]
autoapi_type = "python"
autoapi_dirs = ["../../nlptest"]
autoapi_root = "reference/_autosummary"
autoapi_template_dir = "_templates/_autoapi"
autoapi_add_toctree_entry = False
# autoapi_member_order = "groupwise"
autoapi_keep_files = True
autoapi_ignore = exclude_patterns
# autoapi_generate_api_docs = False
# autoapi_python_use_implicit_namespaces = True

add_module_names = False

# -- More Configurations -----------------------------------------------------

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"