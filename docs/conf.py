# Configuration file for the Sphinx documentation builder.

# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/page/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import shutil
import sys
from datetime import datetime
from importlib.metadata import metadata
from pathlib import Path

from sphinxcontrib import katex

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "extensions"))


# -- Project information -----------------------------------------------------

# NOTE: If you installed your project in editable mode, this might be stale.
#       If this is the case, reinstall it to refresh the metadata
info = metadata("liana")
project_name = info["Name"]
author = info["Author"]
copyright = f"{datetime.now():%Y}, the LIANA+ developers"
version = info["Version"]
urls = dict(pu.split(", ") for pu in info.get_all("Project-URL"))
repository_url = urls["Source"]

# The full version, including alpha/beta/rc tags
release = info["Version"]

bibtex_bibfiles = ["references.bib"]
templates_path = ["_templates"]
nitpicky = True  # Warn about broken links
# The notebooks under `docs/tutorials` come from the liana-tutorials submodule and are not editable from here; these three subtypes only fire inside them.
# Everything else is built with `-W`.
suppress_warnings = ["myst.xref_missing", "image.not_readable", "myst.directive_unknown"]
needs_sphinx = "4.0"

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "scverse",
    "github_repo": "liana",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
# They can be extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "myst_nb",
    "sphinxcontrib.mermaid",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.katex",
    "sphinx_autodoc_typehints",
    "sphinx_design",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinxext.opengraph",
    "scanpydoc",  # theme + elegant type hints; needs to be before linkcode
    *[p.stem for p in (HERE / "extensions").glob("*.py")],
]

autosummary_generate = True
autodoc_member_order = "groupwise"
default_role = "literal"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
myst_heading_anchors = 6  # create anchors for h1-h6
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
]
myst_url_schemes = ("http", "https", "mailto")
# Render ```mermaid fenced blocks (GitHub-native) via the mermaid directive
myst_fence_as_directive = ["mermaid"]
# securityLevel 'loose' is required for clickable nodes (click events) to navigate.
# `mermaid_init_config` is the key for sphinxcontrib-mermaid >=1; `mermaid_init_js`
# is the equivalent for older (<1) releases. Both are set for robustness.
mermaid_init_config = {"startOnLoad": False, "securityLevel": "loose"}
mermaid_init_js = "mermaid.initialize({startOnLoad:true, securityLevel:'loose'});"
nb_output_stderr = "remove"
nb_execution_mode = "off"
nb_merge_streams = True
typehints_defaults = "braces"

always_use_bars_union = True  # use `|` instead of `Union` in types even when building with Python ≤3.14

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "mudata": ("https://mudata.readthedocs.io/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "plotnine": ("https://plotnine.org/", None),
    # add more as needed
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "tutorials/README.md",  # the submodule's own readme, not a docs page
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "scanpydoc"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

html_title = project_name

html_logo = "_static/logo.png"
html_favicon = "_static/logo.png"

html_theme_options = {
    "repository_url": repository_url,
    "repository_branch": "main",
    "use_repository_button": True,
    "use_issues_button": True,
    "path_to_docs": "docs/",
    "navigation_with_keys": False,
    # The crimson of the logo. scanpydoc exposes it as the `--accent-color` CSS variable, which colors the mobile header and the project name.
    "accent_color": "#ba1b57",
    "show_toc_level": 2,
    "footer_content_items": ["copyright.html"],
}

pygments_style = "default"
katex_prerender = shutil.which(katex.NODEJS_BINARY) is not None

nitpick_ignore = [
    ("py:class", "callable"),
    # Internal keyword-argument shapes; documented on the functions that take them
    ("py:class", "liana.method.sc._liana_pipe.SpatialKwargs"),
    ("py:class", "liana.method.sc._liana_pipe.MdataKwargs"),
    # Optional dependencies, absent from the docs build unless `extras` is installed
    ("py:class", "corneto._core.Graph"),
    ("py:class", "corneto.backend._base.ProblemDef"),
    ("py:class", "cell2cell.tensor.tensor.InteractionTensor"),
    ("py:class", "optional"),
    ("py:class", "array-like"),
    ("py:class", "csr_matrix"),
    ("py:class", "tuples"),
    # inherited from MuData/AnnData, whose own docstrings these come from
    ("py:class", "MuData"),
    ("py:class", "zarr.abc.store.Store"),
    ("py:class", "corneto.Graph"),
    # Base classes for *defining* methods, deliberately not in the public API
    ("py:class", "liana.method.sc._Method.Method"),
    ("py:attr", "n_obs"),
    ("py:attr", "n_vars"),
    ("py:attr", "n_var"),
    ("py:attr", "obs"),
    ("py:attr", "obsm"),
    ("py:attr", "var"),
    ("py:attr", "varm"),
    # add more as needed
]


qualname_overrides = {
    "pandas.core.series.Series": "pandas.Series",
    "numpy._typing._array_like.NDArray": ("py:data", "numpy.typing.NDArray"),
    "numpy._typing._array_like.ArrayLike": ("py:data", "numpy.typing.ArrayLike"),
}
