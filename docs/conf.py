from datetime import datetime
from importlib.metadata import metadata

from scverse_doc.config import DEFAULTS
from scverse_doc.registry import intersphinx

info = metadata("liana")
project = info["Name"]
author = info["Author"]
copyright = f"{datetime.now():%Y}, the LIANA+ developers"
version = info["Version"]
urls = dict(pu.split(", ") for pu in info.get_all("Project-URL"))
source_repository = urls["Source"]
release = info["Version"]

bibtex_bibfiles = ["references.bib"]
templates_path = ["_templates"]
# The notebooks under `docs/tutorials` come from the liana-tutorials submodule and are not editable from here; these three subtypes only fire inside them.
# Everything else is built with `-W`.
suppress_warnings = ["myst.xref_missing", "image.not_readable", "myst.directive_unknown"]

extensions = [
    "scverse_doc",
    "sphinxcontrib.mermaid",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
    "IPython.sphinxext.ipython_console_highlighting",
]

myst_url_schemes = ("http", "https", "mailto")
# Render ```mermaid fenced blocks (GitHub-native) via the mermaid directive
myst_fence_as_directive = ["mermaid"]
# securityLevel 'loose' is required for clickable nodes (click events) to navigate.
# `mermaid_init_config` is the key for sphinxcontrib-mermaid >=1; `mermaid_init_js`
# is the equivalent for older (<1) releases. Both are set for robustness.
mermaid_init_config = {"startOnLoad": False, "securityLevel": "loose"}
mermaid_init_js = "mermaid.initialize({startOnLoad:true, securityLevel:'loose'});"
typehints_defaults = "braces"
always_use_bars_union = True  # use `|` instead of `Union` in types even when building with Python ≤3.14

intersphinx_mapping = intersphinx() | {
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "plotnine": ("https://plotnine.org/", None),
}

exclude_patterns = [
    *DEFAULTS.exclude_patterns,
    "tutorials/README.md",  # the submodule's own readme, not a docs page
]

html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_logo = "_static/logo.png"
html_favicon = "_static/logo.png"
html_theme_options = {"secondary_accent": True}

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
    ("py:class", "liana.method.sc._Method.MethodMeta"),
    ("py:attr", "n_obs"),
    ("py:attr", "n_vars"),
    ("py:attr", "n_var"),
    ("py:attr", "obs"),
    ("py:attr", "obsm"),
    ("py:attr", "var"),
    ("py:attr", "varm"),
    # add more as needed
]


xref_overrides = {
    "pandas.core.series.Series": "pandas.Series",
    "numpy._typing._array_like.NDArray": ("py:data", "numpy.typing.NDArray"),
    "numpy._typing._array_like.ArrayLike": ("py:data", "numpy.typing.ArrayLike"),
}
