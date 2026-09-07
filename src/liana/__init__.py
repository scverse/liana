import sys
from importlib.metadata import version

from scanpy._utils import annotate_doc_types

from liana import datasets as ds
from liana import method as mt
from liana import multisample as ms
from liana import plotting as pl
from liana import preprocessing as pp
from liana import resource as rs

__version__ = version("liana")
__all__ = ["ds", "ms", "mt", "pl", "pp", "rs"]

# register short aliases as importable modules (`import liana.mt`)
sys.modules.update({f"{__name__}.{m}": globals()[m] for m in __all__})
annotate_doc_types(sys.modules[__name__], "liana")

del sys, annotate_doc_types, version
