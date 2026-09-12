from collections.abc import Callable
from functools import wraps
from typing import Any, cast

from scverse_misc import Deprecation, deprecated

from ._annulus import annulus
from ._circle_plot import circle
from ._connectivity_plot import connectivity
from ._dotplot import dotplot, dotplot_by_sample
from ._elbow import elbow
from ._feature_by_group import feature_by_group
from ._lric_plot import lric_divergence, lric_lineplot
from ._misty_plots import misty_contributions, misty_interactions, misty_target_metrics
from ._tileplot import tileplot

__all__ = [
    "annulus",
    "circle",
    "connectivity",
    "dotplot",
    "dotplot_by_sample",
    "elbow",
    "feature_by_group",
    "lric_divergence",
    "lric_lineplot",
    "misty_contributions",
    "misty_interactions",
    "misty_target_metrics",
    "tileplot",
]

_RENAMED_IN = "2.0.0"
"""The release that settled the plotting names.

Plot functions are bare nouns, as in :mod:`scanpy.pl`, and carry the prefix of the method they belong to when they only apply to that method.
"""


def _renamed_to[F: Callable[..., Any]](current: F, old_name: str) -> F:
    """Expose ``current`` under the ``old_name`` it went by before :data:`_RENAMED_IN`.

    The alias carries the old name so that the warning names the function that was actually called, and it is a distinct object so that :func:`~warnings.deprecated` marks only the alias -- which also lets a type checker flag the old name without flagging the new one.
    """

    @wraps(current)
    def alias(*args: Any, **kwargs: Any) -> Any:
        return current(*args, **kwargs)

    alias.__name__ = alias.__qualname__ = old_name
    message = Deprecation(_RENAMED_IN, f"Use `liana.pl.{current.__name__}` instead.")

    return cast("F", deprecated(message)(alias))


annulus_plot = _renamed_to(annulus, "annulus_plot")
circle_plot = _renamed_to(circle, "circle_plot")
contributions = _renamed_to(misty_contributions, "contributions")
interactions = _renamed_to(misty_interactions, "interactions")
lric_divergence_plot = _renamed_to(lric_divergence, "lric_divergence_plot")
target_metrics = _renamed_to(misty_target_metrics, "target_metrics")
