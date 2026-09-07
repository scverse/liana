import numpy as np
from anndata import AnnData
from mudata import MuData
from pandas import DataFrame, concat

from liana._core._common import _check_if_installed
from liana._core._constants import DefaultValues as V
from liana._core._docs import d
from liana._core._pipe_utils import prep_check_adata
from liana._core._types import get_obsm_frame, get_var
from liana.preprocessing import obsm_to_adata


@d.dedent
def estimate_metalinks(
    adata: AnnData,
    resource: DataFrame,
    pd_net: DataFrame,
    t_net: DataFrame | None = None,
    x_name: str = "metabolite",
    y_name: str = "receptor",
    use_raw: bool = V.use_raw,
    layer: str | None = V.layer,
    verbose: bool = V.verbose,
    **kwargs: object,
) -> MuData:
    """
    Estimate Metabolites from anndata object, and return a MuData object of metabolites and receptors.

    Parameters
    ----------
    adata
        Annotated data matrix.
    resource
        Resource to use for ligand-receptor inference.
    pd_net
        Metabolic production-degradation network to use.
    t_net
        Transport set to use.
    x_name
        Name of the metabolite modality.
    y_name
        Name of the receptor modality. Must be present as a column in the resource.
    %(use_raw)s
    %(layer)s
    %(verbose)s
    **kwargs
        Additional arguments to pass to the decoupler-py functions.
        Method-specific arguments are not supported.

    Returns
    -------
    A MuData object with metabolite & receptor assays.

    Examples
    --------
    `pd_net` is a metabolite-to-enzyme network whose weights say whether a gene produces (+) or degrades (-) a metabolite, and `resource` links metabolites to their receptors.
    Both normally come from MetalinksDB (see :func:`liana.rs.get_metalinks`); toy ones are built here so the example stays offline:

    >>> import numpy as np
    >>> import pandas as pd
    >>> import liana as li
    >>> adata = li.ds.generate_toy_adata()
    >>> genes = adata.var_names[:16].tolist()
    >>> pd_net = pd.DataFrame({"source": np.repeat(["HMDB0000122", "HMDB0000148"], 8), "target": genes, "weight": 1.0})
    >>> resource = pd.DataFrame({"source": ["HMDB0000122", "HMDB0000148"], "receptor": ["CD4", "ITGB2"]})
    >>> mdata = li.mt.estimate_metalinks(adata, resource=resource, pd_net=pd_net)

    Metabolite abundances are estimated from the enzyme expression and returned in
    a `'metabolite'` modality, next to the receptors in a `'receptor'` one. Pass
    `t_net` to additionally require a transporter for metabolites that cannot cross
    the membrane on their own. The result is the input to
    ``liana.mt.bivariate`` or to any single-cell method, with
    `x_mod='metabolite'` and `y_mod='receptor'`.
    """
    dc = _check_if_installed(package_name="decoupler")
    ad = prep_check_adata(
        adata,
        layer=layer,
        use_raw=use_raw,
        verbose=verbose,
        groupby=None,
        min_cells=None,
        uns=dict(adata.uns),
        obsm=adata.obsm,
    )
    dc.mt.ulm(ad, net=pd_net, raw=False, verbose=verbose, **kwargs)
    met_est = get_obsm_frame(ad, "score_ulm")

    if t_net is not None:
        dc.mt.waggr(ad, t_net, times=0, raw=False, verbose=verbose, fun="wmean", **kwargs)

        out_est = get_obsm_frame(ad, "score_waggr")
        intersect = np.intersect1d(met_est.columns, out_est.columns)
        out_est = out_est[intersect]

        out_mask = out_est > 0

        mask = np.ones(out_est.shape)
        mask[out_mask == 0] = 0

        # mask those with transporters
        mmat = met_est[intersect] * mask

        # concat the rest
        coldiff = np.setdiff1d(met_est.columns, mmat.columns)
        mmat = concat([mmat, met_est[coldiff]], axis=1)

    else:
        mmat = met_est

    resource = resource[resource["source"].isin(mmat.columns)].copy()
    receptor = ad[:, get_var(ad).index.isin(np.unique(resource[y_name]))]

    ad.obsm["mmat"] = mmat
    metabolites = obsm_to_adata(ad, "mmat")

    return MuData({x_name: metabolites, y_name: receptor})
