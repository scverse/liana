# MISTy: multi-view spatial modelling

Aim: for each target feature in a spot or cell (the `intra` view), how much do features in other
views (the same location, the immediate neighbours, the wider neighbourhood, another modality)
explain its expression, and which predictors matter. Unsupervised, no LR resource needed unless
you want one. Tutorial: `misty`.

## Build the views

```python
misty = li.mt.lrMistyData(adata, bandwidth=200)
# receptors -> intra view, ligands -> extra view; scores all ligand x receptor combinations
misty = li.mt.genericMistyData(intra=comps, extra=acts, bandwidth=200, cutoff=0.05)
# intra + juxta (n_neighs nearest) + para (kernel-weighted within bandwidth) from any AnnData(s)
misty = li.mt.MistyData({"intra": a, "para": b}, obs=mdata.obs, enforce_obs=False)
# any dict of views; the key "intra" is mandatory; non-intra views carry their own connectivities
```

`bandwidth` is in coordinate units (kernel `misty_rbf` by default). With `enforce_obs=False` views
may have different observations; their connectivities then sit in `.obsm` (as written by
`li.pp.spatial_neighbors(view, reference=intra_coords)`).

## Fit and read

```python
from liana.method.sp import RandomForestModel, LinearModel, RobustLinearModel
misty(model=RandomForestModel)                # or LinearModel / RobustLinearModel; bypass_intra=True for extra-only
misty.uns["target_metrics"]   # per target: intra_R2, multi_R2, gain_R2, one contribution column per view
misty.uns["interactions"]     # per view/predictor/target: importances
li.pl.target_metrics(misty, stat="gain_R2", top_n=20)
li.pl.contributions(misty)
li.pl.interactions(misty, view="para", top_n=15, filter_fn=lambda d: d["importances"] > 0.5)
```

- `model` is a class, not an instance. Random forest importances are impurity decreases (always
  positive); `LinearModel` importances are t-values, so a negative value is a negative association.
- `bypass_intra=True` skips using intra features as predictors (useful when intra is a different
  modality than the extra views). `maskby="obs_col"` fits each group separately.
- Plots return plotnine objects and take `filter_fn`, `top_n`, `ascending`.
- Save with `misty.write_h5mu(path)`; reload with `mudata.read_h5mu(path)` and wrap it in
  `li.mt.MistyData(mdata)` again (`.uns` results are preserved since 2.0.0; older versions dropped them).
- No built-in multi-sample aggregation: run per slide and concatenate `target_metrics` /
  `interactions` yourself before comparing conditions.
- Requires scikit-learn and statsmodels (core deps); squidpy is not needed.
