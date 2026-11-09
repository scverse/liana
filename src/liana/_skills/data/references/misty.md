# MISTy: multi-view spatial modelling

Aim: for each target feature (the `intra` view), how much do other views (same location, immediate
neighbours, wider neighbourhood, another modality) explain it, and which predictors matter.
Unsupervised; an LR resource is optional. Tutorial: `misty`.

## Build the views

- `li.mt.lrMistyData(adata, bandwidth=...)`: receptors -> `intra`, ligands -> `extra`; fit it with
  `bypass_intra=True` so receptors are not predicted from each other.
- `li.mt.genericMistyData(intra=..., extra=..., bandwidth=...)`: `intra` + `juxta` (`n_neighs` nearest)
  + `para` (kernel-weighted within `bandwidth`) from any AnnData(s), e.g. cell-type proportions and
  pathway activities.
- `li.mt.MistyData({"intra": a, "para": b}, obs=..., enforce_obs=False)`: any dict of views (`intra`
  mandatory); with `enforce_obs=False` views may have different observations, their connectivities in
  `.obsm` (from `li.pp.spatial_neighbors(view, reference=intra_coords)`).

`bandwidth` uses the `misty_rbf` kernel.

## Fit and read

```python
from liana.method.sp import RandomForestModel, LinearModel
misty = li.mt.lrMistyData(adata, bandwidth=50)
misty(model=LinearModel, bypass_intra=True)          # RandomForestModel: non-linear, slower
misty.uns["target_metrics"]    # per target: intra_R2, multi_R2, gain_R2, one contribution column per view
misty.uns["interactions"]      # per view, predictor, target: importances
li.pl.misty_target_metrics(misty, stat="gain_R2", top_n=20)
li.pl.misty_contributions(misty)
li.pl.misty_interactions(misty, view="extra", top_n=15)
```

- `model` is a class, not an instance. Random forest importances are impurity decreases (always
  positive); linear importances are t-values (sign = direction).
- `maskby="obs_col"` fits each group separately. Plots take `filter_fn`, `top_n`, `ascending`.
- Save with `misty.write_h5mu(path)`; reload with `mudata.read_h5mu(path)` and wrap in `li.mt.MistyData(mdata)`.
- No multi-sample aggregation: run per slide and concatenate `target_metrics`/`interactions` before
  comparing conditions.
