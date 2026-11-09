# Troubleshooting: message -> cause -> fix

Quote the message to the user, name every listed cause, then fix.

**`ValueError: mat contains negative values`** (single-cell methods, `lric`)
- Cause: scaled or z-scored input, in `.X`, the chosen `layer`, or `.raw` with `use_raw=True`.
- Fix: point at non-negative log1p expression (`layer=` or `.X`); never scale for these methods. Only
  spatial methods with `x_transform`/`y_transform` (`bivariate`, `inflow`) take scaled input.

**`ValueError: mat contains values too large to have been log-transformed`** (`logfc`, `rank_aggregate`)
- Cause: raw counts; the max exceeds where inverting the log overflows (~709 for natural log).
- Fix: `sc.pp.normalize_total` + `sc.pp.log1p` into `.X` or a layer.

**Warning `mat does not look log1p-normalised`** (single-cell methods)
- Cause: all-integer values or a max above 50, i.e. likely counts. Scores are still produced but meaningless.
- Fix: as above. Do not ignore it.

**`ValueError: Please check if appropriate organism/ID type was provided!`**
- Cause 1: `var_names` are not gene symbols (Ensembl IDs, the wrong matrix or modality, protein names).
- Cause 2: non-human data with the human `consensus` resource.
- Fix: map IDs to symbols; for mouse `resource_name="mouseconsensus"`; for a fuller map or other
  organisms, translate the resource with HCOP (`li.rs.get_hcop_orthologs`).

**``ValueError: The `map_df` DataFrame must have two columns named 'source' and 'target'!``**
- Cause: the full HCOP table (15 columns) passed to `li.rs.translate_resource`/`translate_column`.
- Fix: `li.rs.get_hcop_orthologs(columns=["human_symbol", "mouse_symbol"])`, then rename to `source`, `target`.

**MISTy results (`target_metrics`, `interactions`) missing after a MuData round trip**
- Cause: liana < 1.8.1 dropped `.uns` when wrapping a `MuData` in `li.mt.MistyData`.
- Fix: upgrade liana; on old versions re-run the fit after reloading.
