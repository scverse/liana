# Changelog

## 2.0.0 (unreleased)

liana+ now has a new home under the scverse organisation.

### Changed

- **`matplotlib.pyplot.show` is banned in `src`.** The hang fixed in `li.pl.annulus` was invisible to CI, which runs headless and so turns `show` into a no-op -- the call only blocks where someone has a display. A lint rule catches the next one at the point it is written rather than at the point a user runs it.

- **LRIC groups its edge list by counting sort.** The group key is a cell-type pair crossed with a radius tile, so it spans a few hundred values over tens of millions of edges; placing each edge in one pass beats paying a factor of `log(n_edges)` for the same order. Ascending traversal keeps ties in input order, so the result matches the stable sort it replaces exactly, and the group offsets fall out of the histogram rather than a second search over the sorted keys. `li.mt.lric(groupby=...)` goes from 5.4 s to 3.2 s on 14k spots over 36M edges.

- **Binning that edge list no longer doubles peak memory.** Dropping self-pairs, assigning tiles and compacting used to be a chain of numpy expressions, each allocating a full-length intermediate; one pass that counts and then fills allocates only what it returns. Peak drops from 1954 MB to 1013 MB for the same 579 MB of edges, and the result is unchanged.

- **The permutation null no longer carries an untested fallback.** The compiled trimean kernel assumed a non-negative expression matrix, so anything else -- a scaled `layer`, say -- fell back to aggregating gathered rows, a path no test ever reached. Splicing the implicit zeros in at the position they sort to, rather than assuming they come first, covers negative values too, which removes the fallback, its dispatch and `joblib` from the module.

- **`MethodMeta` no longer keeps a registry of every instance ever built.** The class held a list of weak references, appended to in `__init__` and never pruned, only to answer `li.mt.get_method_scores()`: 20 entries for the 9 methods liana ships, since a `Method` and the `MethodMeta` it wraps each registered, growing without bound as methods are constructed, and defining a custom method silently changed the scores reported for the whole process. The scores are known where the methods are defined, so they are read from there. This also drops the import-order constraint `liana/__init__.py` documented.

- **Permutation nulls are built by compiled kernels instead of `joblib`.** Both the mean and the trimean null read the CSR buffers directly, one pass over the non-zeros per permutation, and never materialise a permuted copy of the matrix; the trimean sorts each gene's stored entries rather than densifying the group. On 50k cells x 600 genes x 200 permutations, the mean null goes from 2.8 s to 0.45 s and CellChat's trimean null from 68 s to 4.5 s (8 threads). `n_jobs` previously made the permutations *slower* than serial, because a task per permutation re-pickled the sparse matrix each time. Results are unchanged for the trimean and now depend only on `seed`, never on `n_jobs`; the mean null sums in double precision where it previously inherited scipy's single-precision accumulation.

- **A sample carrying a single cluster now yields `p = 1` throughout.** Every permutation leaves that cluster's membership untouched, so it has to score exactly as the observation does, but the observed and permuted sides are accumulated by different routines and the tie did not survive that. Permuted scores within single-precision resolution of the observed one now count as tied. Only reachable where a `sample_key` split, or `min_cells`, leaves one cluster standing; on the toy data this moved 9 of 2115 `by_sample` rows off values that were pure float noise.

- **`liana.pl` plot names follow one convention.** Plot functions are bare nouns, as in `scanpy.pl`, and carry the prefix of the method they belong to when they only apply to it. The old names still resolve, via `scverse_misc.deprecated`, so a type checker flags them and calling one raises a `FutureWarning`:

  | Was | Now |
  |---|---|
  | `li.pl.circle_plot` | `li.pl.circle` |
  | `li.pl.annulus_plot` | `li.pl.annulus` |
  | `li.pl.lric_divergence_plot` | `li.pl.lric_divergence` |
  | `li.pl.target_metrics` | `li.pl.misty_target_metrics` |
  | `li.pl.contributions` | `li.pl.misty_contributions` |
  | `li.pl.interactions` | `li.pl.misty_interactions` |

- **`liana_pipe` was split into named stages.** Assembling the ligand-receptor statistics, scoring them and aggregating across methods are now three functions rather than one 616-line one with five underscore-prefixed pseudo-private parameters. The consensus path has its own entry point (`liana_pipe_consensus`), so `liana_pipe` no longer dispatches on `_score.method_name == "Rank_Aggregate"` and always returns a `DataFrame`. Internal only -- `li.mt.*` and `li.mt.rank_aggregate` are unchanged.

- **`li.mt.lric(pair_chunk=...)` is deprecated and ignored.** The weighted numerator is accumulated by a compiled kernel that holds no per-chunk temporaries, so there is nothing left to tune for memory. The same change makes it about 10x faster (3.7 s to 0.4 s on 2M edges x 500 pairs).

- Locating ligands, receptors and cluster labels in the expression matrix uses `Index.get_indexer` instead of a `numpy.where` scan per interaction, which was quadratic in the number of interactions (1.42 s to 0.005 s for 60k interactions over 2k genes). An interaction naming a gene absent from `adata.var_names` now raises `KeyError` instead of silently indexing from the end.

- Permutation progress bars track completed permutations. They previously wrapped the submission generator, so the bar filled immediately and then stalled.

- **`li.ms.nmf` no longer draws the elbow plot** (#99). The error curve is stored in `adata.uns["nmf_errors"]` as before; plot it with the new `li.pl.elbow`, which returns the figure like every other `li.pl` function.

- **Breaking: the public namespaces were reorganised to match scverse-style.** The top-level API is now `li.ds`, `li.ms`, `li.mt`, `li.pl`, `li.pp`, `li.rs` (`li.ut`, `li.mu` and `li.testing` are gone; `li.ds`, `li.pp` and `li.ms` are new). The functions themselves are unchanged — only their import path moved:

  | Was | Now | Moved |
  |---|---|---|
  | `li.ut` (`utils`) | **removed**, split three ways | — |
  | `li.ut.spatial_neighbors` / `spatial_pair_proximity` / `obsm_to_adata` / `interpolate_adata` / `expand_coordinates` / `query_bandwidth` / `neg_to_zero` / `zi_minmax` | **`li.pp`** (`preprocessing`, new) | preprocessing/coordinate utilities |
  | `li.ut.get_factor_scores` / `get_variable_loadings` / `mdata_to_anndata` | **`li.ms`** (`multisample`) | multi-sample helpers |
  | `li.ut.get_lric_auc` / `get_lric_divergence` | **`li.mt`** | live with the LRIC method |
  | `li.mu` (`multi`) | **renamed `li.ms`** (`multisample`) | `nmf` / `estimate_elbow`, `adata_to_views` / `lrs_to_views` / `lrdata_to_mudata` / `filter_view_markers`, `to_tensor_c2c` |
  | `li.mu.df_to_lr` | **`li.mt.df_to_lr`** | sits with the methods |
  | `li.testing` | **renamed `li.ds`** (`datasets`) | `kang_2018`, `generate_toy_adata` / `generate_toy_mdata` / `generate_toy_spatial`, `sample_lrs` |
  | `li.mt.build_prior_network` | **`li.rs.build_prior_network`** | it builds a resource, not a method result (`li.mt.find_causalnet` stays) |

- The six namespaces are also importable directly (`import liana.ms`, `import liana.pp`, …); the removed aliases (`import liana.ut` / `liana.mu` / `liana.testing`) no longer resolve — update both attribute access and direct imports.
- **Breaking: `use_raw` now defaults to `False` (was `True`) everywhere.** Methods read `adata.X` by default instead of `adata.raw.X`, aligning with the scverse ecosystem (scanpy auto/`None`, squidpy/decoupler `False`), where log-normalised expression is expected in `.X`. Pass `use_raw=True` explicitly to keep reading `.raw`. Relatedly, `li.ds.generate_toy_adata`/`generate_toy_spatial` now ship log-normalised expression in `.X` (matching `generate_toy_mdata`), so the default path works on valid data.
- **Internal: shared machinery consolidated into a private `liana._core` package.** `liana._common`, `_constants` and `_docs` moved under `liana._core`, and the pipeline internals (`_pipe_utils`: `_pre`, `_aggregate`, `_get_mean_perms`, …) moved out of `liana.method` into `liana._core`. The public subpackages now depend on `_core` rather than reaching into one another, removing cross-imports between `method`/`multisample`/`plotting`/`preprocessing`/`resource`. No user-facing symbols changed.
- **Breaking: spatial proximity weighting in the single-cell methods is opt-in** (#255). `spatial_key` now defaults to `None` for all `li.mt` methods and `rank_aggregate` (the methods previously weighted silently whenever `obsm["spatial"]` existed; `rank_aggregate` never did). Passing a key that is not in `adata.obsm` raises `KeyError` instead of silently skipping the weighting.
- **Typed codebase (#255).** Synced with the scverse cookiecutter template; `mypy` runs in pre-commit and CI; `.toarray()`/`.A` replaced by `fast-array-utils`. Output is unchanged. Two `_expm1_base` test expectations were corrected: the old tests passed `(base, X)` in swapped order.
- `docrep` replaced by a small in-house docstring processor; an unknown placeholder now raises at import instead of warning.
- **Proportion thresholds standardised.** `nz_prop` (proportion of *all* observations with a non-zero value) and `expr_prop` (proportion *within each cell-type group*) are now distinct, and both defaults live in `DefaultValues` (`liana._core._constants`). `expr_prop` now defaults to `0.05` (was `0.1`) across all single-cell methods, `rank_aggregate`, `li.mt.df_to_lr` and `li.mt.lric`'s directed (`groupby`) mode; `nz_prop` defaults to `0.05` and is shared by `li.mt.bivariate`, `li.mt.inflow` (was `0.001`) and `li.mt.lric`'s cell-type-agnostic mode (previously unmasked, `0.0`). `li.mt.lric` gained `nz_prop` for its agnostic mode, while its `expr_prop` now only applies when `groupby` is set.

- liana+ repository moved from saezlab/liana-py to scverse/liana.

### Added

- **Claude Code Agent Skill** bundled in `src/liana/_skills`, and `liana-install-skills` console script that copies it to `~/.claude/skills/liana`. Run it once after installing; re-run with `--force` after upgrading. See the README section "Claude Code Skill".

### Fixed

- **The single-cell methods reject negative expression instead of silently scoring it as `NaN`.** `li.mt` assumes non-negative input in three places -- `np.sqrt(ligand_means)` in SingleCellSignalR, `np.log2(mean + 1)` in `_calc_log2fc` and `gmean` in Geometric Mean -- each of which yields `NaN` on scaled or centred data. On pbmc68k's scaled `.X` the default `rank_aggregate` returned `lrscore` 93.1% `NaN` and **`magnitude_rank` 100% `NaN`**, with no error and no warning; `geometric_mean` returned a `NaN` statistic for 93.1% of rows and, because a `NaN` never satisfies `perm >= obs`, gave every one of them `gmean_pvals = 0.0` -- the most significant value in the range, and 98.4% of the column. `prep_check_adata` gained `block_negatives`, which the single-cell pipe passes and which raises naming `use_raw=True` / `layer=` as the way out; it defaults to `False`, so `li.mt.bivariate`, MISTy and LRIC keep working on signed data. More reachable since `use_raw` became `False` by default (below), which pointed the default path at `.X`.

- **CellChat no longer corrupts the permutation null of every method that follows it in a consensus.** `_get_means_perms` wrote its `X / mat_max` normalisation back into the shared `adata`, so a later permutation method built its null on the shrunken matrix while its observed statistic came from the unnormalised one. This was a silient bug what did not affect any default results, but it could if the aggregate_rank was constructed in a different order.

- **`li.rs.get_metalinks` and `li.rs.get_hcop_orthologs` no longer download into the working directory.** Both wrote their file to `os.getcwd()`, so calling either from a checkout dropped an untracked artifact into the repo, changing directory silently re-downloaded, and two processes in one directory raced on the same path. Both go through :func:`pooch.retrieve` now, as the rest of scverse does, caching under :attr:`scanpy.settings.datasetdir` alongside what `li.ds` fetches; `_download_metalinksdb` takes a `cache_dir` for callers that want their own. MetaLinksDB is checked against a pinned `sha256`, so a truncated or corrupted copy is re-fetched rather than served from the cache forever -- the previous code only rejected a file of length zero. Neither call had passed a timeout, so a stalled server blocked indefinitely.

- `li.rs.get_metalinks_values` opened two connections to the database and closed one.

- **`li.pl.annulus` returns its figure instead of calling `matplotlib.pyplot.show`.** Showing from inside a library takes the decision away from the caller, and under an interactive backend it blocks in the GUI event loop -- which hung the function indefinitely in any script, and hid only because a headless backend turns `show` into a no-op. It takes `return_fig` and returns a `Figure`, as the rest of `li.pl` does; a notebook still renders it, and a script decides for itself when to show.

- **Argument validation no longer runs on `assert`.** Eight checks on user input were assertions, which `python -O` strips, letting bad input through silently; several carried no message. They raise `ValueError` or `KeyError` now, as do the six places that raised `AssertionError` for a bad argument -- `except ValueError` around a liana call catches those. `liana.ms.filter_view_markers` warns with `UserWarning` rather than bare `Warning`, so the warning can be filtered by category.

- **Spatial proximity weighting now reaches the p-values of the permutation-based methods.** `spatial_key` weighted both the observed score and the permuted null by the same per-interaction factor, which cancels out of `perm * w >= obs * w` -- so on toy data 92.5% of CellPhoneDB p-values were bit-identical with and without weighting, and the rest only moved because a zero weight forced them to 1. Only the observed statistic is weighted now, so a spatially distant pair needs a correspondingly stronger expression signal to clear the null. Affects `li.mt.cellphonedb`, `li.mt.cellchat` and `li.mt.geometric_mean` when `spatial_key` is passed; magnitudes are unchanged.

- **`rank_aggregate`'s `magnitude_rank` now ranks each score column once.** Connectome and NATMI share `expr_prod` as their magnitude score, and the consensus previously ranked it once per method, so the second pass ranked the ranks and reversed its contribution. `magnitude_rank` now agrees with the individual magnitude scores it aggregates; `specificity_rank` and all per-method scores are unchanged.
- **`rank_aggregate(n_perms=None)` now returns `specificity_rank`.** Skipping the permutations used to drop the consensus to `Magnitude` only, so the column was missing altogether. Specificity is now aggregated over the scores that need no permutations -- with the default methods, Connectome's `scaled_weight`, log2FC's `lr_logfc` and NATMI's `spec_weight` -- while a permutation p-value (`cellphone_pvals`) is left out for that run only. `magnitude_rank` is unchanged. A consensus that ends up with a single specificity score warns under `verbose=True`.
- `li.rs.get_metalinks(source="...")` filtered per character of the string; it now filters on the whole value (#255).
- `return_all_lrs=True` works under pandas 3 (chained `fillna(inplace=True)` was a no-op under Copy-on-Write); the `pandas<3` pin from #244 is lifted.

### Packaging

- **`requests` dropped from `[extras]`.** Nothing imports it since the downloads moved to `pooch`, which brings it along in any case.

- **`mudata>=0.4` required.** With anndata ≥ 0.13, `mudata<0.4` fails on `write_h5mu` (`AttributeError: 'NoneType' object has no attribute 'startswith'`, the `None` layer key that anndata now exposes), which also affected saving `MistyData` objects. `muon>=0.1.9` is required alongside it, older muon cannot import with `mudata>=0.4`.
- **Version derived from git tags** via [hatch-vcs](https://github.com/ofek/hatch-vcs), as in scanpy and pertpy. `bumpversion` and `.bumpversion.cfg` are gone; `liana.__version__` now reads the installed metadata via `importlib.metadata.version("liana")`. Releases are made by publishing a `vX.Y.Z` tag on GitHub.
- **Requires Python ≥ 3.12, anndata ≥ 0.13, scanpy ≥ 1.12** (#255). scanpy < 1.12 cannot import liana's PEP 695 type aliases.

- **Tutorial CI dependency recipes.** The tutorial notebooks (`docs/tutorials/notebooks`, see Documentation) are now runnable from declared extras rather than ad-hoc `pip install` lines, with a committed `uv.lock` for reproducibility. Two install targets cover all 14 notebooks: `uv sync --extra tutorials` (12 CPU notebooks) and `uv sync --extra tutorials-gpu` (the two heavy ones, `inflow_mofaflex` + `liana_c2c`). `tutorials` layers `liana[extras]` with the notebook-only viz/runtime packages (`matplotlib`, `seaborn`, `adjustText`, `marsilea`, `pycrosstalker`); `tutorials-gpu` adds `tensorly`, `mofaflex` and `torch`. Naming follows pertpy/scvi-tools conventions.
- **`squidpy` added to `[extras]`** — it backs `li.mt.MistyData` and `li.pp.spatial_neighbors` (lazy-imported) and was the one optional-feature dependency the extra never declared.
- **`torch` is routed to the CPU wheel index** via `[tool.uv.sources]`, keeping tutorial CI off the ~2.5 GB CUDA build; swap the index url for cu124 when GPU CI lands. **`mofaflex` is pinned to git `@main`** there — `inflow_mofaflex.ipynb` needs the unreleased 0.2.0 terms/priors API, which PyPI 0.1.2 does not provide; the override is uv-only, so published metadata stays PyPI-clean.

### Documentation

- **Tutorials moved to a dedicated repository** ([dbdimitrov/liana-tutorials](https://github.com/dbdimitrov/liana-tutorials)) and pulled back in as a git submodule at `docs/tutorials` (the pertpy-tutorials pattern). `docs/notebooks/` was removed; the toctree now lives in `docs/tutorials.md` and renders the notebooks from `docs/tutorials/notebooks/*.ipynb`. Rendered tutorial URLs move from `…/notebooks/<name>.html` to `…/tutorials/notebooks/<name>.html`. RTD builds the submodule (`submodules: include: all`); the tutorial-execution extras (`tutorials` / `tutorials-gpu`) stay in liana-py.
- All 14 tutorials were re-run and their headings normalised to a consistent hierarchy.

## 1.10.0 (27.08.2026)

### Changed

- **LRIC & cross-PCF reworked onto an analytical null and one shared, exact binning** (#250, by @AtheerAS). `li.mt.lric` and `li.mt.cross_pcf` now compute `g(r)` against a closed-form random-labelling null conditioned on the observed cell positions, replacing the CSR area expectation with bounding-box edge correction; cell-type-pairwise LRIC decomposes the full coupling into architecture-only (`g_pcf`, identical to `cross_pcf`) and expression-only (`g_expr`) components. Numerator and denominator are binned on a single shared partition of disjoint `radius_step`-wide tiles, with each output annulus reconstructed as `annulus_steps` consecutive tiles — fixing deflated `g` under overlapping annuli, bin-edge convention mismatches on gridded coordinates, and zero-distance pairs. The float `annulus_width` parameter is replaced by `annulus_steps` (int ≥ 1) in `lric`, `cross_pcf` and `annulus_plot`.
- Internal logging and result-resolution helpers were consolidated into `liana._common`; resolution consistently prefers `adata` over `liana_res` and raises a `ValueError` when neither is given.
- `li.mt.cross_pcf` gained `groupby_pairs`, matching `li.mt.lric`: it restricts the emitted cell-type combinations (matched regardless of orientation, since `g(r)` is symmetric) and folds the referenced cell types into `cell_types`. Both methods now warn when `groupby_pairs` names a cell type that is not in the data, or matches nothing at all, instead of silently returning an empty result.
- The three `g(r)` variants (`cross_pcf`, agnostic and pairwise `lric`) now share their geometry prelude, edge grouping, random-labelling null, LR weighting, cell-type indexing and long-format output instead of repeating them, so numerator and denominator cannot drift apart between variants. Output is unchanged.
- **LRIC / cross-PCF results are long-format DataFrames.** Both methods return/store a tidy frame (`source`, `target`, `ligand_complex`, `receptor_complex`, `interaction`, `radius`, `g`, plus `g_expr`/`g_pcf` for pairwise LRIC) in `adata.uns[key_added]`, column-compatible with the dotplot family; `cross_pcf` emits each unordered cell-type pair once. The LRIC tutorial was rewritten for the new API.

### Added

- **`li.ut.get_lric_auc`** — ranks interactions by the span-normalised area under `transform_fn(g(r))` (default: log2 with `g` floored at `0.05`; pass `np.log2` for the strict behaviour that drops non-finite bins), and reports `peak_radius`, the radius of the largest deviation from the null; its output feeds `li.pl.dotplot` directly. When the result is empty, a warning logs why (too few radius bins in-window, or too few finite bins per interaction).
- **`li.ut.get_lric_divergence`** — the span-normalised area between two `g(r)` curves and the radius where their separation peaks. Curves are selected as `{column: value}` dicts over any columns of the result, so concatenated results from several samples/conditions (e.g. with a `condition` column) support cross-condition comparison of the same interaction; unpinned replicate rows average into one curve. Same floored-log2 default transform as `get_lric_auc`.
- **`li.pl.lric_lineplot`** — the `g(r)` profile of a single interaction, with the pairwise decomposition drawn as separate curves.
- **`li.pl.lric_divergence_plot`** — the two `transform_fn(g(r))` curves behind a `get_lric_divergence` result, with the area between them shaded and `r_star` marked.

## 1.9.0 (19.08.2026)

### Added

- **`Examples` sections across the public API (#192).** Minimal runnable calls on `liana.testing` toy data that point at where the result lands, following pertpy's style; `hatch run doctest:run` executes them (the few that need a download are shown as literal blocks).

### Fixed

- **MiSTy's `LinearModel` applied `n_jobs` to the first target only, then forked a worker per core for every other one.** `fit` *popped* `n_jobs` from state shared across targets, so all but the first fell back to the `-1` default -- spending ~4s on joblib pool startup to cross-validate a linear regression. It is now read rather than popped, defaults to `1`, and is documented; results are bit-identical.
- **`import liana.mu` raised `ModuleNotFoundError`.** `mu` was the one short alias missing from the `sys.modules` registration, so it failed while its four siblings resolved.
- **`_calc_log2fc` raised a bare `ZeroDivisionError` when a group had nothing to compare against (#93).** A `sample_key` group holding a single `groupby` category leaves the "rest" side empty; a `ValueError` now names the cause.
- **Dropped the `MAML2-NOTCH1/2/3/4` rows from the consensus resource (#207, PR #247).** MAML2 is a nuclear transcriptional co-activator, not a surface ligand, so these were a curation artifact; a regression test keeps them out.
- **Corrected the `CD38-PECAM1` direction in the consensus resource (#218).** The pair is directed `PECAM1` (ligand) -> `CD38` (receptor), as in CellPhoneDB and the literature (PMID: 7542249); the consensus row was flipped. Also guarded against `SMAD3` (a transcription factor) appearing as a consensus receptor. Regression tests keep both in check.
- **`_get_means_perms` mutated the caller's matrix and upcast it to float64.** `adata.X /= norm_factor` wrote into a buffer that can be shared with `adata.raw.X`; the division is now out-of-place and cast back to the original dtype, halving peak memory.
- **Three plotting bugs surfaced by the new tests:** `li.pl.dotplot`/`li.pl.tileplot` constructed `ValueError`s for a missing `orderby`/`orderby_ascending` but never raised them; `li.pl.feature_by_group` called `_logg.warning(...)` on a function, which would have raised `AttributeError`; `li.pl.contributions` assumed a categorical `target` and failed on a plain string column.

### Changed

- **Breaking: the public namespaces no longer export internals.** `Method` and `MethodMeta` are now `_Method`/`_MethodMeta` (base classes for defining methods, not user-facing API); `explode_complexes` and `filter_reassemble_complexes` stay behind the private `_reassemble_complexes` module and left `docs/api.md`; the `LRIC` class is no longer exported -- call `li.mt.lric` or `li.mt.cross_pcf`; and the duplicate `li.multi.process_scores` was dropped in favour of `li.mt.process_scores`.
- **Tests now mirror the package layout and share their data via fixtures (#194).** `tests/` follows `src/liana` with one directory per public namespace (`method/{sc,sp}`, `multi`, `plotting`, `resource`, `utils`); private subpackages are not mirrored, matching decoupler and squidpy. Module-level test objects were replaced by fixtures in `tests/conftest.py`, so no test inherits another's mutations, and the download fixtures in `tests/resource/conftest.py` cache to `tests/.cache`. Plotting tests were extended to assert on the plot's underlying data rather than only that a figure was produced.
- **Tests that need the internet are marked `network`,** so `pytest -m "not network"` runs the suite offline; `--strict-markers` is enabled.
- **Assertions that could not fail were replaced or removed** -- membership checks against a `Series` (which test the index, not the values), `assert ... is not None` on always-present AnnData attributes, and checks made against a test's input rather than its output. `liana.testing._sample_target_metrics` and `_sample_interactions` now require a `seed`, so the misty plot tests no longer depend on global RNG state.

## 1.8.1 (15.07.2026)

### Added

- **`li.ut.expand_coordinates`** — utility that lays out the spatial coordinates of multiple samples side-by-side on a non-overlapping grid, enabling multi-sample spatial analyses (e.g. a joint `spatial_neighbors` graph) without cross-sample coordinate overlap. Exposed in `li.ut` and the API reference. (#238)
- **MOFA-Flex inflow tutorial** (`inflow_mofaflex.ipynb`) showing how to combine the inflow score with MOFA-Flex to extract spatially-resolved, single-cell-derived cell-cell communication programs.

### Changed

- **LRIC / cross-PCF memory & performance refactor** (#245, by @AtheerAS). `li.mt.lric` and `li.mt.cross_pcf` now route preprocessing through `prep_check_adata`, build per-annulus sparse scale matrices and multiply them against the weight matrices in chunked (`pair_chunk`) column slices — bounding peak memory to a few hundred MB on large datasets — and use SciPy `sparse_distance_matrix` / `searchsorted` for distance binning. This also fixes a `.raw`-subsetting bug in feature extraction, which slightly changes LRIC output values (test reference values updated accordingly). The LRIC tutorial was re-run to reflect the new numerics.

### Fixed

- **`MistyData` now preserves more than `.uns` on `MuData` round-trips (#242).** Converting a `MuData` back to `MistyData` previously dropped `.uns`, breaking downstream plots such as `li.pl.contributions`; the conversion now carries over `uns`, `obsm`, `varm`, `obsp` and `varp`.
- **`rank_aggregate` / `by_sample` dependency compatibility (#244).** The AnnData `dtype=` removal (AnnData ≥0.11) is handled in preprocessing. pandas 3.0 additionally breaks the consensus path — Copy-on-Write turns a chained `inplace` fillna into a no-op, and string-typed columns coerce an internal `None`-labelled score column to `'nan'` — so `pandas<3` is pinned until liana gains full pandas-3.0 support.

## 1.8.0 (29.06.2026)

### Added

- **`li.mt.lric` — Ligand-Receptor Interaction Correlation (LRIC).** A new spatial method for single-cell-resolution data that computes an expression-weighted cross pair-correlation function: each cell's contribution at distance `r` is weighted by its ligand (sender) and receptor (receiver) expression, so the resulting `g(r)` reflects whether ligand- and receptor-expressing cells are spatially co-enriched at distance `r`, beyond what cell-type co-localisation alone predicts. Uses distance-binned annuli with bounding-box edge correction. (`src/liana/method/sp/_LRIC.py`)
- **`li.mt.cross_pcf` — cross pair-correlation function (cross-PCF).** The classical point-pattern statistic underlying LRIC: the distance-resolved `g(r)` for every directed sender→receiver cell-type pair, using cell positions only (no expression). Inspired by the cross-PCF in the MuSpAn toolbox (Bull et al., 2024, doi:10.1101/2024.12.06.627195).
- New plots: `li.pl.annulus_plot` (visualise per-annulus interaction profiles) (`src/liana/plotting/_annulus.py`)
- **pyCrossTalkeR integration tutorial** (`liana_pyCrossTalkeR.ipynb`) showing network-based differential CCC analysis, plus a dedicated LRIC tutorial (`LRIC_tutorial.ipynb`).
- Mermaid diagram rendering in the docs (`sphinxcontrib-mermaid` doc dependency, `myst_fence_as_directive`/`mermaid_init_config` in `conf.py`); reworked the README decision tree with clickable nodes, colour-coded branches, and the new LRIC / spatially-constrained / pyCrossTalkeR entry points.
- Expanded `docs/api.md` to document previously-undocumented public functions (`compute_global_specificity`, `filter_view_markers`, `circle_plot`, `feature_by_group`, `spatial_pair_proximity`, `query_bandwidth`, `filter_reassemble_complexes`, `translate_resource`, `translate_column`, `get_hcop_orthologs`) alongside the new spatial methods and plots.

### Fixed

- Improved numerical stability of the weighted Pearson/Spearman correlations in `li.mt.bivariate`: the variance denominators are now zeroed relative to their sum-of-squares scale (`<= 1e-6 * ss`) rather than against a fixed `1e-6` absolute threshold, avoiding spurious near-zero correlations from float accumulation. (`src/liana/method/sp/_bivariate/_local_functions.py`)

### Changed

- Standardised `compute_global_specificity` docstring to NumPy format and removed stale `mask_negatives`/`add_categories` parameter references from the `inflow` docstring.

## 1.7.3 (26.05.2026)

- Fixed top-level `import corneto` in `liana/method/fun/_causalnet.py` which caused ReadTheDocs builds to fail (`no module named liana.method`) because `corneto` is an optional dependency not installed in the doc environment. Removed the top-level import and the now-unnecessary `corneto.*` type annotations from function signatures; runtime loading already used `_check_if_installed("corneto")`.
- Updated `inflow_score.ipynb` to use the new `target_organism='mouse'` parameter for `li.rs.get_hcop_orthologs` instead of the defunct EBI FTP `url`.

## 1.7.2 (14.05.2026)

- Fixed `get_hcop_orthologs` to use the HGNC Google Cloud Storage bucket instead of the defunct EBI FTP mirror, resolving 404 errors in CI.
- Added `target_organism` parameter (default `"mouse"`) to `get_hcop_orthologs`, enabling homology mapping to any of the 19 species available in the HCOP database.
- Updated documentation notebook (`prior_knowledge.ipynb`) to use the new `target_organism` API.
- Updated `sc_multi.ipynb` metabolite-receptor section for decoupler v2: renamed `pd_net`/`t_net` columns to `source`/`target`/`weight` and removed deprecated `source`/`target`/`weight`/`min_n` kwargs from `estimate_metalinks` (replaced by `tmin`).
- Standardized all public docstrings to NumPy format and added type annotations across public modules (#219).
- Added mypy type-checking to pre-commit hooks (`--no-strict-optional --ignore-missing-imports`).
- Added `build.yaml` CI workflow: validates the package build with `uv build` + `twine check --strict` on every push and pull request.
- Renamed `.github/workflows/main.yml` → `test.yml`.

## 1.7.1 (24.01.2026)

- Fixed issue with Metalinks download due to User-Agent restrictions.
- Added scanpy version compatibility using getattr to handle both _set_default_colors_for_categorical_obs (old) and set_default_colors_for_categorical_obs (new).


## 1.7.0 (07.01.2026)

- Inflow implementation and tutorial #221 by @AtheerAS
- Global specificity calculation #221 by @AtheerAS
-  The integration of spatial proximity weighting into scoring and permutation-based p-value calculations, new user-facing parameters for spatial analysis, and enhancements to the documentation to reflect these features. #222. The main cell-cell communication pipeline (`liana_pipe`) and scoring methods now support spatial proximity weighting. This includes new arguments (`spatial_key`, `spatial_kwargs`) and logic to compute and merge spatial proximity scores into LR (ligand-receptor) results, and to adjust permutation-based p-value calculations accordingly. (`src/liana/method/sc/_liana_pipe.py`)
- Expanded docstrings and parameter documentation to cover new spatial analysis arguments, including detailed descriptions of spatial proximity options and kernel/bandwidth settings.
- Updated the notebook index and documentation to reference new spatial analysis notebooks, such as `inflow_score.ipynb`.
- Bumped the package version to 1.7.0 across configuration files, and updated dependencies for `decoupler`.
- Added Python 3.13 support in classifiers. #216
- Added Installation instructions in `installation.md`. #217
- Properly check if a passed (cell type) labels in plotting are a string #220
- Fixed an issue where MetalinksDB download would fail due to User-Agent restrictions.

## 1.6.1 (28.09.2025)

- Comply with AnnData CSR matrix changes
- Bump Python version to <=3.13

## 1.6.0 (09.07.2025)

- Adapted and bumped requirements to decopler-py \>=2.0.0 \| PR #178 by
  \@robinfallegger addresses [#179](https://github.com/scverse/liana/issues/179)
- Removed upper Python version requirement [#172](https://github.com/scverse/liana/issues/172) [#170](https://github.com/scverse/liana/issues/170)
- Minor adjustment to SpatialDM Global Moran\'s R description [#176](https://github.com/scverse/liana/issues/176)
- Fix feature name warning logic [#169](https://github.com/scverse/liana/issues/169)
- Use scverse cookiecutter [#180](https://github.com/scverse/liana/issues/180)
- Address count issue with circle plot [#185](https://github.com/scverse/liana/issues/185)

## 1.5.1 (13.02.2025)

- liana will now require Python \>= 3.10
- Removed AnnData upper version restrictions
- Merged PR #161 for numpy2.0 compatibility
- Minor documentation improvements for circle_plot.

## 1.5.0 (17.01.2025)

- New `circle_plot` is now available (Merged #139). Thanks to
  \@WeipengMO.
- Update bivariate metrics to no longer save in place but rather return
  the AnnData
- Issue related to .A for a csr_matrix after a certain scipy version
  #155, #135
- Removed inplace paramter from `li.mt.bivariate` Related to #147. It
  will now by default return an AnnData object.

## 1.4.0 (02.09.2024)

- Now published at Nat Cell Bio.
- Correctly referred to PK tutorial for orthology conversion

\- Added `batch_key` and `min_var_nbatches` to control te way batches
are selected in `li.multi.lrs_to_views`. This might result in minor
differences of how many interactions are considered per view, as I also
changed the order of filtering.

- Changed `max_neighbours` in `li.ut.spatial_neighbors` to be a fixed
  number (default=100), rather than a fraction of the spots as this was
  making RAM explode for large spatial formats.

## 1.3.0 (12.07.2024)

- Minor improvements to documentation, specifically changed to the furo
  theme. Resolved issues with latex not being rendered and plot sizes
  being off.
- An exception will now be reaised if `nz_prop` is too high in
  `li.mt.bivariate`. #121
- Updated MetalinksDB to v0.4.5 (the latest version of the MetalinksDB
  paper), extended to also include production-degradation information.
- Fixed some edgecases where an external `resource` or `interactions`
  can have duplicated entries, also resolving a pandas name index issue
  (#120)
- Added simple tutorial how to process multi-omics and multi-modal (e.g.
  metabolite inference) data with LIANA+. #41 #124

## 1.2.1 (11.06.2024)

- Added +1 to the max_neighbours to account for the spot itself in the
  spatial connectivities.
- Replaced Squidpy\'s neighbourhood graph with liana\'s radial basis
  kernel, but with a fixed number of neighbours for each spot. This does
  not account for edges, but differences are minimal does not require
  squidpy as a dependency. One can easily replace it on demand. (#
  <https://github.com/scverse/liana/issues/112>)
- Fixed Python version range between 3.8 and 3.12 (Merged #112)
- Improved the Differential Expression Vignette be more explicit about
  the causal subnetwork search results (related to #66)

## 1.2.0 (24.05.2024)

\- Added inbuilt orthology conversion functions to convert between
species in the ligand-receptor resources (addressing #76) These include:
`li.rs.get_hcop_orthology` to obtain a dataframe of orthologs from
\[HCOP\](<https://www.genenames.org/tools/hcop/>),
`li.rs.translate_column` to translate a single column in a dataframe,
and `li.rs.translate_resource` as a simple wrapper from the latter
function to be applied on dataframes.

- Merged #109 to address a backward compatibility issue with plotnine\'s
  facets.
- Updated MOFAcell & MOFAtalk tutorials, by making some parameters a bit
  more explicit (#102), and using decoupler\'s association plot to do
  ANOVA + plot metadata associations.
- The mean rank returned by `rank_aggregate` when `aggregate_metod` =
  \'mean\' is now normalized by the total number of interactions.
- Fixed a minor logic issue when calculating analytical p-values for
  Moran\'s R

## 1.1.0 (12.04.2024)

- Added a check for the subset of cell types in li.multi.dea_to_lr.
  Related to #92.
- Split Local and Global Bivariate metrics. Specifically, I reworked
  completely the underlying code, though the API should remain
  relatively unchanged. With the exceptions of: 1) `lr_bivar` is now
  removed and `bivar` has been renamed to `bivariate`. This allowed me
  to remove a lot of redundancies between the two functions. 2)
  `nz_threshold` has been renamed to `nz_prop` for consistency with
  `expr_prop` in the remainder of the package. Related to #44.
- `li.mt.bivariate` parameter `mod_added` has been renamed to
  `key_added` due to this now refer to both `.obsm` and `.mod` -
  depedening whether an AnnData or MuData object is passed.
- Added Global \[Lee\'s
  statistic\](<https://onlinelibrary.wiley.com/doi/abs/10.1111/gean.12106>),
  along with a note on weighted product that upon z-scaling it is
  equivalent to Lee\'s local statistic.
- The Global \[L
  statistic\](<https://onlinelibrary.wiley.com/doi/abs/10.1111/gean.12106>)
  and Global \[Moran\'s
  R\](<https://www.nature.com/articles/s41467-023-39608-w>) are
  themselves basically identical. See Eq.22 from Lee and Eq.1 in Supps
  of SpatialDM.
- Changed the `li.mt.bivar` parameter `function_name` to `local_name`
  for consistency and to avoid ambiguity with the newly-added
  `global_name` parameter.
- Added `bumpversion` to manage versioning. Related to #73.
- Added `max_runs` and `stable_runs` parameters to enable the inference
  of robust causal networks with CORNETO. Related to #82.
- Optimized MISTy such that the matrix multiplication by weights is done
  only once, rather than for each target. Users can now obtain the
  weighted matrix via the `misty.get_weighted_matrix` function.
- MISTy models are now passed externally, rather than being hardcoded.
  This allows for more flexibility in the models used. As an example, I
  also added a RobustLinearModel from statsmodels. Related to #74.
- Removed forced conversion to sparse csr_matrix matrices in MISTy.
  Related to #57.

## 1.0.5 (25.02.2024)

- Added ScSeqComm Method, implemented by \@BaldanMatt (#68)

\- Added functions to query a metabolite-receptor interactions database
(\[MetalinksDB\](<https://github.com/biocypher/metalinks>)), including:
=\> `li.rs.get_metalinks` to get the database =\>
`li.rs.get_metalinks_values` to get the distinct annotation values of
the database =\> `describe_metalinks` to get a description of the
database

- Added a metabolite-mediated CCC tutorial in spatially-resolved
  multi-omics data (#45).
- Changed hardcoded constants to be defined in
  [constants.py]{#constants.py}
- Excluded CellChat from the default `rank_aggregate` method
- Fixed return logic of SpatialBivariate
- `li.mt.process_scores` is now exported to `li.mt`
- Changed the default `max_neighbours` in `li.ut.spatial_neighbors` to
  1/10 of the number of spots.

## 1.0.4 (17.01.2024)

- Moved the Global score summaries of `SpatialBivariate` from .uns to
  .var
- `df_to_lr` will now also return the expression and proportion of
  expression for the interactions
- `li.multi.nfm` will now also accept a DataFrame as input
- Filtered putative interactions in the Consensus resource, mostly such
  coming from CellTalkDB.
- Changed `filter_lambda` parameter to `filter_fun` for consistency and
  now any function can be passed to be applied as a row-wise filter.
- Global results of `SpatialBivariate` will now be saved to `.var`
- Added `li.ut.interpolate_adata` utility function to interpolate the
  data to a common space.
- MISTy will also work with directly non-aligned data with spatial
  connectivities from one modality to the other being passed via `obsm`
  rather than `obsp`. Making use of `li.ut.spatial_neighbors` by passing
  reference coordinates.
- Fixed a bug where `li.ut.obsm_to_adata` would assign var as a method
  rather than DataFrame
- Fixed a bug where p-values for Global Moran\'s were not calculated
  correctly.
- Enabled `cell_pairs` of interest to be passed to single-cell methods.
- Enabled Parallelization of Permutation-based methods.
- Local categories will now be only calculated for positive interactions
  (not non-ambigous as before).
- Names of source and target panels can now be passed to
  `li.pl.tileplot`.
- `li.rs.explode_complexes` is now consistently exported to `li.rs` (as
  previous versions)
- `li.mt.find_causalnet`: changed the noise assigned to nodes to be
  proportional to the minimum penalty of the model. Also, added noise to
  the edges to avoid multiple solutions to the same problem.

## 1.0.3 (06.11.2023)

- Added `filterby` and `filter_lambda` parameters to
  `li.pl.interactions` and `li.pl.target_metrics` to allow filtering of
  interactions and metrics, respectively.
- Removed unnecessary `stat` parameter from `li.pl.contributions`
- Added tests to ensure both `lr_bivar` and single-cell methods throw an
  exception when the resource is not covered by the data.
- `estimate_elbow` will add the errors and the number of patterns to
  `.uns` when inplace is True.
- When `groupby` or `sample_key` are not categorical liana will now
  print a warning before converting them to categorical. Related to #28
- Various documentation improvements, including using `docrep` to ensure
  consistency.
- `__version__` will now correctly reflect the version in pyproject.toml
- Exported repeated value definitions to `_constants.py`
- Renamed some `*_separator` columns to `*_sep` for consistency.
- Added `li.ut.query_bandwidth` to query the bandwidth of the spatial
  connectivities (used in spatial bivariate tutorial)
- Added **pre-commit** hooks adapted from scverse\'s cookiecutter.

## 1.0.2 (13.10.2023)

- Added as `seed` param to `find_causalnet`, used to a small amount of
  noise to the nodes in to avoid obtaining multiple solutions to the
  same problem when multiple equal solutions are possible.
- Updated `installation.rst` to refer to `pip install liana[common]` and
  `liana[full]` for extended installations.
- Fixed a bug which would cause `bivar` to crash when an AnnData object
  was passed

Merged #61 including the following:

- Added `standardize` parameter to spatial_neighbors, used to
  standardize the spatial connectivities such that each spot\'s
  proximity weights to 1. Required for non-standardized metrics (such as
  `product`)
- Fixed edge case in `assert_covered` to handle interactions not present
  in `adata` nor the resource.

\- Added simple product (scores ranging from -inf, +inf) and
norm_product (scores ranging from -1, +1). The former is a simple
product of x and y, while the latter standardized each variable to be
between 0 and 1, following weighing by spatial proximity, and then
multiplies them. Essentially, it diminishes the effect of spatial
proximity on the score, while still taking it into account. We observed
that this is useful for e.g. border zones.

## 1.0.1 Stable Release (30.09.2023)

- Bumped CORNETO version and it\'s now installed via PyPI.

## 1.0.0a2 (19.09.2023)

- Interactions names in `tileplot` and `dotplot` will now be sorted
  according to `orderby` when used; related to #55
- Added `filter_view_markers` function to filter view markers considered
  background in MOFAcellular tutorial
- Added `keep_stats` parameter to `adata_to_views` to enable pseudobulk
  stats to be kept.
- Replace `intra_groupby` and `extra_groupby` with `maskby` in misty.
  The spots will now only be filtered according to `maskby`, such that
  both intra and extra both contain the same spots. The extra views are
  multiplied by the spatial connectivities prior to masking and the
  model being fit
- Merge MOFAcell improvements; related to #42 and #29
- Targets with zero variance will no longer be modeled by misty.
- Resolve #46 - refactored misty\'s pipeline
- Resolved logging and package import verbosity issues related to #43
- Iternal .obs\[\'label\'\] placeholder renamed to the less generic
  .obs\[\'@label\'\]; related to #53
- Minor Readme & tutorial text improvements.

## 1.0.0a1 Biorxiv (30.07.2023)

- `positive_only` in bivariate metrics was renamed to `mask_negatives`
  will now mask only negative-negative/low-low interactions, and not
  negative-positive interactions.
- Replaced MSigDB with transcription factor activities in MISTy\'s
  tutorial
- Enable sorting according to ascending order in misty-related plots
- Enable `cmap` to be passed to tileplot & dotplots
- Minor Readme & tutorial improvements.

## 1.0.0a0 (27.07.2023)

LIANA becomes LIANA+.

Major changes have been made to the repository, however the API visible
to the user should be largely consistent with previous versions, except
minor exceptions: - `li.fun.generate_lr_geneset` is now called via
`li.rs.generate_lr_geneset`

- the old \'li.funcomics\' model is now renamed to something more
  general: `li.utils`
- `get_factor_scores` and `get_variable_loadings` were moved to
  `li.utils`

LIANA+ includes the following new features:

### Spatial

- A sklearn-based implementation to learn spatially-informed multi-view
  models, i.e.
  \[MISTy\](<https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02663-5>)
  models.
- A new tutorial that shows how to use LIANA+ to build and run MISTy
  models.
- Five vectorized local spatially-informed bivariate clustering and
  similarity metrics, such as \[Moran\'s
  R\](<https://www.biorxiv.org/content/10.1101/2022.08.19.504616v1.full>),
  Cosine, Jaccard, Pearson, Spearman. As well as a numba-compiled
  \[Masked
  Spearman\](<https://www.nature.com/articles/s41592-020-0885-x>) local
  score.

\- A new tutorial that shows how to use LIANA+ to compute
spatially-informed bivariate metrics, permutations-based p-values,
interaction categoriez, as well as how to summarize those into patterns
using NMF.

\- A radial basis kernel is implemented to calculate spot/cell
connectivities (spatial connectivities); this is used by the
spatially-informed bivariate metrics and MISTy. It mirrors
\[squidpy\'s\](<https://squidpy.readthedocs.io/en/stable/>)
`sq.gr.spatial_neighbors` function, and is hence interchangeable with
it.

### Handling multiple modalities

\- LIANA+ will now work with multi-modal data, i.e. it additionally
support MuData objects as well as AnnData objects. The API visible to
the user is the same, but the underlying implementation is different.

- These come with a new tutorial that shows how to use LIANA+ with
  multi-modal (CITE-Seq) data, along with inbuilt transformations.
- The same API is also adapted by the local bivariate metrics, i.e. they
  can also be used with multi-modal data.

### Multi-conditions

\- A utility function has been added that will take any dataframe with
various statistics and append it to information from AnnData objects;
thus creating a multi-condition dataframe in the format of LIANA.

- A new tutorial that shows how to use PyDESeq2 together with this
  utility function has been added, essentially a tutorial on
  \"Hypothesis-driven CCC\".

### Visualizations

- A tileplot (`li.pl.tileplot`) has been added to better visualize
  ligands and receptors independently.
- MISTy-related visualizations have been added to vislualize view
  contributions and performance, and interaction
  coefficients/importances.
- A simple plot `li.pl.connectivity` is added to show spatial
  connectivities

### Others

- A Causal Network inference function has been added to infer downstream
  signalling networks. This is currently placed in the tutorial with
  PyDESeq2.
- An elbow approximation approach has been added to the NMF module, to
  help with the selection of the number of patterns.
- Various utility functions to simplify AnnData extraction/conversion,
  Matrix transformations, etc (added to `li.ut`)

Note: this is just an overview of the new features, for details please
refer to the tutorials, API, and documentation.

## 0.1.9 (06.06.2023)

- Fixed issues with deprecated params of pandas.DataFrame.to_csv &
  .assert_frame_equal in tests
- `multi.get_variable_loadings` will now return all factors
- Added source & target params to `fun.generate_lr_geneset`

\- Refactored `sc._Method._get_means_perms` & related scoring functions to be more efficient.

:   `None` can now be passed to n_perms to avoid permutations - these
    are only relevant if specificity is assumed to be relevant.

- LIANA\'s aggregate method can now be customized to include any method
  of choice (added an example to basic_usage).
- Removed \'Steady\' aggregation from rank_aggregate
- Changed deprecated np.float to np.float32 in `liana_pipe`, relevant
  for CellChat `mat_max`.
- Method results will now be ordered by magnitude, if available, if not
  specificity is used.
- Added `ligand_complex` and `receptor_complex` filtering to liana\'s
  dotplot
- MOFAcellular will now work only with decoupler\>=1.4.0 which
  implements edgeR-like filtering for the views.

## 0.1.8 (24.03.2023)

- Removed walrus operator to support Python 3.7
- Added a tutorial that shows the repurposed use of MOFA with liana to
  obtain intercellular communication programmes, inspired by
  Tensor-cell2cell
- Added a tutorial that shows the repurposed use of MOFA to the analysis
  of multicellular programmes as in Ramirez et al., 2023
- Added `key_added` parameter to save liana results to any
  `adata.uns``slot, and`uns_key`to use liana results from any`adata.uns\`\`
  slot
- `inplace` now works as intended (i.e. only writes to `adata.uns` if
  `inplace` is True).

## 0.1.7 (08.02.2023)

- Fixed an edge case where subunits within the same complex with
  identical values resulted in duplicates. These are now arbitrarily
  removed according to random order.
- All methods\' complexes will now be re-assembled according to the
  closest stat to expression that each method uses, e.g. `cellchat` will
  use `trimeans` and the rest `means`.
- Added a basic liana to Tensor-cell2cell tutorial as a solution to
  liana issue #5
- Updated the basic tutorial
- Referred to CCC chapter from Theis\' best-practices book

## 0.1.6 (23.01.2023)

- Fixed issue with duplicate subunits for non-expressed LRs when
  `return_all_lrs` is True
- `min_prop` when working with `return_all_lrs` is now filled with 0s
- Added `by_sample` function to class Method that returns a long-format
  dataframe of ligand-receptors, for each sample
- Added `dotplot_by_sample` function to visualize ligand-receptor
  interactions across samples
- Refractored preprocessing of `dotplot` and `dotplot_by_sample` to a
  separate function
- Changed \"pvals\" of geometric_mean method to \"gmean_pvals\" for
  consistency
- `to_tensor_c2c` utility function to convert a long-format dataframe of
  ligand-receptor interactions by sample to Tensor-cell2cell tensor.
- Added a list to track the instances of `MethodMeta` class
- Added `generate_lr_geneset` function to generate a geneset of
  ligand-receptors for different prior knowledge databases

## 0.1.5 (11.01.2023)

- Hotfix `return_all_lrs` specificity_rank being assigned to NaN
- Add test to check that `specificity_rank` of `lrs_to_keep` is equal to
  min(specificity_rank)

## 0.1.4 (11.01.2023)

- `rank_aggregate` will now sort interactions according to
  `magnitude_rank`.
- Fixed `SettingWithCopyWarning` warning when `return_all_lrs` is True
- Minor text improvements to the basic tutorial notebook
- Removed \'Print\' from a verbose print message in `_choose_mtx_rep`

## 0.1.3 (07.12.2022)

- Added `supp_columns` parameter to allow any column from liana to be
  returned.
- Added `return_all_lrs` parameter to allow all interactions to be
  returned with a `lrs_to_filter` flag for the interaction that do not
  pass the `expr_prop`, and each of those interactions is assigned to
  the worst **present** score from the ones that do pass the threshold.
- Fixed a bug where an exception was not thrown by `assert_covered`
- Raise explicit exceptions as text in multiple places.
- Changed cellphonedb p-values column name from \"pvals\" to
  \"cellphone_pvals\".

## 0.1.2

- Added CellChat and GeometricMean methods

## 0.1.1

- Add progress bar to permutations
- Deal with adata copies to optimize RAM
- change copy to inplace, and assign to uns, rather than return adata
- remove unnecessary filtering in [pre]{#pre} + extend units tests

## 0.1.0

- Restructure API further
- Submit to PIP

## 0.0.3

- Added a filter according to `min_cells` per cell identity
- prep_check_adata will now assert that `groupby` exists
- extended test_pre.py tests
- restructured the API to be more scverse-like

## 0.0.2

- Added `dotplot` as a visualization option
- Added `basic_usage` tutorial

## 0.0.1

First release alpha version of **liana-py**

-

  Re-implementations of:

  :   - CellPhoneDB
      - NATMI
      - SingleCellSignalR
      - Connectome
      - logFC
      - Robust aggregate rank

- Ligand-receptor resources as generated via OmniPathR.
