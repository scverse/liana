# Agent Skill evaluation scenarios

Manual evaluations for `src/liana/_skills/data` (not run by pytest: they need a Claude Code
session and are not deterministic). Each is grounded in real GitHub issues. Run each prompt in a
fresh headless session twice, once without the skill and once with it in `.claude/skills/liana/`,
on toy data from `li.ds`, and check the expected behaviour:

```bash
claude -p --model sonnet --allowedTools "Bash,Read,Write,Edit,Glob,Grep,Skill" < prompt.txt
```

Re-run after changing a public signature, a default, or the routing table in SKILL.md.

| # | Prompt (user's words) | Data | Expected behaviour | Issues |
|---|---|---|---|---|
| 1 | "PBMC scRNA-seq, one condition, cell types in obs. log1p in .X, scaled data in .raw. Run ligand-receptor inference and tell me the top hits and how to read the scores." | dissociated, human | reads single-cell-lr.md; `rank_aggregate` on `.X`, never on scaled `.raw`; explains magnitude vs specificity ranks (lower is better); mentions `expr_prop`; plots with `li.pl.dotplot`; cites LIANA+ | #205 #229 #107 #52 #35 |
| 2 | "4 samples in 2 conditions. I ran rank_aggregate per condition and compared magnitude ranks. Valid? What instead?" | dissociated, multi-sample | reads multisample.md; says per-condition rank comparison is indicative only and one sample per condition cannot be tested; offers the three routes (pseudobulk DE + `df_to_lr`, MOFA+ via `lrs_to_views`, Tensor-cell2cell via `to_tensor_c2c`) and picks one with a reason; runs `by_sample` first | #88 #156 #5 #206 #117 |
| 3 | "Visium, coordinates in full-res pixels, scalefactors in uns. Spatially informed LR scores: which bandwidth, and what is in the returned object?" | spot-level spatial | reads spatial-bivariate.md; converts pixels to µm, anchors bandwidth on ~100 µm or the 6-neighbour ring via `query_bandwidth`; `set_diag=True` for spots; explains the returned AnnData (`.X` local, `.var` global, `.layers` pvals/cats) | #133 #154 #176 #106 |
| 4 | "Xenium, one cell type per cell, counts in layers, log-normalised in X, coordinates in µm. Which cell types signal to which via which LR pairs, taking distance into account? bivariate gave no cell types." | single-cell spatial | reads inflow.md; runs `spatial_neighbors` then `li.mt.inflow` (or the sc methods with `spatial_key`), then `compute_global_specificity`; does not hand-roll a proximity merge; notes the unpublished status when citing | #159 #212 #131 #153 |
| 5 | "Mouse data (Cxcl9, Cd4). liana throws 'Please check if appropriate organism/ID type was provided!'. How do I run this on mouse?" | dissociated, mouse | names both causes of the error (resource vs var_names mismatch, not only organism); uses `resource_name="mouseconsensus"`; mentions HCOP translation as the fuller alternative | #51 #105 #76 #4 |
| 6 | "I cannot share the data (patient material). MuData for MISTy; after passing it back into MistyData the .uns results are gone. Work it out from my description." | described only | does not demand the file; asks a short bounded set of questions (version, how the MuData was built, exact call, traceback); reasons from source; proposes a synthetic reprex | #242 #93 #143 |
| 7 | "PBMC scRNA-seq, all genes. I already ran the standard LR analysis. Beyond protein ligands, can liana say anything about small-molecule or metabolite signalling? I only have RNA." | dissociated, RNA only | reads metabolites.md; `li.rs.get_metalinks` filtered to blood, `li.mt.estimate_metalinks`, then `rank_aggregate` with `mdata_kwargs` (`x_mod="metabolite"`, `y_mod="receptor"`, `zi_minmax` transforms); states the linear-enzyme and independence caveats; cites MetalinksDB (Farr 2024) | #215 #190 |
| 8 | "CITE-seq MuData with rna and prot modalities, cell types in the rna modality's obs. Ligands from RNA, receptors from the measured protein." | dissociated, MuData | reads single-cell-lr.md Variants; lifts the label onto `mdata.obs`; `rank_aggregate(mdata, mdata_kwargs={x_mod: rna, y_mod: prot, zi_minmax transforms})`; maps antigen names to HGNC symbols and prefixes them (`AB:`) in both modality and resource; strips `_TotalSeqB` (complex separator) | #93 #143 |
| 9 | "PBMC scRNA-seq, all genes. Give me as complete a picture as liana can of how these cell types communicate. Run what is appropriate and tell me what else could be done." | dissociated, RNA only | runs `rank_aggregate` and plots; without being asked, offers metabolite-mediated CCC via MetalinksDB as a next step (the "widen the question" line in SKILL.md); says what the data does not support (no samples, no coordinates) | #215 |
