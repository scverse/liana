# Prior knowledge: LR resources, organisms, custom lists

Tutorial: `prior_knowledge`.

```python
li.rs.show_resources()                       # e.g. consensus, mouseconsensus, cellphonedb, cellchatdb
res = li.rs.select_resource("consensus")     # DataFrame with columns ligand, receptor (human symbols)
```

`consensus` merges CellPhoneDB, CellChat, ICELLNET, connectomeDB2020 and CellTalkDB. Every method takes `resource_name=`, `resource=` (any DataFrame with
`ligand` and `receptor` columns) or `interactions=[("L", "R"), ...]`, which overrides the others.

## Other organisms

- Mouse: `resource_name="mouseconsensus"` (homologene-derived, misses some genes).
- Anything else, or a fuller mouse map:

```python
map_df = li.rs.get_hcop_orthologs()                                      # mouse by default; downloads to cwd
map_df = map_df.rename(columns={"human_symbol": "source", "mouse_symbol": "target"})
res_mm = li.rs.translate_resource(res, map_df=map_df, columns=["ligand", "receptor"])
```

`map_df` must have exactly `source` and `target` columns. `min_evidence` (number of agreeing
databases) and `one_to_many` (orthologs kept per gene) decide coverage; the tutorial uses
`one_to_many=1` for mouse and 3 for zebrafish. `li.rs.translate_column(df, map_df, column=...)`
does the same for one column, e.g. MetalinksDB gene symbols.

## Derived knowledge

- `li.rs.generate_lr_geneset(resource, net)` turns a gene-set table (`source`, `target`, optional
  `weight`) into LR sets for enrichment: an LR is kept only if both partners sit in the same set
  with a coherent sign. `weight=None` for unweighted sets.
- `li.rs.build_prior_network(ppis, input_nodes, output_nodes)` builds the CORNETO graph for
  `li.mt.find_causalnet`; see `multisample.md`.
- MetalinksDB: `li.rs.get_metalinks(...)`; see `metabolites.md`.
