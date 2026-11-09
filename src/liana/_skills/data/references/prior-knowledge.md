# Prior knowledge: LR resources, organisms, custom lists

Tutorial: `prior_knowledge`.

```python
li.rs.show_resources()                    # consensus, mouseconsensus, cellphonedb, cellchatdb and more
resource = li.rs.select_resource()        # consensus: ligand, receptor columns, human symbols
```

- `consensus` merges CellPhoneDB, CellChat, ICELLNET, connectomeDB2020 and CellTalkDB.
- Every method takes `resource_name=`, `resource=` (any DataFrame with `ligand`, `receptor`) or
  `interactions=[("L", "R"), ...]`, which overrides the other two.
- Complex subunits are joined by `_` in the resource; feature names containing `_` are misread as complexes.

## Other organisms

- Mouse: `resource_name="mouseconsensus"` (homologene-derived; misses some genes).
- Any other organism, or a fuller mouse map: translate with HCOP orthologs.

<!-- skip-exec: network -->
```python
map_df = li.rs.get_hcop_orthologs(columns=["human_symbol", "mouse_symbol"])   # rat: target_organism="rat", "rat_symbol"
map_df = map_df.rename(columns={"human_symbol": "source", "mouse_symbol": "target"})
resource_mm = li.rs.translate_resource(resource, map_df=map_df, columns=["ligand", "receptor"])
```

`min_evidence` (agreeing databases) and `one_to_many` (orthologs kept per gene) set coverage; the
tutorial uses `one_to_many=1` for mouse, 3 for zebrafish. `li.rs.translate_column(df, map_df,
column=...)` translates one column, e.g. MetalinksDB `gene_symbol`.

## Derived knowledge

- `li.rs.generate_lr_geneset(resource, net)`: gene-set table (`source`, `target`, `weight`) to LR
  sets; an LR is kept only if both partners sit in the same set with a coherent sign. `weight=None` for unweighted sets.
- `li.rs.build_prior_network(ppis, input_nodes, output_nodes)` builds the graph for `li.mt.find_causalnet`.
