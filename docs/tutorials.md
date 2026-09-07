# Tutorials

Each tutorial is a notebook that runs top to bottom on a public dataset from {mod}`liana.ds`.

## Where to start

The tree below goes from the kind of data you have to the methods that apply to it.
Click a node to open its tutorial.

```{include} ../README.md
:start-after: <!-- decision-tree-start -->
:end-before: <!-- decision-tree-end -->
```

The tree is a guide rather than an exhaustive map, since the methods are modular and can be combined across data types and questions.

## Getting started

How a method is called, and where the prior knowledge it scores comes from.

```{toctree}
:maxdepth: 1

tutorials/notebooks/basic_usage.ipynb
tutorials/notebooks/prior_knowledge.ipynb
```

## Dissociated single-cell data

Inference in one sample, and across samples or conditions.

```{toctree}
:maxdepth: 1

tutorials/notebooks/targeted.ipynb
tutorials/notebooks/liana_pyCrossTalkeR.ipynb
tutorials/notebooks/mofatalk.ipynb
tutorials/notebooks/mofacellular.ipynb
tutorials/notebooks/liana_c2c.ipynb
```

## Spatially-resolved data

Interactions restricted to, or modelled from, spatial coordinates.

```{toctree}
:maxdepth: 1

tutorials/notebooks/bivariate.ipynb
tutorials/notebooks/misty.ipynb
tutorials/notebooks/inflow_score.ipynb
tutorials/notebooks/inflow_mofaflex.ipynb
tutorials/notebooks/LRIC_tutorial.ipynb
```

## Multi-modal data

Interactions between modalities, such as transcriptome and surface protein, or transcriptome and metabolite.

```{toctree}
:maxdepth: 1

tutorials/notebooks/sc_multi.ipynb
tutorials/notebooks/sma.ipynb
```
