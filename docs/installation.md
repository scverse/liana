# Installation

LIANA+ requires Python 3.12 or newer.

::::{tab-set}

:::{tab-item} pip

```bash
pip install liana
```

:::

:::{tab-item} uv

```bash
uv add liana
```

:::

:::{tab-item} conda

```bash
conda install bioconda::liana
```

:::

::::

This installs the ligand-receptor methods, the spatial and multi-modal metrics, the prior knowledge resources and the plots.

## Extras

Functionality that only part of the user base needs sits behind the `extras` group:

```bash
pip install 'liana[extras]'
```

It pulls in [decoupler](https://decoupler.readthedocs.io/), [muon](https://muon.readthedocs.io/), mofax and mofapy2 for multi-view and multi-sample analyses, [omnipath](https://omnipathdb.org/) to query prior knowledge, [pydeseq2](https://pydeseq2.readthedocs.io/) for differential expression, [gseapy](https://gseapy.readthedocs.io/) for enrichment, corneto, cvxpy and PySCIPOpt for the causal network inference, [squidpy](https://squidpy.readthedocs.io/) for the spatial neighborhoods, as well as cell2cell and kneed.

LIANA+ imports these when they are first used, so a missing one surfaces when you call the function that needs it.

## Running the tutorials

The notebooks need a few plotting packages on top of the extras:

```bash
pip install 'liana[tutorials]'
```

The two heaviest notebooks (`inflow_mofaflex` and `liana_c2c`) additionally need torch, mofaflex and tensorly:

```bash
pip install 'liana[tutorials-gpu]'
```

## Development install

```bash
git clone https://github.com/scverse/liana.git
cd liana
uv sync --all-extras
```

The {doc}`contributing guide <contributing>` describes the environments, the test matrix and the docs build.

## Claude Code Skill

LIANA+ bundles an Agent Skill for Claude Code. After installing liana, run `liana-install-skills` once to copy it to `~/.claude/skills/liana/`; Claude Code picks it up automatically in every project (type `/skills` to confirm) and consults it for liana tasks. Refresh the copy with `liana-install-skills --force` after upgrading liana.
