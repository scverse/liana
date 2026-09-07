# LIANA+

LIANA+ is an all-in-one framework for cell-cell communication.
It adapts and extends existing methods and prior knowledge to single-cell, spatially-resolved and multi-modal data, on {class}`~anndata.AnnData` and {class}`~mudata.MuData` objects, and is part of the [scverse ecosystem](https://scverse.org).

```{image} _static/abstract.png
:alt: Overview of the analyses that LIANA+ supports
:class: liana-hero
```

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`desktop-download;1.5em;sd-mr-1` Installation
:link: installation
:link-type: doc

Install LIANA+ with pip, uv or conda, with or without the optional extras.
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Quickstart
:link: tutorials/notebooks/basic_usage
:link-type: doc

Infer ligand-receptor interactions in a dissociated single-cell dataset.
:::

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Tutorials
:link: tutorials
:link-type: doc

A decision tree, and a runnable notebook for each type of data and question.
:::

:::{grid-item-card} {octicon}`code-square;1.5em;sd-mr-1` API reference
:link: api
:link-type: doc

Every method, preprocessing step, resource, dataset and plot.
:::

:::{grid-item-card} {octicon}`comment-discussion;1.5em;sd-mr-1` Discussion
:link: https://discourse.scverse.org/

Ask a question on the scverse forum.
:::

:::{grid-item-card} {octicon}`mark-github;1.5em;sd-mr-1` GitHub
:link: https://github.com/scverse/liana

Read the source, report a bug, or open a pull request.
:::

::::

## Citation

```{eval-rst}
.. include:: about/cite.md
    :start-line: 2
    :parser: myst
```

## NumFOCUS

[//]: # "numfocus-fiscal-sponsor-attribution"

LIANA+ is part of the scverse® project ([website](https://scverse.org), [governance](https://scverse.org/about/roles)) and is fiscally sponsored by [NumFOCUS](https://numfocus.org/).
If you like scverse® and want to support our mission, please consider making a tax-deductible [donation](https://numfocus.org/donate-to-scverse) to help the project pay for developer time, professional services, travel, workshops, and a variety of other needs.

<div align="center">
<a href="https://numfocus.org/project/scverse">
  <img
    src="https://raw.githubusercontent.com/numfocus/templates/master/images/numfocus-logo.png"
    width="200"
  >
</a>
</div>

```{toctree}
:caption: General
:hidden:
:maxdepth: 1

installation
api
changelog
contributing
references
```

```{toctree}
:caption: Gallery
:hidden:
:maxdepth: 2

tutorials
```

```{toctree}
:caption: About
:hidden:
:maxdepth: 1

about/background
about/cite
GitHub <https://github.com/scverse/liana>
Discourse <https://discourse.scverse.org/>
```
