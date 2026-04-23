# Diminimal Unicyclic Graphs

This repository accompanies research on diminimal unicyclic graphs. It provides a compact computational workflow for constructing and visualizing a specific weighted diminimal matrix associated with a generalized sunlet-type unicyclic graph. The repository is intended both as reproducible computational support for a research article and as a citable software reference.

## Overview

The current codebase includes:

- construction of Jacobi matrices from spectral data;
- assembly of a generalized sunlet matrix through `generalized_sunlet_matrix`;
- numerical inspection of the distinct spectrum of the resulting matrix;
- visualization of the matrix as a heatmap and as a weighted unicyclic graph;
- export of weighted graph drawings to TikZ for direct inclusion in LaTeX manuscripts.

## Repository Structure

```text
.
|-- Notebooks/
|   `-- Diminimal Matrices.ipynb
`-- src/
    `-- Utils/
        `-- utils.py
```

## Main Construction

The core routine of the repository is the function `generalized_sunlet_matrix(k, p, lambdas, mu)`, implemented in `src/Utils/utils.py`.

At a high level, the construction works as follows:

1. two Jacobi matrices are reconstructed from interlacing spectral data using `construir_matriz_jacobi`;
2. these blocks are combined along a generalized sunlet-style unicyclic structure;
3. the resulting weighted matrix can then be analyzed spectrally and visualized graphically.

The main parameters are:

- `k`: number of attachment positions around the unique cycle;
- `p`: block-size parameters controlling the attached branches;
- `lambdas` and `mu`: spectral sequences used in the Jacobi reconstruction.

## Reproducing the Notebook Example

The notebook [`Notebooks/Diminimal Matrices.ipynb`](Notebooks/Diminimal%20Matrices.ipynb) contains a complete example:

```python
import sys
import numpy as np
import os
sys.path.append(os.path.join("..", "src"))

from Utils.utils import (
    generalized_sunlet_matrix,
    plot_heatmap_nn,
    plot_weighted_sunlet_graph_from_matrix,
)

A_sunlet = generalized_sunlet_matrix(
    k=6,
    p=[3, 2, 1, 3, 1, 1],
    lambdas=[1, 2, 3],
    mu=[1.5, 2.5, 3.5],
)

print("DSpec(A)  =", set(np.round(np.linalg.eigvals(A_sunlet).real, 3).tolist()))
print("|DSpec(A)| =", len(set(np.round(np.linalg.eigvals(A_sunlet).real, 3).tolist())))
```

For this example, the code produces a `28 x 28` weighted matrix and computes its distinct spectrum before plotting the associated weighted unicyclic graph.

## Installation

The project currently depends on:

- `numpy`
- `matplotlib`
- `networkx`
- `torch`
- `torch-geometric`
- `jupyter`

A minimal environment can be created with:

```bash
pip install numpy matplotlib networkx torch torch-geometric jupyter
```

Then open the notebook:

```bash
jupyter notebook "Notebooks/Diminimal Matrices.ipynb"
```

## Outputs

The utilities in `src/Utils/utils.py` allow you to:

- generate the weighted matrix associated with the construction;
- plot the matrix as a heatmap via `plot_heatmap_nn`;
- draw the corresponding weighted unicyclic graph via `plot_weighted_sunlet_graph_from_matrix`;
- export the drawing to TikZ via `weighted_sunlet_graph_to_tikz`.

These outputs are intended to support both computational verification and figure preparation for academic writing.

## Citation

If you use this repository in academic work, please cite it as software accompanying the related article. A suggested BibTeX entry is:

```bibtex
@misc{scaratti2026diminimalunicyclicgraphs,
  author       = {Bruno Scaratti},
  title        = {Diminimal-Unicyclic-Graphs: Computational Construction of a Specific Diminimal Matrix for Unicyclic Graphs},
  year         = {2026},
  howpublished = {\url{https://github.com/brunoscaratti99/Diminimal-Unicyclic-Graphs}},
  note         = {GitHub repository, accessed 2026-04-23, commit 01224ce}
}
```

For precise reproducibility, citing a specific commit or future tagged release is recommended.

## Purpose

This repository is meant to serve as:

- a computational companion to an article on diminimal unicyclic graphs;
- a transparent record of the matrix construction used in that work;
- a bibliographic software reference for readers who want to reproduce the construction or generate related figures.
