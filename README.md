Kernel and feature in-context learners on single-index targets (arXiv 2610.01712), a small JAX slice: the two attention reductions, the degree-1 and degree-2 feature map, and the infinite-pretraining error formulas.

```
/home/nixos/arxiv-impl-work/bin/py -m pip install -e .
/home/nixos/arxiv-impl-work/bin/py -m pytest -q
```

Does not match the paper: no replica saddle point, no Figure 2, no N=12288 pretraining run. The degree-2 map keeps the trace-zero redundancy, so its width is one larger than D_2.
