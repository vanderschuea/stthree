# Are Straight-Through gradients and Soft-Thresholding all you need for Sparse Training?
Source Code for submission at the WACV conference (arXiv link coming)

## Abstract

Turning the weights to zero when training a neural network helps in reducing the computational complexity at
inference. To progressively increase the sparsity ratio in the network without causing sharp weight discontinuities
during training, our work combines soft-thresholding and straight-through gradient estimation to update the raw, i.e.
non-thresholded, version of zeroed weights. Our method,named ST-3 for straight-through/soft-thresholding/sparse-
training, obtains SoA results, both in terms of accuracy/sparsity and accuracy/FLOPS trade-offs, when progressively increasing the sparsity ratio in a single training cycle. In particular, despite its simplicity, ST-3 favorably compares to the most recent methods, adopting differentiable formulations (ProbMask) or bio-inspired neuroregeneration principles (GraNet). This suggests that the key ingredients for
effective sparsification primarily lie in the ability to give the weights the freedom to evolve smoothly across the zero
state while progressively increasing the sparsity ratio.

## Instructions

The pieces of code proper to the ST-3(sigma) method described in the paper are available in the following 2 files:
* `quantized.py`: the `STEPrunedLayer` is the layer-type allowing for straight-through estimation as well as soft-thresholding
* `pruning.py`: the `l1` and `l1std` are responsible for zeroing out the correct weights either based on l1-magnitued (for ST-3) or l1-magnitude normalized by layer width (ST-3-sigma). The quadratic pruning-ratio increase is also handled in this file.

In those files a lot of code is also there to reproduce results of ProbMask, GMP and LRR and can be ignored, they are included for completeness. The other files are mostly boilerplate and part of a bigger framework that isn't necessary to understand ST-3.

## Execution
`$ ./launch <path/to/configs>`

## Weights
The weights will be made available soon
