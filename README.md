# Are Straight-Through gradients and Soft-Thresholding all you need for Sparse Training?
Source Code for submission 1614 of the WACV conference

The pieces of code proper to the ST-3(sigma) method described in the paper are available in the following 2 files:
* `quantized.py`: the `STEPrunedLayer` is the layer-type allowing for straight-through estimation as well as soft-thresholding
* `pruning.py`: the `l1` and `l1std` are responsible for zeroing out the correct weights either based on l1-magnitued (for ST-3) or l1-magnitude normalized by layer width (ST-3-sigma). The quadratic pruning-ratio increase is also handled in this file.

In those files a lot of code is also there to reproduce results of ProbMask, GMP and LRR and can be ignored, they are included for completeness. The other files are mostly boilerplate and part of a bigger framework that isn't necessary to understand ST-3.

# Execution
`$ ./launch <path/to/configs>`

# Weights
The weights aren't stored on an anonymized server and thus will be made available after publication
