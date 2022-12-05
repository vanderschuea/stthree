# ST-3: Are Straight-Through gradients and Soft-Thresholding all you need for Sparse Training?
Source Code for accepted paper at the upcoming WACV 2022 conference ([paper here](https://arxiv.org/abs/2212.01076))

## General Overview

Pruning litterature has turned to more and more complex methods to prune weights during training based on (ProbMask), sometimes even taking cues from biological neurogeneration (GraNet). This work aims at taking a **simpler approach** (that nevertheless surpasses [previous SoA](https://paperswithcode.com/sota/network-pruning-on-imagenet-resnet-50-90)) based on minimizing the *mismatch* between forward and backward propagation that occurs when a Straight-through-estimator is used to update the weights.
To reduce this disparity, *soft thresholding* and *weight rescaling* are applied during forward propagation only and the pruning ratio is cubicly increased during training to allow for a smoother transition. 

## Results on ImageNet
These are the result of training on ImageNet for 100 epochs (no longer), w/ only RandomCropping and RandomFlipping aas data augmentation during training as to align with results in the literature. ST-3 uses l1-magnitude pruning, ST-3 $^\sigma$ uses scaled l1-magnitude pruning as to force the pruning to be more uniform accross layers. ST-3 $^\sigma$  tends do have a better flops reduction although it tends to produce results that are slightly worse than ST-3 w/o constraints.

### ResNet-50
| Method | Accuracy \[%\] | Sparsity \[%\] | GFLOPS |
| --- | --- | --- | --- |
| Baseline | 77.10 | 0 | 4089 |
| ST-3 | 76.95 | 80 | 1215 |
| ST-3 $^\sigma$ | 76.44 | 80 | 739 |
| ST-3 | 76.03 | 90 | 764 |
| ST-3 $^\sigma$ | 75.28 | 90 | 397 |
| ST-3 | 74.46 | 95 | 436 |
| ST-3 $^\sigma$ | 73.69 | 95 | 219 |
| ST-3 | 73.31 | 96.5 | 351 |
| ST-3 $^\sigma$ | 72.62 | 96.5 | 167 |
| ST-3 | 70.46 | 98 | 220 |
| ST-3 $^\sigma$ | 69.75 | 98 | 116 |
| ST-3 | 63.88 | 99 | 120 |
| ST-3 $^\sigma$ | 63.25 | 99 | 69 |

## Instructions

The pieces of code proper to the ST-3(sigma) method described in the paper are available in the following 2 files:
* `quantized.py`: the `STEPrunedLayer` is the layer-type allowing for straight-through estimation as well as soft-thresholding
* `pruning.py`: the `l1` and `l1std` functions are responsible for zeroing out the correct weights either based on l1-magnitued (for ST-3) or l1-magnitude normalized by layer 'width' (ST-3-sigma). The cubic pruning-ratio increase is also handled in this file.

In those files a lot of code is also there to reproduce results of ProbMask, GMP and LRR and can be ignored, they are included for completeness. The other files are mostly boilerplate and part of a bigger framework that isn't necessary to understand ST-3.



## Execution
`$ ./launch <path/to/configs>`

## Weights
The weights will be made available soon
