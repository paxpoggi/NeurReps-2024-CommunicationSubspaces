# Communication Subspaces Align in ANNs

## Abstract
Communication subspaces have recently been identified as a promising mechanism for selectively routing information between brain areas. In this study, we explored whether communication subspaces develop with training in artificial neural networks (ANNs) and explored differences across connection types. Specifically, we analyzed the subspace angles between activations and weights and between pairs of weight layers in ResNet-50 before and after training. We found that after training, activations were more aligned to the weight layers, although this effect decreased in deeper layers. We also found that for all branching, direct, and skip connections, weight layer pairs were more geometrically aligned in trained versus untrained models throughout the entire network. These findings motivate further exploration into whether learning induces similar subspace alignment in biological systems. Moreover, they highlight communication subspaces as a compelling framework for investigating distinct informational communication patterns associated with complex connections in the brain.

## Main results figure from ResNet Analysis
![Main Results Downsized](https://github.com/user-attachments/assets/be06c1c9-7216-4523-bbec-8a04be71d5e5)

Figure 1. Top: mean alignment between activation-to-weight connections. Bottom: mean alignment for weight-to-weight connections across layers.
