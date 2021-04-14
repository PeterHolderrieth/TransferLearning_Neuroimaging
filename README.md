# Transfer Learning for Neuroimaging via Re-Use of Deep Neural Network Features

<img align="right" src="https://github.com/PeterHolderrieth/TransferLearning_Neuroimaging/blob/main/visualization/plots/brain_manifold.png" width="400" height="700">

This repository provides code for an efficient transfer learning method designed for structural MRI scans of brains.
 
As a deep learning model, we use the Simple Fully Connected Neural Networks (SFCNs) by [Peng et al (2021)](https://www.sciencedirect.com/science/article/pii/S1361841520302358)
As pre-training data set, we use the [UK Biobank](https://www.nature.com/articles/nn.4393) neuroimaging data set giving T1-weighted structural MRI data. As target data sets,
we use [OASIS-3](https://www.oasis-brains.org/), [IXI](https://brain-development.org/ixi-dataset/), and [ABIDE I and II](http://fcon_1000.projects.nitrc.org/indi/abide/). 



## Installation

To use this repository, one can simply clone it and  install requirements specified in `requirements.txt`.
Please note this installs additional packages used for visualization. In particular, we use [PyTorch](https://https://pytorch.org/) 
as a library for automatic differentation.

## Structure of the repository
The core implementation of SFCNs can be found in `hello`.

