# Transfer Learning for Neuroimaging via Re-Use of Deep Neural Network Features

This repository provides code for an efficient transfer learning method designed for structural MRI scans of brains.

         |  
As a deep learning model, we use the Simple Fully Connected Neural Networks by [Peng et al (2021)](https://www.sciencedirect.com/science/article/pii/S1361841520302358)
As pre-training data set, we use the [UK Biobank](https://www.nature.com/articles/nn.4393) neuroimaging data set giving T1-weighted structural MRI data. As target data sets,
we use [OASIS-3](https://www.nature.com/articles/nn.4393), [IXI](https://www.nature.com/articles/nn.4393), and [ABIDE I and II](https://www.nature.com/articles/nn.4393).  | <img src="https://github.com/PeterHolderrieth/TransferLearning_Neuroimaging/blob/main/visualization/plots/brain_manifold.png" width="400" height="700">






![Features of deep neural networks](https://github.com/PeterHolderrieth/TransferLearning_Neuroimaging/blob/main/visualization/plots/preliminary_visualization_features.png)


## Repository

This library provides an implementation of SteerCNPs and the code for two experiments.

We tested our model on two data sets: a Gaussian process regression task and real-world weather data.
Below, the model predicts the wind in a cyclic region of 500km radius around Memphis in the South of the US.
It gets measurements of wind, temperature and pressure from places marked in red.

![ERA5Predictions](https://github.com/PeterHolderrieth/Steerable_CNPs/blob/master/plots/era5/ERA5_predictions.png?raw=true)



## Installation

To use this repository, one can simply clone it and  install requirements specified in `requirements.txt`.
Please note this installs additional packages used for visualization. In particular, we use:

- [PyTorch](https://https://pytorch.org/) as a library for automatic differentation.
- the [UK Biobank](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=overview) neuroimaging data set giving T1-weighted structural MRI data.

## Structure of the repository
The core implementation of SteerCNPs are all files in the root. The folder "tasks" gives the two main tasks (data sets+ data loading scripts) which we have given our model: GP vector field data and
real-world weather data. The folder "experiments" gives the main execution file per task. 
The folder CNP gives an implementation of [Conditional Neural Processes](https://arxiv.org/abs/1807.01613)
to compare our results.

