# DL-LA: Deep Learning Leakage Assessment
This repository contains source code related to the article [DL-LA: Deep Learning Leakage Assessment - A modern roadmap for SCA evaluations](https://eprint.iacr.org/2019/505.pdf) and provides download links for obtaining associated trace files.

## Short Description of DL-LA
DL-LA is a leakage assessment methodology based on deep learning. Simply put, a neural network is trained to become a classifier in order to distinguish between two groups of side-channel measurements which have been acquired while supplying a cryptographic implementation with one of two distinct fixed inputs (fixed-vs-fixed). In the subsequent validation phase the classification capability of the network is evaluated. In case the classifier succeeds with a (statistically) significantly higher percentage of correct classifications than it could be achieved by a randomly guessing binary classifier, it can be concluded that the side-channel measurements seen by the network during the training phase reveal enough input-dependent information to confidently distinguish the two groups.

## Content of the Repository
In this repository we share simple Python (and C++) scripts to perform deep learning leakage assessment on your own side-channel data and partially reproduce the results reported in [DL-LA](https://eprint.iacr.org/2019/505.pdf). Please note, that all given scripts are reduced to their basic functionality and have been kept short and simple on purpose. For reproducibility of (a part of) the experimental results presented in the paper we have hosted the underlying leakage traces for two case studies, namely CS3 and CS5 (see [DL-LA](https://eprint.iacr.org/2019/505.pdf)). Those trace files can be downloaded here:
- CS3: Serialized PRESENT implementation, randomized clock (heavily misaligned), measured on SAKURA-G FPGA board: [LINK](https://ruhr-uni-bochum.sciebo.de/s/uapVSe9CxOxxwis)
- CS5: Serialized PRESENT threshold implementation, bad trigger (slightly misaligned), measured on SAKURA-G FPGA board: [LINK](https://ruhr-uni-bochum.sciebo.de/s/7kNH7o8nPnmNPTI)

## Getting Started
The simplest way of getting started is to download the side-channel traces linked above and in the respective folders. However, you may also start right away with your own SCA data. In any case, just follow the steps below to get everything up and running:
- Step 1: edit/check `traces.yml` and make sure that all trace sets which should be analyzed are correctly defined
- Step 2: edit/check `extract_mean_and_std_deviation.py` and make sure that all trace sets which should be analyzed are listed under "names"
- Step 3: execute the `extract_mean_and_std_deviation.py` script - this may take some time depending on the size of your SCA data but only needs to be executed once at the beginning or whenever the trace sets or their definitions have changed
- Step 4: choose one of the three DL-LA scripts and adapt, if necessary, the name of the trace set and the training/validation parameters:
  1. `DL-LA_MLP_SA_inputs`: DL-LA using the standard MLP network with final sensitivity analysis based on the network inputs
  2. `DL-LA_MLP_SA_first_layer_weights`: DL-LA using the standard MLP network with final sensitivity analysis based on the first layer weights
  3. `DL-LA_CNN_SA_inputs`: DL-LA using the standard CNN network with final sensitivity analysis based on the network inputs
- Step 5: execute the chosen and adapted script
- Step 6: find the validation accuracy results in `val_acc.log` (ASCII) and the sensitivity analysis results in `sensi.dat` (BINARY)
- Step 7: you may want to use the C++ multi-precision log probability calculator (Boost C++ library required) in order to convert the validation accuracy values and the size of the validation set into log probabilities
- Step 8: Repeat from Step 4 for further analyses on the same trace sets

## Contact and Support
Please contact Thorben Moos (thorben.moos@rub.de) if you have any questions, comments or if you found a bug that should be corrected.

## Licensing
Please see `LICENSE` for licensing instructions.

## Publication
[DL-LA: Deep Learning Leakage Assessment - A modern roadmap for SCA evaluations](https://eprint.iacr.org/2019/505.pdf).
