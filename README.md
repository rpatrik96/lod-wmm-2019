# Weight Matrix Modification - LOD2019 paper supplement
### Authors: Patrik Reizinger & Bálint Gyires-Tóth

## General description:
The project contains the source files (without the datasets) which implement WMM _(Weight Matrix Modification,)_ a weight matrix-based regularization technique for Deep Neural Networks. In the following the proposed methods are shortly introduced, including the evaluation framework.

### Weight shuffling
Weight shuffling is based on the assumption that locally the coefficients of a weight matrix are correlated. Based on this, we hypothesize that shuffling the weight within a rectangular window - which is under the beforementioned assumption a way of adding correlated noise to the weights - may help reduce overfitting.


### Weight reinitialization
Weight reinitialization aims to reduce overfitting while partially reinitializing the weight matrix, thus in the case of a non-representative training set it may reduce the over-/underestimation of the significance regarding specific input data.

## Usage:
The code can be run with typing the following command:
```
python ignite_main.py --model MODEL --dataset DS --num-trials TRIALS
```
Where `MODEL` can be one of following:
- `mnistnet`
- `seqmnistnet`
- `cifar10net`
- `lstmnet`
- `jsbchoralesnet`

While `DS` should be (mismatch check is included in the code, the code structure was decided to be that way to enable the usage of multiple networks for the same dataset):
- MNIST
- CIFAR10
- SIN (for the synthetic data) or SIN-NOISE (for noisy variant)

The default is to use Weight Shuffling, Weight Reinitialization can be selected by specifying `--choose-reinit`. `TRIALS` gives the number of runs by the hyper-optimization engine.
The result of the hyper-optimization will be a `.tsv` file containing essential information about each training run.

More arguments concerning e.g. checkpointing or logging can be found in `ignite_main.py`.

The parameters of the optimizer (e.g. learning rate, momentum) can be set up in the `ModelParameters` class in `descriptors.py`.

## Results
The 20 best results for each dataset and each method is included in the `results` directory, where the most important parameters are also included beside performance metrics.