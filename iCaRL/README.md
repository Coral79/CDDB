# iCaRL PyTorch implementation

This is a fully working implementation of the iCaRL code that can be used to reproduce **CIFAR-100** experiments.

This code currently reproduces the CIFAR-100 *10 classes* per batch experiment (**figure 2.a, top right**).

Beware that this is not the official iCaRL code. The original Theano-Lasagne code for the CIFAR-100 experiments can be found in the [official repo](https://github.com/srebuffi/iCaRL).

Beware that this code won't work with the iILSVRC dataset as it uses a modified version version of the **ResNet** which is specific to CIFAR-100.

This implementation has been tested with Python 3.8.6 and PyTorch 1.7.1 (see the [conda environment](environment.yml) for more details).


## About this implementation

The code was manually translated from the original Theano-Lasagne one. This includes the **modified ResNet** and the custom **weight initialization** procedure. The code was translated by analyzing the original code and the Lasagne documentation in order to correctly port the correct default values for optimizers, layers, weight initialization procedures, etc. Beware that discrepancies with the original code may still exist.

This implementation is far from being efficient. **A more efficient, general and tested version is coming to the [Avalanche](https://github.com/ContinualAI/avalanche) Continual Learning library very soon!**

The reference conda environment can be found in [`environment.yml`](environment.yml)


## Reading the code

The entry point for the experiments is [`main_icarl.py`](main_icarl.py). Some annotations have been left allowing for a mapping between the Theano-Lasagne and the PyTorch code.

Various values are hard coded (for instance, the replay buffer size), but it should be easy to adapt them at will.

The `NCProtocol` class is the grandfather of the `NCScenario` class found in [Avalanche](https://github.com/ContinualAI/avalanche). Its role is to return batches of classes by splitting the CIFAR-100 dataset. Changing the dataset and the amount of classes allocated to each batch is very simple.

Tensorboard logging is not supported. Results are returned by using the same method used in the iCaRL original code, that is by saving tensors to disk (`top1_acc_list_cumul_icarl_cl10` and `top1_acc_list_ori_icarl_cl10`).

Again, it is **strongly encouraged** to use the Avalanche implementation of iCaRL, which will be integrated very soon and will surely be more general and tested.