# Beyond RNNs and CNNs: A Unified Transformer for Spatiotemporal Prediction

## Installation

```
# Exp Setting: PyTorch: 2.1.0+ Python 3.10
conda env create -f environment.yml  
conda activate torch
```

## Data Preparation

Take the Moving MNIST dataset as an example, the `data` folder has the following structure:

```
|-- data
   |-- moving_mnist
      |-- mnist_cifar_test_seq.npy
      |-- mnist_test_seq.npy
      |-- train-images-idx3-ubyte.gz
```

We provide a [Gdrive](https://drive.google.com/drive/folders/1RohIA_RJFvFjDU3VdRfgv3QzirKceXPA?usp=sharing) to download the Moving MNIST datasets.



## Train

We provide a training script. Simply run the following command in the project root:

```
# Moving MNIST
bash scripts/mmnist/MeldFormer_train.sh
```

By default, the model is trained for 2000 epochs with a learning rate of 5e-4.

## Acknowledgments

Our code is based on [OpenSTL](https://github.com/chengtan9907/OpenSTL). We sincerely appreciate for their contributions.


