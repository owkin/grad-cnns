# Efficient Per-Example Gradient Computations in Convolutional Neural Networks

*By Gaspar Rochette \<gaspar.rochette@ens.fr\>, Andre
Manoel \<andre.manoel@owkin.com\> and Eric W. Tramel \<eric.tramel@owkin.com\>*

*For more details, please check the report at [[arXiv:1912.06015]](https://arxiv.org/abs/1912.06015).*

Computing per-example gradients is not a trivial task in deep learning
frameworks. The naive approach, of simply looping through the examples, is
often unpractical: for a batch size of $B$, this approach typically makes
the code $O(B)$ times slower to run.

Other strategies have been discussed in reports and online forums (see
manuscript for more details). The first report to discuss such an issue is
[Goodfellow (2015)](https://arxiv.org/abs/1510.01799).
However, the technique it presents to deal with the issue is not applicable
to convolutional networks (CNNs).

A different solution which works better for CNNs was presented at this
[Github discussion on the Tensorflow
repository](https://github.com/tensorflow/tensorflow/issues/4897#issuecomment-290997283),
also by Goodfellow. We shall call this solution `multi`. It is essentially
the fastest solution found to this day, and is the one used by e.g.
`tf.privacy`, through the `tf.vectorized_map` function (see code
[here](https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/optimizers/dp_optimizer_vectorized.py#L124))
A good summary of the different strategies, with benchmarks in PyTorch, was
presented [here](https://github.com/pytorch/pytorch/issues/7786) by
Künstner.

In our report, we rediscuss these strategies, performing a series
of benchmarks. We focus on the differential privacy use case, and measure
how long does it take to process a few batches using a
differentially-private optimizer, see [Abadi et al.
(2016)](https://arxiv.org/abs/1607.00133). Most importantly, we
*introduce a new strategy*, which leverages the `groups` argument in
PyTorch's convolution: by considering samples in the batches as different
input channels, we are able to process them in parallel, while still
obtaining results for each of the channels, i.e. they are not automatically
reduced, as done for the batch dimension.

## Running benchmarks

*NOTE: all benchmarks were done on a `n1-standard-8` instance on GCP, with a
Nvidia P100 GPU; the image used was based on Ubuntu 18.04, with Python 3.5.3
and PyTorch 1.1 installed*

For obtaining similar benchmarks as those shown in the report, you
should first install the `gradcnn` package, then run
`code/benchmarks/run_test.sh`:

```
cd code
pip3 install --user .
cd benchmarks
./run_test.sh
```

The benchmark consists in iterating the training of a given CNN for a few
batches; parameters are specified inside the script, please check `run.py` for
more information. Three different strategies will be used: `naive`, `multi` and
`crb`. For instance, on a `n1-standard-8` instance at GCP with a P100, each
run of `naive` should take around 0.5s, each run of `multi` 0.3s, and each
run of `crb`, 0.1s. Pickle files containing more information on the
benchmarks will be created.

Note that this effect isn't noticeable if there is no GPU on your machine. On a
CPU, all three strategies lead to comparable runtimes.

Alternatively, one can run `run_gradcheck.sh` to print the average gradient
obtained using all the three strategies, during the first 5 batches. This
indicates that all of them provide approximately (but not exactly) the same
gradient.

Finally, the script `run_all.sh` will run exact the same benchmarks as in the
Figure 1 of the manuscript. After generating the pickle files with this script,
Figure 1 can be reproduced by means of `plot.py`.

## MNIST example

There is also a MNIST example in the `examples` folder, built with minimal
changes to the [default PyTorch MNIST
example](https://github.com/pytorch/examples/blob/master/mnist/main.py). It
consists in training a network with 4 layers, 2 of which are convolutional.
To run the standard example, without DP, just type `python3 mnist.py`. In a
`n1-standard-8` instance at GCP with a P100, one gets, after one epoch

```
Test set: Average loss: 0.0397, Accuracy: 9877/10000 (99%)
Elapsed time: 19.46s
```

Running `python3 mnist.py --dp` gives

```
Test set: Average loss: 0.5463, Accuracy: 8840/10000 (88%)
Elapsed time: 20.36s
```

While the accuracy is reduced--as expected, due to the noisy, norm-bounded
gradients--the runtime is virtually the same.

We also note that, in order to have the training to be
differentially-private, two changes were performed to the script:

1. Replacing `import torch.nn as nn` by `from gradcnn import crb as nn`,
   thus using our code to compute per-example gradients automatically when
   backprop is done. In practice, this leads to the creation of a `bgrad`
   attribute for each parameter, which contains the per-example gradients.
2. Replacing the optimizer by a new one which is built using the following
   code

    ```python
    model = Net().to(device)
    if args.dp:
      model.get_detail(True)
      Optimizer = make_optimizer(
          cls=optim.Adam,
          noise_multiplier=1.1,
          l2_norm_clip=1.0,
      )
    else:
      Optimizer = optim.Adam
    ```
