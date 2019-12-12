'''
Benchmark differentially-private training (see Abadi et al. 2016) on CNNs of
arbitrary size. Inputs are generated at random.
'''

# Author: Andre Manoel <andre.manoel@owkin.copm>
# License: BSD 3 clause

import argparse
import pickle
import time
import uuid

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm, trange

from gradcnn import make_optimizer, replicate_model
from gradcnn import crb as nn


class Net(nn.Module):
    '''Create CNN of arbitrary size'''
    def __init__(self, input_size=(1, 32, 32), n_layers=2, n_channels=25,
            factor=1.0, kernel_size=5):
        super().__init__()
        self.input_size = input_size

        # Layers grow at factor `factor`
        conv_layers = []
        for i in range(n_layers):
            channels_in = int(n_channels * (factor ** (i - 1))) if i > 0 else \
                input_size[0]
            channels_out = int(n_channels * (factor ** i))

            conv_layers.append(nn.Conv2d(channels_in, channels_out,
                kernel_size, 1, padding=2))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.conv_output = self.conv_block(torch.randn(1, *input_size)).size(1)

        self.fc = nn.Linear(self.conv_output, 10)

    def conv_block(self, x):
        for i, conv in enumerate(self.conv_layers):
            x = F.relu(conv(x))
            if i % 2:  # MaxPooling every other layer
                x = F.max_pool2d(x, 2, 2)
        return x.view(len(x), -1)

    def forward(self, x):
        x = self.fc(self.conv_block(x))
        return x



def run_benchmark(model, optimizer, input_size, n_batches=20, batch_size=32,
        naive=False, check_grad=False):
    '''Iterate training over a few batches'''

    device = next(model.parameters()).device

    model.train()

    batches = range(n_batches) if check_grad else trange(n_batches)
    for b in batches:
        data = torch.randn(batch_size, *input_size).float().to(device)
        target = torch.randint(10, (batch_size,)).to(device)

        # Reset optimizer and do backward/forward passes
        if naive:
            for i in range(batch_size):
                optimizer.zero_grad()
                output = model(data[i].unsqueeze(0))
                loss = F.cross_entropy(output, target[i].unsqueeze(0))
                loss.backward()

                # Per-example gradients are always stored in `p.bgrad`
                for p in model.parameters():
                    if i == 0:
                        p.bgrad = torch.empty((batch_size, *p.grad.shape)).to(device)
                    p.bgrad[i] = p.grad.clone() / batch_size
        else:
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            if hasattr(model, 'models'):  # for `multi`: grad from bgrad
                model.reduce_batch()

        # Print average of gradient at each layer
        if check_grad:
            grad_avg = np.mean([p.bgrad.mean().item() for p in model.parameters()])
            print('GRADIENT AVG AT BATCH {}: {}'.format(b + 1, grad_avg))

        optimizer.step()
        if hasattr(model, 'models'):  # for 'multi': recreate pointers
            model.reassign_params()

    model.eval()


def run_experiment(params):
    '''Initialize all structures and repeats training a number of times'''

    # initialize flags
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(params.seed)
    if use_cuda:
        torch.cuda.manual_seed(params.seed + 1)

    results = vars(params)
    print('PARAMETERS:', results)

    device = torch.device('cuda' if use_cuda else 'cpu')

    # generate model and optimizer
    input_size = (3, params.input_size, params.input_size)

    MNet = replicate_model(net_class=Net, batch_size=params.batch_size) if params.multi else Net
    model = MNet(input_size=input_size, n_layers=params.layers,
            n_channels=params.channels, factor=params.factor,
            kernel_size=params.kernel_size).to(device)
    # print(model)

    optimizer_class = optim.Adam if params.adam else optim.SGD
    if not params.nodp:
        if not params.naive:
            model.get_detail(True)  # activate per-sample computation
        optimizer_class = make_optimizer(
            cls=optimizer_class,
            noise_multiplier=params.noise_multiplier,
            l2_norm_clip=params.l2_norm_clip,
        )

    optimizer_params = {'lr' : params.lr}
    optimizer = optimizer_class(model.parameters(), **optimizer_params)

    elapsed = []

    # perform training
    for trial in tqdm(range(1, params.trials + 1)):
        # initialize counter
        if use_cuda:
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        run_benchmark(model, optimizer, input_size,
                n_batches=params.n_batches, batch_size=params.batch_size,
                naive=params.naive, check_grad=params.check_grad)

        # compute overall training time
        if use_cuda:
            torch.cuda.synchronize()
        elapsed_time = time.perf_counter() - start_time

        tqdm.write('TRIAL {}:'.format(trial))
        tqdm.write('\tELAPSED: {:.3f}'.format(elapsed_time))

        elapsed.append(elapsed_time)

    # put results into dictionary
    results['elapsed'] = elapsed.copy()

    # save results to pickle file
    with open('benchmarks_{}.pickle'.format(uuid.uuid1()), 'wb') as output:
        pickle.dump(results, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--multi', action='store_true', default=False)
    parser.add_argument('--naive', action='store_true', default=False)
    parser.add_argument('--adam', action='store_true', default=False)
    parser.add_argument('--nodp', action='store_true', default=False)
    parser.add_argument('--check_grad', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_batches', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--input_size', type=int, default=32)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--factor', type=float, default=1.0)
    parser.add_argument('--channels', type=int, default=25)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--noise_multiplier', type=float, default=1.1)
    parser.add_argument('--l2_norm_clip', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=1.0)
 
    params = parser.parse_args()

    run_experiment(params)
