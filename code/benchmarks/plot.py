'''
This script processes the output of `run_all.sh`. The name of the folder
containing the pickle files must be given as argument.
'''

# Author: Andre Manoel <andre.manoel@owkin.copm>
# License: BSD 3 clause

import glob
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# All .pickle files must be inside a given folder
if len(sys.argv) > 1:
    folder = sys.argv[1]
else:
    raise ValueError('folder name not given')

# Load results into dataframe
data = pd.DataFrame(columns=['layers', 'factor', 'elapsed'])
for pickle_file in glob.glob('{}/benchmarks_*.pickle'.format(folder)):
    d = pickle.load(open(pickle_file, 'rb'))
    data = data.append({
        'layers' : d['layers'],
        'factor' : d['factor'],
        'multi' : d['multi'],
        'elapsed_mean' : np.array(d['elapsed']).mean(),
        'elapsed_std' : np.array(d['elapsed']).std(),
    }, ignore_index=True)

fig, axs = plt.subplots(1, 3, figsize=(10,4))

layers = [2, 3, 4]
for k, layer in enumerate(layers):
    data_filtered = data[data['layers'] == layer]

    data_crb = data_filtered[data_filtered['multi'] == 0]
    data_multi = data_filtered[data_filtered['multi'] == 1]

    axs[k].errorbar(data_crb['factor'], data_crb['elapsed_mean'],
            yerr=data_crb['elapsed_std'], fmt='o', label='crb')
    axs[k].errorbar(data_multi['factor'], data_multi['elapsed_mean'],
            yerr=data_crb['elapsed_std'], fmt='o', label='multi')

    axs[k].set_title('{} convolutional layers'.format(layer))
    axs[k].set_xlabel('channel rate')

axs[0].set_ylabel('runtime (s)')
axs[0].legend()

fig.tight_layout()
fig.savefig('{}.pdf'.format(folder))
