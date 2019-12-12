#!/bin/bash
echo -e "\nChecking gradients are approximately the same for each strategy..."
echo -e "\nTrying method 'naive'"
python3 run.py --trials 1 --n_batches 5 --check_grad --channels 25 --layers 2 --kernel_size 5 --input_size 32 --batch_size 10 --naive
echo -e "\nTrying method 'multi'"
python3 run.py --trials 1 --n_batches 5 --check_grad --channels 25 --layers 2 --kernel_size 5 --input_size 32 --batch_size 10 --multi
echo -e "\nTrying method 'crb'"
python3 run.py --trials 1 --n_batches 5 --check_grad --channels 25 --layers 2 --kernel_size 5 --input_size 32 --batch_size 10
