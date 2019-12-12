#!/bin/bash
echo "WARNING: time differences won't be significant unless one uses a GPU"
echo -e "\nTrying method 'naive'"
python3 run.py --channels 25 --layers 2 --kernel_size 5 --input_size 32 --batch_size 10 --naive
echo -e "\nTrying method 'multi'"
python3 run.py --channels 25 --layers 2 --kernel_size 5 --input_size 32 --batch_size 10 --multi
echo -e "\nTrying method 'crb'"
python3 run.py --channels 25 --layers 2 --kernel_size 5 --input_size 32 --batch_size 10
