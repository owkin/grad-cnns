#!/bin/bash
python3 run.py --layers 2 --factor 1 --input_size 256 --kernel_size 5
python3 run.py --layers 2 --factor 2  --input_size 256 --kernel_size 5
python3 run.py --layers 2 --factor 4 --input_size 256 --kernel_size 5
python3 run.py --layers 2 --factor 8 --input_size 256 --kernel_size 5
python3 run.py --layers 2 --factor 16 --input_size 256 --kernel_size 5
python3 run.py --layers 2 --factor 1 --multi --input_size 256 --kernel_size 5
python3 run.py --layers 2 --factor 2 --multi --input_size 256 --kernel_size 5
python3 run.py --layers 2 --factor 4 --multi --input_size 256 --kernel_size 5
python3 run.py --layers 2 --factor 8 --multi --input_size 256 --kernel_size 5
python3 run.py --layers 2 --factor 16 --multi --input_size 256 --kernel_size 5
python3 run.py --layers 3 --factor 1 --input_size 256 --kernel_size 5
python3 run.py --layers 3 --factor 2 --input_size 256 --kernel_size 5
python3 run.py --layers 3 --factor 3 --input_size 256 --kernel_size 5
python3 run.py --layers 3 --factor 4 --input_size 256 --kernel_size 5
python3 run.py --layers 3 --factor 5 --input_size 256 --kernel_size 5
python3 run.py --layers 3 --factor 1 --multi --input_size 256 --kernel_size 5
python3 run.py --layers 3 --factor 2 --multi --input_size 256 --kernel_size 5
python3 run.py --layers 3 --factor 3 --multi --input_size 256 --kernel_size 5
python3 run.py --layers 3 --factor 4 --multi --input_size 256 --kernel_size 5
python3 run.py --layers 3 --factor 5 --multi --input_size 256 --kernel_size 5
python3 run.py --layers 4 --factor 1 --input_size 256 --kernel_size 5
python3 run.py --layers 4 --factor 1.75 --input_size 256 --kernel_size 5
python3 run.py --layers 4 --factor 2.5 --input_size 256 --kernel_size 5
python3 run.py --layers 4 --factor 3.25 --input_size 256 --kernel_size 5
python3 run.py --layers 4 --factor 4 --input_size 256 --kernel_size 5
python3 run.py --layers 4 --factor 1 --multi --input_size 256 --kernel_size 5
python3 run.py --layers 4 --factor 1.75 --multi --input_size 256 --kernel_size 5
python3 run.py --layers 4 --factor 2.5 --multi --input_size 256 --kernel_size 5
python3 run.py --layers 4 --factor 3.25 --multi --input_size 256 --kernel_size 5
python3 run.py --layers 4 --factor 4 --multi --input_size 256 --kernel_size 5
