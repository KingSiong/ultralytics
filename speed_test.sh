#!/bin/bash

for ((i=0;i<1;i++));
do
echo $i'-th val:'
CUDA_VISIBLE_DEVICES='0' python -u test.py
done