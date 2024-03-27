#!/bin/bash

for ((i=0;i<3;i++));
do
echo $i'-th val:'
CUDA_VISIBLE_DEVICES='1' python -u test.py
done