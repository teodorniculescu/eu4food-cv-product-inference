#!/bin/bash

model=resnet18
device=cuda:0
num_epochs=100
num_workers=4

for batch_size in 32 64 128
do
	for learning_rate in 0.001 0.01 0.1
	do
		for weight_decay in 1e-4 1e-3 1e-2
		do
			for momentum in 0.9 0.95 0.99
			do
			    python train.py resnet18 eu4food-dataset/Images/ \
				    --balanced_dataset \
				    --augment \
				    --batch_size $batch_size \
				    --learning_rate $learning_rate \
				    --weight_decay $weight_decay \
				    --momentum $momentum \
				    --device $device \
				    --num_epochs $num_epochs \
				    --num_workers $num_workers
			done
		done
	done
done
