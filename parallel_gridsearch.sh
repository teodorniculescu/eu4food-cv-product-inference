#!/bin/bash

model=resnet18
num_epochs=4
num_workers=4

parallel=12

# Initialize a counter variable and a process ID array
counter=0
pids=()

# Loop over different values of batch size, learning rate, weight decay, and momentum
for batch_size in 32 64 128
do
        for learning_rate in 0.001 0.01 0.1
        do
                for weight_decay in 1e-4 1e-3 1e-2
                do
                        for momentum in 0.9 0.95 0.99
                        do
                                # Determine which GPU to use based on the counter variable
                                if [ $((counter % 2)) -eq 0 ]; then
                                        device=cuda:0
                                else
                                        device=cuda:1
                                fi

                                # Run the Python script with the specified hyperparameters and GPU
                                python train.py $model eu4food-dataset/Images/ \
                                        --balanced_dataset \
                                        --augment \
                                        --batch_size $batch_size \
                                        --learning_rate $learning_rate \
                                        --weight_decay $weight_decay \
                                        --momentum $momentum \
                                        --device $device \
                                        --num_epochs $num_epochs \
                                        --num_workers $num_workers \
                                        &

                                pids+=($!)
                                counter=$((counter + 1))

                                # Wait until there are no more than parallel active processes
                                while [ ${#pids[@]} -ge $parallel ]
                                do
                                        wait -n ${pids[*]}
                                        pids=(${pids[@]/$?/})
                                done
                        done
                done
        done
done

# Wait for all the remaining background processes to complete
wait ${pids[*]}

