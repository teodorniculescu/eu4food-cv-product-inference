#!/bin/bash

ENV_NAME=eu4food_cv_product_inference_venv 
dataset_path=dataset_gcloud/20_products/
source $ENV_NAME/bin/activate

model=resnet18
num_epochs=50
num_workers=2
parallel=12
save_path=train_results/gridsearch14May2023

# Initialize a counter variable and a process ID array
counter=0
pids=()

# Function to clean up child processes
function cleanup() {
        echo "Cleaning up child processes..."
        for pid in "${pids[@]}"
        do
                kill "$pid"
        done
        exit
}

# Trap the SIGINT signal and call the cleanup function
trap cleanup SIGINT

for augment_train in None RandAugment AugMix TrivialAugmentWide Custom
do
	for augment_valid in None RandAugment AugMix TrivialAugmentWide Custom
	do
		if [ "$augment_valid" != "None" ] && [ "$augment_valid" != "$augment_train" ]; then
			continue
		fi

		# Loop over different values of batch size, learning rate, weight decay, and momentum
		for batch_size in 128 64 32
		do
			for learning_rate in 0.0001 0.001 0.01 0.1
			do
				for weight_decay in 0.01
				do
					for momentum in 0.9
					do
						for use_mean_std in no yes
						do
							if [ "$use_mean_std" = "yes" ]; then
							    use_mean_std="--use_mean_std"
							else
							    use_mean_std=""
							fi

							# Determine which GPU to use based on the counter variable
							if [ $((counter % 2)) -eq 0 ]; then
								device=cuda:0
							else
								device=cuda:1
							fi

							sleep 0.1
							# Run the Python script with the specified hyperparameters and GPU
							python train.py $model $dataset_path \
								$use_mean_std \
								--dont_save_model \
								--save_path $save_path \
								--preload_images \
								--augment_train $augment_train \
								--augment_valid $augment_valid \
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

							if [ ${#pids[@]} -eq 12 ]; then
								for pid in ${pids[@]}; do
									wait $pid
								done
								pids=()
							fi
						done
					done
				done
                        done
                done
        done
done

if [ ${#pids[@]} -ne 0 ]; then
	for pid in ${pids[@]}; do
		wait $pid
	done
	pids=()
fi

