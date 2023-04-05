#!/bin/bash

save_path=cpy_train_results

: << COMMENT
BUCKET_NAME=gs://eu4food-dataset
DATASET_PATH=eu4food-dataset/Images
ENV_NAME=eu4food_cv_product_inference_venv 

model=resnet18
batch_size=64
learning_rate=0.01
weight_decay=1e-3
momentum=0.95
device=cpu
#num_epochs=100
num_epochs=2
num_workers=3

# Download files from the bucket
gsutil -m cp -r $BUCKET_NAME .

# Check the exit status of the gsutil command
if [ $? -eq 0 ]; then
	echo "Downloaded files from $BUCKET_NAME"
else
	echo "ERROR: Could not download files from $BUCKET_NAME"
	exit 1
fi

source $ENV_NAME/bin/activate
python3.8 train.py $model $DATASET_PATH \
	--save_path save_path \
	--balanced_dataset \
	--augment \
	--batch_size $batch_size \
	--learning_rate $learning_rate \
	--weight_decay $weight_decay \
	--momentum $momentum \
	--device $device \
	--num_epochs $num_epochs \
	--num_workers $num_workers

# Check the exit status of the train command
if [ $? -eq 0 ]; then
	echo "Finished training"
else
	echo "ERROR: Training failed"
	exit 1
fi

# Remove downloaded dataset
rm -rf $DATASET_PATH

COMMENT

most_recent_train=$(ls -v1 $save_path | tail -n 1)
if [ $? -eq 0 ]; then
	echo "Most recent train is $most_recent_train"
else
	echo "ERROR: Could not get most recent train"
	exit 1
fi

