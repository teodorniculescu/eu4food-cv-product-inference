#!/bin/bash

save_path=cpy_train_results

DATASET_BUCKET_NAME=gs://eu4food-dataset
MODEL_BUCKET_NAME=gs://eu4food-public
DATASET_PATH=eu4food-dataset/Images
ENV_NAME=eu4food_cv_product_inference_venv 

: << COMMENT
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
gsutil -m cp -r $DATASET_BUCKET_NAME .

# Check the exit status of the gsutil command
if [ $? -eq 0 ]; then
	echo "Downloaded files from $DATASET_BUCKET_NAME"
else
	echo "ERROR: Could not download files from $DATASET_BUCKET_NAME"
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

most_recent_train=$save_path/$(ls -v1 $save_path | tail -n 1)
if [ $? -eq 0 ]; then
	echo "Most recent train is $most_recent_train"
else
	echo "ERROR: Could not get most recent train"
	exit 1
fi

gsutil ls $MODEL_BUCKET_NAME >/dev/null 2>&1
if [ $? -eq 0 ]; then
	echo "Bucket $MODEL_BUCKET_NAME exists"
else
	echo "ERROR: No such bucket $MODEL_BUCKET_NAME"
	exit 1
fi

upload=1

gsutil ls $MODEL_BUCKET_NAME/best_model/
if [ $? -eq 1 ]; then
	echo "First model being uploaded"
	upload=0

else
	echo "Compare current train performance with uploaded model performance"

	gsutil -m cp -r $MODEL_BUCKET_NAME/best_model/scores.json uploaded_model_scores.json

	if [ $? -eq 0 ]; then
		echo "Copied scores"
	else
		echo "ERROR: Could not copy scores"
		exit 1
	fi

	uploaded_f1=$(cat uploaded_model_scores.json | jq -r '.test.f1')
	local_f1=$(cat $most_recent_train/scores.json | jq -r '.test.f1')

	if (( $(echo "$local_f1 > $uploaded_f1" | bc -l) )); then
		echo "Local F1 is greater than uploaded F1"
		upload=0
	else
		echo "Local F1 is not greater than uploaded F1. No need to upload files."
		upload=1
	fi

fi

if [ $upload -eq 0 ]; then
	gsutil -m cp -r $most_recent_train $MODEL_BUCKET_NAME
	if [ $? -eq 0 ]; then
		echo "Successfuly uploaded new model"
	else
		echo "ERROR: Could not upload model"
		exit 1
	fi
fi
