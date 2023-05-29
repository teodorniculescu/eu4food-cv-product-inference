#!/bin/bash

save_path=train_results

DATASET_BUCKET_NAME=gs://eu4food-dataset
MODEL_BUCKET_NAME=gs://eu4food-public
DATASET_PATH=dataset/
#DATASET_PATH=dataset_gcloud/20_products/
#DATASET_PATH=dataset_gcloud/100_products/
ENV_NAME=eu4food_cv_product_inference_venv 
DOWNLOAD_AND_DELETE=false

model=resnet18
batch_size=64
learning_rate=0.1
weight_decay=0.01
momentum=0.9
device=cuda:0
num_epochs=100
num_workers=0
augment_train=RandAugment
augment_valid=$augment_train
preload_images=yes
use_mean_std=no

if [ "$use_mean_std" = "yes" ]; then
    use_mean_std="--use_mean_std"
else
    use_mean_std=""
fi

if [ "$preload_images" = "yes" ]; then
    preload_images="--preload_images"
else
    preload_images=""
fi

source $ENV_NAME/bin/activate

if [ "$DOWNLOAD_AND_DELETE" = true ]; then
	echo "Download dataset"
	#mkdir $DATASET_PATH
	#gsutil -m cp -r $DATASET_BUCKET_NAME/20_products/* $DATASET_PATH
	python download_dataset.py $DATASET_PATH
fi

# Check the exit status of the gsutil command
if [ $? -eq 0 ]; then
	echo "Downloaded files from $DATASET_BUCKET_NAME"
else
	echo "ERROR: Could not download files from $DATASET_BUCKET_NAME"
	exit 1
fi

python3.8 train.py $model $DATASET_PATH \
	$use_mean_std \
	$preload_images \
	--augment_train $augment_train \
	--augment_valid $augment_valid \
	--save_path $save_path \
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

if [ "$DOWNLOAD_AND_DELETE" = true ]; then
	echo "Remove dataset"
	rm -rf $DATASET_PATH
fi

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
	echo "Compare current class names with uploaded class names"

	gsutil -m cp -r $MODEL_BUCKET_NAME/best_model/obj.names uploaded_obj.names

	if [ $? -eq 0 ]; then
		echo "Copied obj.names"
	else
		echo "ERROR: Could not obj.names"
		exit 1
	fi

	sort uploaded_obj.names > sorted_uploaded_obj.names
	sort $most_recent_train/obj.names > sorted_local_obj.names

	if diff sorted_uploaded_obj.names sorted_local_obj.names > /dev/null; then
		echo "The local obj.names and the uploaded obj.names are the same"

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
	else
		echo "The obj.names are different"
		upload=0
	fi


fi

if [ $upload -eq 0 ]; then
	gsutil -m cp -r $most_recent_train/* $MODEL_BUCKET_NAME/best_model
	if [ $? -eq 0 ]; then
		echo "Successfuly uploaded new model"
	else
		echo "ERROR: Could not upload model"
		exit 1
	fi
fi

exit 0
