#!/bin/bash

BUCKET_NAME=gs://eu4food-dataset
DATASET_PATH=eu4food-dataset/Images

# Download files from the bucket
gsutil -m cp -r $BUCKET_NAME .

# Check the exit status of the gsutil command
if [ $? -eq 0 ]; then
	echo "Downloaded files from $BUCKET_NAME"
else
	echo "ERROR: Could not download files from $BUCKET_NAME"
	exit 1
fi
