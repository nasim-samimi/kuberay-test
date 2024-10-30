#!/bin/bash

# Directory containing Ray job files
JOB_FOLDER="mobilenet_imagenet/hcbs-test/"

# Loop through each YAML file in the job folder
for JOB_FILE in "$JOB_FOLDER"/*.yaml; do
    echo "Processing job file: $JOB_FILE"

    # Submit the Ray job
    kubectl create -f "$JOB_FILE"
    
    # Extract job name to monitor its completion
    JOB_NAME=$(kubectl get -f "$JOB_FILE" -o jsonpath='{.metadata.name}')
    
    # Wait for the Ray job to complete
    echo "Waiting for Ray job $JOB_NAME to complete..."
    kubectl wait --for=condition=complete job/"$JOB_NAME" 

    # Clean up the job
    echo "Cleaning up Ray job $JOB_NAME..."
    kubectl delete -f "$JOB_FILE"
    
    echo "Completed job $JOB_FILE."
done

