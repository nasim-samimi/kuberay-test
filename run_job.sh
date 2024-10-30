#!/bin/bash

print_usage() {
  echo "Usage: $0 [JOB_FOLDER]"
  echo
  echo "Arguments:"
  echo "  JOB_FOLDER   Optional. Path to the folder containing Ray job YAML files."
  echo "               Defaults to 'mobilenet_imagenet/cfs-test/' if not specified."
  echo
  echo "Example:"
  echo "  $0 my_ray_jobs_folder"
  exit 1
}

# Check for help flag
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  print_usage
fi

# Directory containing Ray job files
JOB_FOLDER="${1:-mobilenet_imagenet/hcbs-test/}"

# Verify that the folder exists
if [ ! -d "$JOB_FOLDER" ]; then
  echo "Error: Folder '$JOB_FOLDER' does not exist."
  exit 1
fi

# Loop through each YAML file in the job folder
for JOB_FILE in "$JOB_FOLDER"/*.yaml; do
    echo "Processing job file: $JOB_FILE"

    # Submit the Ray job
    kubectl create -f "$JOB_FILE"
    
    # Start stress-ng in the background with CPU load, capture its PID
    nice -n 10 stress-ng --cpu 9 --cpu-load 95 &
    STRESS_PID=$!
    
    # Extract job name to monitor its completion
    JOB_NAME=$(kubectl get -f "$JOB_FILE" -o jsonpath='{.metadata.name}')
    
    # Wait for the Ray job to complete
    echo "Waiting for Ray job $JOB_NAME to complete..."
    MAX_ATTEMPTS=1000  # Adjust based on expected job duration
    DELAY=60  # Check every 60 seconds

    # Poll the RayJob status until it is 'SUCCEEDED' or 'FAILED'
    for ((i=1; i<=MAX_ATTEMPTS; i++)); do
        JOB_STATUS=$(kubectl get rayjob "$JOB_NAME" -o jsonpath='{.status.jobStatus}')
        echo "Attempt $i: RayJob status is $JOB_STATUS"
        
        if [[ "$JOB_STATUS" == "SUCCEEDED" ]]; then
            echo "RayJob $JOB_NAME completed successfully."
            break
        elif [[ "$JOB_STATUS" == "FAILED" ]]; then
            echo "RayJob $JOB_NAME failed."
            break
        fi

        if [[ $i -eq $MAX_ATTEMPTS ]]; then
            echo "Reached maximum attempts without detecting job completion. Exiting with status $JOB_STATUS."
            exit 1
        fi

        sleep $DELAY
    done

    echo "Stopping stress-ng process with PID $STRESS_PID..."
    kill $STRESS_PID

    # Clean up the job
    echo "Cleaning up Ray job $JOB_NAME..."
    kubectl delete -f "$JOB_FILE"
    
    echo "Completed job $JOB_FILE."
    sleep 10  # Add a delay between jobs
done

