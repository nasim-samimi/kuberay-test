#!/bin/bash

# Check for Ray start parameters and modify behavior accordingly
if [[ "$@" == *"--head"* ]]; then
    echo "Starting Ray head node with SCHED_RR policy"
    # Apply chrt for the head node
    exec chrt -r 99 ray start --head "$@"
elif [[ "$@" == *"--address"* ]]; then
    echo "Starting Ray worker node with SCHED_RR policy"
    # Apply chrt for worker nodes
    exec chrt -r 99 ray start "$@"
else
    # For other commands, just run normally
    exec "$@"
fi
