#!/bin/bash

# Find PIDs of all relevant processes (adjust criteria as needed)
pids=$(pgrep -f 'ray')

# Loop through each PID and apply `chrt`
for pid in $pids; do
  echo "Applying chrt to PID $pid"
  chrt -r -p 90 $pid || echo "Failed to change scheduling for PID $pid"
done
