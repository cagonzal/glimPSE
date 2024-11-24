#!/bin/bash

for dir in */; do
    if [ -d "$dir" ]; then
        echo "Processing folder: $dir"
        cd "$dir"
        
        echo "Running cleanup..."
        ./cleanup
        
        echo "Submitting job..."
        sbatch submit.slurm
        
        cd ..
    fi
done

echo "All folders processed."
