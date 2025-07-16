#!/bin/bash

# Configuration
USERNAME="imw2293"
NODES=("c1" "c2" "c3" "c4" "c5" "c6" "c7" "c8" "c9" "c10")  # Add more nodes as needed
WORK_DIR="/mount/sirocco1/imw2293/GROUP_CAT/SelfCalGroupFinder"
PYTHON_CMD="python3 py/exec.py mcmc 14" # number is node number minus 1
OUTPUT_FILE="y1full8_mcmc.out"

# Function to process a single node
process_node() {
    local node=$1
    local node_num=${node//[!0-9]/}
    local node_output_file="${OUTPUT_FILE/.out/_$node.out}"
       echo "Processing node: $node"
    
    ssh -T $node << EOF
        echo "Connected to $node"
        
        # Kill existing python3 processes for user
        echo "Killing existing python3 processes for $USERNAME on $node..."
        pkill -u $USERNAME python3
        sleep 2
        
        # Change to working directory
        cd $WORK_DIR
        
       # Start the job
        echo "Starting MCMC job on $node..."
        nohup $PYTHON_CMD x$(($node_num - 1)) &> "$node_output_file" &
        
        # Show the new process
        sleep 1
        echo "New python3 processes on $node:"
        pgrep -u $USERNAME python3
EOF
    
    echo "----------------------------------------"
}

# Main execution
echo "Starting distributed MCMC job deployment..."
echo "Target nodes: ${NODES[@]}"
echo "Command: $PYTHON_CMD"
echo "=========================================="

# Process each node
for node in "${NODES[@]}"; do
    process_node $node
done

echo "All nodes processed!"
echo "To check status, run:"
echo "for node in ${NODES[@]}; do echo \"=== \$node ===\"; ssh \$node 'pgrep -u $USERNAME python3 | wc -l'; done"