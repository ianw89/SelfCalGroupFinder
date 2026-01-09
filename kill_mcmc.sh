#!/bin/bash

# Configuration
USERNAME="imw2293"
NODES=("c1" "c2" "c3" "c4" "c5" "c6" "c7" "c8" "c9" "c10")  # Add more nodes as needed
WORK_DIR="/mount/sirocco1/imw2293/GROUP_CAT/SelfCalGroupFinder"
PYTHON_CMD="python3 py/scripts/exec.py mcmc 12" # number is node number minus 1
OUTPUT_FILE="/mount/sirocco1/imw2293/GROUP_CAT/LOGS/y1full8_v1_mcmc.out"


kill_node() {
    local node=$1
    local node_num=${node//[!0-9]/}
    local node_output_file="${OUTPUT_FILE/.out/_$node.out}"
       echo "Processing node: $node"
    
    ssh -T $node << EOF
        echo "Connected to $node"
        
        # Kill python3 processes and kdGroupFinder_omp for user
        echo "Killing existing python3 processes for $USERNAME on $node..."
        pkill -u $USERNAME python3
        # Also kill group finder
        pkill -u $USERNAME kdGroupFinder_omp
        sleep 5
        
        echo "python3 processes on $node:"
        pgrep -u $USERNAME python3
EOF
    
    echo "----------------------------------------"
}
# Main execution
echo "Killing MCMC jobs..."
echo "Target nodes: ${NODES[@]}"
echo "=========================================="

# Process each node
for node in "${NODES[@]}"; do
    kill_node $node
done

echo "All nodes killed!"
echo "To check status, run:"
echo "for node in ${NODES[@]}; do echo \"=== \$node ===\"; ssh \$node 'pgrep -u $USERNAME python3 | wc -l'; done"