#!/bin/bash

# Helper script to list available training experiments and checkpoints
# Usage: ./list_checkpoints.sh [dataset] [seed]

BASE_DIR="/home/xiaos7/data_areas/lmr-ihb-imaging/xiaos7/experiments/cytodiff"
DATASET=${1:-"custom_wbc"}
SEED=${2:-"seed0"}

echo "=========================================="
echo "Available Training Experiments"
echo "=========================================="
echo ""

# Check if dataset directory exists
if [ ! -d "$BASE_DIR/$DATASET" ]; then
    echo "Error: Dataset directory not found: $BASE_DIR/$DATASET"
    exit 1
fi

echo "Dataset: $DATASET"
echo "Base directory: $BASE_DIR/$DATASET"
echo ""

# List all training configurations
echo "Available training runs:"
ls -1 "$BASE_DIR/$DATASET/" 2>/dev/null || echo "No training runs found"
echo ""

# For each training configuration, show details
for RUN_DIR in "$BASE_DIR/$DATASET"/*; do
    if [ -d "$RUN_DIR" ]; then
        RUN_NAME=$(basename "$RUN_DIR")
        echo "=========================================="
        echo "Run: $RUN_NAME"
        echo "=========================================="
        
        # List learning rate / epoch combinations
        for LR_EP_DIR in "$RUN_DIR"/*; do
            if [ -d "$LR_EP_DIR" ]; then
                LR_EP=$(basename "$LR_EP_DIR")
                echo ""
                echo "  Configuration: $LR_EP"
                echo "  Path: $LR_EP_DIR"
                echo ""
                
                # Count classes with trained weights
                NUM_CLASSES=$(ls -1d "$LR_EP_DIR"/*/ 2>/dev/null | wc -l)
                echo "  Number of trained classes: $NUM_CLASSES"
                echo ""
                
                # Sample one class to check checkpoints
                SAMPLE_CLASS=$(ls -1d "$LR_EP_DIR"/*/ 2>/dev/null | head -n 1)
                if [ -n "$SAMPLE_CLASS" ]; then
                    CLASS_NAME=$(basename "$SAMPLE_CLASS")
                    echo "  Sample class: $CLASS_NAME"
                    
                    # Check for final weights
                    if [ -f "$SAMPLE_CLASS/pytorch_lora_weights.safetensors" ]; then
                        echo "    ✓ Final weights available"
                    else
                        echo "    ✗ Final weights NOT found"
                    fi
                    
                    # List checkpoints
                    CHECKPOINTS=$(ls -1d "$SAMPLE_CLASS"/checkpoint-*/ 2>/dev/null | xargs -n 1 basename 2>/dev/null | sort -V)
                    if [ -n "$CHECKPOINTS" ]; then
                        echo "    Available checkpoints:"
                        for CHECKPOINT in $CHECKPOINTS; do
                            STEP=$(echo $CHECKPOINT | sed 's/checkpoint-//')
                            echo "      - $STEP"
                        done
                    else
                        echo "    No intermediate checkpoints found"
                    fi
                fi
                echo ""
            fi
        done
    fi
done

echo "=========================================="
echo "Usage Instructions"
echo "=========================================="
echo ""
echo "To use a specific checkpoint in generation, edit generation_shpc.sbatch:"
echo ""
echo "  CHECKPOINT=100     # Use checkpoint-100"
echo "  CHECKPOINT=200     # Use checkpoint-200"
echo "  CHECKPOINT=None    # Use final weights (default)"
echo ""
echo "Or run directly:"
echo "  python main.py --checkpoint=100 --fewshot_seed=$SEED --dataset=$DATASET ..."
echo ""
