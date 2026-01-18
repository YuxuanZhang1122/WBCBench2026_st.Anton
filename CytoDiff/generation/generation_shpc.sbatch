#!/bin/bash

#SBATCH --time=1-00:00:00                # Maximum runtime (1 day)
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --partition=batch_gpu          # GPU partition
#SBATCH --qos=1d                        # Quality of Service (1 day)
#SBATCH -o /home/xiaos7/projects/CytoDiff/experiments/%x_%j/slurm-%x_%j.out
#SBATCH -e /home/xiaos7/projects/CytoDiff/experiments/%x_%j/slurm-%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --mail-type=END,FAIL           # Send email on job end or failure

EXPERIMENT_NAME="${SLURM_JOB_NAME}_${SLURM_JOB_ID}"

# Create experiment directories
EXPERIMENT_DIR="/home/xiaos7/projects/CytoDiff/experiments/$EXPERIMENT_NAME"
mkdir -p "$EXPERIMENT_DIR"

source /apps/rocs/init.sh 2020.08
source /apps/rocs/2020.08/cascadelake/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh

### Helping Guide : https://bioinformatics_core.ascgitlab.helmholtz-muenchen.de/it_hpc_documentation/index.html 

conda activate cytodiff

# Redirect Hugging Face cache and model downloads to data area
export HF_HOME="/home/xiaos7/data_areas/lmr-ihb-imaging/xiaos7/experiments/cytodiff/hf_cache"
export HF_HUB_CACHE="/home/xiaos7/data_areas/lmr-ihb-imaging/xiaos7/experiments/cytodiff/hub_cache"
export TORCH_HOME="/home/xiaos7/data_areas/lmr-ihb-imaging/xiaos7/experiments/cytodiff/torch_cache"

# Create cache directories
mkdir -p "$HF_HOME"
mkdir -p "$HF_HUB_CACHE"
mkdir -p "$TORCH_HOME"

# Variables
GPU=0
N_SET_SPLIT=1
SPLIT_IDX=0

BS=10                    # Batch size for generation
NIPC=500                 # Number of images per class to generate
SD="sd2.1"               # Stable Diffusion version
GS=7.5                   # Guidance scale (higher = more adherence to prompt)

N_SHOT=15                # Must match training (15 images per class)
N_TEMPLATE=1             # Must match training

MODE="datadream"         # Use fine-tuned LoRA weights
DD_LR=1e-4               # Must match training learning rate
DD_EP=300                # Must match training epochs (you have epoch100 folder)

DATASET="custom_wbc"     # Your custom white blood cell dataset
IS_DATASETWISE=False     # Generate per-class models (not dataset-wise)
FEWSHOT_SEED="seed6"     # Must match training seed
CHECKPOINT=None          # Checkpoint to use (e.g., 100, 200, etc.) or None for final weights


echo "Starting synthetic image generation for $DATASET dataset..."
echo "Mode: $MODE | Images per class: $NIPC | Guidance scale: $GS"
echo "Training params: N_SHOT=$N_SHOT, LR=$DD_LR, EPOCHS=$DD_EP, SEED=$FEWSHOT_SEED"
if [ "$CHECKPOINT" != "None" ]; then
    echo "Using checkpoint: $CHECKPOINT"
else
    echo "Using final weights (no checkpoint specified)"
fi

# Build the Python command based on whether checkpoint is specified
PYTHON_CMD="CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --bs=$BS \
    --n_img_per_class=$NIPC \
    --sd_version=$SD \
    --mode=$MODE \
    --guidance_scale=$GS \
    --n_shot=$N_SHOT \
    --n_template=$N_TEMPLATE \
    --dataset=$DATASET \
    --n_set_split=$N_SET_SPLIT \
    --split_idx=$SPLIT_IDX \
    --fewshot_seed=$FEWSHOT_SEED \
    --datadream_lr=$DD_LR \
    --datadream_epoch=$DD_EP \
    --is_dataset_wise_model=$IS_DATASETWISE"

# Add checkpoint parameter if specified
if [ "$CHECKPOINT" != "None" ]; then
    PYTHON_CMD="$PYTHON_CMD --checkpoint=$CHECKPOINT"
fi

# Execute the command
eval $PYTHON_CMD

echo "Generation completed! Check output directory for synthetic images."
