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

source /apps/rocs/init.sh 2020.08
source /apps/rocs/2020.08/cascadelake/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh

### Helping Guide : https://bioinformatics_core.ascgitlab.helmholtz-muenchen.de/it_hpc_documentation/index.html 

conda activate cytodiff

# Setup wandb for monitoring
export WANDB_PROJECT="cytodiff-training"
export WANDB_ENTITY="shuhan-xiao"
export WANDB_MODE="online"

# Redirect Hugging Face cache and model downloads to data area (memory-heavy files)
export HF_HOME="/home/xiaos7/data_areas/lmr-ihb-imaging/xiaos7/experiments/cytodiff/hf_cache"
export HF_HUB_CACHE="/home/xiaos7/data_areas/lmr-ihb-imaging/xiaos7/experiments/cytodiff/hub_cache"
export TORCH_HOME="/home/xiaos7/data_areas/lmr-ihb-imaging/xiaos7/experiments/cytodiff/torch_cache"

# Create cache directories
mkdir -p "$HF_HOME"
mkdir -p "$HF_HUB_CACHE"
mkdir -p "$TORCH_HOME"

# Variablesl
GPU=0
SET_SPLIT=1
SPLIT_IDX=15

DATASET="custom_wbc" #"matek"
N_CLS=13  # custom_wbc has 13 classes (0-12)
FEWSHOT_SEED="seed6"  # Using seed0 to match your data structure
N_SHOT=15    # You have 15 images per class
NUM_TRAIN_EPOCH=300  # As per paper specification

# Storage location for weights and outputs (memory-heavy files)
WEIGHTS_OUTPUT_DIR="/home/xiaos7/data_areas/lmr-ihb-imaging/xiaos7/experiments/cytodiff"

# CLASS_IDXS - Train all classes (0 to N_CLS-1)
CLASS_IDXS=($(seq 0 $((N_CLS - 1))))
echo "CLASS_IDXS: ${CLASS_IDXS[@]}"
echo "Training classes: 0 to $((N_CLS - 1))"

# Create weights output directory
mkdir -p "$WEIGHTS_OUTPUT_DIR"

# Train one model per class (per-class fine-tuning as in the paper)
for CLASS_IDX in "${CLASS_IDXS[@]}"; do
    echo "Training class $CLASS_IDX..."
    CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --dataset=$DATASET \
    --n_template=1 \
    --fewshot_seed=$FEWSHOT_SEED \
    --train_batch_size=8 \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-4 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=100 \
    --num_train_epochs=$NUM_TRAIN_EPOCH \
    --report_to="wandb" \
    --train_text_encoder=True \
    --is_tqdm=True \
    --output_dir="$WEIGHTS_OUTPUT_DIR" \
    --n_shot=$N_SHOT \
    --target_class_idx=$CLASS_IDX \
    --resume_from_checkpoint=None \
    --validation_prompt="$(python -c "from util_data import SUBSET_NAMES, PROMPTS_BY_CLASS, FOLDER_TO_PROMPT_KEY; class_name = SUBSET_NAMES['custom_wbc'][$CLASS_IDX]; prompt_key = FOLDER_TO_PROMPT_KEY.get(class_name, class_name); print(PROMPTS_BY_CLASS.get(prompt_key, 'white blood cell under microscope'))")" \
    --num_validation_images=4 \
    --validation_epochs=25 \
    --checkpointing_steps=100
    
    echo "Completed training for class $CLASS_IDX"
done
