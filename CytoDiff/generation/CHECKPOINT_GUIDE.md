# Checkpoint Selection Guide for CytoDiff Generation

## Overview
You can now select specific training checkpoints for generating synthetic images instead of always using the final weights. This is useful when:
- You want to compare image quality at different training stages
- Early stopping might produce better results for some classes
- You're running multiple experiments with different hyperparameters

## Quick Start

### 1. List Available Checkpoints
```bash
cd /home/xiaos7/projects/CytoDiff/generation
./list_checkpoints.sh
```

This will show:
- All training runs (different seeds, shots, templates)
- Learning rate and epoch configurations
- Available checkpoints for each experiment
- Whether final weights exist

### 2. Select a Checkpoint

**Option A: Edit the SBATCH script (recommended for cluster jobs)**
```bash
# Edit generation_shpc.sbatch
CHECKPOINT=100      # Use checkpoint at step 100
CHECKPOINT=200      # Use checkpoint at step 200
CHECKPOINT=None     # Use final weights (default)
```

**Option B: Pass directly to Python (for testing)**
```bash
python main.py \
    --checkpoint=100 \
    --fewshot_seed=seed0 \
    --dataset=custom_wbc \
    --n_shot=15 \
    --datadream_lr=1e-4 \
    --datadream_epoch=100 \
    ...
```

## Understanding Checkpoint Structure

Your training outputs are organized as:
```
/home/xiaos7/data_areas/lmr-ihb-imaging/xiaos7/experiments/cytodiff/
└── custom_wbc/
    └── shot15_seed0_tpl1/
        └── lr0.0001_epoch100/
            ├── basophil/
            │   ├── checkpoint-100/
            │   │   └── pytorch_lora_weights.safetensors
            │   └── pytorch_lora_weights.safetensors  (final weights)
            ├── blast/
            ├── eosinophil/
            └── ... (all 13 classes)
```

## Checkpoint Types

### Final Weights (Default)
- File: `pytorch_lora_weights.safetensors` in the class folder
- Created after full training completes
- Use when: Training converged properly

### Intermediate Checkpoints
- Directory: `checkpoint-{step}/` (e.g., `checkpoint-100/`)
- Saved at intervals during training (controlled by `--checkpointing_steps`)
- Use when: You want to evaluate quality at different training stages

## Multiple Experiments Workflow

### Scenario 1: Different Seeds
```bash
# Training
FEWSHOT_SEED="seed0"  # Training run 1
FEWSHOT_SEED="seed6"  # Training run 2

# Generation - Make sure to match!
FEWSHOT_SEED="seed0"  # Must match training seed
CHECKPOINT=100        # Optional: select checkpoint
```

### Scenario 2: Different Training Lengths
```bash
# You might have:
# - lr0.0001_epoch100  (100 epochs, checkpoints: 100)
# - lr0.0001_epoch200  (200 epochs, checkpoints: 100, 200)
# - lr0.0001_epoch300  (300 epochs, checkpoints: 100, 200, 300)

# In generation_shpc.sbatch:
DD_EP=300            # Must match the epoch folder
CHECKPOINT=200       # Can select any available checkpoint
```

### Scenario 3: Comparing Checkpoints
```bash
# Generate with checkpoint-100
sbatch --job-name=gen_cp100 generation_shpc.sbatch  # Set CHECKPOINT=100

# Generate with checkpoint-200
sbatch --job-name=gen_cp200 generation_shpc.sbatch  # Set CHECKPOINT=200

# Generate with final weights
sbatch --job-name=gen_final generation_shpc.sbatch  # Set CHECKPOINT=None
```

## Important Parameters to Match

When generating images, these parameters **must match** your training:
- `FEWSHOT_SEED`: Must match the seed used during training
- `N_SHOT`: Must match the number of shots (15 for your case)
- `DD_LR`: Must match the learning rate (1e-4)
- `DD_EP`: Must match the epoch folder (100, 200, 300, etc.)

These parameters can be **different** from training:
- `CHECKPOINT`: Select any checkpoint or None for final weights
- `NIPC`: Number of images to generate (doesn't affect loading)
- `GS`: Guidance scale (generation parameter only)
- `BS`: Batch size (generation parameter only)

## Examples

### Example 1: Use Final Weights
```bash
# In generation_shpc.sbatch:
FEWSHOT_SEED="seed0"
DD_EP=100
CHECKPOINT=None
```

### Example 2: Use Intermediate Checkpoint
```bash
# In generation_shpc.sbatch:
FEWSHOT_SEED="seed0"
DD_EP=100
CHECKPOINT=100
```

### Example 3: Check if Checkpoint Exists
```bash
# The script will automatically fall back to final weights if checkpoint doesn't exist
FEWSHOT_SEED="seed0"
DD_EP=100
CHECKPOINT=999  # If checkpoint-999 doesn't exist, uses final weights
```

## Troubleshooting

### "Checkpoint not found" Warning
- The script will automatically use final weights
- Check available checkpoints with `./list_checkpoints.sh`
- Make sure `DD_EP` matches the epoch folder

### "Final weights not found"
- Training might still be in progress
- Check the training logs to see if it completed
- Look for intermediate checkpoints instead

### Mismatched Parameters
If you get errors loading weights:
```
KeyError: 'shot15_seed0_tpl1'  → Check FEWSHOT_SEED matches
FileNotFoundError               → Check DD_EP matches epoch folder
```

## Advanced: Checkpoint Selection Strategy

For best results:
1. **Start with final weights** - Usually the most converged
2. **Try different checkpoints** if final weights:
   - Produce artifacts or unrealistic images
   - Show signs of overfitting
   - Have quality degradation
3. **Compare multiple checkpoints** - Generate small batches (NIPC=10) with different checkpoints
4. **Select best checkpoint** - Use the one with best visual quality for full generation

## Monitoring Training Progress

To check training checkpoints while running:
```bash
# Check what's being saved
ls -lh /home/xiaos7/data_areas/lmr-ihb-imaging/xiaos7/experiments/cytodiff/custom_wbc/*/lr*/basophil/

# Check latest checkpoint
ls -lt /home/xiaos7/data_areas/lmr-ihb-imaging/xiaos7/experiments/cytodiff/custom_wbc/*/lr*/basophil/checkpoint-*/
```
