# WBC2026

Image classification pipeline with advanced feature extraction and class balancing.

## Pipeline

1. **Data Augmentation** - Standard image transformations
2. **Feature Extraction** - ResNet50 + DinoBloom-B
3. **Training** - Oversampling + Weighted Cross-Entropy

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python train.py

# Generate predictions
python generate_predictions.py
```

## Structure

- `src/` - Core modules (models, data loaders, augmentation)
- `configs/` - Training configurations
- `experiments/` - Experiment outputs
- `raw_data/` - Dataset (excluded from git)
