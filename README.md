# Temporal Link Prediction with Graph Transformers

This project implements a full pipeline for temporal link prediction on the CollegeMsg dataset using Graph Transformer models.

## Features
- Temporal graph construction from edge list
- Node and temporal feature engineering
- Graph Transformer model (PyTorch Geometric)
- Training, evaluation, and monitoring

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place `CollegeMsg.txt` in the project root (already present).
3. Run the training pipeline:
   ```bash
   python train.py
   ```

## Files
- `data_utils.py`: Data loading and preprocessing
- `features.py`: Feature engineering
- `model.py`: Graph Transformer model
- `dataset.py`: Dataset and dataloader
- `train.py`: Training and evaluation loop
- `evaluate.py`: Evaluation metrics 