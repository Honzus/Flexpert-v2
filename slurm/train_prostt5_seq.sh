#!/bin/bash
# Train Flexpert-Seq (ProstT5, AA-only) on ATLAS.
# Adjust --partition / --gres / --time and the environment activation for your cluster.
# Submit from the repository root; logs go to outputs/logs/, the final weights to
# outputs/checkpoints/<run_name>_<timestamp>/final_model/pytorch_model.bin.
#SBATCH --job-name=flexpert_seq
#SBATCH --output=outputs/logs/slurm-%x-%j.out
#SBATCH --error=outputs/logs/slurm-%x-%j.err
# --partition: change to your cluster's GPU partition.
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=120G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate          # or: conda activate flexpert
export WANDB_MODE=offline

python3 train.py \
    --run_name atlasFlexpertProstT5 \
    --batch_size 8 \
    --data_path data/rmsf_atlas_data_prottransready.txt \
    --fasta_path data/atlas_sequences.fasta \
    --splits_path data/atlas_splits.json
