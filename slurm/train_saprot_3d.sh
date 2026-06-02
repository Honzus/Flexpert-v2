#!/bin/bash
# Train Flexpert-3D (SAProt) on ATLAS.
# Adjust --partition / --gres / --time and the environment activation for your cluster.
#SBATCH --job-name=flexpert_3d
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=120G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate          # or: conda activate flexpert
export WANDB_MODE=offline

python3 train_3d.py \
    --run_name atlasFlexpertSaprot \
    --batch_size 8 \
    --data_path data/rmsf_atlas_data_prottransready.txt \
    --fasta_path data/atlas_sa_sequences.fasta \
    --splits_path data/atlas_splits.json
