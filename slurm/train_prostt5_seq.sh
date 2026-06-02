#!/bin/bash
# Train Flexpert-Seq (ProstT5, AA-only) on ATLAS.
# Adjust --partition / --gres / --time and the environment activation for your cluster.
#SBATCH --job-name=flexpert_seq
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

python3 train.py \
    --run_name atlasFlexpertProstT5 \
    --batch_size 8 \
    --data_path data/rmsf_atlas_data_prottransready.txt \
    --fasta_path data/atlas_sequences.fasta \
    --splits_path data/atlas_splits.json
