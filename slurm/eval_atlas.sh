#!/bin/bash
# Run both Flexpert models on the ATLAS test split and print a correlation table.
# Set SEQ_WEIGHTS / THREE_D_WEIGHTS to your checkpoints (downloaded or self-trained).
#SBATCH --job-name=flexpert_eval
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=120G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export WANDB_MODE=offline

SEQ_WEIGHTS=${SEQ_WEIGHTS:-models/weights/flexpert_seq_prostt5_aa_weights.bin}
THREE_D_WEIGHTS=${THREE_D_WEIGHTS:-models/weights/flexpert_3d_saprot_weights.bin}

python3 run_inference.py --backbone prostt5 \
    --weights_path "$SEQ_WEIGHTS" \
    --fasta_path data/atlas_sequences.fasta \
    --splits_path data/atlas_splits.json --split test \
    --output_path prediction_results/seq_atlas_test.txt --batch_size 16

python3 run_inference.py --backbone saprot \
    --weights_path "$THREE_D_WEIGHTS" \
    --fasta_path data/atlas_sa_sequences.fasta \
    --splits_path data/atlas_splits.json --split test \
    --output_path prediction_results/3d_atlas_test.txt --batch_size 16

python3 evaluate_ablations.py \
    --data_path data/rmsf_atlas_data_prottransready.txt \
    --splits_path data/atlas_splits.json \
    --pred_files \
        "Flexpert-Seq:prediction_results/seq_atlas_test.txt" \
        "Flexpert-3D:prediction_results/3d_atlas_test.txt"
