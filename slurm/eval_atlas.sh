#!/bin/bash
# Run both Flexpert models on the ATLAS test split and print a correlation table.
# Set SEQ_WEIGHTS / THREE_D_WEIGHTS to your checkpoints (default: the released weights fetched by
# `python3 download_flexpert_weights.py` into weights/; or e.g.
# outputs/checkpoints/<run>_<ts>/final_model/pytorch_model.bin for your own).
# Submit from the repository root (logs go to outputs/logs/).
#SBATCH --job-name=flexpert_eval
#SBATCH --output=outputs/logs/slurm-%x-%j.out
#SBATCH --error=outputs/logs/slurm-%x-%j.err
# --partition: change to your cluster's GPU partition.
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=120G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export WANDB_MODE=offline

SEQ_WEIGHTS=${SEQ_WEIGHTS:-weights/flexpert_seq_prostt5_aa_weights.bin}
THREE_D_WEIGHTS=${THREE_D_WEIGHTS:-weights/flexpert_3d_saprot_weights.bin}

python3 run_inference.py --backbone prostt5 \
    --weights_path "$SEQ_WEIGHTS" \
    --fasta_path data/atlas_sequences.fasta \
    --splits_path data/atlas_splits.json --split test \
    --output_path outputs/predictions/flexpert_seq_atlas_test.txt --batch_size 16

python3 run_inference.py --backbone saprot \
    --weights_path "$THREE_D_WEIGHTS" \
    --fasta_path data/atlas_sa_sequences.fasta \
    --splits_path data/atlas_splits.json --split test \
    --output_path outputs/predictions/flexpert_3d_atlas_test.txt --batch_size 16

python3 -m evaluation.evaluate_ablations \
    --data_path data/rmsf_atlas_data_prottransready.txt \
    --splits_path data/atlas_splits.json \
    --pred_files \
        "Flexpert-Seq:outputs/predictions/flexpert_seq_atlas_test.txt" \
        "Flexpert-3D:outputs/predictions/flexpert_3d_atlas_test.txt" \
        "Flexpert-v1-Seq:data/atlas/precomputed_flexibility_profiles/flexpert_v1_prott5_seq_atlas_test.txt" \
        "Flexpert-v1-3D:data/atlas/precomputed_flexibility_profiles/flexpert_v1_prott5_3d_atlas_test.txt"
