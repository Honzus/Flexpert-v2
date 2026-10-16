#!/bin/bash
# End-to-end check of a Flexpert-v2 checkout on one GPU:
#   1. released weights present (downloads all 9 into weights/ if missing),
#   2. smoke training of Flexpert-Seq (ProstT5, ProtT5) and Flexpert-3D (4 steps each; final_model
#      must load),
#   3. inference with all 9 released weights on the 3 test splits, compared to the reference
#      metrics with `python3 -m evaluation.check_reproduction` (±0.005),
#   4. custom proteins (no splits file): the 6 Flexpert-Seq weights on
#      data/custom_dataset/fasta_dataset.fasta, and, if Foldseek is found, the 3 Flexpert-3D
#      weights on an SA FASTA built from data/PDBs (otherwise SKIP). Each output must have one
#      entry per input record, with the same names.
# Submit from the repository root:  sbatch slurm/verify_core.sh
# Optional: REFERENCE_DIR / REFERENCE_PROTT5_DIR = dirs of older prediction files; also reports
# per-residue differences.
#SBATCH --job-name=verify_core
#SBATCH --output=outputs/logs/slurm-%x-%j.out
# --partition: change to your cluster's GPU partition.
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=120G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1

cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
source .venv/bin/activate
export WANDB_MODE=offline

STATUS=()
step() {  # step <name> <command...>: run, record PASS/FAIL, keep going
    local name=$1; shift
    echo; echo "=== [$name] $*"
    if "$@"; then STATUS+=("PASS  $name"); else STATUS+=("FAIL  $name"); fi
}
same_ids() {  # same_ids <fasta> <predictions>: same record names, in the same order
    diff <(grep '^>' "$1" | sed 's/^>//; s/[[:space:]].*//') <(grep '^>' "$2" | sed 's/^>//')
}

# 1. weights
WEIGHTS=(flexpert_seq_prostt5_aa_weights.bin flexpert_seq_prostt5_aa_mdcath_weights.bin
         flexpert_seq_prostt5_aa_combined_weights.bin flexpert_3d_saprot_weights.bin
         flexpert_3d_saprot_mdcath_weights.bin flexpert_3d_saprot_combined_weights.bin
         flexpert_seq_prott5_weights.bin flexpert_seq_prott5_mdcath_weights.bin
         flexpert_seq_prott5_combined_weights.bin)
have_all() { for w in "${WEIGHTS[@]}"; do [ -s "weights/$w" ] || return 1; done; }
have_all || step "download weights" python3 download_flexpert_weights.py --models all --dataset all
step "weights present" have_all

# 2. smoke training; the saved final_model must load in run_inference.py
step "train Flexpert-Seq (smoke)" python3 train.py --run_name verify_smoke_seq --max_steps 4 --batch_size 2
step "train Flexpert-Seq ProtT5 (smoke)" python3 train.py --backbone prott5 --run_name verify_smoke_prott5 \
    --max_steps 4 --batch_size 2
step "train Flexpert-3D (smoke)" python3 train_3d.py --run_name verify_smoke_3d --max_steps 4 --batch_size 2 \
    --fasta_path data/atlas_sa_sequences.fasta
for run in seq:prostt5:atlas_sequences prott5:prott5:atlas_sequences 3d:saprot:atlas_sa_sequences; do
    IFS=: read -r tag backbone fasta <<< "$run"
    ckpt=$(ls -td outputs/checkpoints/verify_smoke_${tag}_*/final_model 2>/dev/null | head -1)
    step "infer with smoke $tag checkpoint" python3 run_inference.py --backbone "$backbone" \
        --weights_path "${ckpt:-missing}/pytorch_model.bin" --fasta_path "data/${fasta}.fasta" \
        --splits_path data/atlas_splits.json --split validation \
        --output_path "outputs/predictions/verify_smoke_${tag}_atlas_validation.txt"
done

# 3. released weights x test splits
declare -A FASTA_PREFIX=([atlas]=atlas [mdcath]=mdCATH [combined]=combined)
for train in atlas mdcath combined; do
    sfx=$([ "$train" = atlas ] && echo "" || echo "_$train")
    for ds in atlas mdcath combined; do
        p=${FASTA_PREFIX[$ds]}
        step "infer seq$sfx on $ds" python3 run_inference.py --backbone prostt5 \
            --weights_path "weights/flexpert_seq_prostt5_aa${sfx}_weights.bin" \
            --fasta_path "data/${p}_sequences.fasta" --splits_path "data/${p}_splits.json" --split test \
            --output_path "outputs/predictions/flexpert_seq${sfx}_${ds}_test.txt" --batch_size 16
        step "infer 3d$sfx on $ds" python3 run_inference.py --backbone saprot \
            --weights_path "weights/flexpert_3d_saprot${sfx}_weights.bin" \
            --fasta_path "data/${p}_sa_sequences.fasta" --splits_path "data/${p}_splits.json" --split test \
            --output_path "outputs/predictions/flexpert_3d${sfx}_${ds}_test.txt" --batch_size 16
        step "infer seq_prott5$sfx on $ds" python3 run_inference.py --backbone prott5 \
            --weights_path "weights/flexpert_seq_prott5${sfx}_weights.bin" \
            --fasta_path "data/${p}_sequences.fasta" --splits_path "data/${p}_splits.json" --split test \
            --output_path "outputs/predictions/flexpert_seq_prott5${sfx}_${ds}_test.txt" --batch_size 16
    done
done
REF_ARGS=()
[ -n "${REFERENCE_DIR:-}" ] && REF_ARGS+=(--reference_dir "$REFERENCE_DIR")
[ -n "${REFERENCE_PROTT5_DIR:-}" ] && REF_ARGS+=(--reference_dir "$REFERENCE_PROTT5_DIR")
step "reproduce reference metrics" python3 -m evaluation.check_reproduction --require_all "${REF_ARGS[@]}" \
    --json_out outputs/predictions/check_reproduction.json

# 4. custom proteins, no splits file (default output names: <model>_<fasta stem>.txt)
CUSTOM=data/custom_dataset/fasta_dataset.fasta
for w in flexpert_seq_prostt5_aa flexpert_seq_prostt5_aa_mdcath flexpert_seq_prostt5_aa_combined \
         flexpert_seq_prott5 flexpert_seq_prott5_mdcath flexpert_seq_prott5_combined; do
    backbone=$([[ $w == *prott5* ]] && echo prott5 || echo prostt5)
    model=${w/_prostt5_aa/}
    step "custom FASTA: $w" python3 run_inference.py --backbone "$backbone" \
        --weights_path "weights/${w}_weights.bin" --fasta_path "$CUSTOM"
    step "custom FASTA output: $w" same_ids "$CUSTOM" "outputs/predictions/${model}_fasta_dataset.txt"
done
FOLDSEEK_BIN=$(python3 -c "from data.scripts.build_sa_fasta import resolve_foldseek; print(resolve_foldseek(None) or '')")
if [ -n "$FOLDSEEK_BIN" ]; then
    CUSTOM_SA=outputs/custom_sa.fasta
    step "build SA FASTA from data/PDBs" python3 -m data.scripts.build_sa_fasta --pdb_dir data/PDBs \
        --out "$CUSTOM_SA" --force
    for w in flexpert_3d_saprot flexpert_3d_saprot_mdcath flexpert_3d_saprot_combined; do
        model=${w/_saprot/}
        step "custom SA FASTA: $w" python3 run_inference.py --backbone saprot \
            --weights_path "weights/${w}_weights.bin" --fasta_path "$CUSTOM_SA"
        step "custom SA FASTA output: $w" same_ids "$CUSTOM_SA" "outputs/predictions/${model}_custom_sa.txt"
    done
else
    STATUS+=("SKIP  custom Flexpert-3D (no foldseek)")
fi

echo; echo "=========== verify_core summary ==========="
printf '%s\n' "${STATUS[@]}"
! printf '%s\n' "${STATUS[@]}" | grep -q '^FAIL'
