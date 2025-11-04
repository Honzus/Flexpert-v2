#!/bin/bash
#
# Part 1: SBATCH Directives
#
#SBATCH --job-name=ProT5_sweep2      # A descriptive name for your job
#SBATCH --output=slurm-%j.out     # Standard output file (%j is replaced with job ID)
#SBATCH --error=slurm-%j.err  
#SBATCH --partition=gpu 
#SBATCH --nodelist=amd-2         # Specify the resource partition (queue)
#SBATCH --ntasks=1                # Total number of tasks (processes)
#SBATCH --cpus-per-task=2         # Number of CPU cores per task
#SBATCH --mem=16G                  # Amount of memory (e.g., 1GB)
#SBATCH --time=96:00:00           # Maximum job runtime (HH:MM:SS)
#SBATCH --gres=gpu:1

#
# Part 2: Job Commands (The actual work)
#

# Change to the directory where the script was submitted
cd $SLURM_SUBMIT_DIR
echo "Activating virtual environment: venv_torch"
source venv_torch/bin/activate
export WANDB_MODE="offline"

# Your main command(s) to run
echo "Starting job on $(hostname)"
echo "Running my Python script..."

python3 train.py --run_name testrun-Seq --adaptor_architecture no-adaptor --batch_size 2 --run_name ProT5FullBS2R16A16 --lora_r 4 --lora_alpha 4
python3 train.py --run_name testrun-Seq --adaptor_architecture no-adaptor --batch_size 2 --run_name ProT5FullBS2R32A32 --lora_r 4 --lora_alpha 32
python3 train.py --run_name testrun-Seq --adaptor_architecture no-adaptor --batch_size 2 --run_name ProT5FullBS2R32A32 --lora_r 8 --lora_alpha 8
python3 train.py --run_name testrun-Seq --adaptor_architecture no-adaptor --batch_size 2 --run_name ProT5FullBS2R16A16 --lora_r 16 --lora_alpha 16
python3 train.py --run_name testrun-Seq --adaptor_architecture no-adaptor --batch_size 2 --run_name ProT5FullBS2R32A32 --lora_r 32 --lora_alpha 32
echo "Job finished at $(date)"