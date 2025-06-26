#!/bin/bash
#SBATCH --nodes=1  #asked for 1 node
#SBATCH --ntasks=1  #asked for 1 cores
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition gpu  #this job will submit to medium partition
#SBATCH --mem=10G  #this job is asked for 1G of total memory, use 0 if you want to use entire node memory
#SBATCH --time=0-00:15:00 # 15 hours
#SBATCH --output=/home/sshriram2/mi3Testing/hazard_detection_CLIP/hpc_jobs/logs/%x_%j.qlog
#SBATCH --job-name=test1  #the job name
#SBATCH --export=ALL
whoami
echo "Current directory: $(pwd)"
echo "Starting job..."
# This job will use one python input but takes different argument each time per job array



# Activate conda environment
source /home/sshriram2/data/miniconda3/etc/profile.d/conda.sh

conda activate /home/sshriram2/.conda/envs/clip_scores

# Run script
cd "/home/sshriram2/mi3Testing/hazard_detection_CLIP/scripts/setup"
python -u segment_everything.py
