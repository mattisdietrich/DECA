#!/usr/bin/env bash

#SBATCH --job-name=deca
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=slurm_logs/std_out/%j.log
#SBATCH --error=slurm_logs/std_err/%j.log
#SBATCH --gres=gpu:1

cd /home/dietrich/Testing/DECA/DECA

# setup conda
_conda=${HOME}/miniconda3
source ${_conda}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-deca}

echo "HERE: $(hostname) | WD: $(pwd) | python: $(which python)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID} | SLURM_JOB_NAME: ${SLURM_JOB_NAME} | SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK} | SLURM_MEM_PER_CPU: ${SLURM_MEM_PER_CPU}"
echo "ARGS:"
echo $@

#python main_train.py --cfg configs/wo_shape/deca_pretrain.yml
python main_train.py --cfg configs/wo_shape/deca_detail.yml