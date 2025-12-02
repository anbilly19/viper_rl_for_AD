#!/bin/bash
#SBATCH -t 3-0:01:00
#SBATCH --mem=80G
#SBATCH -J vqgan_dmc            # job name
#SBATCH --mail-type=END,FAIL,TIME_LIMIT  # send notification emails
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH -N 1                         # request slots on 1 node
##SBATCH --gpus=v100:1              # request 1 Volta V100 GPU
##SBATCH --partition=informatik-mind      # run on one of our DGX servers
#SBATCH --gres=gpu:a100:1
#SBATCH --output=junk/grid_%j_out.txt         # capture output
#SBATCH --error=junk/grid_%j_err.txt          # and error streams

module load anaconda3/latest
module load nvidia/11.4
. $ANACONDA_HOME/etc/profile.d/conda.sh

conda activate viper
export XLA_PYTHON_CLIENT_PREALLOCATE=0.8

# python scripts/train_vqgan.py -o viper_rl_data/checkpoints/dmc_vqgan -c viper_rl/configs/vqgan/dmc.yaml
# python scripts/train_videogpt.py -o viper_rl_data/checkpoints/dmc_videogpt_l16_s1 -c viper_rl/configs/videogpt/dmc.yaml
python scripts/train_dreamer.py --configs=dmc_vision videogpt_prior_rb --task=dmc_walker_walk --reward_model=dmc_clen16_fskip1 --logdir=/scratch/billimor/logdir_walker_walk

conda deactivate