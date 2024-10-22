#!/bin/bash -l
#SBATCH -o logs/2024.10.14.%A_%a_training_LoRa_replicates
#SBATCH -J pert_seq
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=150G
#SBATCH --cpus-per-task=20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --array=1-30%3
#--partition=gpu

##--array=1-15%3 to train 5 replicates for each dataset 

echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " GPUs per node:= " $SLURM_JOB_GPUS
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo " Run started at:- "
date

export SBATCH_EXPORT=ALL

host=$(hostname)
if [[ $host =~ "gamma" ]]; then
    echo "host is gamma"
    source ~/miniconda3/etc/profile.d/conda.sh
else
    echo "host is not gamma"
    source ~/.bashrc
    cd perturb_seq
fi 
conda activate perturb_seq

pwd

echo $SLURM_ARRAY_TASK_ID

data_names=("adamson" "norman" "replogle_k562_essential")
data_name=${data_names[$((SLURM_ARRAY_TASK_ID%3))]}

if [ $data_name = "replogle_k562_essential" ]; then 
    filter_perturbations=true
else
    filter_perturbations=false
fi 

##LoRa, rank=800
python runner.py --mode=train --data_name=$data_name --filter_perturbations=$filter_perturbations --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/LoRa=800/${data_name}_run_$SLURM_ARRAY_TASK_ID --use_lora=True --lora_rank=800

echo "Run completed at:- "
date
