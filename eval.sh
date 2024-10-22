#!/bin/bash
#SBATCH -o logs/2024.10.16.
#SBATCH -J pert_seq
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=150G
#SBATCH --cpus-per-task=20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

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

null_save_dir="save/null/"

##==============================================
##train models and then evaluate
##==============================================
# # #train from pretrained models/scgpt-pretrained/scGPT_human
# for n in {1..10}
#     do 
#     python runner.py --mode=train --data_name=adamson --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/default_config_baseline/adamson_run_$n
#     python runner.py --mode=train --data_name=norman --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/default_config_baseline/norman_run_$n
#     ##replogle dataset is special because we need to filter the dataloaders to remove perturbations not part of dataset's gene set 
#     python runner.py --mode=train --data_name=replogle_k562_essential --filter_perturbations=True --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/default_config_baseline/replogle_k562_essential_run_$n
#     done 

# ##test out human CP on perturb seq test 
# python runner.py --mode=test --data_name=adamson --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/eval_human_cp_foundation/adamson/
# python runner.py --mode=test --data_name=norman --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/eval_human_cp_foundation/norman/
# python runner.py --mode=test --data_name=replogle_k562_essential --filter_perturbations=True --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/eval_human_cp_foundation/replogle_k562_essential/ 

# ##random shuffle pert index control 
# for n in {1..5}
#     do 
#     python runner.py  --mode=train --random_shuffle=True --data_name=adamson --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/shuffled_pert_token/adamson_run_$n
#     python runner.py  --mode=train --random_shuffle=True --data_name=norman --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/shuffled_pert_token/norman_run_$n
#     python runner.py  --mode=train --random_shuffle=True --data_name=replogle_k562_essential --filter_perturbations=True --load_model=models/scgpt-pretrained/scGPT_human  --save_dir=save/shuffled_pert_token/replogle_k562_essential_run_$n
#     done

# # ##pretraining control (don't load any weights prior to fine tuning)
# for n in {1..10}
#     do
#     python runner.py --mode=train --data_name=adamson --load_model=None --pretrain_control=True --save_dir=save/no_pretraining/adamson_run_$n/
#     python runner.py --mode=train --data_name=norman --load_model=None --pretrain_control=True --save_dir=save/no_pretraining/norman_run_$n/
#     python runner.py --mode=train --data_name=replogle_k562_essential --filter_perturbations=True  --load_model=None --pretrain_control=True --save_dir=save/no_pretraining/replogle_k562_essential_run_$n/ 
    # done 

# # ##transformer encoder control (don't load any transformer encoder weights prior to fine tuning)
# for n in {1..10}
#     do
#     python runner.py --mode=train --data_name=adamson --load_model=models/scgpt-pretrained/scGPT_human --transformer_encoder_control=True --save_dir=save/transformer_encoder_control/adamson_run_$n/
#     python runner.py --mode=train --data_name=norman --load_model=models/scgpt-pretrained/scGPT_human --transformer_encoder_control=True --save_dir=save/transformer_encoder_control/norman_run_$n/
#     python runner.py --mode=train --data_name=replogle_k562_essential --filter_perturbations=True --load_model=models/scgpt-pretrained/scGPT_human --transformer_encoder_control=True --save_dir=save/transformer_encoder_control/replogle_k562_essential_run_$n/ 
#     done 

# # #attention control (don't load any pretrained self-attention weights prior to fine tuning) 
# for n in {1..10}
#     do
#     python runner.py --mode=train --data_name=adamson --load_model=models/scgpt-pretrained/scGPT_human --attention_control=True --save_dir=save/no_pretrained_attention/adamson_run_$n/
#     python runner.py --mode=train --data_name=norman --load_model=models/scgpt-pretrained/scGPT_human --attention_control=True --save_dir=save/no_pretrained_attention/norman_run_$n/
#     python runner.py --mode=train --data_name=replogle_k562_essential --filter_perturbations=True --load_model=models/scgpt-pretrained/scGPT_human --attention_control=True --save_dir=save/no_pretrained_attention/replogle_run_$n/ 
#     done

# # ##input encoder control (don't load any input encoder weights prior to fine tuning)
# for n in {1..10}
#     do
#     python runner.py --mode=train --data_name=adamson --load_model=models/scgpt-pretrained/scGPT_human --input_encoder_control=True --save_dir=save/input_encoder_control/adamson_run_$n/
#     python runner.py --mode=train --data_name=norman --load_model=models/scgpt-pretrained/scGPT_human --input_encoder_control=True --save_dir=save/input_encoder_control/norman_run_$n/
#     python runner.py --mode=train --data_name=replogle_k562_essential --filter_perturbations=True --load_model=models/scgpt-pretrained/scGPT_human --input_encoder_control=True --save_dir=save/input_encoder_control/replogle_k562_essential_run_$n/ 
    # done 

# ##use LoRa for finetuning 
# lora_ranks=(8 80 800)
# for lora_rank in "${lora_ranks[@]}"
#     do 
#     python runner.py --mode=train --data_name=adamson --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/lora/adamson/rank=$lora_rank/ --use_lora=True --lora_rank=$lora_rank 
#     python runner.py --mode=train --data_name=norman --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/lora/norman/rank=$lora_rank/ --use_lora=True --lora_rank=$lora_rank 
#     python runner.py --mode=train --data_name=replogle_k562_essential --filter_perturbations=True --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/lora/replogle_k562_essential/rank=$lora_rank/ --use_lora=True --lora_rank=$lora_rank 
#     done

# #train simple affine model
# #will use default config, which has variables that won't be used or affect anything, but are kept for ease and sake of consistency with scGPT variables - requires less changes to the core code base
# for n in {1..10}
#     do
#     python runner.py --mode=train --model_type=simple_affine --fixed_seed=False --data_name=adamson --load_model=None --pretrain_control=True --save_dir=save/simple_affine/adamson_run_$n/  
#     python runner.py --mode=train --model_type=simple_affine --fixed_seed=False --data_name=norman --load_model=None --pretrain_control=True --save_dir=save/simple_affine/norman_run_$n/  
#     python runner.py --mode=train --model_type=simple_affine --fixed_seed=False --data_name=replogle_k562_essential --filter_perturbations=True --load_model=None --pretrain_control=True --save_dir=save/simple_affine/replogle_k562_essential_run_$n/  
#     done

# ##simple affine 
# python runner.py --mode=analysis --data_name=adamson --model_type=simple_affine --load_model=save/simple_affine/adamson_run_6 --save_dir=save/null
# python runner.py --mode=analysis --data_name=norman --model_type=simple_affine --load_model=save/simple_affine/norman_run_8 --save_dir=save/null
# python runner.py --mode=analysis --data_name=replogle_k562_essential --filter_perturbations=True --model_type=simple_affine --load_model=save/simple_affine/replogle_k562_essential_run_8 --save_dir=save/null

# ##perturBench models 
# for n in {1..10}
#     do
#     python runner.py --mode=benchmark --data_name=adamson --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/perturbench/adamson_run_$n/
#     python runner.py --mode=benchmark --data_name=norman --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/perturbench/norman_run_$n/
#     python runner.py --mode=benchmark --data_name=replogle_k562_essential --filter_perturbations=True --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/perturbench/replogle_k562_essential_run_$n/
#     done

# #plot boxplots over DE genes using best models for each dataset and calculate Wasserstein 
# #baseline and smart mean 
# model_mean_types=("mean_control+perturbed" "mean_control" "mean_perturbed" "smart_mean_control+perturbed" "smart_mean_control" "smart_mean_perturbed")
# for model_mean_type in "${model_mean_types[@]}"
#     do 
#     python runner.py --mode=analysis --data_name=adamson --model_type=$model_mean_type --save_dir=save/null
#     python runner.py --mode=analysis --data_name=norman --model_type=$model_mean_type --save_dir=save/null
#     python runner.py --mode=analysis --data_name=replogle_k562_essential --filter_perturbations=True --model_type=$model_mean_type --save_dir=save/null
#     done
# ##scGPT front running models
# python runner.py --mode=analysis --data_name=adamson --load_model=save/default_config_baseline/adamson_run_3 --save_dir=save/null
# python runner.py --mode=analysis --data_name=norman --load_model=save/default_config_baseline/norman_run_5 --save_dir=save/null
# python runner.py --mode=analysis --data_name=replogle_k562_essential --filter_perturbations=True --load_model=save/default_config_baseline/replogle_k562_essential_run_4 --save_dir=save/null

# python analysis.py

echo "Run completed at:- "
date