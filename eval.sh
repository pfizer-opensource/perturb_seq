#!/bin/bash
#SBATCH -o logs/2024.12.12.
#SBATCH -J pert_seq
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=150G
#SBATCH --cpus-per-task=20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#--partition=gpu or defq

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

##==============================================
##train models and then evaluate
##==============================================

python gears_runner.py

##test out human CP on perturb seq test 
python runner.py --mode=test --data_name=adamson --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/eval_human_cp_foundation/adamson/
python runner.py --mode=test --data_name=norman --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/eval_human_cp_foundation/norman/
python runner.py --mode=test --data_name=replogle_k562_essential --filter_perturbations=True --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/eval_human_cp_foundation/replogle_k562_essential/ 
python runner.py --mode=test --data_name=adam_corrected --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/eval_human_cp_foundation/adam_corrected/
python runner.py --mode=test --data_name=adam_corrected_upr --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/eval_human_cp_foundation/adam_corrected_upr/

# # #train scGPT from pretrained models/scgpt-pretrained/scGPT_human
for n in {1..10}
    do 
    python runner.py --mode=train --data_name=adamson --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/default_config_baseline/adamson_run_$n
    python runner.py --mode=train --data_name=norman --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/default_config_baseline/norman_run_$n
    ##replogle dataset is special because we need to filter the dataloaders to remove perturbations not part of dataset's gene set 
    python runner.py --mode=train --data_name=replogle_k562_essential --filter_perturbations=True --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/default_config_baseline/replogle_k562_essential_run_$n
    python runner.py --mode=train --data_name=adam_corrected --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/default_config_baseline/adam_corrected_run_$n
    python runner.py --mode=train --data_name=adam_corrected_upr --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/default_config_baseline/adam_corrected_upr_run_$n
    done 

# #train simple affine model (no pretraining)
# #will use default config, which has variables that won't be used or affect anything, but are kept for ease and sake of consistency with scGPT variables - requires less changes to the core code base
for n in {1..10}
    do
    python runner.py --mode=train --model_type=simple_affine --fixed_seed=False --data_name=adamson --load_model=None --pretrain_control=True --save_dir=save/simple_affine/adamson_run_$n/  
    python runner.py --mode=train --model_type=simple_affine --fixed_seed=False --data_name=norman --load_model=None --pretrain_control=True --save_dir=save/simple_affine/norman_run_$n/  
    python runner.py --mode=train --model_type=simple_affine --fixed_seed=False --data_name=replogle_k562_essential --filter_perturbations=True --load_model=None --pretrain_control=True --save_dir=save/simple_affine/replogle_k562_essential_run_$n/  
    python runner.py --mode=train --model_type=simple_affine --fixed_seed=False --data_name=adam_corrected --load_model=None --pretrain_control=True --save_dir=save/simple_affine/adam_corrected_run_$n/  
    python runner.py --mode=train --model_type=simple_affine --fixed_seed=False --data_name=adam_corrected_upr --load_model=None --pretrain_control=True --save_dir=save/simple_affine/adam_corrected_upr_run_$n/  
    done

# ##perturBench models 
for n in {1..10}
    do
    python runner.py --mode=benchmark --data_name=adamson --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/perturbench/adamson_run_$n/
    python runner.py --mode=benchmark --data_name=norman --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/perturbench/norman_run_$n/
    python runner.py --mode=benchmark --data_name=replogle_k562_essential --filter_perturbations=True --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/perturbench/replogle_k562_essential_run_$n/
    python runner.py --mode=benchmark --data_name=adam_corrected --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/perturbench/adam_corrected_run_$n/
    python runner.py --mode=benchmark --data_name=adam_corrected_upr --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/perturbench/adam_corrected_upr_run_$n/
    done

# ##train scGPT pretraining control (don't load any weights prior to fine tuning)
for n in {1..10}
    do
    python runner.py --mode=train --data_name=adamson --load_model=None --pretrain_control=True --save_dir=save/no_pretraining/adamson_run_$n/
    python runner.py --mode=train --data_name=norman --load_model=None --pretrain_control=True --save_dir=save/no_pretraining/norman_run_$n/
    python runner.py --mode=train --data_name=replogle_k562_essential --filter_perturbations=True  --load_model=None --pretrain_control=True --save_dir=save/no_pretraining/replogle_k562_essential_run_$n/ 
    python runner.py --mode=train --data_name=adam_corrected --load_model=None --pretrain_control=True --save_dir=save/no_pretraining/adam_corrected_run_$n/
    python runner.py --mode=train --data_name=adam_corrected_upr --load_model=None --pretrain_control=True --save_dir=save/no_pretraining/adam_corrected_upr_run_$n/
    done 

# ##input encoder control (don't load any input encoder weights prior to fine tuning)
for n in {1..10}
    do
    python runner.py --mode=train --data_name=adamson --load_model=models/scgpt-pretrained/scGPT_human --input_encoder_control=True --save_dir=save/input_encoder_control/adamson_run_$n/
    python runner.py --mode=train --data_name=norman --load_model=models/scgpt-pretrained/scGPT_human --input_encoder_control=True --save_dir=save/input_encoder_control/norman_run_$n/
    python runner.py --mode=train --data_name=replogle_k562_essential --filter_perturbations=True --load_model=models/scgpt-pretrained/scGPT_human --input_encoder_control=True --save_dir=save/input_encoder_control/replogle_k562_essential_run_$n/ 
    python runner.py --mode=train --data_name=adam_corrected --load_model=models/scgpt-pretrained/scGPT_human --input_encoder_control=True --save_dir=save/input_encoder_control/adam_corrected_run_$n/
    python runner.py --mode=train --data_name=adam_corrected_upr --load_model=models/scgpt-pretrained/scGPT_human --input_encoder_control=True --save_dir=save/input_encoder_control/adam_corrected_upr_run_$n/
    done 

# ##transformer encoder control (don't load any transformer encoder weights prior to fine tuning)
for n in {1..10}
    do
    python runner.py --mode=train --data_name=adamson --load_model=models/scgpt-pretrained/scGPT_human --transformer_encoder_control=True --save_dir=save/transformer_encoder_control/adamson_run_$n/
    python runner.py --mode=train --data_name=norman --load_model=models/scgpt-pretrained/scGPT_human --transformer_encoder_control=True --save_dir=save/transformer_encoder_control/norman_run_$n/
    python runner.py --mode=train --data_name=replogle_k562_essential --filter_perturbations=True --load_model=models/scgpt-pretrained/scGPT_human --transformer_encoder_control=True --save_dir=save/transformer_encoder_control/replogle_k562_essential_run_$n/ 
    python runner.py --mode=train --data_name=adam_corrected --load_model=models/scgpt-pretrained/scGPT_human --transformer_encoder_control=True --save_dir=save/transformer_encoder_control/adam_corrected_run_$n/
    python runner.py --mode=train --data_name=adam_corrected_upr --load_model=models/scgpt-pretrained/scGPT_human --transformer_encoder_control=True --save_dir=save/transformer_encoder_control/adam_corrected_upr_run_$n/
    done 

##use LoRa for finetuning 
lora_ranks=(800)
for lora_rank in "${lora_ranks[@]}"
    do 
    for n in {1..10}
        do
        python runner.py --mode=train --data_name=adamson --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/LoRa=${lora_rank}/adamson_run_${n} --use_lora=True --lora_rank=$lora_rank 
        python runner.py --mode=train --data_name=norman --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/LoRa=${lora_rank}/norman_run_${n}  --use_lora=True --lora_rank=$lora_rank 
        python runner.py --mode=train --data_name=replogle_k562_essential --filter_perturbations=True --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/LoRa=${lora_rank}/replogle_run_${n}  --use_lora=True --lora_rank=$lora_rank 
        python runner.py --mode=train --data_name=adam_corrected --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/LoRa=${lora_rank}/adam_corrected_run_${n}  --use_lora=True --lora_rank=$lora_rank 
        python runner.py --mode=train --data_name=adam_corrected_upr --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/LoRa=${lora_rank}/adam_corrected_upr_run_${n}  --use_lora=True --lora_rank=$lora_rank 
        done
    done

#train simple affine model (w/ pretraining - keep the gene embeddings from scGPT foundation model)
for n in {1..10}
    do
    python runner.py --mode=train --model_type=simple_affine --fixed_seed=False --data_name=adamson --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/simple_affine_with_pretraining/adamson_run_$n/  
    python runner.py --mode=train --model_type=simple_affine --fixed_seed=False --data_name=norman --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/simple_affine_with_pretraining/norman_run_$n/  
    python runner.py --mode=train --model_type=simple_affine --fixed_seed=False --data_name=replogle_k562_essential --filter_perturbations=True --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/simple_affine_with_pretraining/replogle_k562_essential_run_$n/  
    python runner.py --mode=train --model_type=simple_affine --fixed_seed=False --data_name=adam_corrected --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/simple_affine_with_pretraining/adam_corrected_run_$n/  
    python runner.py --mode=train --model_type=simple_affine --fixed_seed=False --data_name=adam_corrected_upr --load_model=models/scgpt-pretrained/scGPT_human --save_dir=save/simple_affine_with_pretraining/adam_corrected_upr_run_$n/  
    done

##=================================================================================================================================================
##=================================================================================================================================================
##various analyses using best front running models of the 10 independently trained (see find_best_models())
# # ##scGPT front running models (no pre-training)
python runner.py --mode=analysis --data_name=adam_corrected --load_model=save/no_pretraining/adam_corrected_run_6 --save_dir=save/null
python runner.py --mode=analysis --data_name=adam_corrected_upr --load_model=save/no_pretraining/adam_corrected_upr_run_3 --save_dir=save/null
python runner.py --mode=analysis --data_name=adamson --load_model=save/no_pretraining/adamson_run_4 --save_dir=save/null
python runner.py --mode=analysis --data_name=norman --load_model=save/no_pretraining/norman_run_6 --save_dir=save/null
python runner.py --mode=analysis --data_name=replogle_k562_essential --filter_perturbations=True --load_model=save/no_pretraining/replogle_k562_essential_run_4 --save_dir=save/null

# # ##scGPT front running models (with pre-training)
python runner.py --mode=analysis --data_name=adam_corrected --load_model=save/default_config_baseline/adam_corrected_run_1 --save_dir=save/null
python runner.py --mode=analysis --data_name=adam_corrected_upr --load_model=save/default_config_baseline/adam_corrected_upr_run_1 --save_dir=save/null
python runner.py --mode=analysis --data_name=adamson --load_model=save/default_config_baseline/adamson_run_9 --save_dir=save/null
python runner.py --mode=analysis --data_name=norman --load_model=save/default_config_baseline/norman_run_2 --save_dir=save/null
python runner.py --mode=analysis --data_name=replogle_k562_essential --filter_perturbations=True --load_model=save/default_config_baseline/replogle_k562_essential_run_8 --save_dir=save/null

# # ##simple affine (no pretraining)
python runner.py --mode=analysis --data_name=adam_corrected --model_type=simple_affine --load_model=save/simple_affine/adam_corrected_run_2 --save_dir=save/null
python runner.py --mode=analysis --data_name=adam_corrected_upr --model_type=simple_affine --load_model=save/simple_affine/adam_corrected_upr_run_5 --save_dir=save/null
python runner.py --mode=analysis --data_name=adamson --model_type=simple_affine --load_model=save/simple_affine/adamson_run_8 --save_dir=save/null
python runner.py --mode=analysis --data_name=norman --model_type=simple_affine --load_model=save/simple_affine/norman_run_5 --save_dir=save/null
python runner.py --mode=analysis --data_name=replogle_k562_essential --filter_perturbations=True --model_type=simple_affine --load_model=save/simple_affine/replogle_k562_essential_run_8 --save_dir=save/null

# # ##simple affine (with pretraining)
python runner.py --mode=analysis --data_name=adam_corrected --model_type=simple_affine --load_model=save/simple_affine_with_pretraining/adam_corrected_run_2 --save_dir=save/null
python runner.py --mode=analysis --data_name=adam_corrected_upr --model_type=simple_affine --load_model=save/simple_affine_with_pretraining/adam_corrected_upr_run_7 --save_dir=save/null
python runner.py --mode=analysis --data_name=adamson --model_type=simple_affine --load_model=save/simple_affine_with_pretraining/adamson_run_9 --save_dir=save/null
python runner.py --mode=analysis --data_name=norman --model_type=simple_affine --load_model=save/simple_affine_with_pretraining/norman_run_9 --save_dir=save/null
python runner.py --mode=analysis --data_name=replogle_k562_essential --filter_perturbations=True --model_type=simple_affine --load_model=save/simple_affine_with_pretraining/replogle_k562_essential_run_8 --save_dir=save/null

python analysis.py
##=================================================================================================================================================
##=================================================================================================================================================

echo "Run completed at:- "
date