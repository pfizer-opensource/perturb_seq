#!/bin/bash
##bash script for creating a conda environment called perturb_seq with all necessary software, as well as scGPT model files

host=$(hostname)
if [[ $host =~ "gamma" ]]; then
    echo "host is gamma"
    source ~/miniconda3/etc/profile.d/conda.sh
else
    echo "host is not gamma"
    source ~/.bashrc
fi 

##==========================================================================================
##if you need conda, then please install it first at: https://docs.anaconda.com/free/miniconda/ 
##it'll look something like this:
## mkdir -p ~/miniconda3
## wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
## bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
## rm -rf ~/miniconda3/miniconda.sh
## ~/miniconda3/bin/conda init bash
## ~/miniconda3/bin/conda init zsh
##==========================================================================================

##==========================================================================================
##build from loot_methods.yml and modify
conda env create -f loot_methods.yml -n perturb_seq 
conda activate perturb_seq
which pip
pip uninstall torch-sparse
pip uninstall torch-scatter
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
pip install dcor
pip install peft
pip install --upgrade huggingface-hub
pip install matplotlib-venn
pip install venn 
pip install pot
##==========================================================================================

##==========================================================================================
##get scGPT models from the google drive that scGPT authors made: https://drive.google.com/file/d/15TEZmd2cZCrHwgfE424fgQkGUZCXiYrR/view?usp=drive_link
##alternatively you can run this:
mkdir models/
mkdir models/scgpt-pretrained/
mkdir models/scgpt-pretrained/scGPT_CP/
cd models/scgpt-pretrained/scGPT_CP/
pip install gdown
gdown 1jfT_T5n8WNbO9QZcLWObLdRG8lYFKH-Q
gdown 15TEZmd2cZCrHwgfE424fgQkGUZCXiYrR
gdown 1x1SfmFdI-zcocmqWAd7ZTC9CTEAVfKZq
cd ..
##scGPT's perturb seq tutorial makes use of this model instead:
mkdir scGPT_human/
gdown 14AebJfGOUF047Eg40hk57HCtrb0fyDTm
gdown 1H3E_MJ-Dl36AQV6jLbna2EdvgPaqvqcC
gdown 1hh2zGKyWAx3DyovD30GStZ3QlzmSqdk1
cd ..
cd ..
##make relevant directories
mkdir outputs/
mkdir pickles/
mkdir logs/
mkdir gears_models/ ##will need to train your own models using gears_runner.py
mkdir data/
mkdir figures/
mkdir save/ ##will need to train your own scGPT models using runner.py
##==========================================================================================

