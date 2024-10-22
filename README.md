# Simple controls exceed best deep learning predictions and reveal foundation model effectiveness for predicting genetic perturbations

Daniel R. Wong, Abby Hill, Robert Moccia

correspondence: daniel.wong@pfizer.com

Machine Learning and Computational Sciences, Pfizer Worldwide Research Development and Medical, 610 Main Street, Cambridge, Massachusetts 02139, USA

## Software Installation
We build off scGPT version 0.2.1 https://github.com/bowang-lab/scGPT
and GEARS 0.1.2 https://github.com/snap-stanford/GEARS 
We provide a .yml file called loot_methods.yml and an install script that will create a conda environment called perturb_seq with all the necesssary packages

## Scripts 
runner.py: runner script for training, evaluating, and analyzing results
analysis.py: script for generating figures and doing one-off analyses of files generated from runner.py
library.py: contains core function definitions
gears_runner.py: script for training GEARS models
eval.sh: example SLURM script for running jobs 
parallel_eval.sh: parallel way of running SLURM jobs
simple_affine.py: contains class definition for the Simple Affine model 